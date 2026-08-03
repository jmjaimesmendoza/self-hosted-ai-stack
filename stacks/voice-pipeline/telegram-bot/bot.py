import json
import logging
import os
import re

import anthropic
import asyncpg
import httpx
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "")
WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper:9000/v1/audio/transcriptions")
LITELLM_URL = os.getenv("LITELLM_URL", "http://litellm:4000/v1/chat/completions")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/postgres")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-coder:14b")
DB_SCHEMA = os.getenv("DB_SCHEMA", "tractor")
# comma-separated table allowlist to keep the prompt small; empty = all visible tables
SCHEMA_TABLES = [t.strip() for t in os.getenv("SCHEMA_TABLES", "").split(",") if t.strip()]
NUM_CTX = int(os.getenv("NUM_CTX", "16384"))  # ollama context window per request
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "300"))  # cold model load can take minutes on CPU
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))  # cap runaway generation (ollama default is unlimited)
# qwen3 soft switch: skip multi-minute <think> spirals; harmless noise for other models
NO_THINK = os.getenv("NO_THINK", "true").lower() == "true"
# Claude API path: enabled when a key is present; otherwise the LiteLLM path runs
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
# Kimi K3 via Moonshot's Anthropic-compatible endpoint — same pipeline, different base_url
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k3")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/anthropic")
if ANTHROPIC_API_KEY:
    anthropic_client, LLM_MODEL = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY), ANTHROPIC_MODEL
elif KIMI_API_KEY:
    anthropic_client, LLM_MODEL = anthropic.AsyncAnthropic(api_key=KIMI_API_KEY, base_url=KIMI_BASE_URL), KIMI_MODEL
else:
    anthropic_client, LLM_MODEL = None, None
LLM_BACKEND = "anthropic" if ANTHROPIC_API_KEY else "kimi" if KIMI_API_KEY else "local"

MAX_RESULT_CHARS = 8000  # keep SQL results from blowing up the model context
TELEGRAM_MSG_LIMIT = 4096
MAX_SQL_ATTEMPTS = 3     # self-correction: model sees SQL errors and retries
ROUTING_MIN_TABLES = 15  # skip stage-1 table routing for schemas this small
HISTORY_MAX = 10         # messages kept per user (5 exchanges) for follow-up questions
LANGUAGES = {"ES": "Spanish", "EN": "English", "PT": "Portuguese"}

hasher = PasswordHasher()

# Known-good question -> SQL pairs; :farmId/:herdId are placeholders the model
# must replace with the scoped UUIDs (or drop/replace joins when unscoped).
FEW_SHOTS = """
EXAMPLE QUERIES — follow these patterns. Replace :farmId and :herdId with the
selected farm/herd UUIDs from context; if no herd is selected drop the lots
join; if no farm is selected join farms and match by name with ILIKE.
Note: "ageGroup" is camelCase and must be double-quoted; enum values are
uppercase (sex: MALE/FEMALE; ageGroup: CALF/YEARLING/HEIFER/STEER/COW/BULL).

-- ¿Cuál es la búfala más vieja?
SELECT a.physical_id, a.name, a.birth_date,
       DATE_PART('year', AGE(a.birth_date)) AS edad_anios
FROM animals a
JOIN lots l ON l.id = a.lot_id AND l.herd_id = :herdId AND l.is_deleted = false
WHERE a.farm_id = :farmId AND a.is_deleted = false AND a.is_active = true
  AND a.sex = 'FEMALE' AND a."ageGroup" = 'COW' AND a.birth_date IS NOT NULL
ORDER BY a.birth_date ASC LIMIT 1;

-- ¿Cuántas búfalas preñadas y vacías hay? (último examen por animal; sin examen = sin chequeo)
SELECT COUNT(*) FILTER (WHERE ex.result = 'PREGNANT')          AS prenadas,
       COUNT(*) FILTER (WHERE ex.result = 'NOT_PREGNANT')      AS vacias,
       COUNT(*) FILTER (WHERE ex.result = 'POSSIBLY_PREGNANT') AS posiblemente_prenadas,
       COUNT(*) FILTER (WHERE ex.result IS NULL)               AS sin_chequeo
FROM animals a
JOIN lots l ON l.id = a.lot_id AND l.herd_id = :herdId AND l.is_deleted = false
LEFT JOIN LATERAL (
  SELECT ge.result FROM gynecological_exams ge
  WHERE ge.animal_id = a.id AND ge.is_deleted = false
  ORDER BY ge.date DESC LIMIT 1
) ex ON true
WHERE a.farm_id = :farmId AND a.is_deleted = false AND a.is_active = true
  AND a.sex = 'FEMALE' AND a."ageGroup" IN ('COW', 'HEIFER');

-- ¿Cuántos bautes (machos) y bautas (hembras) hay? (añojos = YEARLING)
SELECT COUNT(*) FILTER (WHERE a.sex = 'MALE')   AS bautes,
       COUNT(*) FILTER (WHERE a.sex = 'FEMALE') AS bautas
FROM animals a
JOIN lots l ON l.id = a.lot_id AND l.herd_id = :herdId AND l.is_deleted = false
WHERE a.farm_id = :farmId AND a.is_deleted = false AND a.is_active = true
  AND a."ageGroup" = 'YEARLING';

-- ¿Cuántas búfalas secas y en ordeño? (en ordeño = lactancia abierta: inicio sin fin)
SELECT COUNT(*) FILTER (WHERE ol.animal_id IS NOT NULL) AS en_ordeno,
       COUNT(*) FILTER (WHERE ol.animal_id IS NULL)     AS secas
FROM animals a
JOIN lots l ON l.id = a.lot_id AND l.herd_id = :herdId AND l.is_deleted = false
LEFT JOIN (
  SELECT DISTINCT ls.animal_id
  FROM lactation_start_events ls
  LEFT JOIN lactation_end_events le
    ON le.animal_id = ls.animal_id AND le.birth_event_id = ls.birth_event_id
   AND le.is_deleted = false
  WHERE ls.is_deleted = false AND le.animal_id IS NULL
) ol ON ol.animal_id = a.id
WHERE a.farm_id = :farmId AND a.is_deleted = false AND a.is_active = true
  AND a.sex = 'FEMALE' AND a."ageGroup" = 'COW';

-- ¿Cuántas novillas hay?
SELECT COUNT(*)
FROM animals a
JOIN lots l ON l.id = a.lot_id AND l.herd_id = :herdId AND l.is_deleted = false
WHERE a.farm_id = :farmId AND a.is_deleted = false AND a.is_active = true
  AND a.sex = 'FEMALE' AND a."ageGroup" = 'HEIFER';

-- ¿Cuál es el peso promedio de los bautes? (último pesaje por animal)
SELECT ROUND(AVG(lw.weight), 1) AS peso_promedio_kg,
       COUNT(*)                 AS animales_pesados
FROM animals a
JOIN lots l ON l.id = a.lot_id AND l.herd_id = :herdId AND l.is_deleted = false
JOIN LATERAL (
  SELECT we.weight FROM weigh_in_events we
  WHERE we.animal_id = a.id AND we.is_deleted = false
  ORDER BY we.date DESC LIMIT 1
) lw ON true
WHERE a.farm_id = :farmId AND a.is_deleted = false AND a.is_active = true
  AND a.sex = 'MALE' AND a."ageGroup" = 'YEARLING';

-- ¿Cuántos destetes en los últimos 6 meses?
SELECT COUNT(*)
FROM weigh_in_events we
JOIN animals a ON a.id = we.animal_id AND a.farm_id = :farmId AND a.is_deleted = false
JOIN lots l ON l.id = a.lot_id AND l.herd_id = :herdId AND l.is_deleted = false
WHERE we.type = 'WEANING' AND we.is_deleted = false
  AND we.date >= now() - interval '6 months';
"""

RUN_SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_sql_query",
        "description": "Execute a SQL query against the PostgreSQL database.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {"type": "string", "description": "Valid PostgreSQL SELECT query."}
            },
            "required": ["sql_query"],
        },
    },
}

async def get_db_schema(pool) -> dict:
    """Fetch schema as {table_name: one-line description}."""
    query = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = $1
          AND ($2::text[] IS NULL OR table_name = ANY($2))
        ORDER BY table_name, ordinal_position;
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, DB_SCHEMA, SCHEMA_TABLES or None)
    schema_dict = {}
    for row in rows:
        schema_dict.setdefault(row["table_name"], []).append(f"{row['column_name']} ({row['data_type']})")
    return {t: f"Table `{t}`: {', '.join(c)}" for t, c in schema_dict.items()}

async def org_fetch(pool, org_id: str, query: str, *args):
    """Read-only fetch with the RLS org scope applied."""
    async with pool.acquire() as conn:
        async with conn.transaction(readonly=True):
            await conn.execute("SELECT set_config('app.org_id', $1, true)", org_id)
            return await conn.fetch(query, *args)

async def execute_sql(pool, sql_query: str, org_id: str) -> tuple:
    """Run SQL read-only; RLS (see rls.sql) hides rows outside org_id.
    Returns (result_json, ok)."""
    logger.info(f"SQL [{org_id}]: {sql_query}")
    try:
        rows = await org_fetch(pool, org_id, sql_query)
        logger.info(f"SQL result: {len(rows)} rows")
        result = json.dumps([dict(row) for row in rows], default=str)
        if len(result) > MAX_RESULT_CHARS:
            result = result[:MAX_RESULT_CHARS] + f'"] (truncated, {len(rows)} rows total)'
        return result, True
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        return json.dumps({"error": str(e)}), False

# Same tool in the Anthropic Messages API shape (no "function" wrapper)
CLAUDE_SQL_TOOL = {
    "name": "run_sql_query",
    "description": "Execute a SQL query against the PostgreSQL database.",
    "input_schema": RUN_SQL_TOOL["function"]["parameters"],
}

async def claude_query_pipeline(user_prompt: str, pool, schema_map: dict, org_id: str,
                                locale: str, scope: str = "", history: list = None) -> tuple:
    """SQL generation via the Claude API. Returns (reply, last_executed_sql or None)."""
    schema = "\n".join(schema_map.values())
    # Static block (identical across users/turns) is prompt-cached; the
    # per-user specifics go in a second block after the cache breakpoint.
    static_text = f"""You are an AI with access to a PostgreSQL database.
DATABASE SCHEMA:
{schema}
This is a multi-tenant database. Every query MUST filter by the current user's
organization_id (given below), directly or via a join to a table that has it.
Rows with is_deleted = true must be excluded.
All ids are UUIDs — NEVER guess an id from a name. When the user mentions a farm,
herd, animal, or person by name, match it with a join and ILIKE (e.g.
JOIN farms f ON ... WHERE f.name ILIKE '%san rafael%').
Construct a valid SELECT query and use the 'run_sql_query' tool. NEVER use destructive queries.
If the question is ambiguous or missing information you need (which farm, which period,
which animals), ask a short clarifying question instead of guessing.
{FEW_SHOTS}"""
    dynamic_text = (f"The current user's organization_id is '{org_id}'.{scope}\n"
                    f"Always answer the user in {LANGUAGES.get(locale, 'English')}.")
    system = [
        {"type": "text", "text": static_text, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": dynamic_text},
    ]

    messages = (history or []) + [{"role": "user", "content": user_prompt}]
    executed_sql = None
    sql_runs = 0
    while True:
        kwargs = dict(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            tools=[CLAUDE_SQL_TOOL],
        )
        if sql_runs >= MAX_SQL_ATTEMPTS:
            kwargs["tool_choice"] = {"type": "none"}  # force a final answer
        for retry in range(3):
            try:
                response = await anthropic_client.messages.create(**kwargs)
                break
            except json.JSONDecodeError:
                # ponytail: Moonshot's endpoint sometimes returns 200 with an empty
                # body on slow requests — the SDK won't retry a "successful" status
                logger.warning(f"Non-JSON body from LLM API, retrying ({retry + 1}/3)")
        else:
            return "The AI service returned an empty response — please try again.", executed_sql

        if response.stop_reason == "refusal":
            logger.warning("Claude declined the request (stop_reason=refusal)")
            return "The model declined to answer this question.", executed_sql

        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_blocks:
            reply = "".join(b.text for b in response.content if b.type == "text")
            return reply or "No response generated.", executed_sql

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in tool_blocks:
            sql_query = block.input.get("sql_query", "")
            executed_sql = sql_query
            db_result, ok = await execute_sql(pool, sql_query, org_id)
            if not ok:
                logger.info(f"SQL failed (run {sql_runs + 1}/{MAX_SQL_ATTEMPTS}), Claude will retry")
            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": db_result, "is_error": not ok})
            sql_runs += 1
        messages.append({"role": "user", "content": tool_results})

async def pick_tables(client, headers, question: str, table_names: list) -> list:
    """Stage 1 of schema routing: cheap call with table names only."""
    prompt = (f"Database tables:\n{', '.join(table_names)}\n\n"
              f'Which of these tables are needed to answer: "{question}"?\n'
              "Reply with the table names only, comma-separated.")
    resp = await client.post(LITELLM_URL,
                             json={"model": MODEL_NAME, "num_ctx": NUM_CTX, "max_tokens": MAX_TOKENS,
                                   "messages": [{"role": "user", "content": prompt}]},
                             headers=headers)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"] or ""
    # match known names in the reply — no fragile JSON parsing of a small model
    return [t for t in table_names if re.search(rf"\b{re.escape(t)}\b", text)]

async def query_pipeline(user_prompt: str, pool, schema_map: dict, org_id: str, locale: str,
                         scope: str = "", history: list = None) -> tuple:
    """Dispatch to the Claude API when a key is configured, else local LiteLLM."""
    if anthropic_client:
        return await claude_query_pipeline(user_prompt, pool, schema_map, org_id, locale, scope, history)
    return await litellm_query_pipeline(user_prompt, pool, schema_map, org_id, locale, scope, history)

async def litellm_query_pipeline(user_prompt: str, pool, schema_map: dict, org_id: str, locale: str,
                                 scope: str = "", history: list = None) -> tuple:
    """Orchestrates routing -> LLM -> SQL (with self-correction) -> LLM.
    Returns (reply, last_executed_sql or None)."""
    headers = {"Authorization": f"Bearer {LITELLM_MASTER_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        tables = list(schema_map)
        if len(tables) > ROUTING_MIN_TABLES:
            try:
                picked = await pick_tables(client, headers, user_prompt, tables)
            except Exception as e:
                logger.warning(f"Table routing failed, using full schema: {e}")
                picked = []
            for t in ("farms", "herds"):  # scope filters reference these
                if t in schema_map and t not in picked:
                    picked.append(t)
            if picked:
                logger.info(f"Routing: {len(picked)}/{len(tables)} tables: {', '.join(picked)}")
                tables = picked
        schema = "\n".join(schema_map[t] for t in tables)

        system_prompt = f"""You are an AI with access to a PostgreSQL database.
DATABASE SCHEMA:
{schema}
This is a multi-tenant database. The current user's organization_id is '{org_id}'.{scope}
Every query MUST filter by organization_id = '{org_id}' (directly or via a join
to a table that has it). Rows with is_deleted = true must be excluded.
All ids are UUIDs — NEVER guess an id from a name. When the user mentions a farm,
herd, animal, or person by name, match it with a join and ILIKE (e.g.
JOIN farms f ON ... WHERE f.name ILIKE '%san rafael%').
Construct a valid SELECT query and use the 'run_sql_query' tool. NEVER use destructive queries.
If the question is ambiguous or missing information you need (which farm, which period,
which animals), ask a short clarifying question instead of guessing.
Always answer the user in {LANGUAGES.get(locale, "English")}.
{FEW_SHOTS}{" /no_think" if NO_THINK else ""}"""

        # ponytail: table routing above only sees the current question, not history —
        # feed the last user turn into pick_tables if follow-ups route to wrong tables
        messages = ([{"role": "system", "content": system_prompt}] + (history or [])
                    + [{"role": "user", "content": user_prompt}])
        executed_sql = None

        for attempt in range(MAX_SQL_ATTEMPTS):
            resp = await client.post(LITELLM_URL,
                                     json={"model": MODEL_NAME, "messages": messages, "tools": [RUN_SQL_TOOL],
                                           "tool_choice": "auto", "num_ctx": NUM_CTX, "max_tokens": MAX_TOKENS},
                                     headers=headers)
            resp.raise_for_status()
            message = resp.json()["choices"][0]["message"]
            content = message.get("content") or ""

            if message.get("tool_calls"):
                tool_call = message["tool_calls"][0]
                sql_query = json.loads(tool_call["function"]["arguments"]).get("sql_query", "")
                db_result, ok = await execute_sql(pool, sql_query, org_id)
                messages.extend([message, {"role": "tool", "tool_call_id": tool_call["id"], "content": db_result}])
            else:
                # ponytail: local models often emit the tool call as plain text
                # instead of a structured tool_calls entry — salvage the SQL
                m = re.search(r'"sql_query"\s*:\s*("(?:[^"\\]|\\.)*")', content)
                if not m:
                    logger.info("LLM answered without SQL")
                    return content or "No response generated.", executed_sql
                logger.info("Salvaged inline tool call from text response")
                sql_query = json.loads(m.group(1))
                db_result, ok = await execute_sql(pool, sql_query, org_id)
                messages.extend([{"role": "assistant", "content": content},
                                 {"role": "user", "content": f"Query result: {db_result}"}])

            executed_sql = sql_query
            if ok:
                break
            # self-correction: the model sees the error and gets another shot
            logger.info(f"SQL failed (attempt {attempt + 1}/{MAX_SQL_ATTEMPTS}), asking model to fix it")
            messages.append({"role": "user",
                             "content": "The query failed with the error above. Fix the SQL "
                                        "(check table and column names against the schema) and call run_sql_query again."})

        messages.append({"role": "user", "content": "Answer the original question using the query results, "
                                                    f"in {LANGUAGES.get(locale, 'English')}."})
        final_resp = await client.post(LITELLM_URL, json={"model": MODEL_NAME, "messages": messages,
                                                          "num_ctx": NUM_CTX, "max_tokens": MAX_TOKENS}, headers=headers)
        final_resp.raise_for_status()
        return final_resp.json()["choices"][0]["message"]["content"], executed_sql

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Log in with your account: /login <email> <password>")

async def login(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Authenticate against the app's users table (argon2)."""
    try:
        await update.message.delete()  # don't leave the password in chat history
    except Exception:
        pass
    chat = update.effective_chat
    if len(context.args) != 2:
        await chat.send_message("Usage: /login <email> <password>")
        return

    email, password = context.args
    pool = context.bot_data["db_pool"]
    try:
        # SECURITY DEFINER function from rls.sql: the bot role itself
        # cannot read password hashes.
        row = await pool.fetchrow("SELECT * FROM bot_login($1)", email)
    except asyncpg.UndefinedFunctionError:
        await chat.send_message("Login backend not set up: run rls.sql against the database.")
        return
    verified = False
    if row:
        try:
            hasher.verify(row["password"], password)
            verified = True
        except VerifyMismatchError:
            pass
    if not verified:
        logger.warning(f"Failed login attempt for {email} (tg user {update.effective_user.id})")
        await chat.send_message("Invalid credentials.")
        return
    logger.info(f"Login: {email} (tg user {update.effective_user.id})")

    context.user_data.pop("history", None)  # fresh session, fresh conversation
    context.user_data["org_id"] = row["organization_id"]
    context.user_data["db_user_id"] = row["id"]
    context.user_data["name"] = row["name"]
    context.user_data["email"] = email
    context.user_data["locale"] = row["locale"]
    await chat.send_message(f"Logged in as {row['name']}. Send me a question (text or voice).")

async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("org_id"):
        await update.message.reply_text("Not logged in. Use /login <email> <password>")
        return
    d = context.user_data
    farm = d.get("farm", {}).get("name", "all farms")
    herd = d.get("herd", {}).get("name", "all herds")
    await update.message.reply_text(
        f"Logged in as {d['name']} ({d['email']})\nOrganization: {d['org_id']}\n"
        f"Language: {d.get('locale') or 'ES'}\nScope: {farm} / {herd}"
    )

async def new_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("history", None)
    await update.message.reply_text("New conversation started — previous questions are forgotten.")

async def logout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text("Logged out.")

async def farm_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Pick a farm (then a herd) to auto-scope all questions."""
    org_id = context.user_data.get("org_id")
    if not org_id:
        await update.message.reply_text("Please log in first: /login <email> <password>")
        return
    rows = await org_fetch(context.bot_data["db_pool"], org_id,
                           "SELECT id, name FROM farms WHERE is_deleted = false ORDER BY name")
    if not rows:
        await update.message.reply_text("No farms found for your organization.")
        return
    keyboard = [[InlineKeyboardButton(r["name"], callback_data=f"farm:{r['id']}")] for r in rows]
    keyboard.append([InlineKeyboardButton("🌐 All farms (clear scope)", callback_data="farm:all")])
    await update.message.reply_text("Select a farm:", reply_markup=InlineKeyboardMarkup(keyboard))

async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle farm/herd button presses."""
    q = update.callback_query
    await q.answer()
    org_id = context.user_data.get("org_id")
    if not org_id:
        await q.edit_message_text("Session expired — please /login again.")
        return
    pool = context.bot_data["db_pool"]
    kind, val = q.data.split(":", 1)

    if kind == "farm":
        context.user_data.pop("herd", None)
        if val == "all":
            context.user_data.pop("farm", None)
            await q.edit_message_text("Scope cleared — questions cover all farms.")
            return
        rows = await org_fetch(pool, org_id, "SELECT name FROM farms WHERE id = $1", val)
        if not rows:
            await q.edit_message_text("Farm not found.")
            return
        context.user_data["farm"] = {"id": val, "name": rows[0]["name"]}
        herds = await org_fetch(pool, org_id,
                                "SELECT id, name FROM herds WHERE farm_id = $1 AND is_deleted = false ORDER BY name", val)
        if not herds:
            await q.edit_message_text(f"Scope: farm “{rows[0]['name']}” (it has no herds).")
            return
        keyboard = [[InlineKeyboardButton(h["name"], callback_data=f"herd:{h['id']}")] for h in herds]
        keyboard.append([InlineKeyboardButton("🌐 All herds", callback_data="herd:all")])
        await q.edit_message_text(f"Farm: {rows[0]['name']}. Select a herd:",
                                  reply_markup=InlineKeyboardMarkup(keyboard))
    elif kind == "herd":
        farm = context.user_data.get("farm")
        if val == "all" or not farm:
            context.user_data.pop("herd", None)
            await q.edit_message_text(f"Scope: farm “{farm['name']}”, all herds." if farm else "Scope cleared.")
            return
        rows = await org_fetch(pool, org_id, "SELECT name FROM herds WHERE id = $1", val)
        if not rows:
            await q.edit_message_text("Herd not found.")
            return
        context.user_data["herd"] = {"id": val, "name": rows[0]["name"]}
        await q.edit_message_text(f"Scope: farm “{farm['name']}”, herd “{rows[0]['name']}”.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generic handler for text/voice."""
    org_id = context.user_data.get("org_id")
    if not org_id:
        await update.message.reply_text("Please log in first: /login <email> <password>")
        return

    if context.user_data.get("busy"):
        await update.message.reply_text("Still working on your previous question — one moment.")
        return
    context.user_data["busy"] = True

    pool = context.bot_data["db_pool"]
    is_voice = bool(update.message.voice)
    # one status message, edited in place as stages complete
    status = await update.message.reply_text("🎙 Transcribing…" if is_voice else "🔍 Querying the database…")

    try:
        echo, prompt = "", ""
        if is_voice:
            voice_file = await context.bot.get_file(update.message.voice.file_id)
            voice_bytes = await voice_file.download_as_bytearray()
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(WHISPER_URL, files={"file": ("voice.ogg", bytes(voice_bytes))}, data={"model": "whisper-1"})
                r.raise_for_status()
                prompt = r.json().get("text", "").strip()
            if not prompt:
                await status.edit_text("Could not transcribe the voice message.")
                return
            echo = f"🗣 You asked: {prompt}\n\n"
            await status.edit_text(f"{echo}🔍 Querying the database…")
        else:
            prompt = update.message.text
        logger.info(f"Q [{context.user_data.get('email')}{' voice' if is_voice else ''}]: {prompt}")

        scope = ""
        farm, herd = context.user_data.get("farm"), context.user_data.get("herd")
        if farm:
            scope += (f"\nThe user pre-selected farm '{farm['name']}' (farms.id = '{farm['id']}'). "
                      "Scope every query to this farm unless the question names a different one.")
        if herd:
            scope += (f"\nWithin that farm, herd '{herd['name']}' (herds.id = '{herd['id']}') — "
                      "scope to this herd unless told otherwise.")
        if DEBUG_MODE:
            await update.effective_message.reply_text(f"🔧 LLM: {LLM_BACKEND} ({LLM_MODEL or 'litellm'})")
        reply, executed_sql = await query_pipeline(prompt, pool, context.bot_data["db_schema"], org_id,
                                                   context.user_data.get("locale") or "ES", scope,
                                                   context.user_data.get("history"))
        if reply:
            history = context.user_data.setdefault("history", [])
            history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
            del history[:-HISTORY_MAX]
        if DEBUG_MODE and executed_sql:
            reply = f"{reply or ''}\n\n🔧 SQL:\n{executed_sql}"
        await status.edit_text((echo + (reply or "No response generated."))[:TELEGRAM_MSG_LIMIT])

    except Exception as e:
        logger.exception("Handler Error")
        # record the failed turn so "try again" or a rephrase keeps its context
        if prompt:
            history = context.user_data.setdefault("history", [])
            history += [{"role": "user", "content": prompt},
                        {"role": "assistant",
                         "content": f"(No answer was produced — the request failed with {type(e).__name__}. "
                                    "The user may retry or rephrase.)"}]
            del history[:-HISTORY_MAX]
        if DEBUG_MODE:
            # repr(): timeouts and friends often have an empty str()
            await status.edit_text(f"⚠️ DEBUG ERROR: {e!r}"[:TELEGRAM_MSG_LIMIT])
        else:
            await status.edit_text("Something went wrong with that question. "
                                   "You can say “try again”, rephrase it, or narrow it down.")
    finally:
        context.user_data["busy"] = False

async def post_init(app):
    """Create the pool inside PTB's event loop, cache the schema, set the command menu."""
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    app.bot_data["db_pool"] = pool
    schema_map = await get_db_schema(pool)
    app.bot_data["db_schema"] = schema_map
    total = sum(len(v) for v in schema_map.values())
    if anthropic_client:
        name = "Claude API" if ANTHROPIC_API_KEY else "Kimi (Anthropic-compatible)"
        provider = f"{name} ({LLM_MODEL})"
    else:
        provider = f"LiteLLM ({MODEL_NAME}; num_ctx={NUM_CTX}; " \
                   f"routing {'on' if len(schema_map) > ROUTING_MIN_TABLES else 'off'})"
    logger.info(f"Schema: {len(schema_map)} tables, {total} chars (~{total // 4} tokens); LLM: {provider}")
    await app.bot.set_my_commands([
        BotCommand("login", "Log in: /login <email> <password>"),
        BotCommand("farm", "Choose the farm/herd your questions are about"),
        BotCommand("new", "Start a new conversation (forget previous questions)"),
        BotCommand("whoami", "Show who is logged in and current scope"),
        BotCommand("logout", "Log out"),
    ])
    logger.info("DB pool ready, schema cached.")

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("login", login))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("new", new_conversation))
    app.add_handler(CommandHandler("logout", logout))
    app.add_handler(CommandHandler("farm", farm_menu))
    app.add_handler(CallbackQueryHandler(menu_callback, pattern=r"^(farm|herd):"))
    app.add_handler(MessageHandler((filters.TEXT & ~filters.COMMAND) | filters.VOICE, handle_message))

    logger.info("Bot started...")
    app.run_polling()

if __name__ == "__main__":
    main()
