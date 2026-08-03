import json
import logging
import os
import re

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
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

MAX_RESULT_CHARS = 8000  # keep SQL results from blowing up the model context
TELEGRAM_MSG_LIMIT = 4096
LANGUAGES = {"ES": "Spanish", "EN": "English", "PT": "Portuguese"}

hasher = PasswordHasher()

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

async def get_db_schema(pool) -> str:
    """Fetch public schema structure."""
    query = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = $1
        ORDER BY table_name, ordinal_position;
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, DB_SCHEMA)
    schema_dict = {}
    for row in rows:
        schema_dict.setdefault(row["table_name"], []).append(f"{row['column_name']} ({row['data_type']})")
    return "\n".join([f"Table `{t}`: {', '.join(c)}" for t, c in schema_dict.items()])

async def org_fetch(pool, org_id: str, query: str, *args):
    """Read-only fetch with the RLS org scope applied."""
    async with pool.acquire() as conn:
        async with conn.transaction(readonly=True):
            await conn.execute("SELECT set_config('app.org_id', $1, true)", org_id)
            return await conn.fetch(query, *args)

async def execute_sql(pool, sql_query: str, org_id: str) -> str:
    """Run SQL read-only; RLS (see rls.sql) hides rows outside org_id."""
    logger.info(f"SQL [{org_id}]: {sql_query}")
    try:
        rows = await org_fetch(pool, org_id, sql_query)
        logger.info(f"SQL result: {len(rows)} rows")
        result = json.dumps([dict(row) for row in rows], default=str)
        if len(result) > MAX_RESULT_CHARS:
            result = result[:MAX_RESULT_CHARS] + f'"] (truncated, {len(rows)} rows total)'
        return result
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        return json.dumps({"error": str(e)})

async def query_pipeline(user_prompt: str, pool, schema: str, org_id: str, locale: str, scope: str = "") -> tuple:
    """Orchestrates LLM -> SQL -> LLM. Returns (reply, executed_sql or None)."""
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
Always answer the user in {LANGUAGES.get(locale, "English")}."""

    headers = {"Authorization": f"Bearer {LITELLM_MASTER_KEY}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {"model": MODEL_NAME, "messages": messages, "tools": [RUN_SQL_TOOL], "tool_choice": "auto"}
        resp = await client.post(LITELLM_URL, json=payload, headers=headers)
        resp.raise_for_status()
        message = resp.json()["choices"][0]["message"]

        content = message.get("content") or ""
        if message.get("tool_calls"):
            tool_call = message["tool_calls"][0]
            sql_query = json.loads(tool_call["function"]["arguments"]).get("sql_query", "")
            messages.extend([message, {"role": "tool", "tool_call_id": tool_call["id"],
                                       "content": await execute_sql(pool, sql_query, org_id)}])
        else:
            # ponytail: local models often emit the tool call as plain text
            # instead of a structured tool_calls entry — salvage the SQL
            m = re.search(r'"sql_query"\s*:\s*("(?:[^"\\]|\\.)*")', content)
            if not m:
                logger.info("LLM answered without SQL")
                return content or "No response generated.", None
            logger.info("Salvaged inline tool call from text response")
            sql_query = json.loads(m.group(1))
            db_result = await execute_sql(pool, sql_query, org_id)
            messages.extend([
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Query result: {db_result}\nAnswer the original question using this data."},
            ])

        final_resp = await client.post(LITELLM_URL, json={"model": MODEL_NAME, "messages": messages}, headers=headers)
        final_resp.raise_for_status()
        return final_resp.json()["choices"][0]["message"]["content"], sql_query

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
        echo = ""
        if is_voice:
            voice_file = await context.bot.get_file(update.message.voice.file_id)
            voice_bytes = await voice_file.download_as_bytearray()
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(WHISPER_URL, files={"file": ("voice.ogg", voice_bytes)}, data={"model": "whisper-1"})
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
        reply, executed_sql = await query_pipeline(prompt, pool, context.bot_data["db_schema"], org_id,
                                                   context.user_data.get("locale") or "ES", scope)
        if DEBUG_MODE and executed_sql:
            reply = f"{reply or ''}\n\n🔧 SQL:\n{executed_sql}"
        await status.edit_text((echo + (reply or "No response generated."))[:TELEGRAM_MSG_LIMIT])

    except Exception as e:
        logger.error(f"Handler Error: {e}")
        if DEBUG_MODE:
            await status.edit_text(f"⚠️ DEBUG ERROR: {str(e)}"[:TELEGRAM_MSG_LIMIT])
        else:
            await status.edit_text("An error occurred while processing your query.")
    finally:
        context.user_data["busy"] = False

async def post_init(app):
    """Create the pool inside PTB's event loop, cache the schema, set the command menu."""
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    app.bot_data["db_pool"] = pool
    app.bot_data["db_schema"] = await get_db_schema(pool)
    await app.bot.set_my_commands([
        BotCommand("login", "Log in: /login <email> <password>"),
        BotCommand("farm", "Choose the farm/herd your questions are about"),
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
    app.add_handler(CommandHandler("logout", logout))
    app.add_handler(CommandHandler("farm", farm_menu))
    app.add_handler(CallbackQueryHandler(menu_callback, pattern=r"^(farm|herd):"))
    app.add_handler(MessageHandler((filters.TEXT & ~filters.COMMAND) | filters.VOICE, handle_message))

    logger.info("Bot started...")
    app.run_polling()

if __name__ == "__main__":
    main()
