import asyncio
import contextlib
import html
import json
import logging
import os
import re
import time

import anthropic
import asyncpg
import httpx
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PersistenceInput,
    PicklePersistence,
    filters,
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # DEBUG adds the SDKs' own request/response tracing
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=LOG_LEVEL,
)
for _sdk in ("anthropic", "httpx", "httpcore"):  # they default to WARNING, so DEBUG never reaches them otherwise
    logging.getLogger(_sdk).setLevel(LOG_LEVEL)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "")
WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper:9000/v1/audio/transcriptions")
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY", "")  # `docker exec whisper whisper_manage --getkey`
LITELLM_URL = os.getenv("LITELLM_URL", "http://litellm:4000/v1/chat/completions")
DATABASE_URL = os.getenv("DATABASE_URL")  # no default: the fallback was a superuser URL, and owners bypass RLS
DB_ROLE = os.getenv("DB_ROLE", "speech_sql_user")  # post_init asserts we connect as this role (rls.sql)
# sessions survive restarts; mount a volume here in compose or logins die with every deploy
PERSIST_PATH = os.getenv("PERSIST_PATH", "bot_state.pickle")
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
# stream the Anthropic/Kimi call: a silent socket for a whole long generation gets dropped
LLM_STREAM = os.getenv("LLM_STREAM", "true").lower() == "true"
# Claude API path: enabled when a key is present; otherwise the LiteLLM path runs
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
# Kimi K3 via Moonshot's Anthropic-compatible endpoint — same pipeline, different base_url
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k3")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.ai/anthropic")
if ANTHROPIC_API_KEY:
    anthropic_client, LLM_MODEL = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=LLM_TIMEOUT), ANTHROPIC_MODEL
elif KIMI_API_KEY:
    anthropic_client, LLM_MODEL = anthropic.AsyncAnthropic(api_key=KIMI_API_KEY, base_url=KIMI_BASE_URL,
                                                           timeout=LLM_TIMEOUT), KIMI_MODEL
else:
    anthropic_client, LLM_MODEL = None, None
LLM_BACKEND = "anthropic" if ANTHROPIC_API_KEY else "kimi" if KIMI_API_KEY else "local"
# One connection pool for Whisper + LiteLLM calls (per-request clients paid a TLS handshake every turn)
http_client = httpx.AsyncClient(timeout=LLM_TIMEOUT)

LOG_BODY_CHARS = 2000    # cap logged response bodies
MAX_RESULT_CHARS = 8000  # keep SQL results from blowing up the model context
MAX_ROWS = 1000          # cap fetched rows; a model-emitted cartesian join must not stall the bot
TELEGRAM_MSG_LIMIT = 4096
MAX_SQL_ATTEMPTS = 3     # self-correction: model sees SQL errors and retries
ROUTING_MIN_TABLES = 15  # skip stage-1 table routing for schemas this small
HISTORY_MAX = 10         # messages kept per user (5 exchanges) for follow-up questions
LANGUAGES = {"ES": "Spanish", "EN": "English", "PT": "Portuguese"}
# (query ran, no query ran) — appended to replies so the user knows the turn is over
FOOTERS = {
    "ES": ("\n\n✅ Consulta finalizada — puedes hacer otra pregunta o pedir un ajuste.",
           "\n\n💬 No se consultó la base de datos — responde para continuar."),
    "EN": ("\n\n✅ Query complete — ask another question or request a tweak.",
           "\n\n💬 No database query was run — reply to continue."),
    "PT": ("\n\n✅ Consulta concluída — faça outra pergunta ou peça um ajuste.",
           "\n\n💬 Nenhuma consulta foi executada — responda para continuar."),
}

# All user-facing chrome (the LLM answer itself is localized via LANGUAGES in the prompt).
STRINGS = {
    "ES": {
        "start": "Inicia sesión con tu cuenta: /login <email> <contraseña>",
        "login_first": "Primero inicia sesión: /login <email> <contraseña>",
        "login_usage": "Uso: /login <email> <contraseña>",
        "login_bad": "Credenciales inválidas.",
        "login_ok": "Sesión iniciada como {name}. Envíame una pregunta (texto o voz).",
        "delete_warn": "⚠️ No pude borrar tu mensaje — bórralo manualmente para que tu contraseña no quede en el chat.",
        "busy": "Sigo trabajando en tu pregunta anterior — un momento.",
        "transcribing": "🎙 Transcribiendo…",
        "querying": "🔍 Consultando la base de datos…",
        "running_sql": "📊 Ejecutando la consulta…",
        "writing": "📝 Redactando la respuesta…",
        "transcribe_failed": "No pude transcribir el mensaje de voz.",
        "error": "Algo salió mal con esa pregunta. Puedes decir “intenta de nuevo”, reformularla o acotarla.",
        "error_generic": "Algo salió mal — inténtalo de nuevo.",
        "new_chat": "Nueva conversación — las preguntas anteriores quedan olvidadas.",
        "logged_out": "Sesión cerrada.",
        "select_farm": "Elige una finca:",
        "no_farms": "No hay fincas en tu organización.",
        "all_farms": "🌐 Todas las fincas (sin filtro)",
        "all_herds": "🌐 Todos los rebaños",
        "scope_cleared": "Filtro borrado — las preguntas cubren todas las fincas.",
        "farm_not_found": "Finca no encontrada.",
        "herd_not_found": "Rebaño no encontrado.",
        "select_herd": "Finca: {farm}. Elige un rebaño:",
        "scope_farm_no_herds": "Filtro: finca “{farm}” (no tiene rebaños).",
        "scope_farm_all_herds": "Filtro: finca “{farm}”, todos los rebaños.",
        "scope_farm_herd": "Filtro: finca “{farm}”, rebaño “{herd}”.",
        "session_expired": "Sesión expirada — usa /login de nuevo.",
        "lang_usage": "Uso: /lang ES | EN | PT",
        "lang_set": "Idioma cambiado a {lang}.",
        "help": ("Hazme preguntas sobre tus datos, por texto o nota de voz — por ejemplo:\n"
                 "• ¿Cuántos animales hay en la finca San Rafael?\n"
                 "• Peso promedio del rebaño norte este mes\n\n"
                 "Comandos:\n"
                 "/login <email> <contraseña> — iniciar sesión\n"
                 "/farm — filtrar por finca/rebaño\n"
                 "/new — empezar conversación nueva\n"
                 "/lang ES|EN|PT — cambiar idioma\n"
                 "/whoami — ver sesión y filtro\n"
                 "/logout — cerrar sesión"),
    },
    "EN": {
        "start": "Log in with your account: /login <email> <password>",
        "login_first": "Please log in first: /login <email> <password>",
        "login_usage": "Usage: /login <email> <password>",
        "login_bad": "Invalid credentials.",
        "login_ok": "Logged in as {name}. Send me a question (text or voice).",
        "delete_warn": "⚠️ I couldn't delete your message — delete it manually so your password doesn't stay in the chat.",
        "busy": "Still working on your previous question — one moment.",
        "transcribing": "🎙 Transcribing…",
        "querying": "🔍 Querying the database…",
        "running_sql": "📊 Running the query…",
        "writing": "📝 Writing the answer…",
        "transcribe_failed": "Could not transcribe the voice message.",
        "error": "Something went wrong with that question. You can say “try again”, rephrase it, or narrow it down.",
        "error_generic": "Something went wrong — please try again.",
        "new_chat": "New conversation started — previous questions are forgotten.",
        "logged_out": "Logged out.",
        "select_farm": "Select a farm:",
        "no_farms": "No farms found for your organization.",
        "all_farms": "🌐 All farms (clear scope)",
        "all_herds": "🌐 All herds",
        "scope_cleared": "Scope cleared — questions cover all farms.",
        "farm_not_found": "Farm not found.",
        "herd_not_found": "Herd not found.",
        "select_herd": "Farm: {farm}. Select a herd:",
        "scope_farm_no_herds": "Scope: farm “{farm}” (it has no herds).",
        "scope_farm_all_herds": "Scope: farm “{farm}”, all herds.",
        "scope_farm_herd": "Scope: farm “{farm}”, herd “{herd}”.",
        "session_expired": "Session expired — please /login again.",
        "lang_usage": "Usage: /lang ES | EN | PT",
        "lang_set": "Language switched to {lang}.",
        "help": ("Ask me questions about your data, by text or voice note — for example:\n"
                 "• How many animals are on the San Rafael farm?\n"
                 "• Average weight of the north herd this month\n\n"
                 "Commands:\n"
                 "/login <email> <password> — log in\n"
                 "/farm — scope questions to a farm/herd\n"
                 "/new — start a fresh conversation\n"
                 "/lang ES|EN|PT — switch language\n"
                 "/whoami — show session and scope\n"
                 "/logout — log out"),
    },
    "PT": {
        "start": "Inicie sessão com a sua conta: /login <email> <senha>",
        "login_first": "Inicie sessão primeiro: /login <email> <senha>",
        "login_usage": "Uso: /login <email> <senha>",
        "login_bad": "Credenciais inválidas.",
        "login_ok": "Sessão iniciada como {name}. Envie-me uma pergunta (texto ou voz).",
        "delete_warn": "⚠️ Não consegui apagar a sua mensagem — apague-a manualmente para que a senha não fique no chat.",
        "busy": "Ainda estou na sua pergunta anterior — um momento.",
        "transcribing": "🎙 Transcrevendo…",
        "querying": "🔍 Consultando o banco de dados…",
        "running_sql": "📊 Executando a consulta…",
        "writing": "📝 Escrevendo a resposta…",
        "transcribe_failed": "Não consegui transcrever a mensagem de voz.",
        "error": "Algo deu errado com essa pergunta. Você pode dizer “tente de novo”, reformular ou restringir.",
        "error_generic": "Algo deu errado — tente novamente.",
        "new_chat": "Nova conversa — as perguntas anteriores foram esquecidas.",
        "logged_out": "Sessão encerrada.",
        "select_farm": "Escolha uma fazenda:",
        "no_farms": "Nenhuma fazenda na sua organização.",
        "all_farms": "🌐 Todas as fazendas (sem filtro)",
        "all_herds": "🌐 Todos os rebanhos",
        "scope_cleared": "Filtro removido — as perguntas cobrem todas as fazendas.",
        "farm_not_found": "Fazenda não encontrada.",
        "herd_not_found": "Rebanho não encontrado.",
        "select_herd": "Fazenda: {farm}. Escolha um rebanho:",
        "scope_farm_no_herds": "Filtro: fazenda “{farm}” (não tem rebanhos).",
        "scope_farm_all_herds": "Filtro: fazenda “{farm}”, todos os rebanhos.",
        "scope_farm_herd": "Filtro: fazenda “{farm}”, rebanho “{herd}”.",
        "session_expired": "Sessão expirada — use /login novamente.",
        "lang_usage": "Uso: /lang ES | EN | PT",
        "lang_set": "Idioma alterado para {lang}.",
        "help": ("Faça perguntas sobre os seus dados, por texto ou nota de voz — por exemplo:\n"
                 "• Quantos animais há na fazenda San Rafael?\n"
                 "• Peso médio do rebanho norte neste mês\n\n"
                 "Comandos:\n"
                 "/login <email> <senha> — iniciar sessão\n"
                 "/farm — filtrar por fazenda/rebanho\n"
                 "/new — começar conversa nova\n"
                 "/lang ES|EN|PT — mudar idioma\n"
                 "/whoami — ver sessão e filtro\n"
                 "/logout — encerrar sessão"),
    },
}

def get_locale(context) -> str:
    """Normalized 2-letter locale ('es-ES' -> 'ES'); single ES fallback used everywhere."""
    loc = (context.user_data.get("locale") or "ES").upper()[:2]
    return loc if loc in LANGUAGES else "ES"

def t(context, key: str) -> str:
    return STRINGS[get_locale(context)][key]

# Enum value -> (ES, EN, PT) label, keyed by the column the value comes from.
# Labels copied from tractor-backend src/shared/constants/enum-labels.const.ts — keep in sync.
ENUM_GLOSSARY = {
    "animals.sex": {
        "MALE": ("Macho", "Male", "Macho"),
        "FEMALE": ("Hembra", "Female", "Fêmea"),
    },
    'animals."ageGroup"': {
        "CALF": ("Becerro/a", "Calf", "Bezerro/a"),
        "YEARLING": ("Mauta/e", "Yearling", "Garrote/a"),
        "HEIFER": ("Novilla", "Heifer", "Novilha"),
        "STEER": ("Novillo", "Steer", "Novilho"),
        "COW": ("Vaca", "Cow", "Vaca"),
        "BULL": ("Toro", "Bull", "Touro"),
    },
    "species": {
        "BOS_INDICUS": ("Bos indicus",) * 3,
        "BOS_TAURUS": ("Bos taurus",) * 3,
        "BUBALUS_BUBALIS": ("Bubalus bubalis",) * 3,
    },
    'body condition (animals."bodyCondition", body_condition)': {
        "GOOD": ("Buena", "Good", "Boa"),
        "NORMAL": ("Normal", "Normal", "Normal"),
        "POOR": ("Mala", "Poor", "Ruim"),
    },
    "gynecological_exams.result": {
        "PREGNANT": ("Preñada", "Pregnant", "Prenhe"),
        "NOT_PREGNANT": ("Vacía", "Not pregnant", "Vazia"),
        "POSSIBLY_PREGNANT": ("Posiblemente preñada", "Possibly pregnant", "Possivelmente prenhe"),
    },
    "weigh_in_events.type": {
        "BIRTH": ("Al Nacer", "At Birth", "Ao Nascer"),
        "DRY_OFF": ("Al Secado", "At Dry-Off", "Ao Secar"),
        "WEANING": ("Al Destete", "At Weaning", "Ao Desmame"),
        "REGULAR": ("General", "Regular", "Geral"),
        "SALE": ("Venta", "Sale", "Venda"),
    },
    "lactation_end_events.reason": {
        "CALVING": ("Parto", "Calving", "Parto"),
        "ABORTION": ("Aborto", "Abortion", "Aborto"),
        "DRY_OFF": ("Secado", "Dry-off", "Secagem"),
        "OTHER": ("Otro", "Other", "Outro"),
    },
    "birth_events.birth_type": {
        "NORMAL": ("Normal", "Normal", "Normal"),
        "ASSISTED": ("Asistido", "Assisted", "Assistido"),
        "ABORTION": ("Aborto", "Abortion", "Aborto"),
    },
    "birth_event_calves_information.birth_condition": {
        "NORMAL": ("Normal", "Normal", "Normal"),
        "DYSTOCIC": ("Distócico", "Dystocic", "Distócico"),
        "STILLBORN": ("Mortinato", "Stillborn", "Natimorto"),
    },
    "service_events.service_type": {
        "NATURAL": ("Monta natural", "Natural mating", "Monta natural"),
        "ARTIFICIAL_INSEMINATION": ("Inseminación artificial", "Artificial insemination", "Inseminação artificial"),
        "EMBRYO_TRANSFER": ("Transferencia de embrión", "Embryo transfer", "Transferência de embrião"),
        "IN_VITRO_FERTILIZATION": ("Fertilización in vitro", "In vitro fertilization", "Fertilização in vitro"),
    },
    "mastitis_checks.result": {
        "NEGATIVE": ("Negativo", "Negative", "Negativo"),
        "SIGNS": ("Con signos", "With signs", "Com sinais"),
        "WEAK_POSITIVE": ("Positivo débil", "Weak positive", "Positivo fraco"),
        "STRONG_POSITIVE": ("Positivo fuerte", "Strong positive", "Positivo forte"),
    },
    "animal_creation_events.reason": {
        "BIRTH": ("Nacimiento", "Birth", "Nascimento"),
        "PURCHASE": ("Compra", "Purchase", "Compra"),
        "TRANSFER": ("Traslado", "Transfer", "Transferência"),
        "OTHER": ("Otro", "Other", "Outro"),
    },
}

def enum_labels_text(locale: str) -> str:
    """Per-locale glossary block so the model writes labels, never raw enum codes."""
    i = {"ES": 0, "EN": 1, "PT": 2}.get(locale, 1)
    lines = [f"- {col}: " + ", ".join(f"{v}={labels[i]}" for v, labels in vals.items())
             for col, vals in ENUM_GLOSSARY.items()]
    return ("ENUM LABELS — in your answer never show raw enum codes; "
            "use these labels instead:\n" + "\n".join(lines))

class Stopwatch:
    """Interval timer: each lap records time since the previous lap under a key."""
    KEYS = ("bot_llm", "llm", "bot_db", "db", "format")

    def __init__(self):
        self.t = time.perf_counter()
        self.d = {k: [] for k in self.KEYS}

    def lap(self, key):
        now = time.perf_counter()
        self.d[key].append(now - self.t)
        self.t = now

def fmt_timings(d: dict) -> str:
    """'llm: 1.23s + 0.98s = 2.21s' per interval; empty intervals skipped; total line last."""
    lines = []
    for k, v in d.items():
        if not v:
            continue
        parts = " + ".join(f"{t:.2f}s" for t in v)
        lines.append(f"{k}: {parts}" + (f" = {sum(v):.2f}s" if len(v) > 1 else ""))
    lines.append(f"total: {sum(sum(v) for v in d.values()):.2f}s")
    return "\n".join(lines)

def md_to_html(text: str) -> str:
    """Telegram-safe HTML from the simple Markdown subset the model emits.
    Everything is escaped first, so unmatched markers can never break the send."""
    text = html.escape(text)
    text = re.sub(r"```\w*\n(.*?)```", r"<pre>\1</pre>", text, flags=re.S)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.S)
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.M)
    return text

def chunk_text(text: str, limit: int) -> list:
    """Split at line boundaries into pieces of at most `limit` chars.
    ponytail: a single overlong line is hard-sliced; a tag split across
    chunks falls back to plain text in edit_html."""
    chunks, cur = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if cur and len(cur) + len(line) + 1 > limit:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks or [""]

async def edit_html(status, text: str):
    """Render Markdown as HTML into the status message; long answers continue
    in follow-up messages instead of being truncated (footer survives)."""
    chunks = chunk_text(md_to_html(text), TELEGRAM_MSG_LIMIT)
    try:
        await status.edit_text(chunks[0], parse_mode="HTML")
        for c in chunks[1:]:
            await status.reply_text(c, parse_mode="HTML")
    except BadRequest:  # e.g. a tag split across chunks — resend as plain text
        try:
            plain = chunk_text(text, TELEGRAM_MSG_LIMIT)
            await status.edit_text(plain[0])
            for c in plain[1:]:
                await status.reply_text(c)
        except BadRequest:  # e.g. "message is not modified" — nothing left to salvage
            logger.warning("edit_html fallback failed", exc_info=True)

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

GROUNDING = """Every number, date, or value in your answer must come from a query result
in this conversation — NEVER invent, estimate, or fill in data. On a follow-up question,
first check whether earlier query results already contain the answer; if they don't, run a
NEW query reusing the context (same animal, farm, period). If the database cannot answer
it at all, say so plainly instead of guessing."""

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

# information_schema gives names and types but not joins, and reports every enum as
# "USER-DEFINED" — the model then guesses both. These fill in the two gaps.
COLUMNS_SQL = """
    SELECT table_name, column_name, data_type, udt_schema, udt_name
    FROM information_schema.columns
    WHERE table_schema = $1
      AND ($2::text[] IS NULL OR table_name = ANY($2))
    ORDER BY table_name, ordinal_position;
"""
# same conkey/confkey unnest as rls.sql's FK hop; pg_catalog is not privilege-filtered,
# so format_schema drops references to tables the role can't see
FKS_SQL = """
    SELECT c.relname AS tbl, a.attname AS col,
           pc.relname AS ref_tbl, pa.attname AS ref_col
    FROM pg_constraint con
    JOIN pg_class c ON c.oid = con.conrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    JOIN pg_class pc ON pc.oid = con.confrelid
    JOIN unnest(con.conkey) WITH ORDINALITY AS ck(attnum, ord) ON true
    JOIN unnest(con.confkey) WITH ORDINALITY AS fk(attnum, ord) ON fk.ord = ck.ord
    JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = ck.attnum
    JOIN pg_attribute pa ON pa.attrelid = con.confrelid AND pa.attnum = fk.attnum
    WHERE n.nspname = $1 AND con.contype = 'f';
"""
# keyed by (schema, name): a same-named enum in another schema must not merge labels
ENUMS_SQL = """
    SELECT n.nspname AS schema, t.typname AS name, e.enumlabel AS label
    FROM pg_type t
    JOIN pg_enum e ON e.enumtypid = t.oid
    JOIN pg_namespace n ON n.oid = t.typnamespace
    ORDER BY t.typname, e.enumsortorder;
"""

def format_schema(columns: list, fks: list, enums: list) -> dict:
    """{table_name: one-line description}, with enum values and FK targets inlined.

    e.g. Table `animals`: sex (AnimalSex: MALE|FEMALE), farm_id (text -> farms.id)

    ponytail: an enum's values are listed on its first use only — repeating them in
    every table that uses the type was the bulk of the prompt; the type name links
    later columns back to that one listing."""
    labels = {}
    for row in enums:
        labels.setdefault((row["schema"], row["name"]), []).append(row["label"])

    tables = {}
    for row in columns:
        tables.setdefault(row["table_name"], {})[row["column_name"]] = row

    targets = {(f["tbl"], f["col"]): f"{f['ref_tbl']}.{f['ref_col']}" for f in fks
               if f["tbl"] in tables and f["ref_tbl"] in tables}

    out = {}
    listed = set()
    for table, cols in tables.items():
        parts = []
        for name, row in cols.items():
            kind = row["data_type"]
            if kind == "USER-DEFINED":
                key = (row["udt_schema"], row["udt_name"])
                kind = row["udt_name"]
                values = labels.get(key)
                if values and key not in listed:
                    listed.add(key)
                    kind = f"{kind}: {'|'.join(values)}"
            elif kind == "ARRAY":
                kind = row["udt_name"].lstrip("_") + "[]"
            ref = targets.get((table, name))
            parts.append(f"{name} ({kind}{' -> ' + ref if ref else ''})")
        out[table] = f"Table `{table}`: {', '.join(parts)}"
    return out

async def get_db_schema(pool) -> dict:
    """Fetch schema as {table_name: one-line description}. Startup only."""
    async with pool.acquire() as conn:
        columns = await conn.fetch(COLUMNS_SQL, DB_SCHEMA, SCHEMA_TABLES or None)
        fks = await conn.fetch(FKS_SQL, DB_SCHEMA)
        enums = await conn.fetch(ENUMS_SQL)
    return format_schema(columns, fks, enums)

async def org_fetch(pool, org_id: str, query: str, *args):
    """Read-only fetch with the RLS org scope applied. Rows capped at MAX_ROWS."""
    async with pool.acquire() as conn:
        async with conn.transaction(readonly=True):
            await conn.execute("SELECT set_config('app.org_id', $1, true)", org_id)
            cur = await conn.cursor(query, *args)
            return await cur.fetch(MAX_ROWS)

def shrink_result(rows: list) -> str:
    """JSON payload for the model. Notes the MAX_ROWS cap (the true total is unknown
    past it), drops whole rows to fit MAX_RESULT_CHARS, and char-slices only when a
    single row alone exceeds the budget (readable values beat an empty result)."""
    capped = len(rows) == MAX_ROWS
    total = f"at least {MAX_ROWS}" if capped else str(len(rows))
    kept = [dict(r) for r in rows]
    note = [f"(row cap reached: showing the first {MAX_ROWS} matching rows)"] if capped else []
    result = json.dumps(kept + note, default=str)
    while len(result) > MAX_RESULT_CHARS and len(kept) > 1:
        kept = kept[: len(kept) // 2]
        result = json.dumps(kept + [f"(truncated, showing {len(kept)} of {total} rows)"], default=str)
    if len(result) > MAX_RESULT_CHARS:
        result = result[:MAX_RESULT_CHARS] + " … (row truncated)"
    return result

def check_resp(resp, what: str):
    """raise_for_status(), but log the body first: httpx puts the status code in the
    exception message and drops the body, which is where LiteLLM and Whisper put the
    actual reason. Body at DEBUG on success, ERROR on failure."""
    body = resp.text[:LOG_BODY_CHARS]
    if resp.is_error:
        logger.error(f"{what} HTTP {resp.status_code}: {body}")
    else:
        logger.debug(f"{what} HTTP {resp.status_code}: {body}")
    resp.raise_for_status()

async def execute_sql(pool, sql_query: str, org_id: str, sw: Stopwatch) -> tuple:
    """Run SQL read-only; RLS (see rls.sql) hides rows outside org_id.
    Returns (result_json, ok)."""
    sw.lap("bot_db")
    logger.info(f"SQL [{org_id}]: {sql_query}")
    try:
        rows = await org_fetch(pool, org_id, sql_query)
        logger.info(f"SQL result: {len(rows)} rows")
        result = shrink_result(rows)
        sw.lap("db")
        return result, True
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        sw.lap("db")
        return json.dumps({"error": str(e)}), False

# Same tool in the Anthropic Messages API shape (no "function" wrapper)
CLAUDE_SQL_TOOL = {
    "name": "run_sql_query",
    "description": "Execute a SQL query against the PostgreSQL database.",
    "input_schema": RUN_SQL_TOOL["function"]["parameters"],
}

def build_prompts(schema: str, org_id: str, scope: str, locale: str) -> tuple:
    """(static, dynamic) system-prompt blocks shared by both pipelines.
    static is byte-stable per locale so the Claude path can prompt-cache it (one cache
    entry per language); only org_id and scope vary per user, and they go in dynamic.
    The LiteLLM path concatenates both into one system message."""
    static = f"""You are a helpful assistant for the users of a farm-management system,
with read access to its PostgreSQL database. You hold a normal conversation and you
answer questions from the data.
DATABASE SCHEMA (a column shown as `x -> y.z` is a foreign key; an enum column lists
its allowed values):
{schema}
Most tables are multi-tenant: a query touching them MUST filter by the current user's
organization_id (given below), directly or via a join to a table that has it. A table
with no organization_id and no join path to one is global reference data — query it
directly, do not invent a scope for it.
Rows with is_deleted = true must be excluded.
All ids are UUIDs — NEVER guess an id from a name. When the user mentions a farm,
herd, animal, or person by name, match it with a join and ILIKE (e.g.
JOIN farms f ON ... WHERE f.name ILIKE '%san rafael%').
When the question needs data, construct a valid SELECT query and use the
'run_sql_query' tool. NEVER use destructive queries. When it doesn't need data — a
greeting, a thank-you, asking what you can do, explaining what a field or a metric
means, or discussing a result you already fetched — just reply in plain language, no
tool call.
If the question is ambiguous or missing information you need (which farm, which period,
which animals), ask a short clarifying question instead of guessing.
Never announce that you are going to look something up or run a query — either run it
now with the tool, or ask your clarifying question. Your reply ends the turn.
{GROUNDING}
In your answer, format dates like 15/03/2024 (no timestamps unless asked), round decimals
sensibly (e.g. 512.5 kg, ages and animal counts as whole years), and never show UUIDs.
Format answers in simple Markdown only: **bold**, `code`, and "- " bullet lists.
No headers, tables, links, or italics.
{FEW_SHOTS}
Those are examples of SQL style, not a limit on which tables or topics you can cover —
use any table in the schema above.
Always answer the user in {LANGUAGES.get(locale, 'English')}.
{enum_labels_text(locale)}"""
    dynamic = f"The current user's organization_id is '{org_id}'.{scope}"
    return static, dynamic

async def claude_query_pipeline(user_prompt: str, pool, schema_map: dict, org_id: str,
                                locale: str, scope: str = "", history: list = None,
                                sw: Stopwatch = None, progress=None) -> tuple:
    """SQL generation via the Claude API. Returns (reply, last_executed_sql or None)."""
    # Static block (identical across users/turns) is prompt-cached; the
    # per-user specifics go in a second block after the cache breakpoint.
    static_text, dynamic_text = build_prompts("\n".join(schema_map.values()), org_id, scope, locale)
    system = [
        # 1h TTL: the default 5-minute cache almost never hits on a low-traffic bot
        {"type": "text", "text": static_text, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
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
        if progress and sql_runs:
            await progress("writing")  # tool results are in; the model is composing
        sw.lap("bot_llm")
        for retry in range(3):
            try:
                # logged before the await: a hung call is otherwise indistinguishable from
                # a bot doing nothing — this line is the last thing you see when it stalls
                logger.info(f"LLM call: {LLM_MODEL} attempt={retry + 1}/3 msgs={len(messages)} "
                            f"sys={sum(len(b['text']) for b in system)}c "
                            f"tools={kwargs.get('tool_choice', {}).get('type', 'auto')} "
                            f"timeout={LLM_TIMEOUT}s x{anthropic_client.max_retries + 1} sdk tries "
                            f"stream={LLM_STREAM}")
                logger.debug(f"LLM request: {json.dumps(kwargs, default=str)[:LOG_BODY_CHARS]}")
                t0 = time.monotonic()
                if LLM_STREAM:
                    # a non-streaming call leaves the socket silent for the whole generation —
                    # minutes with Kimi's thinking blocks — and Moonshot reaps it as idle.
                    # SSE events keep bytes flowing; get_final_message() returns the same Message.
                    # ponytail: LLM_STREAM=false reverts if a provider can't do SSE
                    async with anthropic_client.messages.stream(**kwargs) as stream:
                        response = await stream.get_final_message()
                else:
                    response = await anthropic_client.messages.create(**kwargs)
                break
            except (json.JSONDecodeError, anthropic.APIConnectionError) as e:
                # ponytail: JSONDecodeError = Moonshot returning 200 with an empty body on a
                # slow request (the SDK won't retry a "successful" status); APIConnectionError
                # = dropped socket, the SDK's own retries already exhausted. Back off and redo
                # the whole call — messages/tool_results are unchanged, so it is safe to repeat.
                # repr + __cause__: a bare "APIConnectionError" hides whether it was DNS,
                # a refused socket or no route — the exception chain names it
                logger.warning(f"LLM API call failed after {time.monotonic() - t0:.1f}s, "
                               f"retrying ({retry + 1}/3): {e!r}"
                               + (f" <- {e.__cause__!r}" if e.__cause__ else ""))
                await asyncio.sleep(2 ** retry)
            except anthropic.APIStatusError as e:
                # not retried here (the SDK already retried 429/5xx); logged then re-raised so
                # the status, request id and body survive — "Handler Error" alone showed none
                logger.error(f"LLM HTTP {e.status_code} req={getattr(e, 'request_id', None)} "
                             f"after {time.monotonic() - t0:.1f}s: "
                             f"{str(getattr(e, 'body', None))[:LOG_BODY_CHARS]}")
                raise
        else:
            sw.lap("llm")  # close the interval so the failed retries aren't booked as format
            return "The AI service is not responding — please try again.", executed_sql
        sw.lap("llm")  # retries fold into one entry
        u = response.usage
        # cache_r is the number that says whether the 1h prompt cache above is earning its keep
        logger.info(f"LLM resp: stop={response.stop_reason} in={u.input_tokens} out={u.output_tokens} "
                    f"cache_r={getattr(u, 'cache_read_input_tokens', None)} "
                    f"cache_w={getattr(u, 'cache_creation_input_tokens', None)} "
                    f"req={getattr(response, '_request_id', None)} in {time.monotonic() - t0:.1f}s")
        for b in response.content:  # the SDK logs request bodies at DEBUG but never responses
            if b.type == "text":
                logger.info(f"LLM text: {b.text[:LOG_BODY_CHARS]}")
            elif b.type == "tool_use":
                logger.info(f"LLM tool_use: {b.name} {json.dumps(b.input, default=str)[:LOG_BODY_CHARS]}")
            elif b.type == "thinking":
                # Kimi K3 reasons here before picking a table — the only place a wrong
                # table choice is visible before the SQL is already written
                logger.info(f"LLM thinking: {b.thinking[:LOG_BODY_CHARS]}")
            else:  # redacted_thinking and anything the SDK adds later
                logger.info(f"LLM block: {b.type}")

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
            if progress:
                await progress("running_sql")
            db_result, ok = await execute_sql(pool, sql_query, org_id, sw)
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
    check_resp(resp, "litellm/route")
    text = resp.json()["choices"][0]["message"]["content"] or ""
    # match known names in the reply — no fragile JSON parsing of a small model
    return [t for t in table_names if re.search(rf"\b{re.escape(t)}\b", text)]

async def query_pipeline(user_prompt: str, pool, schema_map: dict, org_id: str, locale: str,
                         scope: str = "", history: list = None, sw: Stopwatch = None,
                         progress=None) -> tuple:
    """Dispatch to the Claude API when a key is configured, else local LiteLLM."""
    if anthropic_client:
        return await claude_query_pipeline(user_prompt, pool, schema_map, org_id, locale, scope, history, sw, progress)
    return await litellm_query_pipeline(user_prompt, pool, schema_map, org_id, locale, scope, history, sw, progress)

async def litellm_query_pipeline(user_prompt: str, pool, schema_map: dict, org_id: str, locale: str,
                                 scope: str = "", history: list = None, sw: Stopwatch = None,
                                 progress=None) -> tuple:
    """Orchestrates routing -> LLM -> SQL (with self-correction) -> LLM.
    Returns (reply, last_executed_sql or None)."""
    headers = {"Authorization": f"Bearer {LITELLM_MASTER_KEY}", "Content-Type": "application/json"}
    # nullcontext: reuse the shared http_client without closing it (closed in post_shutdown)
    async with contextlib.nullcontext(http_client) as client:
        tables = list(schema_map)
        if len(tables) > ROUTING_MIN_TABLES:
            try:
                sw.lap("bot_llm")
                picked = await pick_tables(client, headers, user_prompt, tables)
                sw.lap("llm")
            except Exception as e:
                sw.lap("llm")  # close the interval so the failed call isn't booked as bot_llm
                logger.warning(f"Table routing failed, using full schema: {e}")
                picked = []
            for t in ("farms", "herds"):  # scope filters reference these
                if t in schema_map and t not in picked:
                    picked.append(t)
            if picked:
                # ponytail: format_schema lists each enum's values on its first use, so a
                # routed-away table can take them with it — acceptable on the local path,
                # where ENUM_GLOSSARY still names the values that matter
                logger.info(f"Routing: {len(picked)}/{len(tables)} tables: {', '.join(picked)}")
                tables = picked
        static_text, dynamic_text = build_prompts("\n".join(schema_map[t] for t in tables),
                                                  org_id, scope, locale)
        system_prompt = static_text + "\n" + dynamic_text + (" /no_think" if NO_THINK else "")

        # ponytail: table routing above only sees the current question, not history —
        # feed the last user turn into pick_tables if follow-ups route to wrong tables
        messages = ([{"role": "system", "content": system_prompt}] + (history or [])
                    + [{"role": "user", "content": user_prompt}])
        executed_sql = None

        for attempt in range(MAX_SQL_ATTEMPTS):
            sw.lap("bot_llm")
            resp = await client.post(LITELLM_URL,
                                     json={"model": MODEL_NAME, "messages": messages, "tools": [RUN_SQL_TOOL],
                                           "tool_choice": "auto", "num_ctx": NUM_CTX, "max_tokens": MAX_TOKENS},
                                     headers=headers)
            check_resp(resp, "litellm/tool")
            sw.lap("llm")
            message = resp.json()["choices"][0]["message"]
            content = message.get("content") or ""

            if message.get("tool_calls"):
                tool_call = message["tool_calls"][0]
                sql_query = json.loads(tool_call["function"]["arguments"]).get("sql_query", "")
                if progress:
                    await progress("running_sql")
                db_result, ok = await execute_sql(pool, sql_query, org_id, sw)
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
                if progress:
                    await progress("running_sql")
                db_result, ok = await execute_sql(pool, sql_query, org_id, sw)
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
        if progress:
            await progress("writing")
        sw.lap("bot_llm")
        final_resp = await client.post(LITELLM_URL, json={"model": MODEL_NAME, "messages": messages,
                                                          "num_ctx": NUM_CTX, "max_tokens": MAX_TOKENS}, headers=headers)
        check_resp(final_resp, "litellm/final")
        sw.lap("llm")
        return final_resp.json()["choices"][0]["message"]["content"], executed_sql

async def require_login(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Return org_id, or None after telling the user to log in."""
    org_id = context.user_data.get("org_id")
    if not org_id:
        await update.effective_message.reply_text(t(context, "login_first"))
    return org_id

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(t(context, "start"))

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(t(context, "help"))

async def lang(update: Update, context: ContextTypes.DEFAULT_TYPE):
    arg = (context.args[0] if context.args else "").upper()[:2]
    if arg not in LANGUAGES:
        await update.message.reply_text(t(context, "lang_usage"))
        return
    context.user_data["locale"] = arg
    await update.message.reply_text(t(context, "lang_set").format(lang=LANGUAGES[arg]))

async def login(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Authenticate against the app's users table (argon2)."""
    try:
        await update.message.delete()  # don't leave the password in chat history
    except Exception:
        await update.effective_chat.send_message(t(context, "delete_warn"))
    chat = update.effective_chat
    if len(context.args) != 2:
        await chat.send_message(t(context, "login_usage"))
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
        except (VerifyMismatchError, InvalidHashError, TypeError):
            pass  # wrong password, or a NULL/non-argon2 hash (SSO / invited user)
    if not verified:
        logger.warning(f"Failed login attempt for {email} (tg user {update.effective_user.id})")
        await chat.send_message(t(context, "login_bad"))
        return
    logger.info(f"Login: {email} (tg user {update.effective_user.id})")

    context.user_data.pop("history", None)  # fresh session, fresh conversation
    context.user_data["epoch"] = context.user_data.get("epoch", 0) + 1  # invalidate in-flight write-backs
    context.user_data["org_id"] = row["organization_id"]
    context.user_data["db_user_id"] = row["id"]
    context.user_data["name"] = row["name"]
    context.user_data["email"] = email
    context.user_data["locale"] = row["locale"]
    await chat.send_message(t(context, "login_ok").format(name=row["name"]))

async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await require_login(update, context):
        return
    d = context.user_data
    farm = d.get("farm", {}).get("name", "all farms")
    herd = d.get("herd", {}).get("name", "all herds")
    await update.message.reply_text(
        f"Logged in as {d['name']} ({d['email']})\nOrganization: {d['org_id']}\n"
        f"Language: {d.get('locale') or 'ES'}\nScope: {farm} / {herd}"
    )

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Session timing averages (debug mode only; averages of per-request sums)."""
    avg = context.user_data.get("timings_avg")
    if not avg:
        await update.message.reply_text("No timing data this session yet.")
        return
    lines = "\n".join(f"{k}: {s / n:.2f}s avg ({n} req)" for k, (s, n) in avg.items())
    await update.message.reply_text(f"<pre>⏱ session averages\n{lines}</pre>", parse_mode="HTML")

async def new_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("history", None)
    context.user_data["epoch"] = context.user_data.get("epoch", 0) + 1  # invalidate in-flight write-backs
    await update.message.reply_text(t(context, "new_chat"))

async def logout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = t(context, "logged_out")  # read the locale before clear() wipes it
    context.user_data.clear()
    await update.message.reply_text(msg)

async def farm_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Pick a farm (then a herd) to auto-scope all questions."""
    org_id = await require_login(update, context)
    if not org_id:
        return
    rows = await org_fetch(context.bot_data["db_pool"], org_id,
                           "SELECT id, name FROM farms WHERE is_deleted = false ORDER BY name")
    if not rows:
        await update.message.reply_text(t(context, "no_farms"))
        return
    keyboard = [[InlineKeyboardButton(r["name"], callback_data=f"farm:{r['id']}")] for r in rows]
    keyboard.append([InlineKeyboardButton(t(context, "all_farms"), callback_data="farm:all")])
    await update.message.reply_text(t(context, "select_farm"), reply_markup=InlineKeyboardMarkup(keyboard))

async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle farm/herd button presses."""
    q = update.callback_query
    await q.answer()
    org_id = context.user_data.get("org_id")
    if not org_id:
        await q.edit_message_text(t(context, "session_expired"))
        return
    pool = context.bot_data["db_pool"]
    kind, val = q.data.split(":", 1)

    if kind == "farm":
        context.user_data.pop("herd", None)
        if val == "all":
            context.user_data.pop("farm", None)
            await q.edit_message_text(t(context, "scope_cleared"))
            return
        rows = await org_fetch(pool, org_id, "SELECT name FROM farms WHERE id = $1", val)
        if not rows:
            await q.edit_message_text(t(context, "farm_not_found"))
            return
        context.user_data["farm"] = {"id": val, "name": rows[0]["name"]}
        herds = await org_fetch(pool, org_id,
                                "SELECT id, name FROM herds WHERE farm_id = $1 AND is_deleted = false ORDER BY name", val)
        if not herds:
            await q.edit_message_text(t(context, "scope_farm_no_herds").format(farm=rows[0]["name"]))
            return
        keyboard = [[InlineKeyboardButton(h["name"], callback_data=f"herd:{h['id']}")] for h in herds]
        keyboard.append([InlineKeyboardButton(t(context, "all_herds"), callback_data="herd:all")])
        await q.edit_message_text(t(context, "select_herd").format(farm=rows[0]["name"]),
                                  reply_markup=InlineKeyboardMarkup(keyboard))
    elif kind == "herd":
        farm = context.user_data.get("farm")
        if val == "all" or not farm:
            context.user_data.pop("herd", None)
            await q.edit_message_text(t(context, "scope_farm_all_herds").format(farm=farm["name"])
                                      if farm else t(context, "scope_cleared"))
            return
        rows = await org_fetch(pool, org_id, "SELECT name FROM herds WHERE id = $1", val)
        if not rows:
            await q.edit_message_text(t(context, "herd_not_found"))
            return
        context.user_data["herd"] = {"id": val, "name": rows[0]["name"]}
        await q.edit_message_text(t(context, "scope_farm_herd").format(farm=farm["name"], herd=rows[0]["name"]))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generic handler for text/voice."""
    org_id = await require_login(update, context)
    if not org_id:
        return

    if context.user_data.get("busy"):
        await update.message.reply_text(t(context, "busy"))
        return
    context.user_data["busy"] = True

    pool = context.bot_data["db_pool"]
    is_voice = bool(update.message.voice)
    epoch = context.user_data.setdefault("epoch", 0)  # bumped by /login and /new; gone after /logout
    status = None  # created inside the try so a failed send can't leak busy=True
    echo, prompt = "", ""  # before the send: the except block reads prompt
    try:
        # one status message, edited in place as stages complete
        status = await update.message.reply_text(t(context, "transcribing" if is_voice else "querying"))
        if is_voice:
            voice_file = await context.bot.get_file(update.message.voice.file_id)
            voice_bytes = await voice_file.download_as_bytearray()
            r = await http_client.post(WHISPER_URL, files={"file": ("voice.ogg", bytes(voice_bytes))}, data={"model": "whisper-1"},
                                       headers={"Authorization": f"Bearer {WHISPER_API_KEY}"} if WHISPER_API_KEY else {},
                                       timeout=120.0)
            check_resp(r, "whisper")
            prompt = r.json().get("text", "").strip()
            if not prompt:
                await status.edit_text(t(context, "transcribe_failed"))
                return
            echo = f"🗣 {prompt}\n\n"
            await status.edit_text(f"{echo}{t(context, 'querying')}")
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

        async def progress(key):  # per-stage status edits close minutes-long silent gaps
            try:
                await status.edit_text(echo + t(context, key))
            except Exception:
                pass

        sw = Stopwatch()  # ponytail: always runs; laps are ~ns, gating only at report time
        reply, executed_sql = await query_pipeline(prompt, pool, context.bot_data["db_schema"], org_id,
                                                   get_locale(context), scope,
                                                   context.user_data.get("history"), sw, progress)
        # skip the write-back if /login, /new or /logout ran while we awaited the LLM —
        # concurrent_updates means this turn may belong to a session that no longer exists
        if reply and context.user_data.get("epoch") == epoch:
            history = context.user_data.setdefault("history", [])
            history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
            del history[:-HISTORY_MAX]
        if DEBUG_MODE and executed_sql:
            reply = f"{reply or ''}\n\n🔧 SQL:\n```sql\n{executed_sql}\n```"
        footer = FOOTERS[get_locale(context)][0 if executed_sql else 1]
        logger.info(f"A: {(reply or '')[:200]}")
        await edit_html(status, echo + (reply or "No response generated.") + footer)
        sw.lap("format")
        if DEBUG_MODE:
            try:  # the answer is already delivered — a failed report must not reach the outer except
                if context.user_data.get("epoch") == epoch:
                    avg = context.user_data.setdefault("timings_avg", {})
                    for k, v in sw.d.items():
                        if v:
                            s, n = avg.get(k, (0.0, 0))
                            avg[k] = (s + sum(v), n + 1)
                await update.effective_message.reply_text(f"<pre>⏱\n{fmt_timings(sw.d)}</pre>", parse_mode="HTML")
            except Exception:
                logger.exception("Timing report failed")

    except Exception as e:
        logger.exception("Handler Error")
        # record the failed turn so "try again" or a rephrase keeps its context
        if prompt and context.user_data.get("epoch") == epoch:
            history = context.user_data.setdefault("history", [])
            history += [{"role": "user", "content": prompt},
                        {"role": "assistant",
                         "content": f"(No answer was produced — the request failed with {type(e).__name__}. "
                                    "The user may retry or rephrase.)"}]
            del history[:-HISTORY_MAX]
        if status:
            if DEBUG_MODE:
                # repr(): timeouts and friends often have an empty str()
                await status.edit_text(f"⚠️ DEBUG ERROR: {e!r}"[:TELEGRAM_MSG_LIMIT])
            else:
                await status.edit_text(t(context, "error"))
    finally:
        context.user_data.pop("busy", None)  # pop, not =False: never repopulate a dict /logout cleared

async def post_init(app):
    """Create the pool inside PTB's event loop, cache the schema, set the command menu."""
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5,
                                     server_settings={"statement_timeout": "30s"})
    role = await pool.fetchval("SELECT current_user")
    if role != DB_ROLE:
        raise RuntimeError(f"DATABASE_URL connects as '{role}', expected '{DB_ROLE}' — "
                           "table owners bypass RLS, so this would disable tenant isolation (see rls.sql)")
    app.bot_data["db_pool"] = pool
    schema_map = await get_db_schema(pool)
    app.bot_data["db_schema"] = schema_map
    # a low count means a stale rls.sql or a stray SCHEMA_TABLES, not a bad model
    logger.info(f"Schema: {len(schema_map)} tables visible in '{DB_SCHEMA}'")
    for ud in app.user_data.values():  # a crash mid-request must not persist a stuck busy flag
        ud.pop("busy", None)
    total = sum(len(v) for v in schema_map.values())
    if anthropic_client:
        name = "Claude API" if ANTHROPIC_API_KEY else "Kimi (Anthropic-compatible)"
        provider = f"{name} ({LLM_MODEL})"
    else:
        provider = f"LiteLLM ({MODEL_NAME}; num_ctx={NUM_CTX}; " \
                   f"routing {'on' if len(schema_map) > ROUTING_MIN_TABLES else 'off'})"
    logger.info(f"Schema: {len(schema_map)} tables, {total} chars (~{total // 4} tokens); LLM: {provider}")
    commands = [
        BotCommand("login", "Log in: /login <email> <password>"),
        BotCommand("farm", "Choose the farm/herd your questions are about"),
        BotCommand("new", "Start a new conversation (forget previous questions)"),
        BotCommand("lang", "Switch language: /lang ES|EN|PT"),
        BotCommand("help", "How to use the bot, with examples"),
        BotCommand("whoami", "Show who is logged in and current scope"),
        BotCommand("logout", "Log out"),
    ]
    if DEBUG_MODE:
        commands.append(BotCommand("stats", "Timing averages for this session (debug)"))
    await app.bot.set_my_commands(commands)
    logger.info("DB pool ready, schema cached.")

async def post_shutdown(app):
    await http_client.aclose()
    pool = app.bot_data.get("db_pool")  # absent when post_init aborted (e.g. the DB-role assert)
    if pool:
        await pool.close()

async def on_error(update, context):
    """Last-resort handler: most command handlers have no try/except of their own."""
    logger.error("Unhandled error", exc_info=context.error)
    msg = getattr(update, "effective_message", None)
    if msg:
        try:
            await msg.reply_text(t(context, "error_generic"))
        except Exception:
            pass

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN")
    if not DATABASE_URL:
        raise ValueError("Missing DATABASE_URL")

    # bot_data holds the (unpicklable) pool and schema — persist user_data only
    persistence = PicklePersistence(filepath=PERSIST_PATH,
                                    store_data=PersistenceInput(bot_data=False, chat_data=False,
                                                                callback_data=False))
    app = (ApplicationBuilder().token(TELEGRAM_BOT_TOKEN)
           .concurrent_updates(True)  # default processes updates one-by-one: one slow query blocked every user
           .persistence(persistence)
           .post_init(post_init).post_shutdown(post_shutdown).build())
    app.add_error_handler(on_error)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("lang", lang))
    app.add_handler(CommandHandler("login", login))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("new", new_conversation))
    app.add_handler(CommandHandler("logout", logout))
    app.add_handler(CommandHandler("farm", farm_menu))
    if DEBUG_MODE:
        app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CallbackQueryHandler(menu_callback, pattern=r"^(farm|herd):"))
    app.add_handler(MessageHandler((filters.TEXT & ~filters.COMMAND) | filters.VOICE, handle_message))

    logger.info("Bot started...")
    app.run_polling()

if __name__ == "__main__":
    main()
