"""
IRT RAG Slack Bot — v5 Final
Socket Mode: NO URL needed. Just run this script and it connects.

Changes from your v3:
  1. Fixed search_kb: qclient.search() → qclient.query_points() (qdrant >= 1.7)
  2. Visible separators: thin Slack divider replaced with bold header-style separators
  3. Similarity % now shows human-readable label with explanation tooltip in footer
  4. Chat memory: bot remembers last 6 turns per user per DM/channel
  5. Friendly error messages: no raw errors shown to users

Run:
  conda activate bug_tracker
  cd /home/user/workspace/python/script_new/irt_rag
  python irt_rag_slack_bot.py
"""

import os, re, sys, time, logging, threading, json
import urllib.request, urllib.error
from collections import defaultdict, deque
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

SLACK_BOT_TOKEN  = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN  = os.environ.get("SLACK_APP_TOKEN")
IRT_CHANNEL      = os.environ.get("IRT_SUPPORT_CHANNEL_ID", "C08BUMMH9B2")
TICKET_URL       = os.environ.get("TICKET_CREATE_URL", "https://conversight.slack.com/lists")
TICKET_LIST_ID   = os.environ.get("IRT_TICKET_LIST_ID", "")  # Slack List ID for ticket creation
COLLECTION       = "irt_knowledge_base"
EMBED_MODEL      = "all-MiniLM-L6-v2"
STORAGE_DIR      = "./qdrant_storage"
TOP_K            = 5
MIN_SCORE        = 0.30
CHAT_HISTORY_LEN = 6   # last N user+assistant turn pairs to keep per user

# ── Automation API ────────────────────────────────────────────────────────────
AUTOMATION_API_URL = (
    "https://api.conversight.ai/universe-engine/v2/api/resource/action/"
    "crn:prod:us:step_flow:9b505609-832c-453b-9e07-19897c59273e:"
    "standard:irtbot?action=irtbotautomation"
)
AUTOMATION_TOKEN = "JWT eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfZG9jIjp7InVzZXJJZCI6ImIzNDJlYjJmLTM5ZmYtNDY2NS1iOWMwLTg1ZDdiYjM2NDk0OSIsImF0aGVuYUlkIjoiYjM0MmViMmYtMzlmZi00NjY1LWI5YzAtODVkN2JiMzY0OTQ5Iiwib3JnSWQiOiI5YjUwNTYwOS04MzJjLTQ1M2ItOWUwNy0xOTg5N2M1OTI3M2UiLCJkZXZpY2VJZCI6IjEyMzQ1NiIsImRldmljZU5hbWUiOiJCcm93c2VyV2ViIiwiaXNUcmlhbFVzZXIiOmZhbHNlLCJpc0ZpcnN0VGltZUxvZ2luIjpmYWxzZSwic2Vzc2lvbklkIjoiY3MtMzhmMWRjNzQtOTFiZS00ZjJkLWJmZTktOTAxYjZkYTk5NzNkIn0sImlhdCI6MTc3NDUwODY3M30.d1eW5Zatk5bnVqaGnKbaiOh4USq8oLFk0PPZMuddBLk"

app = App(token=SLACK_BOT_TOKEN)

print("⏳ Loading embedding model …")
embedder = SentenceTransformer(EMBED_MODEL)
print("✅ Embedding model ready")

print("⏳ Connecting to Qdrant …")
_qdrant_path = os.path.abspath(STORAGE_DIR)
try:
    qclient = QdrantClient(path=_qdrant_path)
    kb_count = qclient.count(collection_name=COLLECTION).count
except Exception as e:
    msg = str(e).lower()
    if "already accessed" in msg or "alreadylocked" in type(e).__name__.lower() or "resource temporarily unavailable" in msg:
        print()
        print("❌  Embedded Qdrant folder is already open in another process.")
        print(f"    Path: {_qdrant_path}")
        print()
        print("    Fix: use only one tool at a time on this folder, or stop the other Python job:")
        print("      • Quit  irt_rag_query_v2.py  /  irt_rag_build_knowledge_base_v2.py  in other terminals")
        print("      • If you used Ctrl+Z on the bot:  run  jobs  then  kill %<n>  or  kill <PID>")
        print("      • Find the PID:  lsof +D " + _qdrant_path)
        print()
        sys.exit(1)
    raise
print(f"✅ Qdrant ready — {kb_count} documents")

ai = OpenAI()

STEPS = [
    "⏳  _Hold on, looking into this for you…_",
    "🔍  _Found some related cases, analysing…_",
    "✍️   _Almost there, putting together your answer…_",
]

# ── Conversation memory (DM chatbot) ─────────────────────────────────────────
_history: dict = defaultdict(lambda: deque(maxlen=CHAT_HISTORY_LEN * 2))

def _conv_key(user: str, channel: str) -> str:
    return f"{user}::{channel}"

def _get_history(user: str, channel: str) -> list:
    return list(_history[_conv_key(user, channel)])

def _add_history(user: str, channel: str, role: str, content: str):
    _history[_conv_key(user, channel)].append({"role": role, "content": content})

def _clear_history(user: str, channel: str):
    key = _conv_key(user, channel)
    if key in _history:
        del _history[key]


# ── Pending clarifications (thread-based follow-up) ───────────────────────────
# When the bot asks a clarifying question in a channel thread, it stores the
# original query keyed by the bot message's ts. When the user replies in that
# thread, we look up the original query and combine them into a full search.
#
# Structure: { "bot_message_ts": {"query": "...", "user": "...", "channel": "..."} }

_pending: dict = {}

def _save_pending(ts: str, query: str, user: str, channel: str):
    _pending[ts] = {"query": query, "user": user, "channel": channel}

def _get_pending(thread_ts: str) -> dict | None:
    return _pending.get(thread_ts)

def _clear_pending(ts: str):
    _pending.pop(ts, None)


# ── Automation state (agentic loop) ──────────────────────────────────────────
# Keyed by user_id ONLY — so state persists whether user types
# /irt in a channel or messages in DM. One active session per user at a time.

_automation_state: dict = {}

def _get_auto_state(user: str, channel: str = None) -> dict | None:
    return _automation_state.get(user)

def _set_auto_state(user: str, channel: str = None, state: dict = None):
    _automation_state[user] = state

def _clear_auto_state(user: str, channel: str = None):
    _automation_state.pop(user, None)


# ── Last answer memory — for ticket creation context ─────────────────────────
# Stores the last KB answer given to each user so ticket creation
# can pre-fill title, category, severity without asking again.
# Structure: { "user_id": { "question", "answer", "hits": [...] } }

_last_answer: dict = {}

def _save_last_answer(user: str, question: str, answer: str, hits: list):
    _last_answer[user] = {
        "question": question,
        "answer":   answer,
        "hits":     hits[:1],  # only top hit needed for ticket context
    }

def _get_last_answer(user: str) -> dict | None:
    return _last_answer.get(user)


def _clear_last_answer(user: str) -> None:
    _last_answer.pop(user, None)


# ── Ticket state — agentic ticket creation loop ───────────────────────────────
# Keyed by user_id only — same pattern as _automation_state.
# Structure: {
#   "title":       pre-filled from last question,
#   "category":    from top KB hit,
#   "severity":    from top KB hit,
#   "reporter":    collected from user,
#   "environment": collected from user,
#   "notes":       collected from user (optional),
#   "step":        "reporter" | "environment" | "notes" | "confirm"
# }

_ticket_state: dict = {}

def _get_ticket_state(user: str) -> dict | None:
    return _ticket_state.get(user)

def _set_ticket_state(user: str, state: dict):
    _ticket_state[user] = state

def _clear_ticket_state(user: str):
    _ticket_state.pop(user, None)


# ── All 14 automation categories — fields, labels, validation ─────────────────
# Each category defines:
#   fields      : list of (key, label, required, hint)
#   build       : function to construct the "details" payload from collected dict

AUTOMATION_CATEGORIES = {

    "Extend Trail Period": {
        "fields": [
            ("org_id",        "Organisation ID",                        True,  "e.g. org_123"),
            ("extend_period", "New expiry date (YYYY-MM-DD)",           True,  "e.g. 2026-04-28"),
        ],
        "build": lambda c: {"org_id": c["org_id"], "extend_period": c["extend_period"]},
    },

    "Update Refresh Time": {
        "fields": [
            ("org_id",       "Organisation ID",                         True,  "e.g. org_123"),
            ("timezone",     "Timezone abbreviation",                   True,  "e.g. EST, PST, UTC"),
            ("refreshTime",  "Refresh times (comma-separated HH:MM)",  True,  "e.g. 09:00, 15:00"),
        ],
        "build": lambda c: {
            "org_id":      c["org_id"],
            "timezone":    c["timezone"],
            "refreshTime": [t.strip() for t in c["refreshTime"].split(",")],
        },
    },

    "Admin Email Changes": {
        "fields": [
            ("role",       "Role (admin / user)",                       True,  "admin or user"),
            ("old_email",  "Current email address",                     True,  "e.g. old@example.com"),
            ("new_email",  "New email address",                         True,  "e.g. new@example.com"),
            ("user_id",    "User ID (required for admin role)",         False, "e.g. user_123"),
            ("dataset_id", "Dataset ID (required for admin role)",      False, "e.g. ds_456"),
            ("org_id",     "Organisation ID (required for admin role)", False, "e.g. org_789"),
        ],
        "build": lambda c: {k: v for k, v in c.items() if v},
    },

    "Enable Athena Threads": {
        "fields": [
            ("org_id",     "Organisation ID",  True,  "e.g. org_123"),
            ("dataset_id", "Dataset ID",       True,  "e.g. ds_456"),
        ],
        "build": lambda c: {"org_id": c["org_id"], "dataset_id": c["dataset_id"]},
    },

    "Get Entity Count": {
        "fields": [
            ("tenant_id", "Tenant ID", True, "e.g. tenant_001"),
        ],
        "build": lambda c: {"tenant_id": c["tenant_id"]},
    },

    "Activate Dataset": {
        "fields": [
            ("dataset_id",        "Dataset ID",                           True,  "e.g. ds_123"),
            ("org_id",            "Organisation ID",                      True,  "e.g. org_456"),
            ("schema_to_activate","Schema name/ID to activate",           True,  "e.g. v2_schema"),
            ("activate_type",     "Activation type",                      True,
             "current_schema / in_progress_schema / backup_schema"),
        ],
        "build": lambda c: {
            "dataset_id": c["dataset_id"],
            "org_id":     c["org_id"],
            "schema": {
                "schema_to_activate":        c["schema_to_activate"],
                f"activate_{c['activate_type']}": True,
            },
        },
    },

    "Remove SME Duplicates": {
        "fields": [
            ("dataset_id",              "Dataset ID",                          True,  "e.g. ds_123"),
            ("remove_synonym_duplicate","Remove synonym duplicates? (yes/no)", False, "yes or no"),
            ("remove_metadata_duplicate","Remove metadata duplicates? (yes/no)",False, "yes or no"),
        ],
        "build": lambda c: {
            "dataset_id":               c["dataset_id"],
            "remove_synonym_duplicate":  c.get("remove_synonym_duplicate","").lower() == "yes",
            "remove_metadata_duplicate": c.get("remove_metadata_duplicate","").lower() == "yes",
        },
    },

    "Increase Session Timeout": {
        "fields": [
            ("org_id",          "Organisation ID",            True, "e.g. org_123"),
            ("time_in_minutes", "New timeout (in minutes)",   True, "e.g. 60"),
        ],
        "build": lambda c: {"org_id": c["org_id"], "time_in_minutes": c["time_in_minutes"]},
    },

    "Increase User Count": {
        "fields": [
            ("org_id",     "Organisation ID",          True, "e.g. org_123"),
            ("user_count", "Target user count (> 0)",  True, "e.g. 15"),
        ],
        "build": lambda c: {"org_id": c["org_id"], "user_count": int(c["user_count"])},
    },

    "Change Data Fetch Limit": {
        "fields": [
            ("dataset_id",  "Dataset ID",                True, "e.g. ds_123"),
            ("fetch_limit", "Max fetch limit size (> 0)", True, "e.g. 50000"),
        ],
        "build": lambda c: {"dataset_id": c["dataset_id"], "fetch_limit": int(c["fetch_limit"])},
    },

    "Remove Insight Duplicates": {
        "fields": [
            ("dataset_id", "Dataset ID", True, "e.g. ds_123"),
        ],
        "build": lambda c: {"dataset_id": c["dataset_id"]},
    },

    "Change Data Refresh Time": {
        "fields": [
            ("org_id",      "Organisation ID",                       True, "e.g. org_123"),
            ("time_in_utc", "Refresh time in UTC (YYYY-MM-DDTHH:MM:SS)", True, "e.g. 2026-08-01T09:00:00"),
        ],
        "build": lambda c: {"org_id": c["org_id"], "time_in_utc": c["time_in_utc"]},
    },

    "Enable Connector V2 Menu": {
        "fields": [
            ("org_id",     "Organisation ID", True, "e.g. org_123"),
            ("user_id",    "User ID",         True, "e.g. user_456"),
            ("dataset_id", "Dataset ID",      True, "e.g. ds_789"),
        ],
        "build": lambda c: {"org_id": c["org_id"], "user_id": c["user_id"], "dataset_id": c["dataset_id"]},
    },

    "Enable Athena Iq Menu": {
        "fields": [
            ("org_id",     "Organisation ID", True, "e.g. org_123"),
            ("user_id",    "User ID",         True, "e.g. user_456"),
            ("dataset_id", "Dataset ID",      True, "e.g. ds_789"),
        ],
        "build": lambda c: {"org_id": c["org_id"], "user_id": c["user_id"], "dataset_id": c["dataset_id"]},
    },
}

# Aliases — natural language phrases that map to category names
AUTOMATION_ALIASES = {
    "extend trail period":        "Extend Trail Period",
    "extend trial period":        "Extend Trail Period",
    "extend trial":               "Extend Trail Period",
    "update refresh time":        "Update Refresh Time",
    "change refresh schedule":    "Update Refresh Time",
    "admin email":                "Admin Email Changes",
    "change email":               "Admin Email Changes",
    "email change":               "Admin Email Changes",
    "enable athena threads":      "Enable Athena Threads",
    "athena threads":             "Enable Athena Threads",
    "get entity count":           "Get Entity Count",
    "entity count":               "Get Entity Count",
    "activate dataset":           "Activate Dataset",
    "dataset activation":         "Activate Dataset",
    "remove sme duplicates":      "Remove SME Duplicates",
    "sme duplicates":             "Remove SME Duplicates",
    "increase session timeout":   "Increase Session Timeout",
    "session timeout":            "Increase Session Timeout",
    "increase user count":        "Increase User Count",
    "user count":                 "Increase User Count",
    "change data fetch limit":    "Change Data Fetch Limit",
    "fetch limit":                "Change Data Fetch Limit",
    "remove insight duplicates":  "Remove Insight Duplicates",
    "insight duplicates":         "Remove Insight Duplicates",
    "change data refresh time":   "Change Data Refresh Time",
    "data refresh time":          "Change Data Refresh Time",
    "enable connector v2":        "Enable Connector V2 Menu",
    "connector v2 menu":          "Enable Connector V2 Menu",
    "enable athena iq":           "Enable Athena Iq Menu",
    "athena iq menu":             "Enable Athena Iq Menu",
}


# ── Friendly errors ───────────────────────────────────────────────────────────

def _friendly_error(e: Exception) -> str:
    msg = str(e).lower()
    if "rate limit" in msg or "429" in msg:
        return "The bot is receiving too many requests right now. Please try again in a moment."
    if "timeout" in msg or "timed out" in msg:
        return "The request took too long to process. Please try again."
    if "qdrant" in msg or "collection" in msg or "query_points" in msg or "search" in msg:
        return "The knowledge base is temporarily unavailable. Please try again shortly."
    if "openai" in msg or "api key" in msg or "authentication" in msg:
        return "The AI service is temporarily unavailable. Please try again shortly."
    if "channel_not_found" in msg or "not_in_channel" in msg:
        return "The bot doesn't have access to this channel. Please contact your workspace admin."
    return "Something went wrong. Please try again, or contact the IRT team if this persists."


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean(text) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<@[A-Z0-9]+>", "", text)
    text = re.sub(r"<https?://[^>]+>", "[link]", text)
    return text.strip()


def search_kb(query: str) -> list:
    vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    # ✅ Fixed: query_points() replaces search() for qdrant-client >= 1.7
    results = qclient.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=TOP_K,
        with_payload=True,
    ).points
    return [
        {
            "score"            : round(r.score, 3),
            "summary"          : r.payload.get("summary", ""),
            "solution"         : clean(r.payload.get("solution", "")),
            "resolution_status": r.payload.get("final_status",
                                 r.payload.get("resolution_status",
                                 r.payload.get("status", ""))),
            "bug_category"     : r.payload.get("bug_category", ""),
            "severity"         : r.payload.get("severity", ""),
            "references"       : r.payload.get("references", "None"),
            "source"           : r.payload.get("source", "Excel"),
        }
        for r in results
    ]


def call_automation_api(category: str, details: dict) -> dict:
    """
    Calls the IRT Automation API with the built payload.
    Returns {"ok": True, "message": "..."} or {"ok": False, "message": "..."}
    """
    payload = json.dumps({"config": {"category": category, "details": details}}).encode()
    req = urllib.request.Request(
        AUTOMATION_API_URL,
        data    = payload,
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer JWT {AUTOMATION_TOKEN}",
        },
        method = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            log.warning(f"automation API response: {body}")
            return {"ok": True, "message": str(body)}
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        log.error(f"automation API HTTP error {e.code}: {body}")
        return {"ok": False, "message": f"API error {e.code}: {body[:200]}"}
    except Exception as e:
        log.error(f"automation API error: {e}")
        return {"ok": False, "message": str(e)[:200]}


def build_enriched_query(original_question: str, clarification_answer: str) -> str:
    """
    Combines the original question + the user's clarification answer
    into a single natural, complete search query.

    Example:
      original:  "dataset stuck with DictionaryRequested status"
      answer:    "v2"
      result:    "v2 dataset stuck with DictionaryRequested status"

      original:  "dataload failed"
      answer:    "v1, in production environment"
      result:    "v1 dataload failed in production environment"
    """
    resp = ai.chat.completions.create(
        model      = os.environ.get("OPENAI_MODEL_SLACK", "gpt-4o"),
        max_tokens = 60,
        messages   = [
            {"role": "system", "content":
                "Combine the original question and the clarification answer into ONE "
                "natural complete sentence that can be used as a search query. "
                "Put the clarification answer details first (e.g. version, environment), "
                "then the issue description. "
                "Return ONLY the combined query — no explanation, no quotes, no extra text."},
            {"role": "user", "content":
                f"Original question: {original_question}\n"
                f"Clarification answer: {clarification_answer}"},
        ],
    )
    result = (resp.choices[0].message.content or "").strip()
    log.warning(f"build_enriched_query: '{original_question[:40]}' + '{clarification_answer}' → '{result}'")
    return result if result else f"{clarification_answer} {original_question}"


def _extract_field_value(key: str, label: str, hint: str, message: str) -> str | None:
    """
    Extracts a specific field value from a natural language user message.
    Returns None if the value is not clearly present.
    """
    resp = ai.chat.completions.create(
        model      = os.environ.get("OPENAI_MODEL_SLACK", "gpt-4o"),
        max_tokens = 30,
        messages   = [
            {"role": "system", "content":
                f"Extract the value for field '{label}' (key: {key}) from the message.\n"
                f"Field hint: {hint}\n"
                f"Common patterns to look for:\n"
                f"  - 'for org X', 'for this org X', 'org_id X', 'org X' → org_id = X\n"
                f"  - 'for dataset X', 'dataset_id X' → dataset_id = X\n"
                f"  - 'for tenant X', 'tenant X' → tenant_id = X\n"
                f"  - 'for user X', 'user_id X' → user_id = X\n"
                f"  - date patterns like '2026-04-28', 'April 28 2026' → date value\n"
                f"Rules:\n"
                f"  - Return ONLY the raw extracted value (e.g. 'trailtest01', '2026-04-28')\n"
                f"  - If NOT found, return exactly: NOT_FOUND\n"
                f"  - Never return the full sentence or unrelated words"},
            {"role": "user", "content": message},
        ],
    )
    result = (resp.choices[0].message.content or "").strip()
    log.warning(f"_extract_field_value: key={key} msg='{message[:50]}' → '{result}'")
    if not result or result == "NOT_FOUND":
        return None
    return result


def _fallback_tenant_id(message: str) -> str | None:
    """If GPT extraction misses (e.g. typo 'enant_001'), accept obvious tenant ids."""
    s = message.strip()
    if re.match(r"^tenant_[\w-]+$", s, re.I):
        return s
    if re.match(r"^enant_[\w-]+$", s, re.I):
        return "t" + s
    return None


def _should_interrupt_automation_for_chat(query: str, awaiting_confirm: bool) -> bool:
    """
    Greetings / small talk should not continue a pending automation slot-fill
    (e.g. user says 'Hi' while bot was waiting for Tenant ID).
    """
    t = query.strip()
    if len(t) > 80:
        return False
    tl = t.lower()
    if awaiting_confirm:
        if tl in (
            "ok", "yes", "y", "confirm", "proceed", "cancel", "no", "abort", "stop", "exit",
        ):
            return False
    # single-line greetings / thanks (not automation answers)
    if re.match(
        r"^(hi|hello|hey|hii|yo|sup|thanks?|thank you|thx|ty|bye|goodbye|gm|good (morning|afternoon|evening))[\s!.,]*$",
        tl,
    ):
        return True
    if tl in ("hi", "hello", "hey", "thanks", "thx", "bye"):
        return True
    return False


def _extract_fields_from_message(state: dict, message: str, category: str):
    """
    When the user's initial trigger message contains field values
    (e.g. "extend trail period for org trailtest01"),
    extract each field value individually using _extract_field_value.
    Saves found values directly into state["collected"].
    """
    fields = AUTOMATION_CATEGORIES[category]["fields"]
    log.warning(f"_extract_fields_from_message: category={category} msg='{message[:60]}'")

    for key, label, required, hint in fields:
        value = _extract_field_value(key, label, hint, message)
        if value and value != "NOT_FOUND":
            state["collected"][key] = value
            log.warning(f"_extract_fields_from_message: saved {key}='{value}'")


def automation_agent(user: str, channel: str, message: str, category: str = None) -> str:
    """
    The agentic loop for automation.

    On each call it either:
      - Starts a new automation session (if category given)
      - Handles cancel/confirm commands
      - Collects the next missing field
      - Executes the API when all fields are ready and confirmed

    Returns a Slack-formatted string to send back to the user.
    """
    state = _get_auto_state(user, channel)

    # ── Handle cancel at any point ────────────────────────────────────────────
    if message.lower().strip() in ("cancel", "abort", "stop", "exit"):
        if state:
            _clear_auto_state(user, channel)
        return "❌ Automation cancelled. Ask me anything else!"

    # ── Start new session ─────────────────────────────────────────────────────
    if category and not state:
        if category not in AUTOMATION_CATEGORIES:
            return f"⚠️ Unknown automation category: *{category}*"
        state = {"category": category, "collected": {}, "awaiting_confirm": False, "just_started": True}
        # Extract any field values already present in the trigger message FIRST
        # e.g. "extend trail period for org trailtest01" → org_id = "trailtest01"
        _extract_fields_from_message(state, message, category)
        # Save AFTER extraction so collected fields are included in state
        _set_auto_state(user, channel, state)

    if not state:
        return "⚠️ No active automation session. Try asking me to perform a specific action."

    category   = state["category"]
    collected  = state["collected"]
    cat_config = AUTOMATION_CATEGORIES[category]
    fields     = cat_config["fields"]

    # ── Handle confirmation ───────────────────────────────────────────────────
    if state.get("awaiting_confirm"):
        if message.lower().strip() in ("confirm", "yes", "proceed", "ok", "y"):
            try:
                details = cat_config["build"](collected)
            except Exception as e:
                _clear_auto_state(user, channel)
                return f"⚠️ Failed to build payload: {e}"

            result = call_automation_api(category, details)
            _clear_auto_state(user, channel)

            if result["ok"]:
                return (
                    f"✅ *{category}* executed successfully!\n\n"
                    f"_Response:_ {result['message'][:300]}"
                )
            else:
                return (
                    f"❌ *{category}* failed.\n\n"
                    f"_Error:_ {result['message']}\n\n"
                    "_Please check the details and try again._"
                )
        else:
            _clear_auto_state(user, channel)
            return "❌ Automation cancelled. Ask me anything else!"

    # ── This is a follow-up message — try to save it to the current missing field
    # Only runs when state already exists (not the first trigger message)
    if not state.get("just_started"):
        for key, label, required, hint in fields:
            if key not in collected:
                extracted = _extract_field_value(key, label, hint, message)
                if not extracted and key == "tenant_id":
                    extracted = _fallback_tenant_id(message)
                if extracted:
                    collected[key] = extracted
                    state["collected"] = collected
                    _set_auto_state(user, channel, state)
                break  # only try one field per turn

    # Clear just_started flag
    state.pop("just_started", None)
    _set_auto_state(user, channel, state)

    # ── Determine effective required fields ──────────────────────────────────
    effective_fields = fields
    if category == "Admin Email Changes":
        role = collected.get("role", "").lower()
        effective_fields = [
            (k, l, True if k in ("role", "old_email", "new_email") else (role == "admin"), h)
            for k, l, _, h in fields
        ]

    # ── Find all still-missing required fields ────────────────────────────────
    missing = [(k, l, h) for k, l, req, h in effective_fields if req and k not in collected]

    if missing:
        _set_auto_state(user, channel, state)

        # Show already collected context so user knows what was saved
        collected_lines = ""
        if collected:
            lines = [f"   ✅ *{l}:* `{collected[k]}`"
                     for k, l, _, _ in fields if k in collected]
            collected_lines = "\n".join(lines) + "\n\n"

        # Ask for the next missing field clearly
        next_key, next_label, next_hint = missing[0]

        # If more than one missing, show what's still needed
        if len(missing) > 1:
            remaining = ", ".join(f"*{l}*" for _, l, _ in missing[1:])
            after_note = f"\n_After that I'll also need: {remaining}_"
        else:
            after_note = ""

        return (
            f"{collected_lines}"
            f"📝 Please provide *{next_label}*\n"
            f"_{next_hint}_{after_note}"
        )

    # ── All fields collected — show confirmation summary ──────────────────────
    summary_lines = [f"   *{label}:* `{collected.get(key, '—')}`"
                     for key, label, _, _ in fields if key in collected]
    summary = "\n".join(summary_lines)

    state["awaiting_confirm"] = True
    _set_auto_state(user, channel, state)

    return (
        f"🔧 *Ready to execute: {category}*\n\n"
        f"{summary}\n\n"
        f"Type *confirm* to proceed or *cancel* to abort."
    )


def detect_automation_intent(query: str, history: list) -> str | None:
    """
    Uses GPT to detect if the user's message is an automation request.
    Returns the exact category name if matched, or None if it's a normal question.

    Uses a targeted prompt — cheap and fast.
    """
    categories_list = "\n".join(f"- {c}" for c in AUTOMATION_CATEGORIES.keys())

    resp = ai.chat.completions.create(
        model      = os.environ.get("OPENAI_MODEL_SLACK", "gpt-4o"),
        max_tokens = 60,
        messages   = [
            {"role": "system", "content": f"""You detect automation intent for an IRT bot.

Supported categories:
{categories_list}

If the user message is requesting to perform one of these operations, reply with EXACTLY:
AUTOMATE: <exact category name>

If it is NOT an automation request (e.g. asking about a bug, greeting, question), reply with:
NONE

No other text. Nothing else."""},
            {"role": "user", "content": query},
        ],
    )
    result = (resp.choices[0].message.content or "").strip()
    log.warning(f"detect_automation_intent: {repr(result)}")

    if result.upper().startswith("AUTOMATE:"):
        category = result[9:].strip()
        if category in AUTOMATION_CATEGORIES:
            return category

    return None


def analyze_query(query: str, history: list) -> dict:
    """
    Single GPT call that decides what to do with the user's message.
    Has a keyword pre-check to catch clear automation phrases before GPT routing.
    """
    # ── Fast keyword pre-check — catches obvious automation requests ──────────
    # This prevents GPT from misrouting "extend the trail period for org X" to SEARCH.
    q_lower = query.lower()
    # Avoid substring hits like "high *entity count*" in unrelated bug descriptions.
    if re.search(r"\b(get entity count|entity count for|count of entities)\b", q_lower):
        log.warning("analyze_query keyword match: explicit Get Entity Count phrase")
        return {"action": "automate", "text": "Get Entity Count"}
    _automation_triggers = {
        "extend trail":           "Extend Trail Period",
        "extend trial":           "Extend Trail Period",
        "trail period":           "Extend Trail Period",
        "trial period":           "Extend Trail Period",
        "update refresh time":    "Update Refresh Time",
        "refresh schedule":       "Update Refresh Time",
        "change email":           "Admin Email Changes",
        "email change":           "Admin Email Changes",
        "admin email":            "Admin Email Changes",
        "athena threads":         "Enable Athena Threads",
        "enable athena thread":   "Enable Athena Threads",
        "activate dataset":       "Activate Dataset",
        "dataset activation":     "Activate Dataset",
        "sme duplicate":          "Remove SME Duplicates",
        "remove sme":             "Remove SME Duplicates",
        "session timeout":        "Increase Session Timeout",
        "increase timeout":       "Increase Session Timeout",
        "user count":             "Increase User Count",
        "increase user":          "Increase User Count",
        "fetch limit":            "Change Data Fetch Limit",
        "data fetch":             "Change Data Fetch Limit",
        "insight duplicate":      "Remove Insight Duplicates",
        "remove insight":         "Remove Insight Duplicates",
        "data refresh time":      "Change Data Refresh Time",
        "refresh time":           "Change Data Refresh Time",
        "connector v2 menu":      "Enable Connector V2 Menu",
        "enable connector":       "Enable Connector V2 Menu",
        "athena iq":              "Enable Athena Iq Menu",
        "enable athena iq":       "Enable Athena Iq Menu",
    }
    for trigger, category in _automation_triggers.items():
        if trigger in q_lower:
            log.warning(f"analyze_query keyword match: '{trigger}' → AUTOMATE: {category}")
            return {"action": "automate", "text": category}

    # ── Ticket creation trigger ───────────────────────────────────────────────
    _ticket_triggers = [
        "create ticket", "create a ticket", "raise ticket", "raise a ticket",
        "log ticket", "log this", "log the issue", "open a ticket",
        "report this issue", "report this", "file a ticket", "file a bug",
        "create bug", "raise bug", "ticket for this", "ticket for the above",
        "create the ticket", "make a ticket",
    ]
    if any(t in q_lower for t in _ticket_triggers):
        log.warning(f"analyze_query keyword match: ticket creation")
        return {"action": "ticket", "text": ""}

    history_text = ""
    if history:
        for msg in history:
            role = "User" if msg["role"] == "user" else "Bot"
            history_text += f"{role}: {msg['content']}\n"

    # Short v1/v2 answer after a version clarify — never send a second CLARIFY (GPT sometimes does)
    qstrip = query.strip()
    ql = qstrip.lower()
    if history and len(qstrip) <= 32:
        last_assistant = None
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                last_assistant = (msg.get("content") or "")
                break
        if last_assistant:
            la = last_assistant.lower()
            version_ask = (
                "v1 or v2" in la
                or "which version" in la
                or bool(re.search(r"\bv1\b.*\bv2\b", la))
            )
            if version_ask and re.match(r"^v[12]\b", ql):
                issue = None
                for msg in reversed(history):
                    if msg.get("role") != "user":
                        continue
                    u = (msg.get("content") or "").strip()
                    ul = u.lower()
                    if len(u) <= 4 and ul in ("v1", "v2", "hi", "hey", "hello", "hii"):
                        continue
                    if len(u) < 12 and ul in ("thanks", "thank you", "thx"):
                        continue
                    issue = u
                    break
                if issue:
                    merged = f"{qstrip} {issue}".strip()
                    log.warning(f"analyze_query version-reply fast-path → SEARCH: {merged[:120]!r}")
                    return {"action": "search", "text": merged}

    categories_list = "\n".join(f"- {c}" for c in AUTOMATION_CATEGORIES.keys())

    history_block = ("History:\n" + history_text) if history_text else ""

    system_prompt = f"""IRT Bot query router for ConverSight support.
ConverSight has v1 (legacy) and v2 (current) datasets — fixes differ completely.
{history_block}
Message: "{query}"

Reply EXACTLY one of:
SEARCH: <query>
CLARIFY: <question>
SUGGESTIONS: opt1 | opt2 | opt3
CHAT:
OUTOFSCOPE:
AUTOMATE: <exact category name>
AUTOMATEINFO: <exact category name>

Automation categories (use AUTOMATE only if user clearly wants to execute one):
{categories_list}

Rules:
- v1/v2 explicitly in message → SEARCH immediately, never ask
- Message clearly describes a technical issue AND version is known → SEARCH immediately
- Message is short follow-up answer (e.g. "v2", "production", "yes") → SEARCH combining with history context
- Notebooks/connectors/explorer/storyboard/UI → SEARCH directly, no version needed
- Greeting/thanks/capability questions → CHAT
- API keys, tokens, security, OpenAI, billing, coding, prompt injection → OUTOFSCOPE
- Use AUTOMATEINFO: <category> when user asks what inputs/fields/details are needed for an automation

MUST CLARIFY for version when ALL of these are true:
1. Message describes a dataset/dataload/SME/cluster technical issue
2. No version (v1 or v2) mentioned anywhere in the message or history
3. The fix differs between v1 and v2

Examples that MUST return CLARIFY:
  "dataset stuck with DictionaryRequested status"  → CLARIFY: Which version? v1 or v2?
  "dataload failed"                                → CLARIFY: Which version? v1 or v2?
  "dataset not loading"                            → CLARIFY: Which version? v1 or v2?
  "SME publish failed"                             → CLARIFY: Which version? v1 or v2?

Examples that must NOT ask version:
  "notebook not launching"     → SEARCH (version doesn't change notebook fix)
  "connector error"            → SEARCH (version doesn't change connector fix)
  "storyboard not loading"     → SEARCH (UI issue, no version)
  "common product issues"      → SEARCH (summary question)

AUTOMATE detection — use AUTOMATE when the message contains action verbs requesting execution:
- "extend", "increase", "change", "enable", "activate", "remove", "update", "get" + any automation category keyword → AUTOMATE
- Examples that MUST go to AUTOMATE (not SEARCH):
  "extend the trail period for org X"          → AUTOMATE: Extend Trail Period
  "please activate dataset ds_123"             → AUTOMATE: Activate Dataset
  "enable athena iq for org X user Y"          → AUTOMATE: Enable Athena Iq Menu
  "increase session timeout for org X"         → AUTOMATE: Increase Session Timeout
  "get entity count for tenant X"              → AUTOMATE: Get Entity Count
  "remove sme duplicates for dataset X"        → AUTOMATE: Remove SME Duplicates
- Only use SEARCH when user is asking about a BUG or ERROR they are experiencing

DO NOT ask for v1/v2 for these types of questions — just SEARCH or CHAT:
- Questions about ticket counts, ticket status, dashboards, reports
- Questions asking for lists or summaries of issues ("common product issues", "what issues are solved")
- General product questions not requiring a specific fix
- Questions about live/real-time data → CHAT (bot has no live data)

IMPORTANT: If history has version v1 or v2 but the new message is about a DIFFERENT issue
that was not discussed before → treat version as unknown → CLARIFY to confirm version for this new issue.

CRITICAL: If the last bot message asked for v1 vs v2 and the user message is ONLY "v1", "v2",
or similar (short version answer) → return SEARCH: <v1 or v2> plus the user's issue text from history,
never CLARIFY again."""

    resp = ai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=120,        # enough for CLARIFY + SUGGESTIONS line
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query}
        ],
    )
    result = (resp.choices[0].message.content or "").strip()
    log.warning(f"analyze_query result: {repr(result)}")

    # GPT sometimes returns literal \n instead of real newlines — normalise both
    result = result.replace("\\n", "\n")
    lines  = [l.strip() for l in result.strip().splitlines() if l.strip()]

    if not lines:
        return {"action": "search", "text": query}

    if lines[0].upper().startswith("CLARIFY:"):
        question    = lines[0][8:].strip()
        suggestions = []
        for line in lines[1:]:
            if line.upper().startswith("SUGGESTIONS:"):
                raw         = line[12:].strip()
                suggestions = [s.strip() for s in raw.split("|") if s.strip()]
                break
        return {"action": "clarify", "text": question, "suggestions": suggestions}

    if lines[0].upper().startswith("SEARCH:"):
        return {"action": "search", "text": lines[0][7:].strip() or query}

    if lines[0].upper().startswith("CHAT"):
        return {"action": "chat", "text": ""}

    if lines[0].upper().startswith("OUTOFSCOPE"):
        return {"action": "outofscope", "text": ""}

    if lines[0].upper().startswith("AUTOMATE:"):
        category = lines[0][9:].strip()
        if category in AUTOMATION_CATEGORIES:
            return {"action": "automate", "text": category}
        return {"action": "search", "text": query}

    if lines[0].upper().startswith("AUTOMATEINFO:"):
        category = lines[0][13:].strip()
        if category in AUTOMATION_CATEGORIES:
            return {"action": "automateinfo", "text": category}
        return {"action": "search", "text": query}

    return {"action": "search", "text": query}


def clarify_blocks(question: str, suggestions: list) -> list:
    """
    Builds a Slack block with the clarifying question and
    clickable suggestion buttons. Each button gets a unique action_id.
    """
    blocks = [
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "_IRT Support Bot · quick question_"}],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"🤔 {question}"},
        },
    ]

    if suggestions:
        buttons = []
        for i, s in enumerate(suggestions[:5]):
            buttons.append({
                "type": "button",
                "text": {"type": "plain_text", "text": s, "emoji": True},
                "action_id": f"clarify_reply_{i}",   # unique per button
                "value": s,
            })
        blocks.append({
            "type": "actions",
            "elements": buttons
        })

    return blocks


def generate_answer(query: str, hits: list, history: list = None) -> str:
    context = ""
    for i, h in enumerate(hits, 1):
        sol = h["solution"]
        if not sol or sol in ("nan", "None", "No solution documented."):
            sol = "(No specific solution recorded)"
        tag = " [RCA]" if h.get("source") == "RCA" else ""
        context += f"""
Case {i}{tag} (relevance: {h['score']:.2f})
  Issue    : {h['summary']}
  Category : {h['bug_category']}
  Status   : {h['resolution_status']}
  Solution : {sol}
  Refs     : {h['references']}
"""

    system_prompt = f"""You are IRT Bot, an Incident Response Team support assistant for ConverSight.
You help engineers and support staff diagnose and resolve product issues.

ConverSight has two dataset versions — v1 (legacy) and v2 (current). Always tailor your answer to the correct version based on the context.

Relevant past cases:
{context}

How to respond:
1. If there is conversation history, use it to understand follow-up questions.
2. Start with "Yes, the IRT team has seen this before." OR
   "This looks like a new issue — please raise it with the IRT team."
3. For each relevant case write ONE sentence:
   "In a case where [issue], the fix was [exact solution]."
   Then: "This was a *permanent fix*." or "This was a *workaround*."
4. Write "*Steps to try:*" then 2-4 bullets ONLY from the Solution fields.
   - Use EXACT actions from Solution — do not invent steps.
   - IRT terms OK: SME publish, republish, vacuum, entity count, org ID, dataset activation.
   - No generic advice.
5. End with: "If this doesn't help, share your *Dataset name*, *Org ID*, *Environment*, and *current status* with the IRT team."

Rules: *bold* key terms. No "knowledge base" or "Case #1". Under 300 words."""

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    resp = ai.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL_SLACK", "gpt-4o"),
        max_tokens=600,
        messages=messages,
    )
    return (resp.choices[0].message.content or "").strip()


def automation_info_response(category: str) -> str:
    """
    Returns a formatted Slack message describing what inputs
    are required for a given automation category.
    Built directly from AUTOMATION_CATEGORIES — always accurate.
    """
    if category not in AUTOMATION_CATEGORIES:
        return f"⚠️ Unknown automation category: *{category}*"

    fields = AUTOMATION_CATEGORIES[category]["fields"]
    lines  = []
    for key, label, required, hint in fields:
        req_tag = "" if required else " _(optional)_"
        lines.append(f"• *{label}*{req_tag} — `{hint}`")

    fields_text = "\n".join(lines)
    return (
        f"*📋 Inputs required for {category}:*\n\n"
        f"{fields_text}\n\n"
        f"_Just say *\"{category.lower()}\"* and I'll guide you through it step by step._"
    )


def _map_to_option_id(user_value: str, options: dict) -> str:
    """
    Maps a user's free-text input to the closest select option ID.
    options = {"label": "OptID", ...}
    Falls back to first option if no match found.
    """
    v = user_value.lower().strip()
    # Exact match first
    for label, opt_id in options.items():
        if v == label.lower():
            return opt_id
    # Partial match
    for label, opt_id in options.items():
        if v in label.lower() or label.lower() in v:
            return opt_id
    # Return first option as fallback
    return list(options.values())[0]


# ── Known option IDs from list F08B5SXE9H8 ───────────────────────────────────
# Fetched from slackLists.items.list — matched to form labels from screenshots

TICKET_CATEGORY_OPTIONS = {
    "Data Load Failure V1":       "OptFI38PO4P",
    "Data Load Failure V2":       "Opt74DUH3XU",
    "Cluster Startup Failure":    "OptJ93QL2JS",
    "Notebook Launch Issues":     "OptUFPFJY50",
    "Scheduled Flows Struck":     "Opt0GZXQQ94",
    "URL Access Issues":          "OptXGU91C0D",
    "Recurring Storyboard Failure": "Opt48UMHNTD",
    "Deployment":                 "OptDHMPQYKR",
    "Cluster Struck Issues":      "OptB4RXY8ET",
    "Others":                     "OptOYUJQ53K",
}

TICKET_TEAM_OPTIONS = {
    "Testing":                    "OptP2OSTCNM",
    "DevOps":                     "OptYX0RVZMF",
    "Engineering (Backend)":      "OptKXV6NQBE",
}

TICKET_ENV_OPTIONS = {
    "AWS Production":             "OptBMCHH1RK",
    "GCP Staging":                "OptCUZ738HM",
    "GCP Production":             "OptYKZKPJWD",
}


def ticket_agent(user: str, message: str, client, channel: str) -> str:
    """
    Agentic ticket creation loop.
    Pre-fills title, category, severity from last KB answer.
    Asks: reporter name, team/department, environment, notes (optional).
    On confirm → creates Slack List item in F08B5SXE9H8.
    """
    state = _get_ticket_state(user)

    # ── Cancel ────────────────────────────────────────────────────────────────
    if message.lower().strip() in ("cancel", "abort", "stop"):
        _clear_ticket_state(user)
        return "❌ Ticket creation cancelled."

    # ── Start new session ─────────────────────────────────────────────────────
    if not state:
        last = _get_last_answer(user)
        if not last:
            return (
                "⚠️ I don't have a recent issue to raise a ticket for.\n"
                "_Please describe your issue first, then request a ticket._"
            )
        top_hit  = last["hits"][0] if last["hits"] else {}
        question = last["question"]

        # Detect category from the question version — don't blindly use KB hit category
        # KB hits for v2 questions often have V1 category from similar old issues
        kb_category = top_hit.get("bug_category", "Others")
        q_lower     = question.lower()
        if "v2" in q_lower and "v1" in kb_category.lower():
            # Swap V1 → V2 in category name
            category = kb_category.replace("V1", "V2").replace("v1", "V2")
        elif "v1" in q_lower and "v2" in kb_category.lower():
            # Swap V2 → V1
            category = kb_category.replace("V2", "V1").replace("v2", "V1")
        else:
            category = kb_category

        severity = (top_hit.get("severity") or "medium").lower()
        if severity not in ("high", "medium", "low"):
            severity = "medium"

        state = {
            "title":       question[:120],
            "category":    category,
            "severity":    severity,
            "description": last["answer"][:500],
            "reporter":    None,
            "team":        None,
            "environment": None,
            "notes":       None,
            "step":        "reporter",
        }
        _set_ticket_state(user, state)

        return (
            f"🎫 *Creating a ticket for:*\n_{state['title']}_\n\n"
            f"I already have:\n"
            f"   *Category:* {state['category']}\n"
            f"   *Severity:* {state['severity'].capitalize()}\n\n"
            f"I just need a few more details:\n\n"
            f"📝 *Your name?* _(Reporter)_"
        )

    step = state.get("step")

    # ── Collect reporter ──────────────────────────────────────────────────────
    if step == "reporter":
        state["reporter"] = message.strip()
        state["step"]     = "team"
        _set_ticket_state(user, state)
        team_opts = " / ".join(TICKET_TEAM_OPTIONS.keys())
        return (
            f"✅ Reporter: *{state['reporter']}*\n\n"
            f"📝 *Team / Department?*\n"
            f"_{team_opts}_"
        )

    # ── Collect team ──────────────────────────────────────────────────────────
    if step == "team":
        state["team"] = message.strip()
        state["step"] = "environment"
        _set_ticket_state(user, state)
        env_opts = " / ".join(TICKET_ENV_OPTIONS.keys())
        return (
            f"✅ Team: *{state['team']}*\n\n"
            f"📝 *Environment?*\n"
            f"_{env_opts}_"
        )

    # ── Collect environment ───────────────────────────────────────────────────
    if step == "environment":
        state["environment"] = message.strip()
        state["step"]        = "notes"
        _set_ticket_state(user, state)
        return (
            f"✅ Environment: *{state['environment']}*\n\n"
            f"📝 *Any additional notes?*\n"
            f"_Type your notes or type *skip* to proceed_"
        )

    # ── Collect notes ─────────────────────────────────────────────────────────
    if step == "notes":
        state["notes"] = "" if message.lower().strip() == "skip" else message.strip()
        state["step"]  = "confirm"
        _set_ticket_state(user, state)
        notes_line = f"\n   *Notes:* {state['notes']}" if state["notes"] else ""
        return (
            f"🎫 *Ready to create ticket:*\n\n"
            f"   *Title:* {state['title']}\n"
            f"   *Category:* {state['category']}\n"
            f"   *Severity:* {state['severity'].capitalize()}\n"
            f"   *Reporter:* {state['reporter']}\n"
            f"   *Team:* {state['team']}\n"
            f"   *Environment:* {state['environment']}"
            f"{notes_line}\n\n"
            f"Type *confirm* to create or *cancel* to abort."
        )

    # ── Confirm and create ────────────────────────────────────────────────────
    if step == "confirm":
        if message.lower().strip() in ("confirm", "yes", "ok", "y"):
            result = create_slack_list_ticket(state, client)
            _clear_ticket_state(user)
            return result
        else:
            _clear_ticket_state(user)
            return "❌ Ticket creation cancelled."

    return "⚠️ Something went wrong. Please try again."


def _rich_text_block(text: str) -> list:
    """Builds a Slack rich_text block value for list text columns."""
    return [
        {
            "type": "rich_text",
            "elements": [{
                "type": "rich_text_section",
                "elements": [{"type": "text", "text": text}]
            }]
        }
    ]


def create_slack_list_ticket(state: dict, client) -> str:
    """
    Creates a ticket as a Slack List item in F08B5SXE9H8.
    No channel post fallback — List only.
    """
    import datetime
    ticket_id   = f"IRT-{int(datetime.datetime.now().timestamp())}"
    notes_text  = state.get("notes") or ""
    severity    = (state.get("severity") or "medium").lower()
    if severity not in ("high", "medium", "low"):
        severity = "medium"

    description = state["description"][:500]
    if notes_text:
        description += f"\n\nAdditional notes: {notes_text}"

    # Map user text to option IDs
    category_id = _map_to_option_id(
        state.get("category", "Others"), TICKET_CATEGORY_OPTIONS
    )
    team_id = _map_to_option_id(
        state.get("team", "Testing"), TICKET_TEAM_OPTIONS
    )
    env_id = _map_to_option_id(
        state.get("environment", "AWS Production"), TICKET_ENV_OPTIONS
    )

    if not TICKET_LIST_ID:
        return (
            "⚠️ *Ticket list not configured.*\n"
            "_Please set `IRT_TICKET_LIST_ID` in your `.env` file._"
        )

    try:
        payload = {
            "list_id": TICKET_LIST_ID,
            "column_values": [
                {"column_id": "Col08B8D4N7NG", "rich_text": _rich_text_block(state["title"])},
                {"column_id": "Col08B32LLAUV", "select": ["new"]},
                {"column_id": "Col08BJJ5S3U1", "select": [severity]},
                {"column_id": "Col08CWKZS679", "select": [category_id]},
                {"column_id": "Col08D8APUF61", "select": [team_id]},
                {"column_id": "Col08DZVB9P0Q", "select": [env_id]},
                {"column_id": "Col08D4V7CF54", "rich_text": _rich_text_block(description)},
            ]
        }
        resp = client.api_call("slackLists.items.create", json=payload)
        log.warning(f"slackLists.items.create response: {resp}")

        if resp.get("ok"):
            item_id    = resp.get("item", {}).get("id", ticket_id)
            ticket_url = (
                f"https://app.slack.com/client/TJKT125D0"
                f"/unified-files/list/{TICKET_LIST_ID}/r/{item_id}"
            )
            return (
                f"✅ *Ticket raised successfully!*\n\n"
                f"   *ID:* `{item_id}`\n"
                f"   *Title:* {state['title']}\n"
                f"   *Category:* {state.get('category', '—')}\n"
                f"   *Severity:* {severity.capitalize()}\n"
                f"   *Reporter:* {state['reporter']}\n"
                f"   *Team:* {state.get('team', '—')}\n"
                f"   *Environment:* {state['environment']}\n"
                f"   *Status:* New\n\n"
                f"<{ticket_url}|📋 Open Ticket>\n\n"
                f"_📎 If you have any screenshots, open the ticket using the link above "
                f"and add them in the ticket thread — it helps the IRT team resolve it faster._"
            )

        # Handle specific known errors with clear messages
        err = resp.get("error", "unknown")
        log.error(f"Slack List creation failed: {err} — full response: {resp}")

        if err in ("max_items_reached", "list_full", "too_many_items"):
            list_url = f"https://app.slack.com/client/TJKT125D0/unified-files/list/{TICKET_LIST_ID}"
            return (
                f"⚠️ *Could not create ticket — the Bugs Tracker list is full.*\n\n"
                f"The list has reached its maximum item limit.\n"
                f"Please ask an admin to archive older items, then try again.\n\n"
                f"<{list_url}|📋 Open Bugs Tracker> to archive old items."
            )

        return (
            f"⚠️ *Ticket creation failed.*\n"
            f"_Error: {err}_\n\n"
            f"Please try again or raise the ticket manually in the Bugs Tracker."
        )

    except Exception as e:
        log.error(f"Slack List ticket error: {e}")
        return (
            f"⚠️ *Something went wrong while creating the ticket.*\n"
            f"_Error: {str(e)[:120]}_\n\n"
            f"Please try again or raise the ticket manually."
        )


def handle_conversational(query: str, history: list = None) -> str:
    """
    Handles greetings and capability questions.
    For capability questions → returns a precise structured response.
    For greetings → GPT generates a friendly reply.
    """
    q = query.lower().strip().rstrip("!?.,")

    # Live / real-time data questions — bot has no live data
    live_data_triggers = [
        "status of today", "today's ticket", "today tickets",
        "current status", "live data", "real time", "realtime",
        "counts of ticket", "ticket count", "how many ticket",
        "tickets closed", "tickets open", "tickets raised",
        "january 2026", "february 2026", "march 2026",  # specific date queries
        "this week", "this month", "last week", "last month",
    ]
    if any(t in q for t in live_data_triggers):
        return (
            "📊 *Live data is not available yet.*\n\n"
            "I currently work with past incident data from the IRT knowledge base. "
            "Real-time ticket counts, daily status, and live metrics are not connected yet.\n\n"
            "_This feature is planned for a future update. "
            "For current ticket status please check the Bug Tracker directly._"
        )

    # Capability questions — return precise hardcoded response
    capability_triggers = {
        "what can you do", "what do you do", "who are you", "what are you",
        "help", "what is this", "how does this work", "tell me about yourself",
        "what can you help with", "what are your capabilities",
        "what can the bot do", "capabilities", "features",
    }
    if q in capability_triggers or any(t in q for t in [
        "what can you", "what do you", "help me with", "capabilities",
        "can you do", "what are you able", "what are you capable",
    ]):
        return (
            "Hi! I'm *IRT Bot* — Conversight Immediate Response Team assistant.\n\n"
            "*🔍 I can help you diagnose issues:*\n"
            "Describe any ConverSight product problem — dataset failures, dataload errors, "
            "SME publish issues, notebook errors, connector failures, dashboard bugs — "
            "and I'll search past incidents to find what worked.\n\n"
            "*⚙️ I can do automations for you:*\n"
            "Like *Activate Dataset*, *Enable Athena IQ*, *Extend Trial Period*, "
            "*Increase User Count*, *Remove SME Duplicates* and more — "
            "just tell me what you need done.\n\n"
            "_Just describe your issue or tell me what you want done — I'll take it from there!_"
        )

    # Regular greeting → GPT handles it with conversation context
    system_prompt = """You are IRT Bot — Conversight Immediate Response Team assistant.
You help with ConverSight product issues and can execute automations.

Respond naturally and conversationally. Keep it short — 2-3 sentences max.
Be friendly and helpful. Use *bold* for emphasis (Slack format).
Never mention "knowledge base" or internal system details.
If the user seems to have a technical issue, encourage them to describe it."""

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    resp = ai.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL_SLACK", "gpt-4o"),
        max_tokens=100,
        messages=messages,
    )
    return (resp.choices[0].message.content or "").strip()


def is_conversational(query: str) -> bool:
    """
    Returns True if the message is a greeting or general chat
    rather than a technical support question.
    """
    q = query.lower().strip().rstrip("!?.,")
    greetings = {
        "hi", "hello", "hey", "hii", "helo", "yo", "sup",
        "good morning", "good afternoon", "good evening",
        "what can you do", "what do you do", "who are you",
        "help", "what is this", "how does this work",
        "how are you", "how r u", "thanks", "thank you",
        "ok", "okay", "cool", "got it", "bye", "goodbye",
        "what are you", "tell me about yourself",
    }
    return q in greetings or len(query.split()) <= 2 and not any(
        kw in query.lower() for kw in [
            "error", "fail", "stuck", "issue", "problem", "not working",
            "dataset", "load", "connector", "notebook", "dashboard",
            "slow", "crash", "broken", "status", "fix", "help with",
        ]
    )


# ── Block builder ─────────────────────────────────────────────────────────────
# KB cards show embedding match as `NN%` next to each similar ticket (higher = closer).

def _slack_question_preview(query: str, max_len: int = 320) -> str:
    """Single-line preview for blockquotes; keeps KB cards readable."""
    t = " ".join((query or "").split())
    if len(t) > max_len:
        return t[: max_len - 1] + "…"
    return t


def _format_reference(ref: str) -> str:
    """
    Renders a reference as a clickable Slack link if URL exists,
    or as readable italic text if it's a meaningful label,
    or hides it if it's generic/useless.
    """
    if not ref or str(ref).strip().lower() in (
        "none", "nan", "", "link", "n/a", "-", "null", "no reference"
    ):
        return ""

    ref = str(ref).strip()

    # Already a Slack mrkdwn link <url|label> — pass through
    if re.match(r"^<https?://[^>]+>$", ref):
        return ref

    # Contains a plain URL — wrap as clickable link
    url_match = re.search(r"https?://\S+", ref)
    if url_match:
        url = url_match.group(0).rstrip(".,)>\"'")
        if "asana.com" in url:
            label = "Asana ticket"
        elif "slack.com" in url:
            label = "Slack thread"
        elif "github.com" in url:
            label = "GitHub"
        elif "jira" in url:
            label = "Jira ticket"
        elif "docs.google" in url:
            label = "Google Doc"
        else:
            label = "Reference"
        return f"<{url}|{label}>"

    # Plain text label with no URL — show as italic if meaningful
    # Hide single generic words that add no value
    if ref.lower() in ("link", "url", "ref", "reference", "ticket", "doc"):
        return ""

    return f"_{ref}_"


def build_blocks(query: str, answer: str, hits: list) -> list:
    icons      = {"Fixed": "✅", "Partial": "⚠️", "Workaround": "⚠️",
                  "Unresolved": "❌", "Rejected": "🚫"}
    res_labels = {"Fixed": "Fixed", "Partial": "Partial fix",
                  "Workaround": "Workaround", "Unresolved": "Unresolved",
                  "Rejected": "Rejected"}

    q_prev = _slack_question_preview(query)
    hit_lines: list[str] = []
    for h in hits[:3]:
        icon  = icons.get(h["resolution_status"], "❓")
        label = res_labels.get(h["resolution_status"], h["resolution_status"])
        pct   = int(h["score"] * 100)
        title = (h.get("summary") or "")[:72]
        if len(h.get("summary") or "") > 72:
            title += "…"
        cat = (h.get("bug_category") or "")[:36]
        rca = " · _RCA_" if h.get("source") == "RCA" else ""
        ref = _format_reference(h.get("references", ""))
        line = f"• {icon} *{title}*{rca}\n    `{pct}%` · {label}"
        if cat:
            line += f" · _{cat}_"
        if ref:
            line += f"\n    {ref}"
        hit_lines.append(line)

    hits_text = "\n".join(hit_lines) if hit_lines else "_No related tickets in the top results._"

    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "📗 Knowledge base result", "emoji": True},
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": "_IRT Support Bot_"},
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*👤 Your question*\n>{q_prev}",
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*🤖 Suggested answer*\n{answer}"},
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*📎 Similar past issues*\n{hits_text}"},
        },
    ]


def step_block(txt: str) -> list:
    """Bot reply: small gray label + body so it reads differently from the user’s messages."""
    return [
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "_IRT Support Bot_"}],
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": txt}},
    ]


_SLACK_PREVIEW_MAX = 300


def slack_notification_preview(blocks: list) -> str:
    """
    Short line for the required top-level `text` field (notifications, search, legacy).
    Must not repeat the full answer — that belongs only in `blocks`.
    """
    if not blocks:
        return "IRT Support Bot"
    for b in blocks:
        bt = b.get("type")
        if bt == "header":
            inner = b.get("text") or {}
            txt = inner.get("text") if isinstance(inner, dict) else None
            if txt:
                return str(txt)[:_SLACK_PREVIEW_MAX]
        if bt == "context":
            for el in b.get("elements") or []:
                if el.get("type") != "mrkdwn":
                    continue
                et = el.get("text") or ""
                if "quick question" in et:
                    return "Clarification needed"
    return "IRT Support Bot"


def message_with_blocks(blocks: list, *, preview: str | None = None) -> dict:
    """
    Slack displays both top-level `text` and `blocks` in the thread; putting the full
    reply in `text` duplicates the body. Keep `text` to a short preview only.
    """
    t = (preview or slack_notification_preview(blocks)).strip() or "IRT Support Bot"
    if len(t) > _SLACK_PREVIEW_MAX:
        t = t[: _SLACK_PREVIEW_MAX - 1] + "…"
    return {"text": t, "blocks": blocks}


# ── Core streaming function ───────────────────────────────────────────────────

def _is_dm_channel(channel: str) -> bool:
    """1:1 DM channels have IDs starting with D."""
    return bool(channel) and channel.startswith("D")


def stream_response(
    client,
    channel: str,
    query: str,
    thread_ts: str = None,
    ephemeral_user: str = None,
    user_id: str = None,
    slack_message_ts: str | None = None,
) -> None:
    """
    For public messages (channel / DM):
      - Posts "Searching…" immediately, animates steps 2-3,
        then replaces with the final answer via chat_update.

    For ephemeral (/irt-test):
      - Skips loading messages entirely (Slack cannot update or delete
        ephemeral messages — they would pile up and never go away).
      - Runs search + generate silently, then posts ONE clean final answer.
    """

    # Only nest under Slack threads the user is already in, or clarify/automation (handled below).
    reply_thread = thread_ts

    # ── Handle reset command — clear ALL session memory first, then reply (in-thread if user reset from a thread)
    if query.lower().strip() in ("reset", "clear", "new", "start over"):
        if user_id:
            _clear_history(user_id, channel)
            _clear_auto_state(user_id, channel)
            _clear_ticket_state(user_id)
            _clear_last_answer(user_id)
        reset_kw = {
            "channel": channel,
            "text": "🔄 Conversation reset. Ask me anything!",
        }
        if thread_ts:
            reset_kw["thread_ts"] = thread_ts
        client.chat_postMessage(**reset_kw)
        return

    # ── Ticket agent path (active session) ───────────────────────────────────
    ticket_state = _get_ticket_state(user_id) if user_id else None
    if ticket_state:
        response = ticket_agent(user_id, query, client, channel)
        kw = {"channel": channel, **message_with_blocks(step_block(response))}
        tt = ticket_state.get("thread_ts") or reply_thread or thread_ts
        if tt:
            kw["thread_ts"] = tt
        client.chat_postMessage(**kw)
        return

    # ── Automation agent path (active session) ───────────────────────────────
    # If user already has an active automation session, route directly to agent.
    # New automation intent is detected inside analyze_query below.
    auto_state = _get_auto_state(user_id, channel) if user_id else None

    if auto_state:
        awaiting = bool(auto_state.get("awaiting_confirm"))
        if _should_interrupt_automation_for_chat(query, awaiting):
            log.warning("automation interrupted: conversational message; clearing state")
            _clear_auto_state(user_id, channel)
            auto_state = None

    if auto_state:
        # Thread anchor comes from state (set when automation ID prompts start in DM)
        auto_thread_ts = auto_state.get("thread_ts") or reply_thread or thread_ts

        if query.lower().strip() in ("cancel", "abort", "stop", "exit"):
            _clear_auto_state(user_id, channel)
            cancel_text = "❌ Automation cancelled. Ask me anything else!"
            kw = {"channel": channel, **message_with_blocks(step_block(cancel_text))}
            if auto_thread_ts:
                kw["thread_ts"] = auto_thread_ts
            client.chat_postMessage(**kw)
            return

        response = automation_agent(user_id, channel, query)
        kw = {"channel": channel, **message_with_blocks(step_block(response))}
        if auto_thread_ts:
            kw["thread_ts"] = auto_thread_ts
        client.chat_postMessage(**kw)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # EPHEMERAL PATH — no loading messages, single final post
    # Slack API hard rule: ephemeral messages cannot be edited or deleted,
    # so any "Searching…" placeholder would stay visible forever above the
    # answer. Solution: run everything silently and post only the final result.
    # ══════════════════════════════════════════════════════════════════════════
    if ephemeral_user:
        try:
            history_channel = (
                channel
                if _is_dm_channel(channel)
                else (f"{channel}:{thread_ts}" if thread_ts else channel)
            )
            history  = _get_history(user_id, history_channel) if user_id else []
            decision = analyze_query(query, history)

            if decision["action"] == "chat":
                answer = handle_conversational(query, history)
                if user_id:
                    _add_history(user_id, history_channel, "user", query)
                    _add_history(user_id, history_channel, "assistant", answer)
                final_text   = answer
                final_blocks = step_block(answer)

            elif decision["action"] == "ticket":
                response     = ticket_agent(user_id, query, client, channel)
                final_text   = response
                final_blocks = step_block(response)

            elif decision["action"] == "automate":
                category = decision["text"]
                response = automation_agent(user_id, channel, query, category=category)
                final_text   = response
                final_blocks = step_block(response)

            elif decision["action"] == "automateinfo":
                final_text   = automation_info_response(decision["text"])
                final_blocks = step_block(final_text)

            elif decision["action"] == "outofscope":
                final_text   = (
                    "⚠️ *This is outside my scope.*\n\n"
                    "I only help with ConverSight product issues — dataset failures, "
                    "dataload errors, SME publish, notebooks, connectors, and similar bugs.\n\n"
                    "For API keys, security, or account questions please contact your admin or raise a ticket."
                )
                final_blocks = step_block(final_text)

            elif decision["action"] == "clarify":
                clarification = decision["text"]
                suggestions   = decision.get("suggestions", [])
                if user_id:
                    _add_history(user_id, history_channel, "user", query)
                    _add_history(user_id, history_channel, "assistant", f"🤔 {clarification}")
                hint = ""
                if suggestions:
                    hint = f"\n_Quick answers: {' · '.join(suggestions)}_"
                final_text   = f"🤔 {clarification}{hint}"
                final_blocks = step_block(final_text)

            else:  # search
                gpt_query = decision["text"]
                if history and len(query.split()) <= 5 and gpt_query:
                    search_q = gpt_query
                elif history and gpt_query and gpt_query.lower() != query.lower():
                    search_q = gpt_query
                else:
                    search_q = query
                log.warning(f"search_q='{search_q[:80]}'")
                hits = search_kb(search_q)
                if not hits or hits[0]["score"] < MIN_SCORE:
                    final_text   = (
                        f"❌ *No similar issues found.*\n\n"
                        "This may be a new issue. Please create a ticket in the Bug Tracker."
                    )
                    final_blocks = step_block(final_text)
                else:
                    answer = generate_answer(search_q, hits, history)
                    if user_id:
                        _add_history(user_id, history_channel, "user", search_q)
                        _add_history(user_id, history_channel, "assistant", answer)
                        _save_last_answer(user_id, search_q, answer, hits)
                    final_text   = answer
                    final_blocks = build_blocks(search_q, answer, hits)

        except Exception as e:
            log.error(f"stream_response (ephemeral) error: {e}")
            final_text   = f"⚠️ {_friendly_error(e)}"
            final_blocks = step_block(final_text)

        client.chat_postEphemeral(
            channel=channel,
            user=ephemeral_user,
            **message_with_blocks(final_blocks),
        )
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC PATH — post loading message, animate steps, replace with answer
    # ══════════════════════════════════════════════════════════════════════════

    # Post step 1 immediately so user sees the bot is alive (DM + memory → under user msg)
    kw = {
        "channel": channel,
        **message_with_blocks(
            step_block(STEPS[0]),
            preview="🔍 Searching knowledge base…",
        ),
    }
    if reply_thread:
        kw["thread_ts"] = reply_thread
    r      = client.chat_postMessage(**kw)
    msg_ts = r.get("ts")

    # Stop flag — animation checks this every 0.1s and halts when set
    stop_flag = {"done": False}

    def animate():
        """
        Cycles through loading steps continuously until stop_flag is set.
        Keeps looping back to step 1 if all steps shown and still processing.
        This ensures the user always sees activity for long-running queries.
        """
        steps = STEPS[1:]   # steps 2 and 3
        idx   = 0
        while True:
            # Wait 2.5s, checking flag every 0.1s
            for _ in range(25):
                if stop_flag["done"]:
                    return
                time.sleep(0.1)
            if stop_flag["done"]:
                return
            try:
                client.chat_update(
                    channel=channel,
                    ts=msg_ts,
                    **message_with_blocks(
                        step_block(steps[idx]),
                        preview="🔍 Searching knowledge base…",
                    ),
                )
            except Exception:
                pass
            idx = (idx + 1) % len(steps)   # cycle back if still running

    anim = threading.Thread(target=animate, daemon=True)
    anim.start()

    # Search + generate
    final_text   = ""
    final_blocks = []
    try:
        # DM: one memory stream for the whole DM; channels use per-thread keys
        history_channel = (
            channel
            if _is_dm_channel(channel)
            else (f"{channel}:{thread_ts}" if thread_ts else channel)
        )
        history  = _get_history(user_id, history_channel) if user_id else []
        decision = analyze_query(query, history)

        if decision["action"] == "chat":
            answer = handle_conversational(query, history)
            if user_id:
                _add_history(user_id, history_channel, "user", query)
                _add_history(user_id, history_channel, "assistant", answer)
            final_text   = answer
            final_blocks = step_block(answer)

        elif decision["action"] == "ticket":
            stop_flag["done"] = True
            anim.join(timeout=1)
            try:
                client.chat_delete(channel=channel, ts=msg_ts)
            except Exception:
                pass
            response = ticket_agent(user_id, query, client, channel)

            is_pure_dm = _is_dm_channel(channel)
            if is_pure_dm or reply_thread or thread_ts:
                kw = {"channel": channel, **message_with_blocks(step_block(response))}
                tt = reply_thread or thread_ts
                if tt:
                    kw["thread_ts"] = tt
                client.chat_postMessage(**kw)
            else:
                # Channel — post anchor then thread ticket conversation under it
                anchor = client.chat_postMessage(
                    channel=channel,
                    **message_with_blocks(
                        [
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"🎫 *Ticket request:* {query}",
                                },
                            }
                        ],
                        preview="🎫 Ticket request",
                    ),
                )
                client.chat_postMessage(
                    channel=channel,
                    **message_with_blocks(step_block(response)),
                    thread_ts=anchor["ts"],
                )
                # Save anchor ts so all follow-up ticket messages stay in thread
                ticket_s = _get_ticket_state(user_id) or {}
                ticket_s["thread_ts"] = anchor["ts"]
                _set_ticket_state(user_id, ticket_s)
            return

        elif decision["action"] == "automate":
            category = decision["text"]
            # Stop animation
            stop_flag["done"] = True
            anim.join(timeout=1)
            try:
                client.chat_delete(channel=channel, ts=msg_ts)
            except Exception:
                pass
            response = automation_agent(user_id, channel, query, category=category)

            # DM: nest org/dataset/tenant prompts under the user's trigger message only
            is_pure_dm = _is_dm_channel(channel)
            if is_pure_dm:
                st = _get_auto_state(user_id, channel)
                kw = {"channel": channel, **message_with_blocks(step_block(response))}
                if st and slack_message_ts:
                    if not st.get("thread_ts"):
                        st["thread_ts"] = slack_message_ts
                        _set_auto_state(user_id, channel, st)
                    kw["thread_ts"] = st["thread_ts"]
                client.chat_postMessage(**kw)
            elif thread_ts:
                kw = {
                    "channel": channel,
                    **message_with_blocks(step_block(response)),
                    "thread_ts": thread_ts,
                }
                client.chat_postMessage(**kw)
            else:
                # Channel — post question anchor then thread automation under it
                anchor = client.chat_postMessage(
                    channel=channel,
                    **message_with_blocks(
                        [
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"⚙️ *Automation request:* {query}",
                                },
                            }
                        ],
                        preview="⚙️ Automation request",
                    ),
                )
                client.chat_postMessage(
                    channel=channel,
                    **message_with_blocks(step_block(response)),
                    thread_ts=anchor["ts"],
                )
                # Save anchor ts so follow-up messages thread correctly
                _set_auto_state(user_id, channel, {
                    **(_get_auto_state(user_id, channel) or {}),
                    "thread_ts": anchor["ts"],
                })
            return

        elif decision["action"] == "automateinfo":
            final_text   = automation_info_response(decision["text"])
            final_blocks = step_block(final_text)

        elif decision["action"] == "outofscope":
            final_text   = (
                "⚠️ *This is outside my scope.*\n\n"
                "I only help with ConverSight product issues — dataset failures, "
                "dataload errors, SME publish, notebooks, connectors, and similar bugs.\n\n"
                "For API keys, security, or account questions please contact your admin or raise a ticket."
            )
            final_blocks = step_block(final_text)

        elif decision["action"] == "clarify":
            clarification = decision["text"]
            suggestions   = decision.get("suggestions", [])

            # Stop animation FIRST
            stop_flag["done"] = True
            anim.join(timeout=1)

            try:
                client.chat_delete(channel=channel, ts=msg_ts)
            except Exception:
                pass

            # DM (anchored to user message), mpim, or any thread: reply in that thread
            if thread_ts:
                anchor_ts = thread_ts
                sent = client.chat_postMessage(
                    channel=channel,
                    thread_ts=anchor_ts,
                    **message_with_blocks(
                        clarify_blocks(clarification, suggestions),
                        preview="Clarification needed",
                    ),
                )
                if user_id:
                    _add_history(user_id, history_channel, "user", query)
                    _add_history(user_id, history_channel, "assistant", f"🤔 {clarification}")
                    _save_pending(
                        ts=sent["ts"], query=query, user=user_id, channel=channel
                    )
                    _save_pending(
                        ts=anchor_ts, query=query, user=user_id, channel=channel
                    )
                    _pending[sent["ts"]]["clarify_ts"] = sent["ts"]
                    _pending[anchor_ts]["clarify_ts"] = sent["ts"]
                    _pending[sent["ts"]]["anchor_ts"] = anchor_ts
                    _pending[anchor_ts]["anchor_ts"] = anchor_ts
            else:
                if _is_dm_channel(channel) and slack_message_ts:
                    # 1:1 DM — nest clarify under the user's message so follow-ups stay in one thread
                    anchor_ts = slack_message_ts
                    sent = client.chat_postMessage(
                        channel=channel,
                        thread_ts=anchor_ts,
                        **message_with_blocks(
                            clarify_blocks(clarification, suggestions),
                            preview="Clarification needed",
                        ),
                    )
                    if user_id:
                        _add_history(user_id, history_channel, "user", query)
                        _add_history(user_id, history_channel, "assistant", f"🤔 {clarification}")
                        _save_pending(
                            ts=sent["ts"], query=query, user=user_id, channel=channel
                        )
                        _save_pending(
                            ts=anchor_ts, query=query, user=user_id, channel=channel
                        )
                        _pending[sent["ts"]]["clarify_ts"] = sent["ts"]
                        _pending[anchor_ts]["clarify_ts"] = sent["ts"]
                        _pending[sent["ts"]]["anchor_ts"] = anchor_ts
                        _pending[anchor_ts]["anchor_ts"] = anchor_ts
                elif _is_dm_channel(channel):
                    sent = client.chat_postMessage(
                        channel=channel,
                        **message_with_blocks(
                            clarify_blocks(clarification, suggestions),
                            preview="Clarification needed",
                        ),
                    )
                    if user_id:
                        _add_history(user_id, history_channel, "user", query)
                        _add_history(user_id, history_channel, "assistant", f"🤔 {clarification}")
                        _save_pending(
                            ts=sent["ts"], query=query, user=user_id, channel=channel
                        )
                else:
                    # Channel — anchor question, then clarify in thread
                    anchor = client.chat_postMessage(
                        channel=channel,
                        **message_with_blocks(
                            [
                                {
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": f"*Question:* {query}",
                                    },
                                }
                            ],
                            preview="Question",
                        ),
                    )
                    anchor_ts = anchor["ts"]
                    sent = client.chat_postMessage(
                        channel=channel,
                        thread_ts=anchor_ts,
                        **message_with_blocks(
                            clarify_blocks(clarification, suggestions),
                            preview="Clarification needed",
                        ),
                    )
                    if user_id:
                        _save_pending(
                            ts      = sent["ts"],
                            query   = query,
                            user    = user_id,
                            channel = channel,
                        )
                        _save_pending(
                            ts      = anchor_ts,
                            query   = query,
                            user    = user_id,
                            channel = channel,
                        )
                        _pending[sent["ts"]]["clarify_ts"] = sent["ts"]
                        _pending[anchor_ts]["clarify_ts"] = sent["ts"]
                        _pending[sent["ts"]]["anchor_ts"] = anchor_ts
                        _pending[anchor_ts]["anchor_ts"] = anchor_ts
            return

        else:  # search
            gpt_query = decision["text"]

            if history and len(query.split()) <= 5 and gpt_query:
                search_q = gpt_query
            elif history and gpt_query and gpt_query.lower() != query.lower():
                search_q = gpt_query
            else:
                search_q = query

            log.warning(f"search_q='{search_q[:80]}'")
            hits = search_kb(search_q)
            if not hits or hits[0]["score"] < MIN_SCORE:
                final_text   = (
                    f"❌ *No similar issues found.*\n\n"
                    "This may be a new issue. Please create a ticket in the Bug Tracker.\n"
                    "_Type another question or type *reset* to start fresh._"
                )
                final_blocks = step_block(final_text)
            else:
                answer = generate_answer(search_q, hits, history)
                if user_id:
                    _add_history(user_id, history_channel, "user", search_q)
                    _add_history(user_id, history_channel, "assistant", answer)
                    _save_last_answer(user_id, search_q, answer, hits)
                final_text   = answer
                final_blocks = build_blocks(search_q, answer, hits)

    except Exception as e:
        log.error(f"stream_response error: {e}")
        final_text   = f"⚠️ {_friendly_error(e)}"
        final_blocks = step_block(final_text)

    # Signal animation to stop FIRST, then wait for it to finish
    stop_flag["done"] = True
    anim.join(timeout=2)

    # Replace loading message with final answer
    try:
        client.chat_update(
            channel=channel,
            ts=msg_ts,
            **message_with_blocks(final_blocks),
        )
    except Exception as e:
        log.error(f"chat_update failed: {e}")
        kw = {"channel": channel, **message_with_blocks(final_blocks)}
        if reply_thread:
            kw["thread_ts"] = reply_thread
        client.chat_postMessage(**kw)


# ── Modal view ────────────────────────────────────────────────────────────────

def irt_modal_view(title="Ask IRT Bot", prefill=""):
    return {
        "type": "modal",
        "callback_id": "irt_modal_submit",
        "title": {"type": "plain_text", "text": title},
        "submit": {"type": "plain_text", "text": "🔍 Search"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text":
                    "*Search the IRT Knowledge Base* 🔍\n"
                    "Describe your issue. The bot will search past cases and generate a fix."}
            },
            {
                "type": "input",
                "block_id": "query_block",
                "label": {"type": "plain_text", "text": "Your question / issue"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "query_input",
                    "multiline": True,
                    "initial_value": prefill,
                    "placeholder": {"type": "plain_text",
                        "text": "e.g. Conversight Usage v2 failed, dataset stuck in DictionaryRequested…"}
                }
            },
            {
                "type": "input",
                "block_id": "visibility_block",
                "label": {"type": "plain_text", "text": "Who sees the answer?"},
                "element": {
                    "type": "static_select",
                    "action_id": "visibility_select",
                    "initial_option": {
                        "text": {"type": "plain_text", "text": "Only me (test)"},
                        "value": "ephemeral"
                    },
                    "options": [
                        {"text": {"type": "plain_text", "text": "Only me (test)"}, "value": "ephemeral"},
                        {"text": {"type": "plain_text", "text": "Whole channel"},  "value": "in_channel"},
                    ]
                }
            }
        ]
    }


# ── Slash commands ────────────────────────────────────────────────────────────

@app.command("/irt")
def handle_irt(ack, command, client):
    ack()
    query   = command.get("text", "").strip()
    channel = command.get("channel_id", "")
    user    = command.get("user_id", "")
    if not query:
        client.chat_postEphemeral(channel=channel, user=user,
            text="Please add a question. Example: `/irt v2 dataset failed`")
        return
    log.warning(f"/irt u={user} q={query[:80]}")
    threading.Thread(
        target=stream_response,
        args=(client, channel, query),
        kwargs={"user_id": user},
        daemon=True
    ).start()


@app.command("/irt-test")
def handle_irt_test(ack, command, client):
    ack()
    query   = command.get("text", "").strip()
    channel = command.get("channel_id", "")
    user    = command.get("user_id", "")
    if not query:
        client.chat_postEphemeral(channel=channel, user=user,
            text="🧪 Test mode — only you see this.\nUsage: `/irt-test v2 dataset failed`")
        return
    log.warning(f"/irt-test u={user} q={query[:80]}")
    threading.Thread(
        target=stream_response,
        args=(client, channel, query),
        kwargs={"ephemeral_user": user, "user_id": user},
        daemon=True
    ).start()


# ── Modal handlers ────────────────────────────────────────────────────────────

@app.shortcut("ask_irt_bot")
def open_irt_modal(ack, shortcut, client):
    ack()
    client.views_open(trigger_id=shortcut["trigger_id"], view=irt_modal_view())


@app.view("irt_modal_submit")
def handle_modal_submit(ack, body, client, view):
    ack()
    user       = body["user"]["id"]
    query      = view["state"]["values"]["query_block"]["query_input"]["value"].strip()
    visibility = view["state"]["values"]["visibility_block"]["visibility_select"]["selected_option"]["value"]
    ephem_user = user if visibility == "ephemeral" else None
    # Use the channel where the shortcut was triggered, fall back to IRT_CHANNEL
    channel    = (body.get("channel") or {}).get("id") or IRT_CHANNEL
    log.warning(f"modal u={user} vis={visibility} ch={channel} q={query[:80]}")
    threading.Thread(
        target=stream_response,
        args=(client, channel, query),
        kwargs={"ephemeral_user": ephem_user, "user_id": user},
        daemon=True
    ).start()


@app.action(re.compile(r"clarify_reply(_\d+)?"))
def handle_clarify_reply(ack, body, client):
    """
    When user clicks a suggestion button (e.g. 'v1', 'v2', 'Production'),
    treat it like a thread reply answer.
    """
    ack()
    user       = body["user"]["id"]
    channel    = body["channel"]["id"]
    value      = body["actions"][0]["value"]
    msg_ts     = body["message"]["ts"]   # ts of the clarification message
    thread_ts  = body["message"].get("thread_ts", msg_ts)
    log.warning(f"clarify_reply u={user} v={value}")

    pending = _get_pending(msg_ts) or _get_pending(thread_ts)
    if pending:
        _clear_pending(msg_ts)
        _clear_pending(thread_ts)
        original_query = pending["query"]
        log.warning(f"clarify_reply: original='{original_query[:60]}' answer='{value}'")
        # Build natural enriched query from original question + button answer
        # e.g. "dataset stuck DictionaryRequested" + "v2"
        #   → "v2 dataset stuck with DictionaryRequested status"
        search_input = build_enriched_query(original_query, value)
    else:
        search_input = value

    threading.Thread(
        target=stream_response,
        args=(client, channel, search_input),
        kwargs={"thread_ts": thread_ts, "user_id": user},
        daemon=True
    ).start()


@app.action("ask_another")
def handle_ask_another(ack, body, client):
    ack()
    tid = body.get("trigger_id")
    if tid:
        try:
            client.views_open(trigger_id=tid, view=irt_modal_view(title="Ask another question"))
        except Exception as e:
            log.error(f"ask_another error: {e}")


@app.action("create_ticket")
def handle_create_ticket(ack):
    ack()


# ── Dedup guard — prevent double responses ────────────────────────────────────
# Both `message` and `app_mention` events can fire for the same @mention.
# Track recently processed message ts to avoid handling the same message twice.
_processed: set = set()
_processed_lock = threading.Lock()

def _already_processed(ts: str) -> bool:
    with _processed_lock:
        if ts in _processed:
            return True
        _processed.add(ts)
        # Keep set small — remove entries older than last 200
        if len(_processed) > 200:
            oldest = list(_processed)[:100]
            for t in oldest:
                _processed.discard(t)
        return False


def _slack_event_is_im(event: dict) -> bool:
    """True for 1:1 DM. Slack sometimes omits channel_type; DM channels IDs start with D."""
    ct = (event.get("channel_type") or "").strip()
    if ct == "im":
        return True
    ch = event.get("channel") or ""
    return not ct and ch.startswith("D")


def _slack_event_is_dm_or_mpim(event: dict) -> bool:
    """True for DM or group DM so we can answer without requiring channel_type."""
    ct = (event.get("channel_type") or "").strip()
    if ct in ("im", "mpim"):
        return True
    ch = event.get("channel") or ""
    if not ct and ch.startswith("D"):
        return True
    return False


@app.event("message")
def handle_dm(event, client):
    if event.get("bot_id") or event.get("subtype"):
        return

    query        = clean(event.get("text", "")).strip()
    channel      = event.get("channel", "")
    user         = event.get("user", "")
    thread_ts    = event.get("thread_ts")
    event_ts     = event.get("ts", "")

    if not query or not user:
        return

    # Dedup — skip if already handled by app_mention handler
    if _already_processed(event_ts):
        return

    # ── Thread reply path ─────────────────────────────────────────────────────
    if thread_ts:
        if not _slack_event_is_im(event):
            pending = _get_pending(thread_ts)
            if pending and pending["user"] == user:
                # User answered the clarifying question
                log.warning(f"thread_reply u={user} original='{pending['query'][:50]}' reply='{query[:50]}'")
                # Clear both keys we saved
                _clear_pending(thread_ts)
                _clear_pending(pending.get("clarify_ts", ""))
                # Build a natural enriched query combining original + answer
                # e.g. "dataset stuck DictionaryRequested" + "v2" 
                #   → "v2 dataset stuck with DictionaryRequested status"
                enriched = build_enriched_query(pending["query"], query)
                threading.Thread(
                    target=stream_response,
                    args=(client, channel, enriched),
                    kwargs={
                        "thread_ts": thread_ts,
                        "user_id": user,
                        "slack_message_ts": event_ts,
                    },
                    daemon=True
                ).start()
            else:
                # Follow-up message in thread using thread context
                log.warning(f"thread_followup u={user} q={query[:80]}")
                threading.Thread(
                    target=stream_response,
                    args=(client, channel, query),
                    kwargs={
                        "thread_ts": thread_ts,
                        "user_id": user,
                        "slack_message_ts": event_ts,
                    },
                    daemon=True
                ).start()
            return

        # ── 1:1 DM + thread_ts: keep the whole turn in this thread until reset or a main-channel message
        pending = _get_pending(thread_ts)
        if pending and pending["user"] == user:
            _clear_pending(thread_ts)
            _clear_pending(pending.get("clarify_ts", ""))
            enriched = build_enriched_query(pending["query"], query)
            threading.Thread(
                target=stream_response,
                args=(client, channel, enriched),
                kwargs={
                    "thread_ts": thread_ts,
                    "user_id": user,
                    "slack_message_ts": event_ts,
                },
                daemon=True
            ).start()
        else:
            # Any other follow-up in the thread (KB, chat, automation, corrections like "sorry it's v1")
            log.warning(f"DM thread follow-up u={user} q={query[:80]}")
            threading.Thread(
                target=stream_response,
                args=(client, channel, query),
                kwargs={
                    "thread_ts": thread_ts,
                    "user_id": user,
                    "slack_message_ts": event_ts,
                },
                daemon=True
            ).start()
        return

    # ── DM / group DM direct message path ────────────────────────────────────
    if not _slack_event_is_dm_or_mpim(event):
        return

    log.warning(f"DM u={user} q={query[:80]}")
    threading.Thread(
        target=stream_response,
        args=(client, channel, query),
        kwargs={"user_id": user, "slack_message_ts": event_ts},
        daemon=True
    ).start()


# ── @mention handler ──────────────────────────────────────────────────────────

@app.event("app_mention")
def handle_mention(event, client):
    text      = re.sub(r"<@[A-Z0-9]+>\s*", "", event.get("text", "")).strip()
    channel   = event.get("channel", "")
    user      = event.get("user", "")
    ts        = event.get("ts")
    thread_ts = event.get("thread_ts")

    # Mark as processed so handle_dm doesn't also handle this message
    if ts:
        _already_processed(ts)

    # ── Thread context — always follow-up, never fresh ────────────────────────
    if thread_ts:
        pending = _get_pending(thread_ts)
        if pending and pending["user"] == user:
            log.warning(f"mention_thread_reply u={user} reply='{text[:60]}'")
            _clear_pending(thread_ts)
            # Build natural enriched query from original question + answer
            enriched = build_enriched_query(pending["query"], text)
            threading.Thread(
                target=stream_response,
                args=(client, channel, enriched),
                kwargs={"thread_ts": thread_ts, "user_id": user},
                daemon=True
            ).start()
        else:
            log.warning(f"mention_thread_followup u={user} q={text[:80]}")
            threading.Thread(
                target=stream_response,
                args=(client, channel, text),
                kwargs={"thread_ts": thread_ts, "user_id": user},
                daemon=True
            ).start()
        return

    if not text:
        client.chat_postMessage(
            channel=channel, thread_ts=ts,
            text="Hi! Use `/irt your question` or DM me directly."
        )
        return

    log.warning(f"mention u={user} q={text[:80]}")
    threading.Thread(
        target=stream_response,
        args=(client, channel, text),
        kwargs={"thread_ts": ts, "user_id": user},
        daemon=True
    ).start()


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    missing = [k for k, v in [
        ("SLACK_BOT_TOKEN", SLACK_BOT_TOKEN),
        ("SLACK_APP_TOKEN", SLACK_APP_TOKEN),
        ("OPENAI_API_KEY",  os.environ.get("OPENAI_API_KEY")),
    ] if not v]
    if missing:
        [print(f"❌  {k} missing in .env") for k in missing]
        exit(1)

    print()
    print("=" * 62)
    print("  🤖  IRT RAG Slack Bot v7  — Agentic Automation")
    print("=" * 62)
    print(f"  /irt <question>      → visible to whole channel  ✅")
    print(f"  /irt-test <question> → only you see it           ✅")
    print(f"  Ask IRT Bot button   → modal + live loading      ✅")
    print(f"  DM the bot           → chatbot with memory         ✅")
    print(f"  @mention bot         → reply in thread           ✅")
    print(f"  Clarify question     → thread reply triggers KB  ✅")
    print(f"  Automation agent     → 14 categories via API     ✅")
    print(f"  Type 'reset' in DM   → clears history + cache + ticket  ✅")
    print(f"  Knowledge base       : {kb_count:,} documents")
    print(f"  Automation token     : {'✅ set' if AUTOMATION_TOKEN else '❌ missing IRT_AUTOMATION_TOKEN'}")
    print(f"  Ticket list          : {'✅ ' + TICKET_LIST_ID if TICKET_LIST_ID else '❌ missing IRT_TICKET_LIST_ID'}")
    print(f"  Chat memory          : last {CHAT_HISTORY_LEN} turns per user")
    print("=" * 62)
    print()

    SocketModeHandler(app, SLACK_APP_TOKEN).start()