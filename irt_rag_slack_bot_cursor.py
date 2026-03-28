"""
IRT RAG Slack Bot — Cursor fork (`irt_rag_slack_bot_cursor.py`)
Socket Mode: NO URL needed. Just run this script and it connects.

This file carries the Cursor-maintained bot (automation threading, scoped state keys,
Extend Trail parsing, analyze_query always runs for new /irt intents, cancel clears
all sessions, etc.). Payload shapes follow ../IRTAutomation_Docs.md.

Historical / base features:
  • search_kb: query_points() (qdrant >= 1.7)
  • Bold header-style separators, similarity labels in footer
  • Chat memory (last N turns per user/channel or thread)
  • Friendly errors to users

Run:
  conda activate bug_tracker
  cd /home/user/workspace/python/script_new/irt_rag
  python irt_rag_slack_bot_cursor.py
"""

import os, re, time, logging, threading, json
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
COLLECTION       = "irt_knowledge_base"
EMBED_MODEL      = "all-MiniLM-L6-v2"
STORAGE_DIR      = "./qdrant_storage"
TOP_K            = 5
MIN_SCORE        = 0.30
CHAT_HISTORY_LEN = 6   # last N user+assistant turn pairs to keep per user

# ── Automation API ────────────────────────────────────────────────────────────
# Use HTTPS: plain HTTP gets 308 Permanent Redirect from nginx; urllib often surfaces
# that as an error for POST instead of following with the body. Override via .env if needed.
AUTOMATION_API_URL = os.environ.get(
    "IRT_AUTOMATION_API_URL",
    "https://api.conversight.ai/universe-engine/v2/api/resource/action/"
    "crn:prod:us:step_flow:9b505609-832c-453b-9e07-19897c59273e:"
    "standard:irtbot?action=irtbotautomation",
)
AUTOMATION_TOKEN = "JWT eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfZG9jIjp7InVzZXJJZCI6ImIzNDJlYjJmLTM5ZmYtNDY2NS1iOWMwLTg1ZDdiYjM2NDk0OSIsImF0aGVuYUlkIjoiYjM0MmViMmYtMzlmZi00NjY1LWI5YzAtODVkN2JiMzY0OTQ5Iiwib3JnSWQiOiI5YjUwNTYwOS04MzJjLTQ1M2ItOWUwNy0xOTg5N2M1OTI3M2UiLCJkZXZpY2VJZCI6IjEyMzQ1NiIsImRldmljZU5hbWUiOiJCcm93c2VyV2ViIiwiaXNUcmlhbFVzZXIiOmZhbHNlLCJpc0ZpcnN0VGltZUxvZ2luIjpmYWxzZSwic2Vzc2lvbklkIjoiY3MtZTEzZmZlYTYtYWRlMy00MjFlLTk4YTctMjhkMjBkODllYWRiIn0sImlhdCI6MTc3NDQ2OTYwNn0.yA3sAmcnKtrfWjM6BTRDQcSHwusEGbmcKMvPzN8VhXU"

app = App(token=SLACK_BOT_TOKEN)

print("⏳ Loading embedding model …")
embedder = SentenceTransformer(EMBED_MODEL)
print("✅ Embedding model ready")

print("⏳ Connecting to Qdrant …")
qclient  = QdrantClient(path=STORAGE_DIR)
kb_count = qclient.count(collection_name=COLLECTION).count
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
# Keys: user::dm:{D…} | user::ch:{C…}:t:{thread_root_ts} | user::eph:{channel} (irt-test)
# _automation_anchors[user::channel] = thread root ts for channel/mpim flows.
# Completely separate from _history — never interferes with KB search flow.
#
# Structure:
# _automation_state[state_key] = {
#     "category":         "Extend Trail Period",
#     "collected":        {"org_id": "x"},
#     "awaiting_confirm": False,
# }

_automation_state: dict = {}

# Channel/mpim: f"{user}::{channel}" -> root message ts (automation continues in that thread)
_automation_anchors: dict[str, str] = {}


def _auto_clear_key(state_key: str | None, user: str, channel: str) -> None:
    if not state_key:
        return
    _automation_state.pop(state_key, None)
    if ":ch:" in state_key and ":t:" in state_key:
        ts = state_key.rsplit(":t:", 1)[-1]
        ak = f"{user}::{channel}"
        if _automation_anchors.get(ak) == ts:
            _automation_anchors.pop(ak, None)


def _clear_all_automation_for_user(user: str) -> None:
    """Clears every automation session and anchor for this user (fixes DM cancel vs /irt in channel)."""
    prefix = f"{user}::"
    for k in list(_automation_state.keys()):
        if k.startswith(prefix):
            _automation_state.pop(k, None)
    for k in list(_automation_anchors.keys()):
        if k.startswith(prefix):
            _automation_anchors.pop(k, None)


def _should_interrupt_automation_for_chat(query: str, awaiting_confirm: bool) -> bool:
    """Greetings should not continue pending automation (e.g. slot-fill for Tenant ID)."""
    t = query.strip()
    if len(t) > 80:
        return False
    tl = t.lower()
    if awaiting_confirm:
        if tl in (
            "ok", "yes", "y", "confirm", "proceed", "cancel", "no", "abort", "stop", "exit",
        ):
            return False
    if re.match(
        r"^(hi|hello|hey|hii|yo|sup|thanks?|thank you|thx|ty|bye|goodbye|gm|good (morning|afternoon|evening))[\s!.,]*$",
        tl,
    ):
        return True
    if tl in ("hi", "hello", "hey", "thanks", "thx", "bye"):
        return True
    return False


def _resolve_auto_state_key(
    user: str,
    channel: str,
    thread_ts: str | None,
    ephemeral_user: str | None,
) -> tuple[str | None, str | None]:
    """
    Returns (state_key, reply_thread_ts). reply_thread_ts is the Slack thread to post in
    for channel automation (None for DM / ephemeral-test keys).
    """
    if ephemeral_user:
        return f"{user}::eph:{channel}", None
    if channel.startswith("D"):
        return f"{user}::dm:{channel}", None
    if thread_ts:
        return f"{user}::ch:{channel}:t:{thread_ts}", thread_ts
    anchor = _automation_anchors.get(f"{user}::{channel}")
    if anchor:
        return f"{user}::ch:{channel}:t:{anchor}", anchor
    return None, None


def _get_auto_state_keyed(state_key: str | None) -> dict | None:
    if not state_key:
        return None
    return _automation_state.get(state_key)


def _set_auto_state_keyed(state_key: str, state: dict) -> None:
    _automation_state[state_key] = state


def _ensure_channel_automation_thread(
    client,
    user: str,
    channel: str,
    category: str,
    reuse_existing: bool,
    slash_thread_root: str | None = None,
) -> tuple[str, str]:
    """
    Returns (state_key, reply_thread_ts). Creates a thread root on the channel when
    /irt is used from the main channel; if /irt is run inside an existing thread,
    posts the header into that thread and uses the thread root as anchor.
    """
    anchor_key = f"{user}::{channel}"
    anchor = _automation_anchors.get(anchor_key)
    old_key = f"{user}::ch:{channel}:t:{anchor}" if anchor else None
    old = _automation_state.get(old_key) if old_key else None

    if reuse_existing and old and old.get("category") == category and anchor:
        return old_key, anchor

    if old_key:
        _auto_clear_key(old_key, user, channel)

    intro = (
        f"🔧 *IRT automation — {category}*\n"
        "_Reply in this thread only — keeps context out of the main channel._"
    )
    if slash_thread_root:
        client.chat_postMessage(
            channel=channel,
            thread_ts=slash_thread_root,
            text=intro,
        )
        new_a = slash_thread_root
    else:
        r = client.chat_postMessage(channel=channel, text=intro)
        new_a = r["ts"]

    _automation_anchors[anchor_key] = new_a
    return f"{user}::ch:{channel}:t:{new_a}", new_a


# ── All 14 automation categories — fields, labels, validation ─────────────────
# Canonical payload specs (category names + details keys/types) live in:
#   ../IRTAutomation_Docs.md  (repo root: IRTAutomation_Docs.md)
# Keep `build` outputs aligned with that validator / automation API.
#
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


def _extract_extend_trail_period(text: str) -> dict:
    """Pull org_id and YYYY-MM-DD from natural language; never returns sentence-sized blobs."""
    out: dict = {}
    if not text or not isinstance(text, str):
        return out
    t = text.strip()
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t)
    if m:
        out["extend_period"] = m.group(1)
    for mm in re.finditer(r"`([^`\n]+)`", t):
        chunk = mm.group(1).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", chunk):
            out["extend_period"] = chunk
        elif not chunk.startswith("http") and len(chunk) <= 120:
            if "org_id" not in out:
                out["org_id"] = chunk
    if "org_id" not in out:
        m = re.search(
            r"\b(?:org|organisation|organization)\s*(?:id)?\s*[:=]?\s*([A-Za-z0-9_.-]+)\b",
            t,
            re.I,
        )
        if m:
            out["org_id"] = m.group(1)
    if "org_id" not in out:
        m = re.search(r"\b(org_[A-Za-z0-9_.-]+)\b", t, re.I)
        if m:
            out["org_id"] = m.group(1)
    if "org_id" not in out:
        m = re.search(
            r"\b(?:for|this)\s+org(?:anisation)?\s+([A-Za-z0-9_.-]+)\b",
            t,
            re.I,
        )
        if m:
            out["org_id"] = m.group(1)
    return out


def _bootstrap_extract(category: str, message: str) -> dict:
    if category == "Extend Trail Period":
        return _extract_extend_trail_period(message)
    return {}


def _extend_trail_intro(collected: dict) -> str:
    if collected.get("org_id") and not collected.get("extend_period"):
        return (
            f"Got it — *Organisation ID:* `{collected['org_id']}`\n\n"
            "Please send the *new expiry date* as `YYYY-MM-DD` "
            "(e.g. `2026-04-28`)."
        )
    if collected.get("extend_period") and not collected.get("org_id"):
        return (
            f"Got it — *New expiry date:* `{collected['extend_period']}`\n\n"
            "Please send the *Organisation ID* (e.g. `org_123` or `trailtest01`)."
        )
    return (
        "🔧 *Extend trail period*\n\n"
        "Please send:\n"
        "• *Organisation ID*\n"
        "• *New expiry date* as `YYYY-MM-DD`\n\n"
        "_You can send both in one line, e.g._ `trailtest01 2026-04-28` _or_ "
        "`org_123` _then the date in your next message._"
    )


def automation_agent(
    user: str,
    channel: str,
    message: str,
    category: str = None,
    *,
    state_key: str,
) -> str:
    """
    The agentic loop for automation.

    On each call it either:
      - Starts a new automation session (if category given)
      - Handles cancel/confirm commands
      - Collects the next missing field
      - Executes the API when all fields are ready and confirmed

    Returns a Slack-formatted string to send back to the user.
    """
    state = _get_auto_state_keyed(state_key)

    # ── Handle cancel at any point ────────────────────────────────────────────
    if message.lower().strip() in ("cancel", "abort", "stop", "exit"):
        if state:
            _clear_all_automation_for_user(user)
            return "❌ Automation cancelled. Ask me anything else!"
        return "Nothing to cancel."

    # ── Start new session ─────────────────────────────────────────────────────
    new_session = bool(category and not state)
    if new_session:
        if category not in AUTOMATION_CATEGORIES:
            return f"⚠️ Unknown automation category: *{category}*"
        boot = _bootstrap_extract(category, message)
        state = {
            "category": category,
            "collected": dict(boot),
            "awaiting_confirm": False,
        }
        _set_auto_state_keyed(state_key, state)

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
                _auto_clear_key(state_key, user, channel)
                return f"⚠️ Failed to build payload: {e}"

            result = call_automation_api(category, details)
            _auto_clear_key(state_key, user, channel)

            if result["ok"]:
                return (
                    f"✅ *{category}* executed successfully!\n\n"
                    f"_Response:_ {result['message'][:300]}"
                )
            return (
                f"❌ *{category}* failed.\n\n"
                f"_Error:_ {result['message']}\n\n"
                "_Please check the details and try again._"
            )
        _auto_clear_key(state_key, user, channel)
        return "❌ Automation cancelled. Ask me anything else!"

    # ── New session: do not dump the whole trigger sentence into one field ───
    skip_slot_fill = False
    if new_session:
        if category == "Extend Trail Period":
            _set_auto_state_keyed(state_key, state)
            for key, label, required, hint in fields:
                if required and key not in collected:
                    return _extend_trail_intro(collected)
            skip_slot_fill = True
        else:
            lines = [f"• *{label}*" + (" _(required)_" if req else "")
                     for key, label, req, hint in fields if req]
            ask = "\n".join(lines) if lines else "the required details"
            _set_auto_state_keyed(state_key, state)
            return (
                f"🔧 *{category}*\n\n"
                f"Please provide:\n{ask}\n\n"
                "_Send values in separate messages or one message per field as prompted below._"
            )

    # ── Slot fill: only after user is answering prompts (not bootstrap turn) ──
    if message.strip() and not skip_slot_fill:
        merged = _bootstrap_extract(category, message) if category == "Extend Trail Period" else {}
        for k, v in merged.items():
            if v and k not in collected:
                collected[k] = v
        if category == "Extend Trail Period" and merged:
            state["collected"] = collected
            _set_auto_state_keyed(state_key, state)
        else:
            for key, label, required, hint in fields:
                if key not in collected:
                    collected[key] = message.strip()
                    state["collected"] = collected
                    _set_auto_state_keyed(state_key, state)
                    break

    # ── Check for next missing required field ─────────────────────────────────
    effective_fields = fields
    if category == "Admin Email Changes":
        role = collected.get("role", "").lower()
        effective_fields = [
            (k, l, True if k in ("role", "old_email", "new_email") else (role == "admin"), h)
            for k, l, _, h in fields
        ]

    for key, label, required, hint in effective_fields:
        if required and key not in collected:
            _set_auto_state_keyed(state_key, state)
            return f"📝 *{label}*\n_{hint}_"

    # ── All fields collected — show confirmation summary ──────────────────────
    summary_lines = [f"*{label}:* `{collected.get(key, '—')}`"
                     for key, label, _, _ in fields
                     if key in collected]
    summary = "\n".join(summary_lines)

    state["awaiting_confirm"] = True
    _set_auto_state_keyed(state_key, state)

    return (
        f"🔧 *Ready to execute: {category}*\n\n"
        f"{summary}\n\n"
        f"Type *confirm* to proceed or *cancel* to abort."
    )


def _run_automation_reply(
    client,
    user_id: str,
    channel: str,
    thread_ts: str | None,
    ephemeral_user: str | None,
    query: str,
    category: str,
) -> tuple[str, list, str | None]:
    """
    Resolves state key / thread, runs automation_agent.
    Returns (text, blocks, slack_thread_ts) — slack_thread_ts is the anchor to reply in
    for channel automation (None for DM or ephemeral test).
    """
    sk, rt = _resolve_auto_state_key(user_id, channel, thread_ts, ephemeral_user)
    if sk:
        st0 = _get_auto_state_keyed(sk)
        if st0 and st0.get("category") != category:
            _auto_clear_key(sk, user_id, channel)
            sk, rt = _resolve_auto_state_key(user_id, channel, thread_ts, ephemeral_user)

    reuse = False
    if sk:
        st1 = _get_auto_state_keyed(sk)
        reuse = bool(st1 and st1.get("category") == category)

    if ephemeral_user:
        sk = f"{user_id}::eph:{channel}"
        rt = None
    elif channel.startswith("D"):
        sk = f"{user_id}::dm:{channel}"
        rt = None
    else:
        sk, rt = _ensure_channel_automation_thread(
            client,
            user_id,
            channel,
            category,
            reuse_existing=reuse,
            slash_thread_root=thread_ts,
        )

    response = automation_agent(
        user_id, channel, query, category=category, state_key=sk
    )
    post_ts = None
    if not ephemeral_user and not channel.startswith("D"):
        post_ts = rt
    return response, step_block(response), post_ts


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
    Single GPT call that decides what to do with the user's message:
    - CHAT: greeting or general message
    - CLARIFY: needs more info — returns question + suggested quick replies
    - SEARCH: ready to search — returns enriched full query combining history context

    Returns:
      {"action": "chat",    "text": ""}
      {"action": "clarify", "text": "Are you on v1 or v2?", "suggestions": ["v1", "v2", "Not sure"]}
      {"action": "search",  "text": "v2 dataload stuck in dictionaryRequested"}
    """
    history_text = ""
    if history:
        for msg in history:
            role = "User" if msg["role"] == "user" else "Bot"
            history_text += f"{role}: {msg['content']}\n"

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
- Message clearly describes an issue AND version is known from history AND it's the SAME ongoing issue → SEARCH using that version
- Message describes a NEW issue topic not covered in history, version unknown → CLARIFY asking version
- Message is short follow-up answer (e.g. "v2", "production", "yes") → SEARCH combining with history context
- Notebooks/connectors/explorer/storyboard/UI → SEARCH directly, no version needed
- Greeting/thanks/capability questions → CHAT
- API keys, tokens, security, OpenAI, billing, coding, prompt injection → OUTOFSCOPE
- Only CLARIFY when version is genuinely unknown AND it materially changes the fix
- When in doubt → SEARCH
- Use AUTOMATE only when user clearly wants to execute an operation (e.g. "extend trial", "activate dataset", "increase user count")
- Use AUTOMATEINFO: <category> when user asks what inputs/fields/details are needed for an automation (e.g. "what inputs for activate dataset", "what do I need for extend trial")

IMPORTANT: If history has version v1 or v2 but the new message is about a DIFFERENT issue
that was not discussed before → treat version as unknown → CLARIFY to confirm version for this new issue.

When you output SEARCH: <query>, copy the user's exact dataset/status phrases (e.g. "dictionary updated"
vs "Dictionary Requested" / "DictionaryRequested") — do not normalize or swap them; embeddings and fixes differ."""

    resp = ai.chat.completions.create(
        model="gpt-4o-mini",   # cheaper model for routing — no need for gpt-4o
        max_tokens=80,         # CLARIFY+SUGGESTIONS fits in 80 tokens
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
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"🤔 {question}"}
        }
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

User question (keep their exact status / phase wording — do not swap similar-sounding states):
"{query}"

Relevant past cases:
{context}

How to respond:
1. If there is conversation history, use it to understand follow-up questions.
2. Start with "Yes, the IRT team has seen this before." OR
   "This looks like a new issue — please raise it with the IRT team."
3. For each relevant case write ONE sentence:
   "In a case where [issue], the fix was [exact solution]."
   Use the **Issue** line text from that case for [issue] when possible — do not rename it.
   Then: "This was a *permanent fix*." or "This was a *workaround*."
4. Write "*Steps to try:*" then 2-4 bullets ONLY from the Solution fields.
   - Use EXACT actions from Solution — do not invent steps.
   - IRT terms OK: SME publish, republish, vacuum, entity count, org ID, dataset activation.
   - No generic advice.
5. End with: "If this doesn't help, share your *Dataset name*, *Org ID*, *Environment*, and *current status* with the IRT team."

**Critical — dataset status names:** "Dictionary updated", "DictionaryUpdated", "dictionary updated",
"Dictionary Requested", "DictionaryRequested", etc. are **different** product states. Never substitute
one for another in your answer. If the user said *dictionary updated*, your prose must reflect that
(and the retrieved Issue summaries), not *Dictionary Requested*, unless that exact requested-state
wording is what the cited case's Issue line says.

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


def handle_conversational(query: str, history: list = None) -> str:
    """
    Handles greetings and capability questions.
    For capability questions → returns a precise structured response.
    For greetings → GPT generates a friendly reply.
    """
    q = query.lower().strip().rstrip("!?.,")

    # Capability questions — return precise hardcoded response
    # so users always get an accurate, up-to-date answer
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

    # Regular greeting → GPT handles it
    system_prompt = """You are IRT Bot, a friendly support assistant for ConverSight's Incident Response Team.
Respond warmly to greetings. Keep it short — 1-2 sentences max.
Mention you can help with product issues and automation tasks.
Use *bold* for emphasis (Slack format)."""

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
#
# Similarity % explained:
#   This is cosine similarity between your question's embedding and the past
#   issue's embedding. It measures how semantically similar the two texts are.
#   85%+ = almost the same question asked before
#   65–84% = clearly related issue
#   50–64% = loosely related, same general area
#   <50% = only vaguely related

def _sim_label(score: int) -> str:
    if score >= 85:
        return f"🟢 *{score}% similarity* — nearly identical issue"
    elif score >= 65:
        return f"🟡 *{score}% similarity* — clearly related"
    elif score >= 50:
        return f"🟠 *{score}% similarity* — loosely related"
    else:
        return f"🔴 *{score}% similarity* — vaguely related"


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

    hits_text = ""
    for h in hits[:3]:
        icon  = icons.get(h["resolution_status"], "❓")
        label = res_labels.get(h["resolution_status"], h["resolution_status"])
        score = int(h["score"] * 100)
        bar   = "█" * (score // 10) + "░" * (10 - score // 10)
        src   = "  📄 _RCA doc_" if h.get("source") == "RCA" else ""
        ref   = _format_reference(h.get("references", ""))

        hits_text += f"{icon} *{h['summary'][:70]}*{src}\n"
        hits_text += f"   `{bar}` {score}% | _{label}_ | {h['bug_category']}\n"
        if ref:
            hits_text += f"   📎 {ref}\n"
        hits_text += "\n"

    return [
        # ── Header ────────────────────────────────────────────────────────────
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🤖 IRT Bot", "emoji": True}
        },

        # ── Question ──────────────────────────────────────────────────────────
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Your question:*\n{query}"}
        },

        {"type": "divider"},

        # ── Answer ────────────────────────────────────────────────────────────
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*💡 Answer:*\n{answer}"}
        },

        {"type": "divider"},

        # ── Similar past issues ───────────────────────────────────────────────
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*📋 Similar past issues:*\n\n{hits_text}"}
        },

        # ── Bold bottom separator — only end border is prominent ──────────────
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬"
            }
        },
    ]


def step_block(txt: str) -> list:
    return [{"type": "section", "text": {"type": "mrkdwn", "text": txt}}]


# ── Core streaming function ───────────────────────────────────────────────────

def stream_response(
    client,
    channel: str,
    query: str,
    thread_ts: str = None,
    ephemeral_user: str = None,
    user_id: str = None,
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

    # ── Handle reset command ──────────────────────────────────────────────────
    if query.lower().strip() in ("reset", "clear", "new", "start over"):
        if user_id:
            _clear_history(user_id, channel)
            _clear_all_automation_for_user(user_id)
        kw = {"channel": channel, "text": "🔄 Conversation reset. Ask me anything!"}
        if thread_ts:
            kw["thread_ts"] = thread_ts
        client.chat_postMessage(**kw)
        return

    # ── Continue active automation (thread reply or DM) — do not run analyze_query ─
    # Otherwise a short answer like "2026-04-28" is mis-routed as SEARCH and clears state.
    # New `/irt …` from the main channel still hits analyze_query below. In the automation
    # thread, type *cancel* first if you need the knowledge base instead.
    if user_id and not ephemeral_user:
        sk_auto = None
        reply_ts = None
        if channel.startswith("D"):
            sk_auto = f"{user_id}::dm:{channel}"
        else:
            an = _automation_anchors.get(f"{user_id}::{channel}")
            if thread_ts and an == thread_ts:
                sk_auto = f"{user_id}::ch:{channel}:t:{thread_ts}"
                reply_ts = thread_ts
        if sk_auto and _get_auto_state_keyed(sk_auto):
            st0 = _get_auto_state_keyed(sk_auto)
            awaiting = bool(st0.get("awaiting_confirm"))
            if _should_interrupt_automation_for_chat(query, awaiting):
                log.warning("automation interrupted: conversational message; clearing state")
                _automation_state.pop(sk_auto, None)
            else:
                response = automation_agent(
                    user_id, channel, query, category=None, state_key=sk_auto
                )
                kw = {"channel": channel, "text": response, "blocks": step_block(response)}
                if reply_ts:
                    kw["thread_ts"] = reply_ts
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
            history_channel = f"{channel}:{thread_ts}" if thread_ts else channel
            history  = _get_history(user_id, history_channel) if user_id else []
            sk_prior, _ = (
                _resolve_auto_state_key(user_id, channel, thread_ts, ephemeral_user)
                if user_id
                else (None, None)
            )
            decision = analyze_query(query, history)

            if (
                decision["action"] in ("search", "clarify", "outofscope")
                and user_id
                and sk_prior
            ):
                _auto_clear_key(sk_prior, user_id, channel)

            if decision["action"] == "chat":
                answer = handle_conversational(query, history)
                if user_id:
                    _add_history(user_id, history_channel, "user", query)
                    _add_history(user_id, history_channel, "assistant", answer)
                final_text   = answer
                final_blocks = step_block(answer)

            elif decision["action"] == "automate":
                category = decision["text"]
                response, final_blocks, _ = _run_automation_reply(
                    client,
                    user_id,
                    channel,
                    thread_ts,
                    ephemeral_user,
                    query,
                    category,
                )
                final_text = response

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
                    final_text   = answer
                    final_blocks = build_blocks(search_q, answer, hits)

        except Exception as e:
            log.error(f"stream_response (ephemeral) error: {e}")
            final_text   = f"⚠️ {_friendly_error(e)}"
            final_blocks = step_block(final_text)

        client.chat_postEphemeral(
            channel=channel, user=ephemeral_user,
            text=final_text, blocks=final_blocks
        )
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC PATH — post loading message, animate steps, replace with answer
    # ══════════════════════════════════════════════════════════════════════════

    # Post step 1 immediately so user sees the bot is alive
    kw = {"channel": channel, "text": STEPS[0], "blocks": step_block(STEPS[0])}
    if thread_ts:
        kw["thread_ts"] = thread_ts
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
                    channel=channel, ts=msg_ts,
                    text=steps[idx], blocks=step_block(steps[idx])
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
        # Use thread_ts as part of history key when in a thread
        # This isolates each thread's conversation from the main channel history
        history_channel = f"{channel}:{thread_ts}" if thread_ts else channel
        history  = _get_history(user_id, history_channel) if user_id else []
        sk_prior, _ = (
            _resolve_auto_state_key(user_id, channel, thread_ts, None)
            if user_id
            else (None, None)
        )
        decision = analyze_query(query, history)

        if (
            decision["action"] in ("search", "clarify", "outofscope")
            and user_id
            and sk_prior
        ):
            _auto_clear_key(sk_prior, user_id, channel)

        if decision["action"] == "chat":
            answer = handle_conversational(query, history)
            if user_id:
                _add_history(user_id, history_channel, "user", query)
                _add_history(user_id, history_channel, "assistant", answer)
            final_text   = answer
            final_blocks = step_block(answer)

        elif decision["action"] == "automate":
            category = decision["text"]
            # Stop animation — automation posts its own message
            stop_flag["done"] = True
            anim.join(timeout=1)
            try:
                client.chat_delete(channel=channel, ts=msg_ts)
            except Exception:
                pass
            response, bl, post_ts = _run_automation_reply(
                client,
                user_id,
                channel,
                thread_ts,
                None,
                query,
                category,
            )
            kw = {"channel": channel, "text": response, "blocks": bl}
            if post_ts:
                kw["thread_ts"] = post_ts
            elif thread_ts:
                kw["thread_ts"] = thread_ts
            client.chat_postMessage(**kw)
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

            if thread_ts:
                anchor_ts = thread_ts
                sent = client.chat_postMessage(
                    channel   = channel,
                    text      = f"🤔 {clarification}",
                    blocks    = clarify_blocks(clarification, suggestions),
                    thread_ts = anchor_ts,
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
                anchor = client.chat_postMessage(
                    channel=channel,
                    text=f"*Question:* {query}",
                    blocks=[{
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Question:* {query}"}
                    }]
                )
                anchor_ts = anchor["ts"]
                sent = client.chat_postMessage(
                    channel   = channel,
                    text      = f"🤔 {clarification}",
                    blocks    = clarify_blocks(clarification, suggestions),
                    thread_ts = anchor_ts,
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
            channel=channel, ts=msg_ts,
            text=final_text, blocks=final_blocks
        )
    except Exception as e:
        log.error(f"chat_update failed: {e}")
        kw = {"channel": channel, "text": final_text, "blocks": final_blocks}
        if thread_ts:
            kw["thread_ts"] = thread_ts
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
    query      = command.get("text", "").strip()
    channel    = command.get("channel_id", "")
    user       = command.get("user_id", "")
    thread_ts  = command.get("thread_ts")
    if not query:
        client.chat_postEphemeral(channel=channel, user=user,
            text="Please add a question. Example: `/irt v2 dataset failed`")
        return
    log.warning(f"/irt u={user} q={query[:80]}")
    threading.Thread(
        target=stream_response,
        args=(client, channel, query),
        kwargs={"user_id": user, "thread_ts": thread_ts},
        daemon=True
    ).start()


@app.command("/irt-test")
def handle_irt_test(ack, command, client):
    ack()
    query      = command.get("text", "").strip()
    channel    = command.get("channel_id", "")
    user       = command.get("user_id", "")
    thread_ts  = command.get("thread_ts")
    if not query:
        client.chat_postEphemeral(channel=channel, user=user,
            text="🧪 Test mode — only you see this.\nUsage: `/irt-test v2 dataset failed`")
        return
    log.warning(f"/irt-test u={user} q={query[:80]}")
    threading.Thread(
        target=stream_response,
        args=(client, channel, query),
        kwargs={"ephemeral_user": user, "user_id": user, "thread_ts": thread_ts},
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
    log.warning(f"modal u={user} vis={visibility} q={query[:80]}")
    threading.Thread(
        target=stream_response,
        args=(client, IRT_CHANNEL, query),
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
    # ANY message inside a thread is ALWAYS a follow-up in that thread's context.
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
                    kwargs={"thread_ts": thread_ts, "user_id": user},
                    daemon=True
                ).start()
            else:
                # Follow-up message in thread using thread context
                log.warning(f"thread_followup u={user} q={query[:80]}")
                threading.Thread(
                    target=stream_response,
                    args=(client, channel, query),
                    kwargs={"thread_ts": thread_ts, "user_id": user},
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
        kwargs={"user_id": user},
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
    print("  🤖  IRT RAG Slack Bot (Cursor)  — threaded automation + RAG")
    print("=" * 62)
    print(f"  /irt <question>      → visible to whole channel  ✅")
    print(f"  /irt-test <question> → only you see it           ✅")
    print(f"  Ask IRT Bot button   → modal + live loading      ✅")
    print(f"  DM the bot           → chatbot with memory       ✅")
    print(f"  @mention bot         → reply in thread           ✅")
    print(f"  Clarify question     → thread reply triggers KB  ✅")
    print(f"  Automation agent     → 14 categories via API     ✅")
    print(f"  Type 'reset' in DM   → clears conversation       ✅")
    print(f"  Knowledge base       : {kb_count:,} documents")
    print(f"  Automation token     : {'✅ set' if AUTOMATION_TOKEN else '❌ missing IRT_AUTOMATION_TOKEN'}")
    print(f"  Chat memory          : last {CHAT_HISTORY_LEN} turns per user")
    print("=" * 62)
    print()

    SocketModeHandler(app, SLACK_APP_TOKEN).start()