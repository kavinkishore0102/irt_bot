"""
INTEGRATION PATCH for irt_rag_slack_bot.py
═══════════════════════════════════════════
This file shows EXACTLY what to change in your existing bot.
It is NOT a standalone file — use it as a reference.

Changes are grouped into 4 sections:
  A. New imports to add at the top
  B. Startup check to add after load_dotenv()
  C. Replace the in-memory _history functions with Redis-backed ones
  D. Update handle_mention to call thread_handler
  E. Add the Close Thread button + block_actions handler

"""

# ═══════════════════════════════════════════════════════════════════════════════
# A. ADD THESE IMPORTS at the top of irt_rag_slack_bot.py
#    (after the existing imports block)
# ═══════════════════════════════════════════════════════════════════════════════

from handlers.thread_handler       import handle_message
from handlers.close_thread_handler import handle_close_thread
from utils.redis_client            import ping_redis


# ═══════════════════════════════════════════════════════════════════════════════
# B. ADD THIS STARTUP CHECK
#    Place it right after:  ai = OpenAI()
# ═══════════════════════════════════════════════════════════════════════════════

print("⏳ Connecting to Redis …")
if ping_redis():
    print("✅ Redis connected — thread memory is ON")
else:
    print("⚠️  Redis not reachable — thread memory will fall back to in-memory")


# ═══════════════════════════════════════════════════════════════════════════════
# C. REPLACE the in-memory _history block
#
# REMOVE this entire block from your bot (lines ~85–100):
#
#   _history: dict = defaultdict(lambda: deque(maxlen=CHAT_HISTORY_LEN * 2))
#
#   def _conv_key(user: str, channel: str) -> str:
#       return f"{user}::{channel}"
#
#   def _get_history(user: str, channel: str) -> list:
#       return list(_history[_conv_key(user, channel)])
#
#   def _add_history(user: str, channel: str, role: str, content: str):
#       _history[_conv_key(user, channel)].append({"role": role, "content": content})
#
#   def _clear_history(user: str, channel: str):
#       key = _conv_key(user, channel)
#       if key in _history:
#           del _history[key]
#
# These are now handled by thread_handler.py + redis_client.py
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# D. UPDATE handle_mention (the @app.event("app_mention") handler)
#
# Find the section inside handle_mention where stream_response is called
# for a regular question (not automation, not ticket, not pending).
# 
# BEFORE (your current code in handle_mention — fresh top-level mention):
#
#   threading.Thread(target=stream_response, args=(client, channel, text),
#       kwargs={"thread_ts": ts, "user_id": user}, daemon=True).start()
#
# AFTER — replace that one line with this helper call:
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_irt_message(client, channel, text, user, workspace_id, thread_ts, is_new_thread=False):
    """
    Drop-in replacement for the threading.Thread(stream_response...) call.

    Calls thread_handler (which loads Redis history, calls LLM, saves back),
    then posts the reply to Slack inside the correct thread.

    Args:
        client       : Slack WebClient
        channel      : Slack channel ID
        text         : cleaned user message text
        user         : Slack user ID
        workspace_id : Slack team/workspace ID (from event['team'])
        thread_ts    : thread parent timestamp to reply inside
        is_new_thread: True if this is the very first message in a new thread
    """
    # ── Show "thinking" indicator ─────────────────────────────────────────────
    loading_msg = client.chat_postMessage(
        channel   = channel,
        thread_ts = thread_ts,
        text      = "⏳ _Looking into this for you…_",
    )
    loading_ts = loading_msg["ts"]

    # ── Optional: get RAG context from your existing search function ──────────
    # Your bot already has a `search()` function in irt_rag_query_v2.py.
    # If you want RAG + thread memory combined, call it here:
    #
    #   rag_hits   = search(text, embedder, qclient, TOP_K)
    #   rag_context = "\n".join(
    #       f"Case {i}: Issue={h['summary']} | Solution={h['solution']}"
    #       for i, h in enumerate(rag_hits[:3], 1) if h['score'] >= MIN_SCORE
    #   ) or None
    #
    # For now, we pass None (pure LLM with thread memory):
    rag_context = None

    # ── Call thread handler ───────────────────────────────────────────────────
    result = handle_message(
        workspace_id  = workspace_id,
        thread_ts     = thread_ts,
        user_message  = text,
        is_new_thread = is_new_thread,
        rag_context   = rag_context,
    )

    # ── Handle expired session notification ───────────────────────────────────
    reply = result["reply"]
    if result["session_expired"]:
        reply = (
            "ℹ️ _Previous session expired. Starting fresh._\n\n" + reply
        )

    # ── Delete the loading message, post the real reply ───────────────────────
    try:
        client.chat_delete(channel=channel, ts=loading_ts)
    except Exception:
        pass  # fine if delete fails

    # ── Post reply with Close Thread button ───────────────────────────────────
    client.chat_postMessage(
        channel   = channel,
        thread_ts = thread_ts,
        text      = reply,
        blocks    = _reply_blocks(reply, thread_ts),
    )


def _reply_blocks(reply_text: str, thread_ts: str) -> list:
    """
    Wraps the reply in a Block Kit layout with a Close Thread button.
    The button's value carries thread_ts so we know what to delete from Redis.
    """
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": reply_text[:2900]},
        },
        {"type": "divider"},
        {
            "type": "actions",
            "elements": [
                {
                    "type"     : "button",
                    "text"     : {"type": "plain_text", "text": "🔒 Close Thread", "emoji": True},
                    "style"    : "danger",
                    "action_id": "close_thread",
                    "value"    : thread_ts,   # ← carries thread_ts into the action handler
                    "confirm"  : {
                        "title"  : {"type": "plain_text", "text": "Close this thread?"},
                        "text"   : {"type": "mrkdwn",     "text": "This will clear the conversation memory for this thread."},
                        "confirm": {"type": "plain_text",  "text": "Yes, close it"},
                        "deny"   : {"type": "plain_text",  "text": "Cancel"},
                    }
                }
            ]
        }
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# E. ADD this block_actions handler to your bot
#    (add it alongside your other @app.action / @app.event handlers)
# ═══════════════════════════════════════════════════════════════════════════════

# Import your app instance (already defined in your bot as: app = App(...))
# from irt_rag_slack_bot import app   ← don't add this line; just place the
#                                        handler IN the same file as your app

# @app.action("close_thread")
# def action_close_thread(ack, body, client):
#     """
#     Fires when user clicks the 🔒 Close Thread button.
#     Deletes the Redis key and posts a confirmation message.
#     """
#     ack()   # ← Always acknowledge Slack actions within 3 seconds
#
#     user         = body["user"]["id"]
#     channel      = body["channel"]["id"]
#     workspace_id = body["team"]["id"]
#     thread_ts    = body["actions"][0]["value"]   # ← we stored thread_ts as button value
#
#     result = handle_close_thread(workspace_id, thread_ts)
#
#     client.chat_postMessage(
#         channel   = channel,
#         thread_ts = thread_ts,
#         text      = result["message"],
#     )


# ═══════════════════════════════════════════════════════════════════════════════
# F. HOW TO WIRE _handle_irt_message INTO YOUR EXISTING handle_mention
#
# In your current handle_mention, find the final section:
#
#   threading.Thread(target=stream_response, args=(client, channel, text),
#       kwargs={"thread_ts": ts, "user_id": user}, daemon=True).start()
#
# Replace it with:
#
#   workspace_id = event.get("team", "default")
#
#   threading.Thread(
#       target=_handle_irt_message,
#       args=(client, channel, text, user, workspace_id, ts),
#       kwargs={"is_new_thread": True},
#       daemon=True
#   ).start()
#
# For the FOLLOW-UP case (reply inside thread), replace:
#
#   threading.Thread(target=stream_response, args=(client, channel, text),
#       kwargs={"thread_ts": thread_ts, "user_id": user}, daemon=True).start()
#
# With:
#
#   workspace_id = event.get("team", "default")
#
#   threading.Thread(
#       target=_handle_irt_message,
#       args=(client, channel, text, user, workspace_id, thread_ts),
#       kwargs={"is_new_thread": False},
#       daemon=True
#   ).start()
#
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# G. ADD TO .env / environment variables
# ═══════════════════════════════════════════════════════════════════════════════

# REDIS_URL=redis://localhost:6379
# THREAD_MAX_TOKENS=8000
# THREAD_MAX_MESSAGES=20
# OPENAI_MODEL_ANSWER=gpt-4.1