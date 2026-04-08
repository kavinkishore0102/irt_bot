"""
handlers/thread_handler.py
──────────────────────────
Handles per-thread conversation logic.

Loads history from Redis, calls the LLM with full context, saves
history back to Redis, and returns the reply to the caller.

Returns:
    dict with keys:
        "reply"          : str  — LLM response text
        "session_expired": bool — True if no prior history was found
"""

import logging
import os

from openai import OpenAI
from dotenv import load_dotenv

from utils.redis_client  import get_thread_history, save_thread_history
from utils.history_manager import (
    build_history,
    trim_history,
    make_system_prompt,
    is_token_limit_exceeded,
    DEFAULT_MAX_MESSAGES,
    DEFAULT_MAX_TOKENS,
)

load_dotenv()
log = logging.getLogger(__name__)

_ai = OpenAI()

OPENAI_MODEL     = os.environ.get("OPENAI_MODEL_ANSWER", "gpt-4.1")
MAX_MESSAGES     = int(os.environ.get("THREAD_MAX_MESSAGES", DEFAULT_MAX_MESSAGES))
MAX_TOKENS       = int(os.environ.get("THREAD_MAX_TOKENS",   DEFAULT_MAX_TOKENS))


def handle_message(
    workspace_id: str,
    thread_ts: str,
    user_message: str,
    is_new_thread: bool = False,
    rag_context: str | None = None,
) -> dict:
    """
    Core per-thread handler.

    Args:
        workspace_id  : Slack workspace/team ID
        thread_ts     : Thread parent timestamp (used as Redis key)
        user_message  : Cleaned user text
        is_new_thread : True if this is the opening message of a thread
        rag_context   : Optional pre-fetched RAG context string to inject

    Returns:
        {
            "reply"          : str   — LLM answer to post in Slack
            "session_expired": bool  — True when we started a fresh session
        }
    """
    session_expired = False

    # ── Load existing history from Redis ──────────────────────────────────────
    history = get_thread_history(workspace_id, thread_ts)

    if is_new_thread or not history:
        if history and is_new_thread:
            log.warning(f"[ThreadHandler] New thread flag set — resetting history for {thread_ts}")
        elif not history and not is_new_thread:
            log.warning(f"[ThreadHandler] No history found for {thread_ts} — treating as new session")
            session_expired = True

        history = [make_system_prompt()]

    # ── Token guard — trim if history is too large ────────────────────────────
    if is_token_limit_exceeded(history, MAX_TOKENS):
        history = trim_history(history, MAX_MESSAGES)

    # ── Inject RAG context into user message if provided ─────────────────────
    user_content = user_message
    if rag_context:
        user_content = (
            f"[Relevant past cases for context]\n{rag_context}\n\n"
            f"[User question]\n{user_message}"
        )

    # ── Append user message and call LLM ─────────────────────────────────────
    history = build_history(history, "user", user_content)

    try:
        response = _ai.chat.completions.create(
            model    = OPENAI_MODEL,
            messages = history,
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"[ThreadHandler] OpenAI error: {e}")
        reply = "⚠️ I ran into an error generating a response. Please try again."

    # ── Append assistant reply and save history ───────────────────────────────
    history = build_history(history, "assistant", reply)
    history = trim_history(history, MAX_MESSAGES)
    save_thread_history(workspace_id, thread_ts, history)

    log.warning(
        f"[ThreadHandler] thread={thread_ts} messages={len(history)} "
        f"session_expired={session_expired}"
    )

    return {
        "reply"          : reply,
        "session_expired": session_expired,
    }
