"""
utils/history_manager.py
────────────────────────
Utility functions for managing LLM conversation history arrays.

These work on plain Python lists — they are NOT tied to Redis.
Redis read/write lives in redis_client.py.
"""

import logging

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MAX_MESSAGES = 50   # keep last N messages (excluding system prompt)
DEFAULT_MAX_TOKENS   = 8000  # rough token estimate before trimming kicks in


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_history(existing_history: list, role: str, content: str) -> list:
    """
    Appends a new message to the history array.

    Args:
        existing_history : current list of { role, content } dicts
        role             : "user" or "assistant"
        content          : message text

    Returns:
        Updated history list (new list, original is not mutated).
    """
    updated = list(existing_history)
    updated.append({"role": role, "content": content})
    return updated


def trim_history(history: list, max_messages: int = DEFAULT_MAX_MESSAGES) -> list:
    """
    Trims history to avoid exceeding the LLM context window.

    Strategy:
      - If the first message is a system prompt  → always keep it
      - Keep the most recent `max_messages` from the rest

    Args:
        history      : list of { role, content } dicts
        max_messages : max number of non-system messages to keep

    Returns:
        Trimmed history list.
    """
    if not history:
        return []

    # Detect and preserve system prompt
    if history[0].get("role") == "system":
        system_msg   = history[0]
        rest         = history[1:]
        trimmed_rest = rest[-max_messages:] if len(rest) > max_messages else rest
        result       = [system_msg] + trimmed_rest
        if len(rest) > max_messages:
            removed = len(rest) - max_messages
            log.warning(f"[HistoryManager] Trimmed {removed} old messages (kept system + last {max_messages})")
        return result

    # No system prompt — just keep the last max_messages
    if len(history) > max_messages:
        removed = len(history) - max_messages
        log.warning(f"[HistoryManager] Trimmed {removed} old messages (kept last {max_messages})")
        return history[-max_messages:]

    return list(history)


def is_token_limit_exceeded(history: list, max_tokens: int = DEFAULT_MAX_TOKENS) -> bool:
    """
    Rough token estimate: total characters / 4 ≈ tokens.
    Returns True if estimated token count exceeds max_tokens.

    Args:
        history    : list of { role, content } dicts
        max_tokens : token threshold (default 8000)

    Returns:
        True if over limit, False if within limit.
    """
    total_chars  = sum(len(str(msg.get("content", ""))) for msg in history)
    est_tokens   = total_chars // 4
    over_limit   = est_tokens > max_tokens
    if over_limit:
        log.warning(f"[HistoryManager] Token limit exceeded: ~{est_tokens} tokens > {max_tokens}")
    return over_limit


def make_system_prompt() -> dict:
    """
    Returns the system prompt message dict used for every IRT thread.
    Edit the content here to change the bot's persona globally.
    """
    return {
        "role": "system",
        "content": (
            "You are an IRT (Incident Response Team) support assistant for ConverSight. "
            "You help IRT team members and clients resolve platform issues based on past cases. "
            "Be concise, specific, and technical. "
            "If you don't know something, say so clearly and suggest escalation."
        )
    }