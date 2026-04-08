"""
utils/redis_client.py
─────────────────────
Redis utility for storing per-thread conversation history.

Key format : thread:{workspace_id}:{thread_ts}
Value      : JSON array of { role, content } messages
TTL        : 86400 seconds (24 hours), reset on every save

Requires:
  pip install redis python-dotenv

Env vars:
  REDIS_URL  (default: redis://localhost:6379)
"""

import json
import logging
import os

import redis
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ── Redis connection (singleton) ──────────────────────────────────────────────

_client: redis.Redis | None = None

def _get_client() -> redis.Redis:
    """
    Returns a shared Redis client.
    Creates it on first call. Raises if REDIS_URL is missing.
    """
    global _client
    if _client is None:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        _client = redis.Redis.from_url(url, decode_responses=True)
        log.warning(f"[Redis] Connected → {url}")
    return _client


def _make_key(workspace_id: str, thread_ts: str) -> str:
    """
    Builds the Redis key for a given thread.
    Example: thread:T12345:1712345678.123456
    """
    return f"thread:{workspace_id}:{thread_ts}"


# ── Public API ────────────────────────────────────────────────────────────────

def get_thread_history(workspace_id: str, thread_ts: str) -> list:
    """
    Loads conversation history for a thread from Redis.

    Returns:
        list of { role, content } dicts — empty list if not found or on error.
    """
    key = _make_key(workspace_id, thread_ts)
    try:
        raw = _get_client().get(key)
        if raw is None:
            log.warning(f"[Redis] GET {key} → not found (new thread or expired)")
            return []
        history = json.loads(raw)
        log.warning(f"[Redis] GET {key} → {len(history)} messages loaded")
        return history
    except Exception as e:
        log.error(f"[Redis] GET {key} failed: {e}")
        return []


def save_thread_history(workspace_id: str, thread_ts: str, history: list) -> bool:
    """
    Saves (overwrites) conversation history for a thread in Redis.
    Resets TTL to 86400 seconds on every save — keeps active threads alive.

    Returns:
        True on success, False on failure.
    """
    key = _make_key(workspace_id, thread_ts)
    try:
        _get_client().setex(key, 259200, json.dumps(history, ensure_ascii=False))
        log.warning(f"[Redis] SETEX {key} → {len(history)} messages saved (TTL reset to 24h)")
        return True
    except Exception as e:
        log.error(f"[Redis] SETEX {key} failed: {e}")
        return False


def delete_thread_history(workspace_id: str, thread_ts: str) -> bool:
    """
    Deletes conversation history for a thread from Redis.
    Called when user clicks 'Close Thread'.

    Returns:
        True on success, False on failure.
    """
    key = _make_key(workspace_id, thread_ts)
    try:
        deleted = _get_client().delete(key)
        if deleted:
            log.warning(f"[Redis] DEL {key} → ✅ deleted")
        else:
            log.warning(f"[Redis] DEL {key} → key did not exist (already expired?)")
        return True
    except Exception as e:
        log.error(f"[Redis] DEL {key} failed: {e}")
        return False


def ping_redis() -> bool:
    """
    Health check — returns True if Redis is reachable.
    Call this at startup to verify the connection early.
    """
    try:
        return _get_client().ping()
    except Exception as e:
        log.error(f"[Redis] PING failed: {e}")
        return False