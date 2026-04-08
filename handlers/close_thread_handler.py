"""
handlers/close_thread_handler.py
─────────────────────────────────
Handles the "Close Thread" button action.

Deletes the Redis key for the thread → clears conversation memory.
Called when user clicks the 🔒 Close Thread button in Slack.
"""

import logging

from utils.redis_client import delete_thread_history

log = logging.getLogger(__name__)


def handle_close_thread(workspace_id: str, thread_ts: str) -> dict:
    """
    Deletes thread history from Redis when user closes a thread.

    Args:
        workspace_id : Slack workspace/team ID
        thread_ts    : The thread timestamp used as part of the Redis key

    Returns:
        dict with keys:
          "success" : bool — True if deleted, False if failed
          "message" : str  — human-readable confirmation to post back to Slack
    """
    log.warning(f"[CloseThread] Closing thread — workspace={workspace_id} thread_ts={thread_ts}")

    success = delete_thread_history(workspace_id, thread_ts)

    if success:
        log.warning(f"[CloseThread] ✅ Redis key deleted for thread_ts={thread_ts}")
        return {
            "success": True,
            "message": (
                "🔒 *Thread closed.* Conversation memory has been cleared.\n"
                "_Start a new message to begin a fresh session._"
            )
        }
    else:
        log.error(f"[CloseThread] ❌ Failed to delete Redis key for thread_ts={thread_ts}")
        return {
            "success": False,
            "message": (
                "⚠️ Could not clear thread memory. "
                "Please try again or contact the IRT team."
            )
        }