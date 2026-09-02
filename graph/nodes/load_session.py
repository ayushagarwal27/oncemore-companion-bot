"""Reattach to (or start) the Postgres session/turn log for this thread.

This is separate from the checkpointer's `messages` window - it's the
durable log that fact/profile/commitment rows point back to via
source_message_id, and it has to work even if the checkpointer is empty.
"""

from __future__ import annotations

from graph.state import ConversationState
from storage import messages as messages_store


async def load_session(state: ConversationState) -> dict:
    session_id = await messages_store.resume_or_start(state["user_id"], state["thread_id"])
    user_row = await messages_store.append(
        session_id, state["user_id"], "user", state["user_message"]
    )
    return {"session_id": session_id, "user_message_id": user_row.id}
