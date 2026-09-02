"""Generate the companion's candidate reply for this turn.

llm.complete only takes system+user, so the message history gets folded
into the user turn as a rendered transcript. Doesn't persist anything -
guard.py does that, after it's had a chance to check (and maybe
regenerate) the reply, so a discarded draft never gets logged.
"""

from __future__ import annotations

import llm
from graph.nodes.transcript import render_transcript
from graph.state import ConversationState


async def respond(state: ConversationState) -> dict:
    transcript = render_transcript(state.get("messages", []), state["user_message"])
    reply = await llm.complete(system=state["system_prompt"], user=transcript)
    return {"response": reply}
