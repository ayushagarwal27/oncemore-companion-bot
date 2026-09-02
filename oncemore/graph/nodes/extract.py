"""Extract candidate facts, profile updates, and persona commitments from
one turn.

Uses llm.parse(ExtractionResult) directly rather than LangMem's
create_memory_manager - that API is built around a flat collection of one
schema, and doesn't fit one combined result whose three lists route to
three different storage backends with different update rules each.
"""

from __future__ import annotations

import llm
from config import settings
from graph.state import ConversationState
from logs import get_logger
from prompts.extraction import EXTRACTION_SYSTEM
from schemas import ExtractionResult

log = get_logger(__name__)


def _render_turn(user_message: str, response: str) -> str:
    return f"User: {user_message}\nCompanion: {response}"


async def extract(state: ConversationState) -> dict:
    turn_text = _render_turn(state["user_message"], state["response"])
    result = await llm.parse(
        schema=ExtractionResult,
        system=EXTRACTION_SYSTEM,
        user=turn_text,
        model=settings.extraction_model,
    )
    log.debug(
        "extracted",
        user_id=state["user_id"],
        candidates=[c.text for c in result.candidates],
        profile_updates=[(u.field, u.value) for u in result.profile_updates],
        persona_commitments=[(c.topic, c.text) for c in result.persona_commitments],
    )
    return {"extraction": result}
