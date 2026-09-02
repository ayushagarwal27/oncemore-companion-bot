"""Apply the cheap-path writes from extraction: profile field overwrites
and persona self-commitments. Neither needs a neighbour search or an LLM
verdict call, so this stays separate from adjudicate.py.
"""

from __future__ import annotations

from config import settings
from graph.state import ConversationState
from logs import get_logger
from schemas import CommitmentCandidate
from storage import commitments, profile

log = get_logger(__name__)


async def _persist_commitment(
    candidate: CommitmentCandidate, *, persona_id: str, source_message_id
) -> None:
    existing = await commitments.by_topic(candidate.topic, persona_id=persona_id)
    new_commitment = await commitments.record(
        candidate, persona_id=persona_id, source_message_id=source_message_id
    )
    if existing:
        await commitments.supersede(existing.id, new_commitment.id)
    log.debug(
        "commitment_persisted",
        persona_id=persona_id,
        topic=candidate.topic,
        text=candidate.text,
        superseded=str(existing.id) if existing else None,
    )


async def persist(state: ConversationState) -> dict:
    extraction = state.get("extraction")
    if not extraction:
        return {}

    if extraction.profile_updates:
        changed = await profile.apply_updates(
            state["user_id"],
            extraction.profile_updates,
            source_message_id=state.get("companion_message_id"),
        )
        if changed:
            log.debug("profile_updated", user_id=state["user_id"], changed=changed)

    persona_id = state.get("persona_id", settings.persona_id)
    for candidate in extraction.persona_commitments:
        await _persist_commitment(
            candidate, persona_id=persona_id, source_message_id=state.get("companion_message_id")
        )

    return {}
