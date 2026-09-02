"""For each extracted FactCandidate, decide against its active neighbours
whether it's NEW, a DUPLICATE, a REFINEMENT, a CONTRADICTION, or
EPISODIC_ONLY, and apply the write.

Skips the adjudicator call and inserts straight away when there are no
related active facts - nothing to adjudicate against, and it saves a call
on the common case of a genuinely new topic.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import llm
from config import settings
from graph.state import ConversationState
from logs import get_logger
from prompts.adjudication import ADJUDICATION_SYSTEM
from schemas import Adjudication, FactCandidate, MemoryFact, Verdict
from storage import facts
from storage.embeddings import embed

log = get_logger(__name__)


def _render_candidate(candidate: FactCandidate) -> str:
    return (
        f"Candidate: {candidate.text}\n"
        f"  subject={candidate.subject!r} predicate={candidate.predicate!r} "
        f"object={candidate.object!r} fact_type={candidate.fact_type.value}"
    )


def _render_neighbours(neighbours: list[MemoryFact]) -> str:
    if not neighbours:
        return "(no related active facts)"
    return "\n".join(f"- id={fact.id} :: {fact.text}" for fact in neighbours)


def _parse_valid_from(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


async def _adjudicate_one(
    user_id: str,
    candidate: FactCandidate,
    *,
    source_message_id,
    source_session_id,
) -> None:
    embedding = await embed(candidate.text)
    neighbours = await facts.get_related_active(user_id, candidate, embedding=embedding)

    if not neighbours:
        await facts.insert_fact(
            user_id,
            candidate,
            embedding=embedding,
            source_message_id=source_message_id,
            source_session_id=source_session_id,
        )
        return

    prompt = (
        f"{_render_candidate(candidate)}\n\n"
        f"Existing active facts on related topics:\n{_render_neighbours(neighbours)}"
    )
    verdict = await llm.parse(
        schema=Adjudication,
        system=ADJUDICATION_SYSTEM,
        user=prompt,
        model=settings.adjudicator_model,
    )

    log.info(
        "adjudicated",
        user_id=user_id,
        candidate=candidate.text,
        verdict=verdict.verdict.value,
        target=verdict.target_fact_id,
    )

    match verdict.verdict:
        case Verdict.NEW:
            await facts.insert_fact(
                user_id,
                candidate,
                embedding=embedding,
                source_message_id=source_message_id,
                source_session_id=source_session_id,
            )
        case Verdict.DUPLICATE if verdict.target_fact_id:
            await facts.mark_duplicate(UUID(verdict.target_fact_id))
        case Verdict.REFINEMENT if verdict.target_fact_id:
            await facts.merge_into(
                UUID(verdict.target_fact_id),
                verdict.merged_text or candidate.text,
                importance=candidate.importance,
            )
        case Verdict.CONTRADICTION if verdict.target_fact_id:
            valid_from = _parse_valid_from(verdict.valid_from)
            new_fact = await facts.insert_fact(
                user_id,
                candidate,
                embedding=embedding,
                valid_from=valid_from,
                source_message_id=source_message_id,
                source_session_id=source_session_id,
            )
            await facts.supersede_fact(
                UUID(verdict.target_fact_id), new_fact.id, valid_to=valid_from
            )
        case Verdict.EPISODIC_ONLY:
            pass  # not durable enough for the ledger; intentionally dropped here
        case _:
            log.warning(
                "adjudication_missing_target",
                verdict=verdict.verdict.value,
                candidate=candidate.text,
            )


async def adjudicate(state: ConversationState) -> dict:
    extraction = state.get("extraction")
    if not extraction or not extraction.candidates:
        return {}

    for candidate in extraction.candidates:
        await _adjudicate_one(
            state["user_id"],
            candidate,
            source_message_id=state.get("companion_message_id"),
            source_session_id=state.get("session_id"),
        )
    return {}
