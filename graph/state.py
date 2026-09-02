"""The state shape threaded through one turn of the graph.

`messages` is the only field the Redis checkpointer persists across turns -
just a running chat window, cheap to lose. Facts/profile/commitments live
in Postgres and get re-read fresh every turn instead.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict
from uuid import UUID

from schemas import Episode, ExtractionResult, PersonaCommitment, ScoredFact


class ConversationState(TypedDict, total=False):
    # identity
    user_id: str
    thread_id: str
    session_id: UUID
    persona_id: str

    # checkpointed across turns for this thread
    messages: Annotated[list[dict[str, str]], add]

    # this turn
    user_message: str

    # retrieve
    retrieved_facts: list[ScoredFact]
    retrieved_commitments: list[PersonaCommitment]
    retrieved_episodes: list[Episode]
    voice_anchors: list[Episode]

    # compose
    system_prompt: str

    # respond
    response: str
    user_message_id: UUID | None
    companion_message_id: UUID | None

    # extract / adjudicate
    extraction: ExtractionResult | None
