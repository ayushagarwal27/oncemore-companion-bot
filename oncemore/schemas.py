"""Typed contracts for every LLM boundary in the system.

All extraction and adjudication models are written to be compatible with
OpenAI Structured Outputs strict mode: no defaults on LLM-facing models, no
bare dicts, unions written as `X | None`.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FactType(str, Enum):
    IDENTITY = "identity"
    RELATIONSHIP = "relationship"
    PREFERENCE = "preference"
    EVENT = "event"
    PLAN = "plan"
    OPINION = "opinion"
    MOOD = "mood"


class FactStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    DUPLICATE = "duplicate"


class Verdict(str, Enum):
    """Output of the adjudicator for one candidate against one existing fact."""

    NEW = "NEW"
    DUPLICATE = "DUPLICATE"
    REFINEMENT = "REFINEMENT"
    CONTRADICTION = "CONTRADICTION"
    EPISODIC_ONLY = "EPISODIC_ONLY"


# Half-life in days per fact type, used by the strength term in ranking.
# Identity never decays; mood decays within a few days.
DECAY_HALF_LIFE_DAYS: dict[FactType, float] = {
    FactType.IDENTITY: 3650.0,
    FactType.RELATIONSHIP: 365.0,
    FactType.PREFERENCE: 180.0,
    FactType.OPINION: 180.0,
    FactType.PLAN: 30.0,
    FactType.EVENT: 90.0,
    FactType.MOOD: 3.0,
}

# ---------------------------------------------------------------------------
# Structured Outputs compatibility
#
# Strict mode drops JSON Schema validation keywords: `minimum`, `maximum`,
# `pattern`, `minLength`, `format` and friends are all stripped before the
# request goes out. A pydantic `Field(ge=0, le=1)` therefore constrains
# nothing at generation time and, on some SDK versions, breaks the schema
# outright. So every LLM-facing model states its range in the description
# and clamps after parsing.
#
# The same rules mean: no bare `dict`, no recursive models, all fields
# required, and a nesting ceiling of five levels.
# ---------------------------------------------------------------------------


def _clamp_unit(value: float) -> float:
    """Pull a model-supplied score back into [0, 1]."""
    return min(max(value, 0.0), 1.0)


# ---------------------------------------------------------------------------
# LLM-facing: extraction
# ---------------------------------------------------------------------------


class FactCandidate(BaseModel):
    """One memory-worthy fact pulled from a turn. Not yet reconciled."""

    subject: str = Field(description="Who the fact is about: 'user', 'companion', or a named person")
    predicate: str = Field(description="Short snake_case relation, e.g. 'works_at', 'feels_about'")
    object: str = Field(description="The value of the relation")
    text: str = Field(description="One natural sentence stating the fact; this is what gets embedded")
    fact_type: FactType
    confidence: float = Field(description="How certain the statement is, from 0.0 to 1.0")
    importance: float = Field(
        description="How much this matters to the relationship, from 0.0 to 1.0"
    )
    is_explicit_remember_request: bool = Field(
        description="True if the user directly asked to be remembered on this"
    )

    _clamp = field_validator("confidence", "importance", mode="after")(_clamp_unit)


class ExtractionResult(BaseModel):
    candidates: list[FactCandidate]
    profile_updates: list["ProfileFieldUpdate"]
    persona_commitments: list["CommitmentCandidate"]


class ProfileFieldUpdate(BaseModel):
    """A change to one strict profile field. Cheap supersession path."""

    field: str = Field(description="Profile column name, e.g. 'relationship_status'")
    value: str | None = Field(description="New value, or null to clear the field")
    reason: str = Field(description="Why this update follows from the turn")


class CommitmentCandidate(BaseModel):
    """Something the companion asserted about itself."""

    topic: str = Field(description="Short topic key, e.g. 'favourite_season'")
    text: str = Field(description="The claim, in one sentence")
    confidence: float = Field(description="Certainty from 0.0 to 1.0")

    _clamp = field_validator("confidence", mode="after")(_clamp_unit)


# ---------------------------------------------------------------------------
# LLM-facing: adjudication
# ---------------------------------------------------------------------------


class Adjudication(BaseModel):
    """Decision about one candidate against the related active facts."""

    verdict: Verdict
    target_fact_id: str | None = Field(
        description="UUID of the existing fact being duplicated, refined or contradicted; null for NEW"
    )
    merged_text: str | None = Field(
        description="For REFINEMENT, the consolidated sentence that replaces both; otherwise null"
    )
    valid_from: str | None = Field(
        description="ISO timestamp for when the new fact became true, if the user implied one"
    )
    reasoning: str = Field(description="One sentence explaining the verdict")


# ---------------------------------------------------------------------------
# LLM-facing: guard node
# ---------------------------------------------------------------------------


class ConsistencyCheck(BaseModel):
    """Does a drafted reply contradict one of the companion's own active
    self-commitments? See graph/nodes/guard.py."""

    conflicts: bool = Field(
        description="True if the drafted reply contradicts one of the listed commitments"
    )
    conflicting_commitment_id: str | None = Field(
        description="UUID of the commitment being contradicted, if conflicts is true; otherwise null"
    )
    reasoning: str = Field(description="One sentence explaining the check result")


# ---------------------------------------------------------------------------
# LLM-facing: episodes
# ---------------------------------------------------------------------------


class EpisodeCandidate(BaseModel):
    """A relational moment worth remembering as an experience, not a fact."""

    title: str = Field(description="Short handle, e.g. 'the night she talked about her dad'")
    observation: str = Field(description="What was happening and what the user disclosed")
    companion_action: str = Field(description="How the companion responded")
    outcome: str = Field(description="How it landed and why that worked")
    salience: float = Field(description="How memorable this moment is, from 0.0 to 1.0")

    _clamp = field_validator("salience", mode="after")(_clamp_unit)


# ---------------------------------------------------------------------------
# Internal: persisted rows
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    """Strict semantic profile. Contradictions here resolve by field overwrite."""

    user_id: str
    preferred_name: str | None = None
    pronouns: str | None = None
    age_range: str | None = None
    location: str | None = None
    occupation: str | None = None
    employer: str | None = None
    relationship_status: str | None = None
    partner_name: str | None = None
    living_situation: str | None = None
    key_people: list[str] = Field(default_factory=list)
    current_focus: str | None = None
    ongoing_stressor: str | None = None
    communication_style: str | None = None
    topics_to_avoid: list[str] = Field(default_factory=list)
    version: int = 1
    updated_at: datetime | None = None

    EDITABLE_FIELDS: ClassVar[set[str]] = {
        "preferred_name",
        "pronouns",
        "age_range",
        "location",
        "occupation",
        "employer",
        "relationship_status",
        "partner_name",
        "living_situation",
        "current_focus",
        "ongoing_stressor",
        "communication_style",
    }

    def as_prompt_block(self) -> str:
        """Rendered into the cached prompt prefix, so keep it stable and short."""
        lines: list[str] = []
        for field in (
            "preferred_name",
            "pronouns",
            "location",
            "occupation",
            "employer",
            "relationship_status",
            "partner_name",
            "living_situation",
            "current_focus",
            "ongoing_stressor",
            "communication_style",
        ):
            value = getattr(self, field)
            if value:
                lines.append(f"- {field.replace('_', ' ')}: {value}")
        if self.key_people:
            lines.append(f"- key people: {', '.join(self.key_people)}")
        if self.topics_to_avoid:
            lines.append(f"- avoid: {', '.join(self.topics_to_avoid)}")
        return "\n".join(lines) if lines else "- (nothing known yet)"


class MemoryFact(BaseModel):
    id: UUID
    user_id: str
    subject: str
    predicate: str
    object: str
    text: str
    fact_type: FactType
    status: FactStatus
    valid_from: datetime
    valid_to: datetime | None
    created_at: datetime
    expired_at: datetime | None
    superseded_by: UUID | None
    confidence: float
    importance: float
    access_count: int
    last_accessed_at: datetime | None
    source_message_id: UUID | None
    source_session_id: UUID | None


class ScoredFact(BaseModel):
    """A retrieval hit with its ranking components kept visible, so a bad
    retrieval can be debugged instead of just observed."""

    fact: MemoryFact
    similarity: float
    lexical_rank: int | None
    rrf: float
    strength: float
    final_score: float


class PersonaCommitment(BaseModel):
    id: UUID
    persona_id: str
    topic: str
    text: str
    status: FactStatus
    valid_from: datetime
    valid_to: datetime | None
    created_at: datetime
    superseded_by: UUID | None
    confidence: float
    source_message_id: UUID | None


class PromptVersion(BaseModel):
    """A candidate or promoted version of one procedural-memory zone.

    'frozen' rows are never written by this system - persona/canon.py
    is the actual source of truth for that zone, edited by hand. Only
    'adaptive' rows get inserted, by prompt_optimizer.py.
    """

    id: UUID
    persona_id: str
    zone: Literal["frozen", "adaptive"]
    content: str
    parent_id: UUID | None
    promoted: bool
    eval_scores: dict | None
    created_at: datetime


class Episode(BaseModel):
    id: UUID
    user_id: str
    kind: Literal["relational", "voice"]
    title: str
    observation: str
    companion_action: str | None
    outcome: str | None
    text: str
    salience: float
    pinned: bool
    access_count: int
    occurred_at: datetime
    created_at: datetime


class StoredMessage(BaseModel):
    id: UUID
    session_id: UUID
    user_id: str
    turn_index: int
    role: Literal["user", "companion"]
    content: str
    created_at: datetime
    trace_id: str | None = None
    feedback: int | None = None
    feedback_reason: str | None = None


ExtractionResult.model_rebuild()
