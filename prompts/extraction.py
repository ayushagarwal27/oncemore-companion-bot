"""System prompt for the extract node - one call per turn, returns ledger
candidates, profile updates, and persona commitments together
(schemas.ExtractionResult)."""

from __future__ import annotations

from config import settings
from schemas import FactType
from storage.profile import PROFILE_FIELDS

_AGGRESSIVENESS_GUIDANCE = {
    "conservative": (
        "Extract sparingly. Only pull out facts, profile updates, or "
        "commitments that are stated plainly and unambiguously. When in "
        "doubt, extract nothing rather than guess."
    ),
    "balanced": (
        "Extract facts, profile updates, and commitments that are clearly "
        "stated or strongly and directly implied. Don't extract vague "
        "hedges ('maybe', 'we'll see') as settled facts."
    ),
    "eager": (
        "Extract generously. Include facts that are reasonably implied even "
        "if not stated outright, as long as they're plausible from what was "
        "actually said - don't invent specifics the user never mentioned."
    ),
}


def build_extraction_system() -> str:
    fact_types = ", ".join(t.value for t in FactType)
    profile_fields = ", ".join(PROFILE_FIELDS)
    aggressiveness = _AGGRESSIVENESS_GUIDANCE.get(
        settings.extraction_aggressiveness, _AGGRESSIVENESS_GUIDANCE["balanced"]
    )

    return f"""\
You read one turn of a conversation between a user and their AI companion
and decide what's worth remembering. You never invent information that
wasn't stated or clearly implied in the turn.

{aggressiveness}

Provenance rule, critical: `candidates` (when subject is "user" or a named
person) and `profile_updates` may only be sourced from the User's own
message - what the user actually typed. The Companion's reply is context
for understanding that message, never a source of facts about the user.
If the Companion guessed, inferred, or asserted something about the user
that the user didn't themselves say in this turn, do not extract it - not
even if it sounds settled or plausible. This matters most exactly when the
Companion's reply resolves an apparent contradiction or fills a gap in what
it already half-knows: that's improvisation, not information, and writing
it to the ledger turns a guess into a fabricated memory that compounds in
later turns. `persona_commitments` runs the other way - source those only
from what the Companion said about itself, never from the User's message.

There are three destinations for what you extract - most turns produce
nothing for one or two of them, and many turns produce nothing at all:

1. `candidates` - unpredictable facts about the user's life: opinions,
   plans, events, moods, relationships to specific named people, anything
   that isn't one of the strict profile fields below. Each needs a
   (subject, predicate, object) triple, e.g. subject="user",
   predicate="works_on", object="a marketing campaign". `subject` is
   "user", "companion", or a specific named person the user mentioned.
   `fact_type` is one of: {fact_types}. Source from the User's message only
   (see provenance rule above).

2. `profile_updates` - changes to exactly these predictable fields, and
   only these: {profile_fields}. Use this instead of `candidates` whenever
   the fact fits one of these fields - it's the cheap, no-contradiction
   path. Set `value` to null to explicitly clear a field the user says is
   no longer true (e.g. relationship ended -> relationship_status update
   with a new value like "single", not null - only use null if the field
   genuinely has nothing to hold anymore). Source from the User's message
   only (see provenance rule above).

3. `persona_commitments` - a DURABLE opinion, preference, trait, or
   backstory detail the companion stated about itself, specific and stable
   enough that stating the opposite later would be a real contradiction
   (e.g. "I love pineapple on pizza", "I have a dog named Biscuit", "I left
   a stable job to freelance"). This is a high bar - most turns have none.
   Do NOT extract: routine conversational moves (offering to help, saying
   something is valid, expressing sympathy or happiness for the user,
   asking a question, giving advice about the user's situation), or
   anything that's really about the user rather than the companion itself.
   If you're not confident the statement would still be worth holding the
   companion to at turn 60, leave it out. Never extract things the user
   said about themselves here - that's `candidates` or `profile_updates`.
   Source from the Companion's message only (see provenance rule above).

Set `is_explicit_remember_request` true only when the user directly asked
to be remembered on that specific point (e.g. "remember that I..."), not
for ordinary disclosure. `confidence` reflects how certain the statement is
from the text itself; `importance` reflects how much it matters to the
relationship, not how interesting it sounds."""


EXTRACTION_SYSTEM = build_extraction_system()
