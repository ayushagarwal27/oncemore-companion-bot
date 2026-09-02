"""LangMem-based optimizer for the 'adaptive' prompt zone only.

Generates candidates and stores them unpromoted - it never touches
persona/canon.py or the 'frozen' zone. Promotion should be gated on eval
scores (candidate vs. currently-promoted version), but `eval/` only
covers store-level metrics today, not reply quality, so `promote()` just
applies a decision someone already made by hand. An optimizer that
promotes its own output with no gate at all is exactly the persona-drift
problem this split exists to prevent, so "requires a human for now" beats
auto-promoting.
"""

from __future__ import annotations

from uuid import UUID

from langmem import create_prompt_optimizer

from config import settings
from storage import messages as messages_store
from storage import prompt_versions


# create_prompt_optimizer has no separate "system context" argument, so the
# `prompt` field it revises is the only place to anchor it. Without a
# guardrail here, a bare placeholder prompt drifted into a generic
# assistant (bullet points, "how can I help you today", disclaimers) even
# with real Mira-voiced trajectories fed in - it's applied fresh on every
# call rather than stored in `content`, so it can't quietly erode across
# edits or end up saved as if it were part of the real system prompt.
_MARKER = "===ADAPTIVE NOTES BELOW==="

_GUARDRAIL_PREAMBLE = f"""\
This is the adaptive interaction-style addendum for Mira, an already fully
defined character (warm, direct, opinionated, never uses bullet points or
numbered lists, replies in 2-4 sentences, never uses therapy-speak or
generic-assistant phrasing). You may ONLY add small, specific
interaction-style refinements grounded in what actually happened in the
conversations below - never redefine who she is, add traits or opinions,
or introduce any of the following, ever, regardless of what the
conversations seem to suggest: bullet points or numbered lists, "How can I
help you today"-style greetings, clarifying-question templates, safety
disclaimers, source-citation requirements, or any other generic-assistant
convention. If nothing specific and well-grounded comes up, return the
notes below unchanged rather than inventing generic advice. Return ONLY
the resulting notes, starting immediately after the marker line below -
do not repeat any of these instructions back.

{_MARKER}
"""

STARTER_ADAPTIVE_NOTES = "No additional interaction notes yet."

_optimizer = None


def _get_optimizer():
    global _optimizer
    if _optimizer is None:
        _optimizer = create_prompt_optimizer(f"openai:{settings.chat_model}", kind="prompt_memory")
    return _optimizer


async def _load_trajectory(user_id: str) -> tuple[list[dict], dict | None]:
    """This user's full transcript as one trajectory, plus the most recent
    feedback found in it, if any. Doesn't split by session or weight recent
    feedback higher - fine for a first pass, not for real multi-session use."""
    transcript = await messages_store.full_transcript(user_id)
    turns = [
        {"role": "user" if m.role == "user" else "assistant", "content": m.content}
        for m in transcript
    ]
    feedback = next(
        (
            {"score": m.feedback, "comment": m.feedback_reason or ""}
            for m in reversed(transcript)
            if m.feedback is not None
        ),
        None,
    )
    return turns, feedback


async def generate_candidate(user_id: str) -> prompt_versions.PromptVersion:
    """Run the optimizer once over this user's history and store the
    result as an unpromoted candidate. Only the notes get saved, not the
    guardrail preamble."""
    current = await prompt_versions.get_active("adaptive")
    current_notes = current.content if current else STARTER_ADAPTIVE_NOTES

    turns, feedback = await _load_trajectory(user_id)
    if not turns:
        raise ValueError(f"no conversation history for {user_id!r} to optimize against")

    guarded_prompt = _GUARDRAIL_PREAMBLE + current_notes
    updated = await _get_optimizer().ainvoke(
        {"trajectories": [(turns, feedback)], "prompt": guarded_prompt}
    )

    # It's told not to echo the preamble back, but strip it just in case.
    if _MARKER in updated:
        updated = updated.rsplit(_MARKER, 1)[1].strip()

    return await prompt_versions.insert_candidate(
        "adaptive",
        updated,
        parent_id=current.id if current else None,
    )


async def promote(version_id: UUID, *, eval_scores: dict | None = None) -> None:
    """Flip a candidate live. Call this after confirming (by eval or by a
    human reading it) that it doesn't regress the character - this just
    applies that decision, it doesn't make it."""
    await prompt_versions.promote(version_id, eval_scores=eval_scores)


async def history(*, persona_id: str | None = None):
    """Every adaptive-zone version ever generated, newest first."""
    return await prompt_versions.history("adaptive", persona_id=persona_id)
