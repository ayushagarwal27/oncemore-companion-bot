"""Assemble the system prompt via llm.build_prompt."""

from __future__ import annotations

import llm
from config import settings
from graph.state import ConversationState
from logs import get_logger
from persona.registry import get_canon
from storage import profile, prompt_versions

log = get_logger(__name__)


def _render_facts_and_episodes(state: ConversationState) -> str:
    lines = [f"- {item.fact.text}" for item in state.get("retrieved_facts", [])]
    lines += [f"- {episode.text}" for episode in state.get("retrieved_episodes", [])]
    return "\n".join(lines)


def _render_commitments(state: ConversationState) -> str:
    return "\n".join(f"- {c.text}" for c in state.get("retrieved_commitments", []))


def _render_voice_anchors(state: ConversationState) -> str:
    return "\n".join(f"- {episode.text}" for episode in state.get("voice_anchors", []))


async def compose(state: ConversationState) -> dict:
    user_profile = await profile.get_profile(state["user_id"])
    adaptive = await prompt_versions.get_active("adaptive")

    persona_id = state.get("persona_id", settings.persona_id)
    profile_block = user_profile.as_prompt_block()
    memory_block = _render_facts_and_episodes(state)
    commitments_block = _render_commitments(state)
    voice_anchors = _render_voice_anchors(state)

    system_prompt = llm.build_prompt(
        persona_canon=get_canon(persona_id),
        voice_anchors=voice_anchors,
        adaptive_notes=adaptive.content if adaptive else "",
        profile_block=profile_block,
        memory_block=memory_block,
        commitments_block=commitments_block,
    )
    log.debug(
        "memory_in_prompt",
        user_id=state["user_id"],
        persona_id=persona_id,
        profile_block=profile_block,
        memory_block=memory_block,
        commitments_block=commitments_block,
        voice_anchors=voice_anchors,
    )
    return {"system_prompt": system_prompt}
