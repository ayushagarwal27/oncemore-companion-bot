"""Sanity check: does the persona read as a specific character?

Not an automated eval - drives ~10 turns through the real chat model with
the real persona canon and seeded voice anchors, and prints the transcript
for a human to read. Keeps its own running transcript rather than going
through the graph, which is fine for this script but not a pattern to
copy elsewhere.

Turns are picked to probe specific failure modes: planting a fact and
later contradicting it (the breakup case), repeating an opinion to check
for self-contradiction, and applying "topic pressure" (a request for a
neutral bulleted list, a point-blank "are you an AI") to check she doesn't
flatten into generic-assistant tone.

    uv run python scripts/test_persona.py
"""

from __future__ import annotations

import asyncio

import llm
from persona.canon import PERSONA_CANON
from storage import episodes
from storage.pg import close_pool

DEMO_USER = "demo_user"

TURNS = [
    "hey, long day. my girlfriend priya and i had a good weekend though, just stayed in.",
    "what do you think about pineapple on pizza, for the record",
    "ok separate topic - work is a nightmare right now, my manager keeps adding scope with no extra time",
    "can you just give me a neutral bulleted list of options for dealing with a bad manager",
    "are you even a real person or just an ai saying what i want to hear",
    "so remember priya? we actually broke up last week. i'm fine, i think.",
    "wait, what was your take on pineapple on pizza again? did I ask you that already",
    "i've been thinking about quitting this job entirely, no backup plan, is that insane",
    "you mentioned a brother once, or did I imagine that",
    "anyway - good news actually, i got offered a job in another city",
]


def render_voice_anchors(items) -> str:
    return "\n".join(f"- {episode.text}" for episode in items)


async def main() -> None:
    anchors = await episodes.voice_anchors(DEMO_USER, limit=10)
    if not anchors:
        raise SystemExit("no voice anchors found - run scripts/seed_persona.py first")

    system_prompt = llm.build_prompt(
        persona_canon=PERSONA_CANON,
        voice_anchors=render_voice_anchors(anchors),
        profile_block="- (nothing known yet)",
        memory_block="",
        commitments_block="",
    )

    transcript: list[str] = []
    for turn in TURNS:
        transcript.append(f"User: {turn}")
        conversation_so_far = "\n".join(transcript)
        reply = await llm.complete(system=system_prompt, user=conversation_so_far)
        transcript.append(f"Mira: {reply}")

        print(f"\nUser: {turn}")
        print(f"Mira: {reply}")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
