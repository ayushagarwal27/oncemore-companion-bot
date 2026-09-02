"""A scripted conversation through the real graph - extract, adjudicate,
and persist included, not direct repository calls - that plants a fact,
revisits it, then contradicts it, and checks the planted fact ends up
superseded rather than sitting next to the new one.

Matches on the fact's actual content ("two months") rather than a
specific (subject, predicate) slot, since predicates are chosen freely by
the extraction LLM and the "dating" fact and the "broke up" fact aren't
guaranteed to share one.

Uses a dedicated test user so a stray fact from other testing doesn't
sneak into the "mentions Priya" filter below.

    uv run python scripts/test_graph.py
"""

from __future__ import annotations

import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from graph.build import build_graph, run_turn
from logs import configure_logging
from storage import facts
from storage.cache import close_redis
from storage.pg import close_pool, ensure_user, init_schema

USER = "graph_test_user"
THREAD = "graph-exit-check"

TURNS = [
    "hey, how's it going",
    "what do you think about pineapple on pizza",
    "so I'm dating someone new, her name's priya, it's going really well",
    "work has been a nightmare this week, my manager keeps piling on scope",
    "anyway yeah priya and I have been together about two months now",
    "we went thrifting together this weekend actually",
    "quick unrelated thing - I've been thinking about learning guitar",
    "remember priya? yeah that's over, we broke up on Tuesday",
    "I'm doing okay about it honestly, more relieved than sad",
    "who am I dating again, did I ever tell you",
]


async def main() -> None:
    configure_logging()
    await init_schema()
    await ensure_user(USER)

    async with InMemorySaver() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        for turn in TURNS:
            result = await run_turn(graph, user_id=USER, thread_id=THREAD, user_message=turn)
            print(f"\nyou:  {turn}")
            print(f"mira: {result['response']}")

    all_facts = await facts.all_for_user(USER)
    priya_facts = [f for f in all_facts if "priya" in f.text.lower()]

    print(f"\n\nall facts mentioning Priya:")
    for f in priya_facts:
        print(f"  [{f.status.value:10}] {f.text}")

    breakup_keywords = ("broke up", "break up", "single", "ended", "no longer")
    stale_active = [
        f
        for f in priya_facts
        if f.status.value == "active" and not any(k in f.text.lower() for k in breakup_keywords)
    ]

    assert priya_facts, "expected to find some fact touching the planted relationship"
    assert not stale_active, (
        "found an active fact that still claims an unresolved ongoing relationship "
        f"after the breakup was disclosed: {[f.text for f in stale_active]!r}"
    )
    print(
        "\nno active fact still claims an unresolved relationship after the "
        "breakup — correct (the adjudicator may merge old + new into one "
        "resolved active fact, which is fine as long as it reflects the breakup)"
    )

    await close_redis()
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
