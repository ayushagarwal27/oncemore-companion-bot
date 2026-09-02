"""End-to-end check of the storage layer against live Postgres and Redis.

Checks two things:
  1. facts persist across a process restart (run it twice, count grows)
  2. a contradiction supersedes the old fact instead of piling up next to it

    uv run python scripts/smoke_test.py

Costs a handful of embedding calls. No chat model is used.
"""

from __future__ import annotations

import asyncio

from schemas import FactCandidate, FactType, ProfileFieldUpdate
from storage import cache, facts, messages, profile
from storage.pg import close_pool, ensure_user, init_schema

USER = "smoke-user"


async def main() -> None:
    await init_schema()
    await ensure_user(USER)

    session_id = await messages.resume_or_start(USER, thread_id="smoke-thread")
    turn = await messages.append(session_id, USER, "user", "I live in Delhi with my girlfriend Priya.")

    # --- the expensive path: unpredictable fact into the ledger -------------
    original = await facts.insert_fact(
        USER,
        FactCandidate(
            subject="user",
            predicate="in_relationship_with",
            object="Priya",
            text="The user is in a relationship with Priya.",
            fact_type=FactType.RELATIONSHIP,
            confidence=0.95,
            importance=0.9,
            is_explicit_remember_request=False,
        ),
        source_message_id=turn.id,
        source_session_id=session_id,
    )
    print(f"inserted   {original.id} :: {original.text}")

    # --- the cheap path: predictable attribute into the profile ------------
    changed = await profile.apply_updates(
        USER,
        [ProfileFieldUpdate(field="relationship_status", value="partnered",
                            reason="stated in turn 0")],
        source_message_id=turn.id,
    )
    print(f"profile    {changed}")

    # --- contradiction ------------------------------------------------------
    replacement = await facts.insert_fact(
        USER,
        FactCandidate(
            subject="user",
            predicate="in_relationship_with",
            object="nobody",
            text="The user broke up with Priya and is now single.",
            fact_type=FactType.RELATIONSHIP,
            confidence=0.95,
            importance=0.95,
            is_explicit_remember_request=False,
        ),
    )
    await facts.supersede_fact(original.id, replacement.id)
    await profile.apply_updates(
        USER,
        [ProfileFieldUpdate(field="relationship_status", value="single",
                            reason="breakup disclosed")],
    )

    # --- assertions ---------------------------------------------------------
    history = await facts.history_for_slot(USER, "user", "in_relationship_with")
    active = [f for f in history if f.status.value == "active"]
    superseded = [f for f in history if f.status.value == "superseded"]

    assert len(active) == 1, f"expected exactly one active fact, got {len(active)}"
    assert superseded and superseded[0].superseded_by == replacement.id
    assert superseded[0].valid_to is not None, "old fact never had its window closed"
    print(f"ledger     {len(active)} active, {len(superseded)} superseded — correct")

    hits = await facts.search(USER, "who is my partner")
    top = hits[0]
    assert top.fact.status.value == "active", "retrieval surfaced a superseded fact"
    print(f"retrieval  {top.fact.text!r}")
    print(f"           rrf={top.rrf:.4f} strength={top.strength:.3f} final={top.final_score:.3f}")

    field_log = await profile.field_history(USER, "relationship_status")
    print(f"profile    {len(field_log)} recorded changes: "
          f"{[(r['old_value'], r['new_value']) for r in field_log]}")

    print(f"\ntotal active facts for {USER}: {await facts.active_count(USER)}")
    print("run again after a restart — the count persists, which is the point")

    await cache.close_redis()
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
