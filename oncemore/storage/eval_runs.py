"""Storage for the eval harness. One row per scenario run in `eval_runs`,
one row per scored check in `eval_results`.
"""

from __future__ import annotations

import json
from uuid import UUID

from storage.pg import get_pool


async def start_run(scenario: str, *, config: dict | None = None) -> UUID:
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "INSERT INTO eval_runs (scenario, config) VALUES (%s, %s::jsonb) RETURNING id",
            (scenario, json.dumps(config or {})),
        )
        row = await cursor.fetchone()
    return row["id"]


async def record_result(
    run_id: UUID,
    probe_id: str,
    metric: str,
    *,
    passed: bool | None,
    score: float | None,
    detail: dict | None = None,
) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO eval_results (run_id, probe_id, metric, passed, score, detail)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            """,
            (run_id, probe_id, metric, passed, score, json.dumps(detail) if detail else None),
        )


async def finish_run(run_id: UUID, summary: dict) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE eval_runs SET ended_at = now(), summary = %s::jsonb WHERE id = %s",
            (json.dumps(summary), run_id),
        )
