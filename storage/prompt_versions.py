"""The 'adaptive' procedural-memory zone: optimizer candidates and the one
promoted version compose.py reads. Nothing in this codebase ever writes a
'frozen' row here - that zone is persona/canon.py, edited by hand.
"""

from __future__ import annotations

import json
from uuid import UUID

from config import settings
from schemas import PromptVersion
from storage.pg import get_pool

_COLUMNS = "id, persona_id, zone, content, parent_id, promoted, eval_scores, created_at"


def _row(row: dict) -> PromptVersion:
    data = dict(row)
    if isinstance(data.get("eval_scores"), str):
        data["eval_scores"] = json.loads(data["eval_scores"])
    return PromptVersion(**{k: v for k, v in data.items() if k in PromptVersion.model_fields})


async def insert_candidate(
    zone: str,
    content: str,
    *,
    persona_id: str | None = None,
    parent_id: UUID | None = None,
) -> PromptVersion:
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            INSERT INTO prompt_versions (persona_id, zone, content, parent_id)
            VALUES (%s, %s, %s, %s)
            RETURNING {_COLUMNS}
            """,
            (persona_id or settings.persona_id, zone, content, parent_id),
        )
        return _row(await cursor.fetchone())


async def promote(version_id: UUID, *, eval_scores: dict | None = None) -> None:
    """Flip one candidate live, demoting whatever was promoted before it
    for the same persona+zone. Callers decide whether promotion is
    warranted - this just applies the decision."""
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.transaction():
            cursor = await conn.execute(
                "SELECT persona_id, zone FROM prompt_versions WHERE id = %s", (version_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                raise ValueError(f"no prompt_versions row with id {version_id}")

            await conn.execute(
                """
                UPDATE prompt_versions SET promoted = FALSE
                 WHERE persona_id = %s AND zone = %s AND promoted
                """,
                (row["persona_id"], row["zone"]),
            )
            await conn.execute(
                "UPDATE prompt_versions SET promoted = TRUE, eval_scores = %s::jsonb WHERE id = %s",
                (json.dumps(eval_scores) if eval_scores is not None else None, version_id),
            )


async def get_active(zone: str, *, persona_id: str | None = None) -> PromptVersion | None:
    """The one promoted row compose.py should actually use for this zone."""
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            SELECT {_COLUMNS} FROM prompt_versions
             WHERE persona_id = %s AND zone = %s AND promoted
             ORDER BY created_at DESC LIMIT 1
            """,
            (persona_id or settings.persona_id, zone),
        )
        row = await cursor.fetchone()
    return _row(row) if row else None


async def history(zone: str, *, persona_id: str | None = None) -> list[PromptVersion]:
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            SELECT {_COLUMNS} FROM prompt_versions
             WHERE persona_id = %s AND zone = %s
             ORDER BY created_at DESC
            """,
            (persona_id or settings.persona_id, zone),
        )
        return [_row(r) for r in await cursor.fetchall()]
