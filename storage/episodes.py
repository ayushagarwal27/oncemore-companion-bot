"""Episodic memory - two kinds sharing one table.

`relational` is a moment worth remembering as an experience rather than a
fact triple ("that sounds like what you were going through in March").
`voice` is a pinned example of the companion handling something well,
carried in the prompt as a few-shot anchor to keep the tone consistent.

Relational episodes are capped - uncapped, they'd swamp retrieval, since an
episode embeds more text than a fact and matches more queries.
"""

from __future__ import annotations

from uuid import UUID

from schemas import Episode, EpisodeCandidate
from storage.embeddings import embed
from storage.pg import get_pool

RELATIONAL_CAP = 20

_COLUMNS = """
    id, user_id, kind, title, observation, companion_action, outcome, text,
    salience, pinned, access_count, occurred_at, created_at
"""


def _row(row: dict) -> Episode:
    return Episode(**{k: row[k] for k in row if k in Episode.model_fields})


def _render(candidate: EpisodeCandidate) -> str:
    return (
        f"{candidate.title}. {candidate.observation} "
        f"I responded by {candidate.companion_action} {candidate.outcome}"
    )


async def record(
    user_id: str,
    candidate: EpisodeCandidate,
    *,
    kind: str = "relational",
    session_id: UUID | None = None,
) -> Episode:
    text = _render(candidate)
    vector = await embed(text)
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            INSERT INTO episodes
                (user_id, kind, title, observation, companion_action, outcome,
                 text, embedding, salience, session_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING {_COLUMNS}
            """,
            (
                user_id, kind, candidate.title, candidate.observation,
                candidate.companion_action, candidate.outcome, text, vector,
                candidate.salience, session_id,
            ),
        )
        episode = _row(await cursor.fetchone())

    if kind == "relational":
        await _enforce_cap(user_id)
    return episode


async def _enforce_cap(user_id: str) -> None:
    """Drop the least salient unpinned episodes past the cap. This is the
    one place the system genuinely forgets - facts never are, but episodes
    are a bounded working set on purpose."""
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            DELETE FROM episodes
             WHERE id IN (
                SELECT id FROM episodes
                 WHERE user_id = %s AND kind = 'relational' AND NOT pinned
                 ORDER BY salience DESC, occurred_at DESC
                OFFSET %s
             )
            """,
            (user_id, RELATIONAL_CAP),
        )


async def search(user_id: str, query: str, *, k: int = 3) -> list[Episode]:
    vector = await embed(query)
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            SELECT {_COLUMNS} FROM episodes
             WHERE user_id = %s AND kind = 'relational'
             ORDER BY embedding <=> %s::vector
             LIMIT %s
            """,
            (user_id, vector, k),
        )
        return [_row(row) for row in await cursor.fetchall()]


async def voice_anchors(user_id: str, *, limit: int = 3) -> list[Episode]:
    """Pinned first, then most salient. Stable ordering keeps the cache warm."""
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            SELECT {_COLUMNS} FROM episodes
             WHERE user_id = %s AND kind = 'voice'
             ORDER BY pinned DESC, salience DESC, created_at
             LIMIT %s
            """,
            (user_id, limit),
        )
        return [_row(row) for row in await cursor.fetchall()]
