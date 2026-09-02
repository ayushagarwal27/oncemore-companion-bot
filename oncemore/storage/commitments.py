"""What the companion has said about itself.

Same bi-temporal shape as the fact ledger, but keyed on persona rather
than user, since the companion is one character across everyone it talks
to. A commitment gets superseded, never rewritten silently, so if it
changes its mind you can see when and why.
"""

from __future__ import annotations

from uuid import UUID

from config import settings
from schemas import CommitmentCandidate, PersonaCommitment
from storage.embeddings import embed
from storage.pg import get_pool

_COLUMNS = """
    id, persona_id, topic, text, status, valid_from, valid_to,
    created_at, superseded_by, confidence, source_message_id
"""


def _row(row: dict) -> PersonaCommitment:
    return PersonaCommitment(
        **{k: row[k] for k in row if k in PersonaCommitment.model_fields}
    )


async def record(
    candidate: CommitmentCandidate,
    *,
    persona_id: str | None = None,
    source_message_id: UUID | None = None,
) -> PersonaCommitment:
    vector = await embed(candidate.text)
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            INSERT INTO persona_commitments
                (persona_id, topic, text, embedding, confidence, source_message_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING {_COLUMNS}
            """,
            (
                persona_id or settings.persona_id,
                candidate.topic,
                candidate.text,
                vector,
                candidate.confidence,
                source_message_id,
            ),
        )
        return _row(await cursor.fetchone())


async def supersede(old_id: UUID, new_id: UUID) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE persona_commitments
               SET status = 'superseded', valid_to = now(), superseded_by = %s
             WHERE id = %s
            """,
            (new_id, old_id),
        )


async def by_topic(topic: str, *, persona_id: str | None = None) -> PersonaCommitment | None:
    """Exact-topic lookup. Catches the clean case before spending a vector search."""
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            SELECT {_COLUMNS} FROM persona_commitments
             WHERE persona_id = %s AND topic = %s AND status = 'active'
             ORDER BY created_at DESC LIMIT 1
            """,
            (persona_id or settings.persona_id, topic),
        )
        row = await cursor.fetchone()
    return _row(row) if row else None


async def search(query: str, *, k: int = 5, persona_id: str | None = None) -> list[PersonaCommitment]:
    """Semantic recall over the persona's own past statements, so it sees
    what it's already claimed before claiming something new."""
    vector = await embed(query)
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            SELECT {_COLUMNS} FROM persona_commitments
             WHERE persona_id = %s AND status = 'active'
             ORDER BY embedding <=> %s::vector
             LIMIT %s
            """,
            (persona_id or settings.persona_id, vector, k),
        )
        return [_row(row) for row in await cursor.fetchall()]


async def all_active(persona_id: str | None = None) -> list[PersonaCommitment]:
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            SELECT {_COLUMNS} FROM persona_commitments
             WHERE persona_id = %s AND status = 'active'
             ORDER BY created_at
            """,
            (persona_id or settings.persona_id,),
        )
        return [_row(row) for row in await cursor.fetchall()]
