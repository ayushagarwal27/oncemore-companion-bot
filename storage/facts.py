"""The fact ledger.

Two rules hold everywhere here: nothing is deleted (a contradiction closes
`valid_to` and sets `superseded_by`, so history stays queryable), and
retrieval never returns a non-active row (superseded facts are still
readable by direct query, just never surfaced to the prompt).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import UUID

from config import settings
from logs import get_logger
from schemas import (
    DECAY_HALF_LIFE_DAYS,
    FactCandidate,
    FactStatus,
    FactType,
    MemoryFact,
    ScoredFact,
)
from storage import cache
from storage.embeddings import embed
from storage.pg import get_pool

log = get_logger(__name__)

_COLUMNS = """
    id, user_id, subject, predicate, object, text, fact_type, status,
    valid_from, valid_to, created_at, expired_at, superseded_by,
    confidence, importance, access_count, last_accessed_at,
    source_message_id, source_session_id
"""

# Same columns, qualified to the `f` alias — needed wherever memory_facts is
# joined against another CTE that also has an `id` column (see `search`).
_COLUMNS_F = ", ".join(
    f"f.{name.strip()}" for name in _COLUMNS.replace("\n", " ").split(",") if name.strip()
)


def _row_to_fact(row: dict) -> MemoryFact:
    return MemoryFact(**{key: row[key] for key in row if key in MemoryFact.model_fields})


def _strength(fact: MemoryFact, now: datetime) -> float:
    """strength = recency_decay * frequency_boost. Decay is a ranking
    signal, never a delete - identity barely fades (10-year half-life),
    mood fades fast (3 days) so a bad afternoon doesn't dominate all week."""
    half_life = DECAY_HALF_LIFE_DAYS[fact.fact_type]
    reference = fact.last_accessed_at or fact.created_at
    age_days = max((now - reference).total_seconds() / 86400.0, 0.0)
    recency = math.pow(0.5, age_days / half_life)
    frequency = 1.0 + math.log1p(fact.access_count) / 4.0
    return min(recency * frequency, 1.0)


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


async def insert_fact(
    user_id: str,
    candidate: FactCandidate,
    *,
    embedding: list[float] | None = None,
    valid_from: datetime | None = None,
    source_message_id: UUID | None = None,
    source_session_id: UUID | None = None,
) -> MemoryFact:
    vector = embedding if embedding is not None else await embed(candidate.text)
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            INSERT INTO memory_facts (
                user_id, subject, predicate, object, text, fact_type,
                embedding, confidence, importance, valid_from,
                source_message_id, source_session_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, now()), %s, %s)
            RETURNING {_COLUMNS}
            """,
            (
                user_id,
                candidate.subject,
                candidate.predicate,
                candidate.object,
                candidate.text,
                candidate.fact_type.value,
                vector,
                candidate.confidence,
                candidate.importance,
                valid_from,
                source_message_id,
                source_session_id,
            ),
        )
        row = await cursor.fetchone()
    await cache.invalidate_user_cache(user_id)
    fact = _row_to_fact(row)
    log.debug(
        "fact_inserted",
        user_id=user_id,
        fact_id=str(fact.id),
        subject=fact.subject,
        predicate=fact.predicate,
        object=fact.object,
        fact_type=fact.fact_type.value,
        importance=fact.importance,
        confidence=fact.confidence,
        valid_from=fact.valid_from.isoformat(),
    )
    return fact


async def supersede_fact(old_id: UUID, new_id: UUID, *, valid_to: datetime | None = None) -> None:
    """Close the old belief's validity window. The row survives."""
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE memory_facts
               SET status        = 'superseded',
                   valid_to      = COALESCE(%s, now()),
                   expired_at    = now(),
                   superseded_by = %s
             WHERE id = %s
            """,
            (valid_to, new_id, old_id),
        )


async def retract_fact(fact_id: UUID) -> None:
    """User says a fact was never true, as opposed to no longer true."""
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE memory_facts
               SET status = 'retracted', expired_at = now(), valid_to = now()
             WHERE id = %s
            """,
            (fact_id,),
        )


async def merge_into(target_id: UUID, merged_text: str, *, importance: float | None = None) -> None:
    """REFINEMENT path: consolidate rather than accumulate near-duplicates."""
    vector = await embed(merged_text)
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE memory_facts
               SET text       = %s,
                   embedding  = %s,
                   importance = COALESCE(%s, importance)
             WHERE id = %s
            """,
            (merged_text, vector, importance, target_id),
        )
    log.debug("fact_merged", fact_id=str(target_id), merged_text=merged_text, importance=importance)


async def mark_duplicate(fact_id: UUID) -> None:
    """Bump the original instead of storing the copy. Repetition is signal."""
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE memory_facts
               SET access_count = access_count + 1,
                   importance   = LEAST(importance + 0.05, 1.0),
                   last_accessed_at = now()
             WHERE id = %s
            """,
            (fact_id,),
        )


async def touch(fact_ids: list[UUID]) -> None:
    """Recall strengthens a memory. Called after facts reach the prompt."""
    if not fact_ids:
        return
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE memory_facts
               SET access_count = access_count + 1, last_accessed_at = now()
             WHERE id = ANY(%s)
            """,
            (list(fact_ids),),
        )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


async def get_related_active(
    user_id: str,
    candidate: FactCandidate,
    *,
    embedding: list[float] | None = None,
    limit: int | None = None,
) -> list[MemoryFact]:
    """Neighbours the adjudicator reasons over: exact (subject, predicate)
    matches for clean overwrites, unioned with vector neighbours for
    paraphrased contradictions that don't share a predicate string."""
    vector = embedding if embedding is not None else await embed(candidate.text)
    k = limit or settings.adjudicator_neighbours
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            (
                SELECT {_COLUMNS}
                  FROM memory_facts
                 WHERE user_id = %(user_id)s
                   AND status = 'active'
                   AND subject = %(subject)s
                   AND predicate = %(predicate)s
                 LIMIT %(k)s
            )
            UNION
            (
                SELECT {_COLUMNS}
                  FROM memory_facts
                 WHERE user_id = %(user_id)s
                   AND status = 'active'
                 ORDER BY embedding <=> %(vector)s::vector
                 LIMIT %(k)s
            )
            """,
            {
                "user_id": user_id,
                "subject": candidate.subject,
                "predicate": candidate.predicate,
                "vector": vector,
                "k": k,
            },
        )
        rows = await cursor.fetchall()
    return [_row_to_fact(row) for row in rows]


async def search(
    user_id: str,
    query: str,
    *,
    final_k: int | None = None,
    fact_types: list[FactType] | None = None,
) -> list[ScoredFact]:
    """Hybrid retrieval: vector + trigram, fused with RRF, reranked by
    importance and decay (not similarity alone)."""
    vector = await embed(query)
    k_vec = settings.retrieval_vector_k
    k_lex = settings.retrieval_lexical_k
    rrf_k = settings.rrf_k
    types = [t.value for t in fact_types] if fact_types else None

    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            WITH vec AS (
                SELECT id,
                       1 - (embedding <=> %(vector)s::vector) AS similarity,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> %(vector)s::vector) AS rank
                  FROM memory_facts
                 WHERE user_id = %(user_id)s
                   AND status = 'active'
                   AND (%(types)s::text[] IS NULL OR fact_type = ANY(%(types)s))
                 ORDER BY embedding <=> %(vector)s::vector
                 LIMIT %(k_vec)s
            ),
            lex AS (
                SELECT id,
                       ROW_NUMBER() OVER (ORDER BY similarity(text, %(query)s) DESC) AS rank
                  FROM memory_facts
                 WHERE user_id = %(user_id)s
                   AND status = 'active'
                   AND (%(types)s::text[] IS NULL OR fact_type = ANY(%(types)s))
                   AND text %% %(query)s
                 ORDER BY similarity(text, %(query)s) DESC
                 LIMIT %(k_lex)s
            ),
            fused AS (
                SELECT COALESCE(vec.id, lex.id) AS id,
                       COALESCE(vec.similarity, 0.0) AS similarity,
                       lex.rank AS lexical_rank,
                       (COALESCE(1.0 / (%(rrf_k)s + vec.rank), 0.0)
                     + COALESCE(1.0 / (%(rrf_k)s + lex.rank), 0.0))::double precision AS rrf
                  FROM vec
                  FULL OUTER JOIN lex ON vec.id = lex.id
            )
            SELECT {_COLUMNS_F}, fused.similarity, fused.lexical_rank, fused.rrf
              FROM fused
              JOIN memory_facts f ON f.id = fused.id
             ORDER BY fused.rrf DESC
            """,
            {
                "user_id": user_id,
                "vector": vector,
                "query": query,
                "k_vec": k_vec,
                "k_lex": k_lex,
                "rrf_k": rrf_k,
                "types": types,
            },
        )
        rows = await cursor.fetchall()

    if not rows:
        return []

    now = datetime.now(timezone.utc)
    max_rrf = max(row["rrf"] for row in rows) or 1.0

    scored: list[ScoredFact] = []
    for row in rows:
        fact = _row_to_fact(row)
        strength = _strength(fact, now)
        final = (
            settings.w_rrf * (row["rrf"] / max_rrf)
            + settings.w_importance * fact.importance
            + settings.w_strength * strength
        )
        scored.append(
            ScoredFact(
                fact=fact,
                similarity=row["similarity"],
                lexical_rank=row["lexical_rank"],
                rrf=row["rrf"],
                strength=strength,
                final_score=final,
            )
        )

    scored.sort(key=lambda item: item.final_score, reverse=True)
    return scored[: (final_k or settings.retrieval_final_k)]


async def history_for_slot(user_id: str, subject: str, predicate: str) -> list[MemoryFact]:
    """Every belief ever held about one slot, newest first - the active row
    plus the superseded chain behind it."""
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            SELECT {_COLUMNS}
              FROM memory_facts
             WHERE user_id = %s AND subject = %s AND predicate = %s
             ORDER BY created_at DESC
            """,
            (user_id, subject, predicate),
        )
        rows = await cursor.fetchall()
    return [_row_to_fact(row) for row in rows]


async def active_count(user_id: str) -> int:
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT count(*) AS n FROM memory_facts WHERE user_id = %s AND status = 'active'",
            (user_id,),
        )
        row = await cursor.fetchone()
    return int(row["n"])


async def all_active(user_id: str) -> list[MemoryFact]:
    """Full store dump. Used by the oracle baseline in the eval harness."""
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"""
            SELECT {_COLUMNS} FROM memory_facts
             WHERE user_id = %s AND status = %s
             ORDER BY created_at
            """,
            (user_id, FactStatus.ACTIVE.value),
        )
        rows = await cursor.fetchall()
    return [_row_to_fact(row) for row in rows]


async def all_for_user(user_id: str) -> list[MemoryFact]:
    """Every fact ever recorded for this user, any status. For debugging
    and tests that need to find a fact by content rather than by a
    specific (subject, predicate) slot - predicates are LLM-chosen and
    aren't a fixed vocabulary, so the same real-world fact won't always
    land on the same one."""
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            f"SELECT {_COLUMNS} FROM memory_facts WHERE user_id = %s ORDER BY created_at",
            (user_id,),
        )
        rows = await cursor.fetchall()
    return [_row_to_fact(row) for row in rows]
