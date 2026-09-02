"""Session and turn log - provenance for every memory, plus the raw
transcript other tools replay. Feedback lives on the message row itself
since it's always about exactly one companion turn.
"""

from __future__ import annotations

from uuid import UUID

from schemas import StoredMessage
from storage.pg import ensure_user, get_pool


async def start_session(user_id: str, thread_id: str) -> UUID:
    await ensure_user(user_id)
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "INSERT INTO sessions (user_id, thread_id) VALUES (%s, %s) RETURNING id",
            (user_id, thread_id),
        )
        row = await cursor.fetchone()
    return row["id"]


async def resume_or_start(user_id: str, thread_id: str) -> UUID:
    """Reattach to an open session for this thread if one exists, so a
    process restart doesn't reset turn indices and split the transcript
    in two."""
    await ensure_user(user_id)
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT id FROM sessions
             WHERE user_id = %s AND thread_id = %s AND ended_at IS NULL
             ORDER BY started_at DESC LIMIT 1
            """,
            (user_id, thread_id),
        )
        row = await cursor.fetchone()
    if row:
        return row["id"]
    return await start_session(user_id, thread_id)


async def end_session(session_id: UUID, summary: str | None = None) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE sessions SET ended_at = now(), summary = COALESCE(%s, summary) WHERE id = %s",
            (summary, session_id),
        )


async def append(
    session_id: UUID,
    user_id: str,
    role: str,
    content: str,
    *,
    trace_id: str | None = None,
) -> StoredMessage:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.transaction():
            cursor = await conn.execute(
                """
                INSERT INTO messages (session_id, user_id, turn_index, role, content, trace_id)
                SELECT %s, %s, COALESCE(MAX(turn_index) + 1, 0), %s, %s, %s
                  FROM messages WHERE session_id = %s
                RETURNING id, session_id, user_id, turn_index, role, content,
                          created_at, trace_id, feedback, feedback_reason
                """,
                (session_id, user_id, role, content, trace_id, session_id),
            )
            row = await cursor.fetchone()
            await conn.execute(
                "UPDATE sessions SET turn_count = turn_count + 1 WHERE id = %s",
                (session_id,),
            )
    return StoredMessage(**row)


async def recent(session_id: UUID, limit: int = 20) -> list[StoredMessage]:
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT id, session_id, user_id, turn_index, role, content,
                   created_at, trace_id, feedback, feedback_reason
              FROM messages WHERE session_id = %s
             ORDER BY turn_index DESC LIMIT %s
            """,
            (session_id, limit),
        )
        rows = await cursor.fetchall()
    return [StoredMessage(**row) for row in reversed(rows)]


async def full_transcript(user_id: str) -> list[StoredMessage]:
    """Every message across every session, oldest first."""
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT m.id, m.session_id, m.user_id, m.turn_index, m.role, m.content,
                   m.created_at, m.trace_id, m.feedback, m.feedback_reason
              FROM messages m JOIN sessions s ON s.id = m.session_id
             WHERE m.user_id = %s
             ORDER BY s.started_at, m.turn_index
            """,
            (user_id,),
        )
        return [StoredMessage(**row) for row in await cursor.fetchall()]


async def record_feedback(message_id: UUID, score: int, reason: str | None = None) -> None:
    """Per-turn thumbs up/down, feeding the adaptive prompt optimizer."""
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE messages SET feedback = %s, feedback_reason = %s WHERE id = %s",
            (score, reason, message_id),
        )
