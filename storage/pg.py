"""Async Postgres access.

psycopg3 rather than asyncpg so the pool shares a driver with LangGraph's
AsyncPostgresSaver if we ever fall back to it from Redis.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async

from config import settings
from logs import get_logger

log = get_logger(__name__)

_pool: AsyncConnectionPool | None = None

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "db" / "schema.sql"


async def _configure(conn) -> None:
    await register_vector_async(conn)


async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=settings.postgres_dsn,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
            configure=_configure,
            open=False,
        )
        await _pool.open(wait=True)
        log.info("pg_pool_opened", min_size=1, max_size=10)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        log.info("pg_pool_closed")


async def init_schema() -> None:
    """Run the DDL. Idempotent, safe on every boot.

    Uses a bare connection instead of the pool - on a fresh DB the `vector`
    extension doesn't exist yet, and the pool tries to register it on every
    connection it opens, which would fail before we get a chance to create it.
    """
    sql = SCHEMA_PATH.read_text()
    async with await psycopg.AsyncConnection.connect(settings.postgres_dsn) as conn:
        await conn.execute(sql)
        await conn.commit()
    log.info("schema_applied", path=str(SCHEMA_PATH))


async def ensure_user(user_id: str) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO users (id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
            (user_id,),
        )


async def get_persona(user_id: str) -> str | None:
    """None means this user hasn't chosen a persona yet - caller should ask."""
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT persona_id FROM users WHERE id = %s", (user_id,)
        )
        row = await cursor.fetchone()
    return row["persona_id"] if row else None


async def set_persona(user_id: str, persona_id: str) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE users SET persona_id = %s WHERE id = %s", (persona_id, user_id)
        )
