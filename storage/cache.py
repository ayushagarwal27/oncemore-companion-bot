"""Redis: the hot, disposable half of the stack.

Nothing in here is a source of truth. If Redis is empty the system is slower
and more expensive, never wrong. That property is what justifies running two
datastores at all.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as redis

from config import settings

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(settings.redis_url, decode_responses=False)
    return _client


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


# --- embedding cache -------------------------------------------------------
# Embeddings are pure functions of (model, text). Caching them removes most
# repeat API calls during eval runs, where the same probe text is embedded
# dozens of times across scenarios.


async def get_cached_embedding(text: str) -> list[float] | None:
    key = f"emb:{settings.embedding_model}:{_digest(text)}"
    raw = await get_redis().get(key)
    return json.loads(raw) if raw else None


async def set_cached_embedding(text: str, vector: list[float]) -> None:
    key = f"emb:{settings.embedding_model}:{_digest(text)}"
    await get_redis().set(key, json.dumps(vector), ex=settings.embedding_cache_ttl_s)


# --- assembled memory block ------------------------------------------------
# Short TTL. Consecutive turns on the same topic reuse the same retrieval,
# which keeps the cached prompt prefix stable and cheap.


async def get_cached_block(user_id: str, query: str) -> str | None:
    key = f"block:{user_id}:{_digest(query)}"
    raw = await get_redis().get(key)
    return raw.decode("utf-8") if raw else None


async def set_cached_block(user_id: str, query: str, block: str) -> None:
    key = f"block:{user_id}:{_digest(query)}"
    await get_redis().set(key, block.encode("utf-8"), ex=settings.memory_block_cache_ttl_s)


async def invalidate_user_cache(user_id: str) -> None:
    """Called after any write that changes what retrieval would return."""
    client = get_redis()
    async for key in client.scan_iter(match=f"block:{user_id}:*", count=200):
        await client.delete(key)


# --- session scratch -------------------------------------------------------


async def set_session_value(session_id: str, field: str, value: Any) -> None:
    await get_redis().hset(f"session:{session_id}", field, json.dumps(value))


async def get_session_value(session_id: str, field: str) -> Any | None:
    raw = await get_redis().hget(f"session:{session_id}", field)
    return json.loads(raw) if raw else None
