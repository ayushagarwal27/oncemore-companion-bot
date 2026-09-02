"""Embeddings, cached. Only OpenAI here, per the stack decision."""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from config import settings
from logs import get_logger
from storage import cache

log = get_logger(__name__)

_embeddings: OpenAIEmbeddings | None = None


def _preview(vector: list[float]) -> list[float]:
    return [round(v, 4) for v in vector[:6]]


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key or None,
        )
    return _embeddings


async def embed(text: str) -> list[float]:
    cached = await cache.get_cached_embedding(text)
    if cached is not None:
        log.debug("embedding_cache_hit", text=text, dims=len(cached), preview=_preview(cached))
        return cached

    vector = await get_embeddings().aembed_query(text)
    await cache.set_cached_embedding(text, vector)
    log.debug(
        "embedding_computed",
        text=text,
        model=settings.embedding_model,
        dims=len(vector),
        preview=_preview(vector),
    )
    return vector


async def embed_many(texts: list[str]) -> list[list[float]]:
    """One API call for every uncached text, cache hits filled in place."""
    if not texts:
        return []

    results: list[list[float] | None] = []
    misses: list[tuple[int, str]] = []

    for index, text in enumerate(texts):
        cached = await cache.get_cached_embedding(text)
        results.append(cached)
        if cached is None:
            misses.append((index, text))

    if misses:
        vectors = await get_embeddings().aembed_documents([text for _, text in misses])
        for (index, text), vector in zip(misses, vectors, strict=True):
            results[index] = vector
            await cache.set_cached_embedding(text, vector)

    log.debug(
        "embedding_batch",
        requested=len(texts),
        cache_misses=len(misses),
        texts=[text for _, text in misses],
    )
    return [vector for vector in results if vector is not None]
