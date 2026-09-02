"""Hybrid retrieval across facts/commitments/episodes, MMR-deduped and
token-budget-truncated before compose sees it.

MMR only runs on facts - commitments and episodes are already capped to a
handful by their own search functions, so there's not much to dedupe there.
"""

from __future__ import annotations

import tiktoken

from config import settings
from graph.state import ConversationState
from logs import get_logger
from schemas import ScoredFact
from storage import commitments, episodes, facts
from storage.embeddings import embed_many

log = get_logger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _mmr_select(
    scored: list[ScoredFact], *, lambda_mult: float, budget_tokens: int
) -> list[ScoredFact]:
    """Greedy MMR: pick whatever maximizes relevance minus similarity to
    what's already picked, until the next pick would blow the token budget."""
    if not scored:
        return []

    texts = [item.fact.text for item in scored]
    vectors = await embed_many(texts)
    relevance = [item.final_score for item in scored]

    picked: list[int] = []
    remaining = list(range(len(scored)))
    used_tokens = 0

    while remaining:
        best_idx, best_score = None, float("-inf")
        for i in remaining:
            similarity_to_picked = max((_cosine(vectors[i], vectors[j]) for j in picked), default=0.0)
            mmr_score = lambda_mult * relevance[i] - (1 - lambda_mult) * similarity_to_picked
            if mmr_score > best_score:
                best_idx, best_score = i, mmr_score

        cost = _token_count(texts[best_idx])
        if picked and used_tokens + cost > budget_tokens:
            break
        picked.append(best_idx)
        used_tokens += cost
        remaining.remove(best_idx)

    return [scored[i] for i in picked]


async def retrieve(state: ConversationState) -> dict:
    user_id = state["user_id"]
    query = state["user_message"]

    persona_id = state.get("persona_id", settings.persona_id)

    scored_facts = await facts.search(user_id, query)
    matched_commitments = await commitments.search(query, k=5, persona_id=persona_id)
    matched_episodes = await episodes.search(user_id, query, k=3)
    anchors = await episodes.voice_anchors(user_id, limit=3)

    kept_facts = await _mmr_select(
        scored_facts,
        lambda_mult=settings.mmr_lambda,
        budget_tokens=settings.memory_token_budget,
    )
    if kept_facts:
        await facts.touch([item.fact.id for item in kept_facts])

    log.debug(
        "retrieved_for_turn",
        user_id=user_id,
        facts=[
            {"text": item.fact.text, "final_score": round(item.final_score, 3)}
            for item in kept_facts
        ],
        commitments=[c.text for c in matched_commitments],
        episodes=[e.text for e in matched_episodes],
        voice_anchors=[a.text for a in anchors],
    )

    return {
        "retrieved_facts": kept_facts,
        "retrieved_commitments": matched_commitments,
        "retrieved_episodes": matched_episodes,
        "voice_anchors": anchors,
    }
