"""Store-level eval metrics: extraction precision/recall, retrieval
recall@k and MRR, supersession accuracy. No LLM judge - just deterministic
checks against real pipeline output.

Pure functions, no I/O - scripts/run_eval.py does the DB/graph calls and
feeds the output in here.
"""

from __future__ import annotations

from eval.scenario import PlantSpec
from schemas import FactCandidate, ScoredFact


def keyword_match(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return all(kw.lower() in lowered for kw in keywords)


def extraction_recall(candidates: list[FactCandidate], plants: list[PlantSpec]) -> float:
    """Of the facts this turn was authored to plant, how many actually
    showed up among the raw extracted candidates (pre-adjudication)."""
    if not plants:
        return 1.0
    found = sum(1 for p in plants if any(keyword_match(c.text, p.keywords) for c in candidates))
    return found / len(plants)


def extraction_precision(candidates: list[FactCandidate], plants: list[PlantSpec]) -> float:
    """Of the candidates the extractor produced this turn, how many match
    something the scenario actually planted. On a `plants: []` turn every
    candidate is a false positive, so this is also the filler-leakage check."""
    if not candidates:
        return 1.0
    matched = sum(1 for c in candidates if any(keyword_match(c.text, p.keywords) for p in plants))
    return matched / len(candidates)


def retrieval_recall_at_k(
    retrieved: list[ScoredFact], expected_ids: set[str], *, k: int | None = None
) -> float:
    """Fraction of the expected gold facts present in the top-k retrieved
    list. `retrieved` is already the post-MMR, budget-truncated list
    retrieve.py handed to compose - k defaults to that full list."""
    if not expected_ids:
        return 1.0
    window = retrieved[:k] if k else retrieved
    hit_ids = {str(item.fact.id) for item in window}
    found = sum(1 for eid in expected_ids if eid in hit_ids)
    return found / len(expected_ids)


def reciprocal_rank(retrieved: list[ScoredFact], expected_ids: set[str]) -> float:
    """1/rank of the first expected gold fact found, 0.0 if none appear."""
    for rank, item in enumerate(retrieved, start=1):
        if str(item.fact.id) in expected_ids:
            return 1.0 / rank
    return 0.0


def supersession_correct(final_status: str, current_text: str, new_keywords: list[str]) -> bool:
    """Did contradicting an old fact actually retire it - either the row is
    no longer active, or it's still active but merged to read as the new
    claim (a REFINEMENT verdict can update text in place instead of
    superseding it, and that's fine too)."""
    if final_status != "active":
        return True
    return keyword_match(current_text, new_keywords)
