"""Run every scenario in eval/scenarios/ through the real graph (same
build_graph + run_turn as scripts/test_graph.py) and score it against the
gold facts the scenario author planted. No LLM judge - every number is a
deterministic check against real pipeline output.

Three metrics:
  - extraction precision/recall, from the raw ExtractionResult each turn
  - retrieval recall@k and MRR, from retrieve.py's output at probe turns
  - supersession accuracy, from the contradicted fact's final DB status

Gold identity is resolved at run time rather than hand-written as UUIDs in
the YAML: after a turn with `plants`, this diffs facts.all_for_user()
against the pre-turn snapshot and keyword-matches the new row(s) to the
declared plant key - predicates are LLM-chosen, not a fixed vocabulary, so
there's no cheaper way to name "the fact this turn was supposed to create."

    uv sync --group eval        # pulls in pyyaml for scenario loading
    uv run python scripts/run_eval.py
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from statistics import mean
from uuid import UUID

from langgraph.checkpoint.memory import InMemorySaver

from eval import metrics
from eval.scenario import Scenario, load_all
from graph.build import build_graph, run_turn
from logs import configure_logging, get_logger
from schemas import MemoryFact
from storage import eval_runs, facts
from storage.cache import close_redis
from storage.pg import close_pool, ensure_user, init_schema

log = get_logger(__name__)


async def _snapshot(user_id: str) -> dict[UUID, MemoryFact]:
    return {f.id: f for f in await facts.all_for_user(user_id)}


async def _run_scenario(scenario: Scenario) -> dict[str, list[float]]:
    user_id = f"eval_{scenario.name}"
    thread_id = f"eval-{scenario.name}"
    await ensure_user(user_id)

    run_id = await eval_runs.start_run(scenario.name)
    scores: dict[str, list[float]] = defaultdict(list)

    key_to_fact_id: dict[str, UUID] = {}
    pending_contradictions: list[tuple[int, list[str], list[str]]] = []
    known = await _snapshot(user_id)

    async with InMemorySaver() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        for turn_idx, turn in enumerate(scenario.turns):
            result = await run_turn(
                graph, user_id=user_id, thread_id=thread_id, user_message=turn.text
            )

            if turn.plants is not None:
                extraction = result.get("extraction")
                candidates = extraction.candidates if extraction else []

                recall = metrics.extraction_recall(candidates, turn.plants)
                precision = metrics.extraction_precision(candidates, turn.plants)
                scores["extraction_recall"].append(recall)
                scores["extraction_precision"].append(precision)
                await eval_runs.record_result(
                    run_id, f"t{turn_idx}", "extraction_recall",
                    passed=recall == 1.0, score=recall,
                )
                await eval_runs.record_result(
                    run_id, f"t{turn_idx}", "extraction_precision",
                    passed=precision == 1.0, score=precision,
                )

            if turn.plants:
                current = await _snapshot(user_id)
                new_rows = [f for fid, f in current.items() if fid not in known]
                for plant in turn.plants:
                    match = next(
                        (f for f in new_rows if metrics.keyword_match(f.text, plant.keywords)),
                        None,
                    )
                    if match:
                        key_to_fact_id[plant.key] = match.id
                    else:
                        log.warning("plant_unresolved", scenario=scenario.name,
                                    turn=turn_idx, key=plant.key)
                known = current

            if turn.probe is not None:
                expected_ids = {
                    str(key_to_fact_id[k]) for k in turn.probe.expects if k in key_to_fact_id
                }
                retrieved = result.get("retrieved_facts", [])
                recall_k = metrics.retrieval_recall_at_k(retrieved, expected_ids, k=turn.probe.k)
                rr = metrics.reciprocal_rank(retrieved, expected_ids)
                scores["retrieval_recall_at_k"].append(recall_k)
                scores["retrieval_mrr"].append(rr)
                await eval_runs.record_result(
                    run_id, f"t{turn_idx}", "retrieval_recall_at_k",
                    passed=recall_k == 1.0, score=recall_k,
                )
                await eval_runs.record_result(
                    run_id, f"t{turn_idx}", "retrieval_mrr",
                    passed=rr > 0.0, score=rr,
                )

            if turn.contradicts:
                new_keywords = turn.plants[0].keywords if turn.plants else []
                pending_contradictions.append((turn_idx, turn.contradicts, new_keywords))

    final = await _snapshot(user_id)
    for turn_idx, old_keys, new_keywords in pending_contradictions:
        for old_key in old_keys:
            fact_id = key_to_fact_id.get(old_key)
            if fact_id is None or fact_id not in final:
                log.warning("contradiction_target_unresolved", scenario=scenario.name,
                            turn=turn_idx, key=old_key)
                continue
            old_fact = final[fact_id]
            correct = metrics.supersession_correct(
                old_fact.status.value, old_fact.text, new_keywords
            )
            scores["supersession_accuracy"].append(1.0 if correct else 0.0)
            await eval_runs.record_result(
                run_id, f"{old_key}->t{turn_idx}", "supersession_accuracy",
                passed=correct, score=1.0 if correct else 0.0,
                detail={"final_status": old_fact.status.value, "final_text": old_fact.text},
            )

    summary = {metric: {"mean": mean(vals), "n": len(vals)} for metric, vals in scores.items()}
    await eval_runs.finish_run(run_id, summary)
    return scores


def _print_table(title: str, scores: dict[str, list[float]]) -> None:
    print(f"\n{title}")
    print(f"  {'metric':<24} {'n':>4} {'mean':>8}")
    for metric_name, vals in sorted(scores.items()):
        print(f"  {metric_name:<24} {len(vals):>4} {mean(vals):>8.2f}")


async def main() -> None:
    configure_logging()
    await init_schema()

    scenarios = load_all()
    if not scenarios:
        print("no scenarios found under eval/scenarios/")
        return

    overall: dict[str, list[float]] = defaultdict(list)
    for scenario in scenarios:
        scores = await _run_scenario(scenario)
        _print_table(f"{scenario.name} ({scenario.description.strip()})", scores)
        for metric_name, vals in scores.items():
            overall[metric_name].extend(vals)

    _print_table(f"overall ({len(scenarios)} scenario(s))", overall)

    await close_redis()
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
