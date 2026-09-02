"""Wires load_session -> retrieve -> compose -> respond -> guard -> extract
-> adjudicate -> persist into one compiled graph (`build_graph`), plus a
shorter `build_fast_graph` that stops after guard so a caller can show the
reply before the write path finishes.

guard sits between respond and extract: it checks the drafted reply
against retrieved persona commitments, regenerates once on conflict, and
is the one that actually logs the final reply - so a discarded first draft
never gets persisted.

Running everything synchronously made the interactive CLI feel slow (4-8
LLM calls before any reply shows up), so `build_fast_graph` stops after
guard and `run_write_path` runs extract/adjudicate/persist afterward as a
background task. It's plain async calls rather than a second compiled
graph since there's no real need for one here.

Trade-off: if the user sends a second message before the previous turn's
background write lands, retrieve won't see that turn's new facts yet.
Scripts that need writes visible immediately (test_graph.py, the eval
harness) should use `build_graph` + `run_turn` instead, which stays fully
synchronous.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from config import settings
from graph.nodes.adjudicate import adjudicate
from graph.nodes.compose import compose
from graph.nodes.extract import extract
from graph.nodes.guard import guard
from graph.nodes.load_session import load_session
from graph.nodes.persist import persist
from graph.nodes.respond import respond
from graph.nodes.retrieve import retrieve
from graph.state import ConversationState
from logs import get_logger

log = get_logger(__name__)


def _base_workflow() -> StateGraph:
    workflow = StateGraph(ConversationState)
    workflow.add_node("load_session", load_session)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("compose", compose)
    workflow.add_node("respond", respond)
    workflow.add_node("guard", guard)

    workflow.add_edge(START, "load_session")
    workflow.add_edge("load_session", "retrieve")
    workflow.add_edge("retrieve", "compose")
    workflow.add_edge("compose", "respond")
    workflow.add_edge("respond", "guard")
    return workflow


def build_graph(checkpointer=None) -> CompiledStateGraph:
    """Full pipeline, synchronous end to end - for scripted tests and the
    eval harness, not the interactive CLI."""
    workflow = _base_workflow()
    workflow.add_node("extract", extract)
    workflow.add_node("adjudicate", adjudicate)
    workflow.add_node("persist", persist)

    workflow.add_edge("guard", "extract")
    workflow.add_edge("extract", "adjudicate")
    workflow.add_edge("adjudicate", "persist")
    workflow.add_edge("persist", END)

    return workflow.compile(checkpointer=checkpointer)


def build_fast_graph(checkpointer=None) -> CompiledStateGraph:
    """load_session -> retrieve -> compose -> respond -> guard only.
    Returns as soon as the reply is ready; call run_write_path(result)
    afterward or the turn's memory never gets written."""
    workflow = _base_workflow()
    workflow.add_edge("guard", END)
    return workflow.compile(checkpointer=checkpointer)


async def run_turn(
    graph: CompiledStateGraph,
    *,
    user_id: str,
    thread_id: str,
    user_message: str,
    persona_id: str = settings.persona_id,
) -> dict:
    """One turn in, full state out, including the write path. Build `graph`
    once per process and reuse it."""
    config = {"configurable": {"thread_id": thread_id}}
    return await graph.ainvoke(
        {
            "user_id": user_id,
            "thread_id": thread_id,
            "user_message": user_message,
            "persona_id": persona_id,
        },
        config=config,
    )


async def run_turn_fast(
    fast_graph: CompiledStateGraph,
    *,
    user_id: str,
    thread_id: str,
    user_message: str,
    persona_id: str = settings.persona_id,
) -> dict:
    """One turn in, reply out - no extract/adjudicate/persist. Pass the
    result to run_write_path yourself, usually as a background task."""
    config = {"configurable": {"thread_id": thread_id}}
    return await fast_graph.ainvoke(
        {
            "user_id": user_id,
            "thread_id": thread_id,
            "user_message": user_message,
            "persona_id": persona_id,
        },
        config=config,
    )


async def run_write_path(state: dict) -> None:
    """extract -> adjudicate -> persist, run as plain async calls so a
    caller can await this on its own (e.g. as a background task). Errors
    are logged, not raised - this usually runs unattended."""
    try:
        state = {**state, **(await extract(state))}
        await adjudicate(state)
        await persist(state)
    except Exception:
        log.exception("write_path_failed", user_id=state.get("user_id"))
