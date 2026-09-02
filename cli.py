"""Interactive REPL - talk to the companion from a terminal.

    uv run companion [user_id] [thread_id]

Defaults to a demo user/thread, so it remembers you across runs with no
setup.

Replies come from the fast graph, then extract/adjudicate/persist run in
the background while you type the next message. Each background write is
awaited before the *next* turn starts (not before the reply is printed) -
otherwise a fast typist could send the next message before the previous
turn's write even started, and it would race that turn's retrieval.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from config import settings
from graph.build import build_fast_graph, run_turn_fast, run_write_path
from logs import configure_logging
from persona.registry import DEFAULT_PERSONA_ID, PERSONAS
from storage.cache import close_redis
from storage.pg import close_pool, ensure_user, get_persona, init_schema, set_persona

DEFAULT_USER = "demo_user"
DEFAULT_THREAD = "cli"


async def _choose_persona(user_id: str) -> str:
    """Ask once per user; after that we just read persona_id from the users table."""
    existing = await get_persona(user_id)
    if existing:
        return existing

    print("pick a companion:")
    ids = list(PERSONAS)
    for i, persona_id in enumerate(ids, start=1):
        meta = PERSONAS[persona_id]
        print(f"  {i}. {meta.name} - {meta.tagline}")

    choice = None
    while choice is None:
        raw = input(f"> [{ids.index(DEFAULT_PERSONA_ID) + 1}] ").strip()
        if not raw:
            choice = DEFAULT_PERSONA_ID
            break
        if raw in PERSONAS:
            choice = raw
            break
        if raw.isdigit() and 1 <= int(raw) <= len(ids):
            choice = ids[int(raw) - 1]
            break
        print(f"didn't catch that - enter a number 1-{len(ids)} or a name")

    await set_persona(user_id, choice)
    print(f"talking to {PERSONAS[choice].name} from now on.\n")
    return choice


async def main() -> None:
    debug = os.environ.get("COMPANION_DEBUG", "").lower() in {"1", "true", "yes"}
    configure_logging(level=logging.DEBUG if debug else logging.INFO)

    user_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_USER
    thread_id = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_THREAD

    await init_schema()
    await ensure_user(user_id)
    persona_id = await _choose_persona(user_id)
    persona_name = PERSONAS[persona_id].name

    print(
        f"talking to {persona_name} as {user_id!r} on thread {thread_id!r}. "
        "ctrl-d or 'quit' to exit.\n"
    )

    pending_write: asyncio.Task | None = None

    async with AsyncRedisSaver.from_conn_string(settings.redis_url) as checkpointer:
        await checkpointer.asetup()
        fast_graph = build_fast_graph(checkpointer=checkpointer)

        while True:
            try:
                user_message = input("you: ").strip()
            except EOFError:
                print()
                break

            if not user_message:
                continue
            if user_message.lower() in {"quit", "exit"}:
                break

            if pending_write is not None:
                await pending_write
                pending_write = None

            result = await run_turn_fast(
                fast_graph,
                user_id=user_id,
                thread_id=thread_id,
                user_message=user_message,
                persona_id=persona_id,
            )
            print(f"{persona_name.lower()}: {result['response']}\n")

            pending_write = asyncio.create_task(run_write_path(result))

        if pending_write is not None:
            print("(finishing up memory writes...)")
            await pending_write

    await close_redis()
    await close_pool()


def app() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    app()
