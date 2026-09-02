"""Generate an adaptive-zone prompt candidate from a user's real
conversation history, print it, and prompt for a manual yes/no before
promoting it live.

    uv run python scripts/optimize_prompt.py [user_id]
"""

from __future__ import annotations

import asyncio
import sys

import prompt_optimizer
from storage.pg import close_pool

DEFAULT_USER = "demo_user"


async def main() -> None:
    user_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_USER

    current = await prompt_optimizer.history()
    print(f"existing adaptive-zone versions for this persona: {len(current)}")

    candidate = await prompt_optimizer.generate_candidate(user_id)
    print(f"\ngenerated candidate {candidate.id} (parent={candidate.parent_id}):\n")
    print(candidate.content)

    answer = input("\npromote this candidate live? [y/N] ").strip().lower()
    if answer == "y":
        await prompt_optimizer.promote(candidate.id)
        print("promoted.")
    else:
        print("left unpromoted - compose.py will keep using whatever was active before.")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
