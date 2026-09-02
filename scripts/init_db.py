"""Create the schema. Idempotent.

    uv run python scripts/init_db.py
"""

from __future__ import annotations

import asyncio

from storage.pg import close_pool, init_schema


async def main() -> None:
    await init_schema()
    print("schema applied")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
