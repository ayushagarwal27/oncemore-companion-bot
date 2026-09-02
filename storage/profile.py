"""The cheap supersession path.

A profile field overwrite is the whole contradiction-handling story for the
~15 attributes we can predict in advance. No embedding, no neighbour search,
no adjudicator call, no chance of the old and new value coexisting. The
history table keeps the previous value so nothing is actually lost.
"""

from __future__ import annotations

import json
from uuid import UUID

from schemas import ProfileFieldUpdate, UserProfile
from storage import cache
from storage.pg import get_pool

PROFILE_FIELDS = (
    "preferred_name",
    "pronouns",
    "age_range",
    "location",
    "occupation",
    "employer",
    "relationship_status",
    "partner_name",
    "living_situation",
    "current_focus",
    "ongoing_stressor",
    "communication_style",
)


async def get_profile(user_id: str) -> UserProfile:
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM user_profile WHERE user_id = %s", (user_id,)
        )
        row = await cursor.fetchone()

    if row is None:
        return UserProfile(user_id=user_id)

    data = dict(row)
    for list_field in ("key_people", "topics_to_avoid"):
        value = data.get(list_field)
        data[list_field] = json.loads(value) if isinstance(value, str) else (value or [])
    return UserProfile(**data)


async def apply_updates(
    user_id: str,
    updates: list[ProfileFieldUpdate],
    *,
    source_message_id: UUID | None = None,
) -> list[tuple[str, str | None, str | None]]:
    """Apply field updates and record what changed.

    Returns the (field, old, new) triples that actually changed, so the caller
    can log or surface them. Unchanged writes are skipped rather than logged,
    which keeps the audit trail meaningful.
    """
    if not updates:
        return []

    valid = [u for u in updates if u.field in PROFILE_FIELDS]
    if not valid:
        return []

    current = await get_profile(user_id)
    changed: list[tuple[str, str | None, str | None]] = []

    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO user_profile (user_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (user_id,),
            )
            for update in valid:
                old_value = getattr(current, update.field)
                if old_value == update.value:
                    continue

                await conn.execute(
                    f"UPDATE user_profile SET {update.field} = %s, "
                    "version = version + 1, updated_at = now() WHERE user_id = %s",
                    (update.value, user_id),
                )
                await conn.execute(
                    """
                    INSERT INTO user_profile_history
                        (user_id, field, old_value, new_value, source_message_id, reason)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (user_id, update.field, old_value, update.value,
                     source_message_id, update.reason),
                )
                changed.append((update.field, old_value, update.value))

    if changed:
        await cache.invalidate_user_cache(user_id)
    return changed


async def add_key_person(user_id: str, name: str) -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO user_profile (user_id, key_people) VALUES (%s, %s::jsonb)
            ON CONFLICT (user_id) DO UPDATE
            SET key_people = (
                SELECT jsonb_agg(DISTINCT value)
                  FROM jsonb_array_elements(user_profile.key_people || EXCLUDED.key_people)
            ),
            updated_at = now()
            """,
            (user_id, json.dumps([name])),
        )
    await cache.invalidate_user_cache(user_id)


async def field_history(user_id: str, field: str) -> list[dict]:
    """Every value this field has held. Used in the demo to show supersession."""
    pool = await get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            SELECT field, old_value, new_value, changed_at, reason
              FROM user_profile_history
             WHERE user_id = %s AND field = %s
             ORDER BY changed_at DESC
            """,
            (user_id, field),
        )
        return [dict(row) for row in await cursor.fetchall()]
