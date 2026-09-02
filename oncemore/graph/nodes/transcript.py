"""Fold the message window into one rendered transcript, since llm.complete
takes a single system+user pair rather than a message list. Used by both
respond.py and guard.py (guard needs it again for regeneration).
"""

from __future__ import annotations


def render_transcript(history: list[dict[str, str]], current: str) -> str:
    lines = [f"{turn['role']}: {turn['content']}" for turn in history]
    lines.append(f"user: {current}")
    return "\n".join(lines)
