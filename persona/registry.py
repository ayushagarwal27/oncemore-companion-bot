"""Maps a persona_id to its frozen canon.

To add a persona: write a module shaped like persona/canon.py or
persona/esha.py, then register it below.
"""

from __future__ import annotations

from persona import canon, esha

DEFAULT_PERSONA_ID = "mira"


class PersonaMeta:
    def __init__(self, persona_id: str, name: str, tagline: str, persona_canon: str) -> None:
        self.persona_id = persona_id
        self.name = name
        self.tagline = tagline
        self.canon = persona_canon


PERSONAS: dict[str, PersonaMeta] = {
    "mira": PersonaMeta(
        "mira", canon.NAME, "direct, warm, a little stubborn", canon.PERSONA_CANON
    ),
    "esha": PersonaMeta(
        "esha", esha.NAME, "quick, funny, teases first and means it after", esha.PERSONA_CANON
    ),
}


def get_canon(persona_id: str) -> str:
    meta = PERSONAS.get(persona_id)
    if meta is None:
        raise ValueError(f"unknown persona_id {persona_id!r}, expected one of {list(PERSONAS)}")
    return meta.canon
