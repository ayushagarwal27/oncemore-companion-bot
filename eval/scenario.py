"""Scenario authoring format for the eval harness.

A scenario is an authored conversation where you know upfront what each
turn should do to memory: which facts it plants, whether a probe turn
should recall them, and which later turn contradicts an earlier plant.
scripts/run_eval.py resolves each `key` to the real row created for it as
the scenario plays out, rather than guessing IDs ahead of time.

`plants: null` (the default) means this turn is unscored filler.
`plants: []` means the turn should extract nothing, so any candidate
counts against precision - use it to check that neutral turns don't leak
facts.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"


class PlantSpec(BaseModel):
    key: str
    keywords: list[str]


class ProbeSpec(BaseModel):
    expects: list[str] = Field(default_factory=list)
    k: int | None = None


class Turn(BaseModel):
    text: str
    plants: list[PlantSpec] | None = None
    probe: ProbeSpec | None = None
    contradicts: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    name: str
    description: str = ""
    turns: list[Turn]


def load_scenario(path: Path) -> Scenario:
    data = yaml.safe_load(path.read_text())
    return Scenario(**data)


def load_all(directory: Path = SCENARIOS_DIR) -> list[Scenario]:
    return [load_scenario(p) for p in sorted(directory.glob("*.yaml"))]
