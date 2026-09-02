"""Every LLM call goes through here.

Extraction/adjudication calls use Structured Outputs (strict JSON schema),
not JSON mode, so we get back a validated pydantic object or an exception,
never almost-valid JSON. Retries are handled by ChatOpenAI's own
`max_retries` for transient errors only - a refusal or schema mismatch
won't fix itself on retry, so those raise straight away.
"""

from __future__ import annotations

from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel

from config import settings
from logs import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_TRANSIENT = (RateLimitError, APIConnectionError, APITimeoutError)

_models: dict[str, ChatOpenAI] = {}


def _get_chat_model(model_name: str) -> ChatOpenAI:
    if model_name not in _models:
        _models[model_name] = ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key or None,
            max_retries=3,  # 3 retries = 4 attempts, same ceiling as before
            timeout=60,
        )
    return _models[model_name]


class StructuredOutputError(RuntimeError):
    """Raised when the model refuses or returns nothing parseable. Not
    retried - a refusal isn't transient, so retrying just spends money."""


async def parse(
    *,
    schema: type[T],
    system: str,
    user: str,
    model: str | None = None,
) -> T:
    """Call the model and get back a validated instance of `schema`."""
    target_model = model or settings.extraction_model
    structured = _get_chat_model(target_model).with_structured_output(
        schema, method="json_schema", strict=True
    )

    try:
        return await structured.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
    except _TRANSIENT:
        raise  # already retried inside ChatOpenAI, nothing left to do
    except Exception as exc:
        log.error(
            "structured_parse_failed",
            model=target_model,
            schema=schema.__name__,
            error=str(exc),
        )
        raise StructuredOutputError(
            f"{target_model} returned no parsed output for {schema.__name__}: {exc}"
        ) from exc


async def complete(
    *,
    system: str,
    user: str,
    model: str | None = None,
) -> str:
    """Free-text generation. Used for the companion's actual replies."""
    target_model = model or settings.chat_model
    response = await _get_chat_model(target_model).ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return response.content


def build_prompt(
    *,
    persona_canon: str,
    voice_anchors: str,
    profile_block: str,
    memory_block: str,
    commitments_block: str,
    adaptive_notes: str = "",
) -> str:
    """Assemble the system prompt in strict cache-prefix order.

    Everything above the CACHE BOUNDARY marker is stable for the whole
    session; everything below varies per turn. Don't reorder these sections
    - moving one line above the boundary kills the cache for the whole
    conversation.
    """
    return "\n".join(
        [
            persona_canon,
            "",
            "## How you sound",
            voice_anchors,
            "",
            "## Interaction notes (may be refined over time; never overrides the traits above)",
            adaptive_notes or "- (none yet)",
            "",
            "## What you know about them",
            profile_block,
            "",
            "<!-- CACHE BOUNDARY: nothing below here is stable -->",
            "",
            "## Things you have said about yourself",
            commitments_block or "- (nothing recorded yet)",
            "",
            "## Relevant memories",
            memory_block or "- (nothing relevant surfaced)",
        ]
    )
