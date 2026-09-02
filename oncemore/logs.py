"""structlog setup. Import `get_logger` everywhere instead of stdlib logging.

The write path runs as a background task, so when something fails there's
no request context to fall back on - call sites bind user_id/fact_id/model
so you can actually tell which turn caused it.
"""

from __future__ import annotations

import logging
import sys

import structlog

_NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "httpx2",
    "httpcore2",
    "openai",
    "langchain",
    "langgraph",
    "redis",
    "redisvl",
    "hpack",
]


def configure_logging(*, json: bool = False, level: int = logging.INFO) -> None:
    """Call once at process startup.

    `level` only applies to our own loggers - the HTTP/SDK libraries stay
    at WARNING, otherwise DEBUG here also turns on httpx/httpcore's raw
    request logging and buries what you actually wanted to see.
    """
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.processors.JSONRenderer()
            if json
            else structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
