"""Normalized stream events.

Both provider adapters translate their SDK-specific stream into these types, so
the engine and any UI (console now, TUI later) only ever see this vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class TurnStarted:
    speaker: str
    turn_no: int


@dataclass
class ThinkingDelta:
    """A chunk of the model's reasoning. Only emitted by providers that expose it."""

    speaker: str
    text: str


@dataclass
class TextDelta:
    """A chunk of the model's visible reply."""

    speaker: str
    text: str


@dataclass
class TurnComplete:
    speaker: str
    usage: Usage | None = None


@dataclass
class Injected:
    """A moderator message dropped into the conversation between turns.

    It enters every persona's view as a `user` message, so the next speaker
    responds to it directly.
    """

    speaker: str
    text: str


@dataclass
class RetryNotice:
    """A turn hit a retryable error (e.g. a rate limit) and will be retried.

    Purely informational: the engine keeps the same turn and tries again after
    `delay` seconds. `attempt` counts from 1.
    """

    speaker: str
    attempt: int
    delay: float
    reason: str


@dataclass
class TurnError:
    """A turn failed and was given up on (non-retryable, or retries exhausted).

    The loop moves on to the next speaker instead of crashing.
    """

    speaker: str
    message: str


StreamEvent = Union[
    TurnStarted,
    ThinkingDelta,
    TextDelta,
    TurnComplete,
    Injected,
    RetryNotice,
    TurnError,
]
