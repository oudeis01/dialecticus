"""The conversation loop.

Holds the shared transcript and decides whose turn it is. For each turn it builds
that persona's view of the conversation (its own lines are `assistant`, everyone
else's are `user`), streams the turn through the persona's adapter, and re-emits
every event so the caller can render it live.

The loop is observable *and* steerable: a UI can pause, single-step, inject a
moderator message, or stop the conversation. All of that is applied at turn
boundaries so it never tears a streaming reply in half.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

from .events import (
    Injected,
    RetryNotice,
    StreamEvent,
    TextDelta,
    TurnError,
    TurnStarted,
)
from .persona import Persona
from .providers.base import Msg, ProviderAdapter

# When a rate limit gives no Retry-After hint, wait this long before retrying.
DEFAULT_RETRY_DELAY = 20.0


def _dig_retry_seconds(obj: object) -> float | None:
    """Recursively search an error body for a retry_after_seconds hint.

    OpenRouter wraps upstream 429s and tucks the delay into nested metadata
    (e.g. error.metadata.retry_after_seconds), so a flat lookup would miss it.
    """
    if isinstance(obj, dict):
        for key in ("retry_after_seconds", "retry_after_seconds_raw"):
            if key in obj:
                try:
                    return float(obj[key])
                except (TypeError, ValueError):
                    pass
        for value in obj.values():
            found = _dig_retry_seconds(value)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _dig_retry_seconds(value)
            if found is not None:
                return found
    return None


@dataclass
class Turn:
    speaker: str
    text: str


class Engine:
    def __init__(
        self,
        personas: list[Persona],
        adapters: dict[str, ProviderAdapter],
        kickoff: str,
        max_turns: int = 6,
        show_thinking: bool = True,
        moderator_name: str = "Moderator",
        context_lengths: dict[str, int] | None = None,
        budget_fraction: float = 0.75,
        max_retries: int = 5,
        max_retry_delay: float = 60.0,
        initial_transcript: list[Turn] | None = None,
        start_turn: int = 0,
    ) -> None:
        if not personas:
            raise ValueError("need at least one persona")
        self.personas = personas
        self.adapters = adapters
        self.kickoff = kickoff
        self.max_turns = max_turns
        self.show_thinking = show_thinking
        self.moderator_name = moderator_name
        # Seed from a prior session when resuming; start_turn keeps the
        # round-robin (personas[turn_no % n]) and the turn budget aligned so the
        # correct persona speaks next and max_turns counts additional turns.
        self.transcript: list[Turn] = list(initial_transcript or [])
        self.start_turn = start_turn

        # Per-persona context budget. Without resolved limits we fall back to a
        # conservative default so long runs still trim instead of overflowing.
        self.context_lengths = context_lengths or {}
        self.budget_fraction = budget_fraction

        # Rate-limit retry policy. Honour a provider's Retry-After when given,
        # otherwise back off by DEFAULT_RETRY_DELAY, capped at max_retry_delay.
        self.max_retries = max_retries
        self.max_retry_delay = max_retry_delay

        # Intervention state, all applied at turn boundaries.
        self.step_mode = False
        self._proceed = asyncio.Event()
        self._proceed.set()  # start running; pause()/step mode clear it
        self._injections: list[str] = []
        self._stopped = False

    # --- controls a UI calls between turns -------------------------------

    def pause(self) -> None:
        """Hold at the next turn boundary (continuous mode)."""
        self._proceed.clear()

    def resume(self) -> None:
        """Resume continuous running. No-op in step mode."""
        if not self.step_mode:
            self._proceed.set()

    def step(self) -> None:
        """Allow exactly one more turn to run."""
        self._proceed.set()

    def set_step_mode(self, on: bool) -> None:
        self.step_mode = on
        if on:
            self._proceed.clear()  # wait for an explicit step() at the next boundary
        else:
            self._proceed.set()  # fall back to continuous flow

    def toggle_step_mode(self) -> None:
        self.set_step_mode(not self.step_mode)

    def is_paused(self) -> bool:
        return not self.step_mode and not self._proceed.is_set()

    def inject(self, text: str) -> None:
        """Queue a moderator message; it lands before the next persona speaks."""
        text = text.strip()
        if text:
            self._injections.append(text)

    def stop(self) -> None:
        self._stopped = True
        self._proceed.set()  # unblock the gate so run() can observe the stop

    # --- error handling --------------------------------------------------

    @staticmethod
    def _rate_limit_delay(exc: Exception) -> float | None:
        """Seconds to wait before retrying, or None if `exc` is not a rate limit.

        Both the Anthropic and OpenAI SDKs raise a 429 with `status_code` set, so
        we sniff that rather than importing either SDK's exception type here.
        """
        if getattr(exc, "status_code", None) != 429:
            return None

        # 1. The standard Retry-After response header.
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                header = resp.headers.get("retry-after")
                if header:
                    return float(header)
            except Exception:
                pass

        # 2. OpenRouter nests retry_after_seconds inside the error body metadata.
        delay = _dig_retry_seconds(getattr(exc, "body", None))
        if delay is not None:
            return delay

        return DEFAULT_RETRY_DELAY

    @staticmethod
    def _error_summary(exc: Exception) -> str:
        text = str(exc).strip() or repr(exc)
        if len(text) > 300:
            text = text[:297] + "…"
        return f"{type(exc).__name__}: {text}"

    async def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep that wakes early if the conversation is stopped."""
        loop = asyncio.get_event_loop()
        end = loop.time() + seconds
        while not self._stopped:
            remaining = end - loop.time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.25, remaining))

    async def _stream_with_retry(
        self, persona: Persona, adapter: ProviderAdapter, messages: list[Msg]
    ) -> AsyncIterator[StreamEvent]:
        """Stream one turn, retrying on rate limits.

        On give-up (non-retryable error or exhausted retries) it yields a
        TurnError instead of raising, so the loop survives and the UI can show
        what happened. Retries only when nothing was emitted yet, so a partially
        streamed reply is never duplicated.
        """
        attempt = 0
        while True:
            produced = False
            try:
                async for event in adapter.stream_turn(
                    messages, persona, self.show_thinking
                ):
                    produced = True
                    yield event
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # adapters raise SDK-specific types
                delay = self._rate_limit_delay(exc)
                if delay is None or produced or attempt >= self.max_retries:
                    yield TurnError(persona.name, self._error_summary(exc))
                    return
                attempt += 1
                delay = min(max(delay, 1.0), self.max_retry_delay)
                yield RetryNotice(persona.name, attempt, delay, "rate limited")
                await self._interruptible_sleep(delay)
                if self._stopped:
                    return

    # --- the loop --------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Heuristic, provider-agnostic: ~4 chars per token, plus a small per-message
        # overhead for role/formatting framing. Deliberately rough; the budget keeps
        # a safety margin so the estimate never has to be exact.
        return len(text) // 4 + 4

    def _build_messages(self, persona: Persona) -> list[Msg]:
        # The kickoff prompt anchors the topic on every turn and guarantees the
        # first message is a user turn (required by both APIs). It and the system
        # prompt are never trimmed.
        kickoff: Msg = {"role": "user", "content": self.kickoff}

        from .context import DEFAULT_CONTEXT

        context = self.context_lengths.get(persona.name, DEFAULT_CONTEXT)
        # Reserve room for the reply. When uncapped, the budget_fraction headroom
        # (the 1 - fraction slice of the window) is what's left for output.
        reserve = persona.max_tokens or 0
        budget = int(context * self.budget_fraction) - reserve
        used = self._estimate_tokens(persona.system_prompt) + self._estimate_tokens(
            self.kickoff
        )

        # Walk newest-to-oldest, keeping as many recent turns as fit. The most
        # recent turn is always kept so the speaker can react to what was just said.
        kept: list[Msg] = []
        for turn in reversed(self.transcript):
            tokens = self._estimate_tokens(turn.text)
            if kept and used + tokens > budget:
                break
            used += tokens
            role = "assistant" if turn.speaker == persona.name else "user"
            kept.append({"role": role, "content": turn.text})
        kept.reverse()

        return [kickoff, *kept]

    async def run(self) -> AsyncIterator[StreamEvent]:
        turn_no = self.start_turn
        while turn_no < self.max_turns:
            # Always yield to the loop once per turn so controls issued from
            # another task (pause / stop / step) are observed promptly, even if
            # an adapter happens to stream without awaiting.
            await asyncio.sleep(0)
            # Gate: honour pause / step mode before committing to a turn.
            await self._proceed.wait()
            if self._stopped:
                break
            if self.step_mode:
                self._proceed.clear()  # one step consumed; wait for the next

            # Drain any moderator messages so the upcoming speaker sees them.
            while self._injections:
                text = self._injections.pop(0)
                self.transcript.append(Turn(self.moderator_name, text))
                yield Injected(self.moderator_name, text)

            persona = self.personas[turn_no % len(self.personas)]
            adapter = self.adapters[persona.name]

            yield TurnStarted(persona.name, turn_no)

            buf: list[str] = []
            errored = False
            async for event in self._stream_with_retry(
                persona, adapter, self._build_messages(persona)
            ):
                if isinstance(event, TextDelta):
                    buf.append(event.text)
                elif isinstance(event, TurnError):
                    errored = True
                yield event

            text = "".join(buf).strip()
            if text:
                self.transcript.append(Turn(persona.name, text))
            elif not errored:
                # Guard against an empty turn becoming an empty message next round,
                # which some providers reject. A failed turn records nothing so it
                # does not pollute the next speaker's view.
                self.transcript.append(Turn(persona.name, "(no response)"))
            turn_no += 1
