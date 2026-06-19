"""Native Anthropic adapter. Highest-fidelity thinking via the Messages API."""

from __future__ import annotations

from typing import AsyncIterator

from anthropic import AsyncAnthropic

from ..events import StreamEvent, TextDelta, ThinkingDelta, TurnComplete, Usage
from ..persona import Persona
from .base import Msg

# The Anthropic Messages API requires max_tokens, so an uncapped persona falls
# back to this rather than streaming without a limit.
ANTHROPIC_DEFAULT_MAX_TOKENS = 4096


class AnthropicAdapter:
    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        # AsyncAnthropic() reads ANTHROPIC_API_KEY from the environment.
        self.client = client or AsyncAnthropic()

    async def stream_turn(
        self,
        messages: list[Msg],
        persona: Persona,
        show_thinking: bool,
    ) -> AsyncIterator[StreamEvent]:
        kwargs: dict = dict(
            model=persona.model,
            max_tokens=persona.max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS,
            system=persona.system_prompt,
            messages=messages,
        )
        if show_thinking:
            # adaptive: the model decides how much to think; summarized: stream a
            # readable summary of the reasoning rather than empty thinking blocks.
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type != "content_block_delta":
                    continue
                delta = event.delta
                if delta.type == "thinking_delta":
                    yield ThinkingDelta(persona.name, delta.thinking)
                elif delta.type == "text_delta":
                    yield TextDelta(persona.name, delta.text)
            final = await stream.get_final_message()

        usage = None
        if final is not None and final.usage is not None:
            usage = Usage(final.usage.input_tokens, final.usage.output_tokens)
        yield TurnComplete(persona.name, usage)
