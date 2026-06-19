"""OpenAI-compatible adapter. Point `base_url` at OpenAI, OpenRouter, a local
server, or any Chat Completions endpoint to reach a huge range of models.

Reasoning exposure is provider-specific and not part of the OpenAI standard:
DeepSeek streams `reasoning_content`, OpenRouter often streams `reasoning`, and
OpenAI's own o-series does not expose raw reasoning at all. We surface it where
present and otherwise stay silent.
"""

from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from ..events import StreamEvent, TextDelta, ThinkingDelta, TurnComplete
from ..persona import Persona
from .base import Msg


class OpenAIAdapter:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.client = client or AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def stream_turn(
        self,
        messages: list[Msg],
        persona: Persona,
        show_thinking: bool,
    ) -> AsyncIterator[StreamEvent]:
        oa_messages = [{"role": "system", "content": persona.system_prompt}, *messages]
        kwargs: dict = dict(
            model=persona.model,
            messages=oa_messages,
            stream=True,
        )
        # Omit max_tokens entirely when uncapped, so the model may generate up to
        # the context limit instead of being truncated mid-reply.
        if persona.max_tokens:
            kwargs["max_tokens"] = persona.max_tokens
        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if show_thinking:
                reasoning = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if reasoning:
                    yield ThinkingDelta(persona.name, reasoning)
            if delta.content:
                yield TextDelta(persona.name, delta.content)
        yield TurnComplete(persona.name, None)
