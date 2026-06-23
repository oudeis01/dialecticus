"""OpenAI-compatible adapter. Point `base_url` at OpenAI, OpenRouter, a local
server, or any Chat Completions endpoint to reach a huge range of models.

Reasoning exposure is provider-specific and not part of the OpenAI standard:
DeepSeek streams `reasoning_content`, OpenRouter often streams `reasoning`, and
OpenAI's own o-series does not expose raw reasoning at all. We surface it where
present and otherwise stay silent.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from openai import AsyncOpenAI

from ..events import (
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnComplete,
)
from ..filetools import FileSandbox, openai_tools, summarize_result
from ..persona import Persona
from .base import Msg

# Safety net so a model that keeps calling tools cannot loop forever. This many
# tool-capable rounds are allowed, then one final round runs with no tools so the
# model is forced to answer from what it read instead of ending an empty turn.
MAX_TOOL_ROUNDS = 8


class OpenAIAdapter:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        sandbox: FileSandbox | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self.client = client or AsyncOpenAI(base_url=base_url, api_key=api_key)
        # When a sandbox is configured, expose the read-only file tools.
        self.sandbox = sandbox
        self.tools = openai_tools() if sandbox else None
        self.max_tool_rounds = max(1, max_tool_rounds)

    async def stream_turn(
        self,
        messages: list[Msg],
        persona: Persona,
        show_thinking: bool,
    ) -> AsyncIterator[StreamEvent]:
        convo: list[Msg] = [
            {"role": "system", "content": persona.system_prompt},
            *messages,
        ]

        for round_no in range(self.max_tool_rounds + 1):
            # The final round offers no tools, forcing the model to answer from
            # what it has already read rather than spending the round on yet more
            # tool calls and ending the turn with empty text.
            final_round = round_no == self.max_tool_rounds
            kwargs: dict = dict(model=persona.model, messages=convo, stream=True)
            # Omit max_tokens entirely when uncapped, so the model may generate up
            # to the context limit instead of being truncated mid-reply.
            if persona.max_tokens:
                kwargs["max_tokens"] = persona.max_tokens
            if self.tools and not final_round:
                kwargs["tools"] = self.tools

            stream = await self.client.chat.completions.create(**kwargs)
            text_parts: list[str] = []
            # Streamed tool calls arrive in fragments keyed by index; accumulate.
            calls: dict[int, dict] = {}
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
                    text_parts.append(delta.content)
                    yield TextDelta(persona.name, delta.content)
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function.arguments:
                            slot["args"] += tc.function.arguments

            if not calls or not self.sandbox or final_round:
                break

            ordered = [calls[i] for i in sorted(calls)]
            # Replay the assistant's tool-call message, then a tool message per call.
            convo.append(
                {
                    "role": "assistant",
                    "content": "".join(text_parts) or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["args"]},
                        }
                        for c in ordered
                    ],
                }
            )
            for c in ordered:
                try:
                    args = json.loads(c["args"]) if c["args"].strip() else {}
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                yield ToolCall(persona.name, c["name"], args)
                output = self.sandbox.execute(c["name"], args)
                ok, summary = summarize_result(c["name"], args, output)
                yield ToolResult(persona.name, c["name"], ok, summary)
                convo.append(
                    {"role": "tool", "tool_call_id": c["id"], "content": output}
                )

        yield TurnComplete(persona.name, None)
