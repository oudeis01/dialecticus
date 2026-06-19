"""Native Anthropic adapter. Highest-fidelity thinking via the Messages API."""

from __future__ import annotations

from typing import AsyncIterator

from anthropic import AsyncAnthropic

from ..events import (
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnComplete,
    Usage,
)
from ..filetools import FileSandbox, anthropic_tools, summarize_result
from ..persona import Persona
from .base import Msg

# The Anthropic Messages API requires max_tokens, so an uncapped persona falls
# back to this rather than streaming without a limit.
ANTHROPIC_DEFAULT_MAX_TOKENS = 4096

# Safety net so a model that keeps calling tools cannot loop forever.
MAX_TOOL_ROUNDS = 8


class AnthropicAdapter:
    def __init__(
        self,
        client: AsyncAnthropic | None = None,
        sandbox: FileSandbox | None = None,
    ) -> None:
        # AsyncAnthropic() reads ANTHROPIC_API_KEY from the environment.
        self.client = client or AsyncAnthropic()
        # When a sandbox is configured, expose the read-only file tools.
        self.sandbox = sandbox
        self.tools = anthropic_tools() if sandbox else None

    async def stream_turn(
        self,
        messages: list[Msg],
        persona: Persona,
        show_thinking: bool,
    ) -> AsyncIterator[StreamEvent]:
        # A turn may take several rounds: text/thinking, then tool calls whose
        # results feed back in, until the model answers without calling a tool.
        convo: list[Msg] = list(messages)
        usage: Usage | None = None

        for _ in range(MAX_TOOL_ROUNDS + 1):
            kwargs: dict = dict(
                model=persona.model,
                max_tokens=persona.max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS,
                system=persona.system_prompt,
                messages=convo,
            )
            if show_thinking:
                # adaptive: the model decides how much to think; summarized: stream
                # a readable summary of the reasoning rather than empty blocks.
                kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
            if self.tools:
                kwargs["tools"] = self.tools

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

            if final is not None and final.usage is not None:
                usage = Usage(final.usage.input_tokens, final.usage.output_tokens)

            tool_uses = [
                b
                for b in (final.content if final is not None else [])
                if getattr(b, "type", None) == "tool_use"
            ]
            if not tool_uses or not self.sandbox:
                break

            # Echo the assistant's tool-call turn back verbatim (including any
            # thinking blocks, which extended-thinking tool use requires), then
            # answer each call with a tool_result.
            convo.append({"role": "assistant", "content": final.content})
            results: list[dict] = []
            for block in tool_uses:
                args = block.input if isinstance(block.input, dict) else {}
                yield ToolCall(persona.name, block.name, args)
                output = self.sandbox.execute(block.name, args)
                ok, summary = summarize_result(block.name, args, output)
                yield ToolResult(persona.name, block.name, ok, summary)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )
            convo.append({"role": "user", "content": results})

        yield TurnComplete(persona.name, usage)
