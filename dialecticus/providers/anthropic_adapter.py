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

# Safety net so a model that keeps calling tools cannot loop forever. This many
# tool-capable rounds are allowed, then one final round runs with no tools so the
# model is forced to answer from what it read instead of ending an empty turn.
MAX_TOOL_ROUNDS = 8

# An ephemeral (5-minute) cache breakpoint. Basic prompt caching is GA, so no
# beta header is needed; only a 1-hour TTL would require one.
_CACHE = {"type": "ephemeral"}


def _cached_system(system_prompt: str) -> object:
    """Mark the system prompt as a cacheable prefix.

    The identity/topic/scope text is identical on every call for a persona, so
    caching it means we pay full input price for it once and 0.1x thereafter.
    Falls back to a plain string when empty (nothing worth a breakpoint).
    """
    if not system_prompt:
        return system_prompt
    return [{"type": "text", "text": system_prompt, "cache_control": _CACHE}]


def _with_cache_breakpoint(messages: list[Msg]) -> list[Msg]:
    """Return a copy of `messages` with an ephemeral breakpoint on the last block.

    Anthropic caches the whole prefix up to a breakpoint and auto-reads the
    longest match, so marking just the final block caches the entire
    conversation so far. Inside the tool loop this is the big win: each extra
    round re-reads the accumulated prefix at 0.1x instead of re-sending it at
    full price. We copy rather than mutate so the engine's transcript dicts stay
    clean. SDK content-block objects (from a prior tool_use turn) are never the
    last message at call time, so we only ever handle strings and dict blocks.
    """
    if not messages:
        return messages
    out = list(messages)
    last = dict(out[-1])
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{"type": "text", "text": content, "cache_control": _CACHE}]
        out[-1] = last
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        blocks = list(content)
        tail = dict(blocks[-1])
        tail["cache_control"] = _CACHE
        blocks[-1] = tail
        last["content"] = blocks
        out[-1] = last
    return out


class AnthropicAdapter:
    def __init__(
        self,
        client: AsyncAnthropic | None = None,
        sandbox: FileSandbox | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        # AsyncAnthropic() reads ANTHROPIC_API_KEY from the environment.
        self.client = client or AsyncAnthropic()
        # When a sandbox is configured, expose the read-only file tools.
        self.sandbox = sandbox
        self.tools = anthropic_tools() if sandbox else None
        self.max_tool_rounds = max(1, max_tool_rounds)

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

        for round_no in range(self.max_tool_rounds + 1):
            # The final round offers no tools, forcing the model to answer from
            # what it has already read rather than spending the round on yet more
            # tool calls and ending the turn with empty text.
            final_round = round_no == self.max_tool_rounds
            kwargs: dict = dict(
                model=persona.model,
                max_tokens=persona.max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS,
                system=_cached_system(persona.system_prompt),
                messages=_with_cache_breakpoint(convo),
            )
            if show_thinking:
                # adaptive: the model decides how much to think; summarized: stream
                # a readable summary of the reasoning rather than empty blocks.
                kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
            if self.tools and not final_round:
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
                u = final.usage
                if usage is None:
                    usage = Usage(0, 0, 0, 0)
                # Accumulate across tool rounds so a multi-round turn reports its
                # whole cost, not just the last API call.
                usage.input_tokens += u.input_tokens or 0
                usage.output_tokens += u.output_tokens or 0
                usage.cache_read_input_tokens += (
                    getattr(u, "cache_read_input_tokens", 0) or 0
                )
                usage.cache_creation_input_tokens += (
                    getattr(u, "cache_creation_input_tokens", 0) or 0
                )

            tool_uses = [
                b
                for b in (final.content if final is not None else [])
                if getattr(b, "type", None) == "tool_use"
            ]
            if not tool_uses or not self.sandbox or final_round:
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
