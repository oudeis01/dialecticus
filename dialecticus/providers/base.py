"""The contract every provider adapter implements.

An adapter's whole job is to turn one provider's streaming API into our normalized
StreamEvent vocabulary. All provider-specific quirks stay behind this boundary.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from ..events import StreamEvent
from ..persona import Persona

# A single chat message in the provider-neutral shape both SDKs accept. Values
# are usually strings (role/content), but tool rounds add richer content: a list
# of content blocks (Anthropic) or a tool_calls list (OpenAI).
Msg = dict[str, object]


class ProviderAdapter(Protocol):
    def stream_turn(
        self,
        messages: list[Msg],
        persona: Persona,
        show_thinking: bool,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one turn for `persona`, yielding normalized events."""
        ...
