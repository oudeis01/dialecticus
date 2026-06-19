"""The contract every provider adapter implements.

An adapter's whole job is to turn one provider's streaming API into our normalized
StreamEvent vocabulary. All provider-specific quirks stay behind this boundary.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from ..events import StreamEvent
from ..persona import Persona

# A single chat message in the provider-neutral shape both SDKs accept.
Msg = dict[str, str]


class ProviderAdapter(Protocol):
    def stream_turn(
        self,
        messages: list[Msg],
        persona: Persona,
        show_thinking: bool,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one turn for `persona`, yielding normalized events."""
        ...
