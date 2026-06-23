"""Gemini (Google AI Studio) adapter. OpenAI-compatible Chat Completions + thinking.

Gemini can be reached through an OpenAI-compatible endpoint:

    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

The API key is a Google AI Studio API key sent as ``Authorization: Bearer``, so
the same ``AsyncOpenAI`` client that powers ``OpenAIAdapter`` works unmodified.

Thinking mode
-------------
When ``show_thinking`` is True the adapter passes ``reasoning_effort="high"``
(the OpenAI-compatible parameter that Gemini maps to its own thinking budget).
For models that support ``include_thoughts`` the adapter requests thought
summaries via ``extra_body.google.thinking_config``.  Thought summaries land in
the ``reasoning_content`` delta field and are surfaced as ``ThinkingDelta``
events by the base ``OpenAIAdapter``.

Model            Context     Notes
───────────────  ──────────  ──────────────────────────────
gemini-3.5-flash   1,048,576  latest flash
gemini-3-flash     1,048,576  prior generation flash
gemini-2.5-pro     1,048,576  flagship reasoning model
gemini-2.5-flash   1,048,576  balanced reasoning / speed
gemini-2.5-flash-lite 1,048,576  faster / cheaper
gemini-2.0-flash   1,048,576  legacy workhorse
"""

from __future__ import annotations

from ..filetools import FileSandbox
from .openai_adapter import MAX_TOOL_ROUNDS, OpenAIAdapter

# The official Gemini OpenAI-compatible endpoint.  A persona can override this
# via ``base_url`` in the YAML; when unset the adapter uses this default.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Known Gemini model context windows.  All current Gemini models ship 1M context;
# the map exists for forward compatibility and explicit auditability.
GEMINI_MODEL_CONTEXTS: dict[str, int] = {
    "gemini-3.5-flash": 1_048_576,
    "gemini-3-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-flash-lite": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
}

# Conservative fallback for unrecognised Gemini model ids.
GEMINI_DEFAULT_CONTEXT = 1_048_576


class GeminiAdapter(OpenAIAdapter):
    """Adapter for Gemini's OpenAI-compatible endpoint.

    Enables thinking mode via ``reasoning_effort`` and requests thought
    summaries so that ``reasoning_content`` is streamed.  All other behaviour
    (streaming, tool calls, token management) is inherited from
    :class:`OpenAIAdapter`.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        sandbox: FileSandbox | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        # Default to the official Gemini OpenAI-compatible endpoint when no
        # explicit base_url is provided.
        super().__init__(
            base_url=base_url or GEMINI_BASE_URL,
            api_key=api_key,
            sandbox=sandbox,
            max_tool_rounds=max_tool_rounds,
        )

    def _thinking_request_params(self) -> dict[str, object]:
        # Gemini's OpenAI-compatible endpoint expects Google-specific
        # parameters wrapped inside a top-level JSON key "extra_body".
        # The OpenAI Python SDK's extra_body kwarg merges its content into
        # the JSON body at top level, so we nest another "extra_body" layer
        # here:
        #   request body → { ..., "extra_body":
        #     { "google": { "thinking_config": { "include_thoughts": ... } } } }
        # "thinking_budget" is used for Gemini 2.5 models; 3.x models
        # use "thinking_level" instead.  include_thoughts=True surfaces
        # thought summaries as reasoning_content in the delta (which the base
        # adapter converts to ThinkingDelta).
        # NOTE: reasoning_effort overlaps with thinking_config and cannot be
        # set simultaneously.
        return {
            "extra_body": {
                "extra_body": {
                    "google": {
                        "thinking_config": {
                            "include_thoughts": True,
                            "thinking_budget": 8192,
                        }
                    }
                }
            },
        }

    @classmethod
    def resolve_context(cls, model: str) -> int:
        """Look up a Gemini model's context window, or return the default (1M)."""
        if model in GEMINI_MODEL_CONTEXTS:
            return GEMINI_MODEL_CONTEXTS[model]
        for prefix, ctx in GEMINI_MODEL_CONTEXTS.items():
            if model.startswith(prefix):
                return ctx
        return GEMINI_DEFAULT_CONTEXT
