"""Z.AI adapter. OpenAI-compatible Chat Completions + thinking mode.

Z.AI's GLM models emit *reasoning_content* when the request body includes
``thinking: {type: "enabled"}``.  The base ``OpenAIAdapter`` already surfaces
``reasoning_content`` as ``ThinkingDelta`` events; this adapter simply enables it
at the source so the stream carries reasoning data to expose.

Model   Context  Notes
──────  ───────  ──────────────────────────────────
glm-5.2  128 K   flagship reasoning model
glm-5.1  128 K   prior generation reasoning model
glm-5    128 K   generic label (may point to glm-5.2)
glm-4.7  128 K   balanced reasoning / speed
glm-4.6  128 K   prior generation
glm-4.5  128 K   general purpose
glm-4.5-air  128 K   faster / cheaper variant
glm-4.5-flash 128 K   fastest / cheapest variant
glm-4-32b-0414-128k  128 K   legacy large model
"""

from __future__ import annotations

from .openai_adapter import OpenAIAdapter

# Known Z.AI (GLM) model context windows.  All current Z.AI models ship 128 K
# context; the map exists for forward compatibility and explicit auditability.
ZAI_MODEL_CONTEXTS: dict[str, int] = {
    "glm-5.2": 128_000,
    "glm-5.1": 128_000,
    "glm-5": 128_000,
    "glm-4.7": 128_000,
    "glm-4.6": 128_000,
    "glm-4.5": 128_000,
    "glm-4.5-air": 128_000,
    "glm-4.5-flash": 128_000,
    "glm-4-32b-0414-128k": 128_000,
}

# Conservative fallback for unrecognised Z.AI model ids.
ZAI_DEFAULT_CONTEXT = 128_000


class ZAIAdapter(OpenAIAdapter):
    """Adapter for Z.AI's OpenAI-compatible GLM model API.

    Enables thinking mode so ``reasoning_content`` is streamed.  All other
    behaviour (streaming, tool calls, token management) is inherited from
    :class:`OpenAIAdapter`.
    """

    def _thinking_request_params(self) -> dict[str, object]:
        # Z.AI's thinking mode is activated via the request body field
        # ``thinking: {type: "enabled"}``.  The OpenAI Python SDK does not expose
        # ``thinking`` as a keyword argument, so we route it through the official
        # ``extra_body`` escape hatch, which the client merges into the JSON body.
        return {"extra_body": {"thinking": {"type": "enabled"}}}

    @classmethod
    def resolve_context(cls, model: str) -> int:
        """Look up a Z.AI model's context window, or return the default (128 K)."""
        if model in ZAI_MODEL_CONTEXTS:
            return ZAI_MODEL_CONTEXTS[model]
        for prefix, ctx in ZAI_MODEL_CONTEXTS.items():
            if model.startswith(prefix):
                return ctx
        return ZAI_DEFAULT_CONTEXT
