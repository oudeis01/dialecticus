"""Resolve each persona's model context window.

Different models have wildly different context limits (free OpenRouter models alone
span 32k..1M), so the engine needs a per-persona budget. Resolution order:

1. an explicit `context_length` in the persona's YAML (always wins),
2. for OpenRouter, the live `context_length` from its models API (fetched once),
3. for Anthropic, a small map of known windows,
4. a conservative default otherwise.

Network lookups happen once at startup and fail soft: if the fetch breaks, we fall
back to the default so a conversation still runs (it just trims more eagerly).
"""

from __future__ import annotations

import json
import urllib.request

from .persona import Persona
from .providers.gemini_adapter import GeminiAdapter
from .providers.zai_adapter import ZAIAdapter

DEFAULT_CONTEXT = 8192

# Known Anthropic windows (substring match on the model id). Conservative on
# purpose; override per-persona in YAML to use a model's full window.
_ANTHROPIC = [
    ("haiku", 200_000),
    ("opus-4", 1_000_000),
    ("sonnet-4", 1_000_000),
    ("fable", 1_000_000),
]
_ANTHROPIC_DEFAULT = 200_000


def _anthropic_context(model: str) -> int:
    for needle, ctx in _ANTHROPIC:
        if needle in model:
            return ctx
    return _ANTHROPIC_DEFAULT


def _fetch_openrouter(base_url: str) -> dict[str, int]:
    """Map model id -> context_length from an OpenRouter-style /models endpoint."""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.load(resp)
    except Exception:
        return {}
    out: dict[str, int] = {}
    for m in data.get("data", []):
        ctx = m.get("context_length")
        if isinstance(m.get("id"), str) and isinstance(ctx, int):
            out[m["id"]] = ctx
    return out


def resolve_context_lengths(personas: list[Persona]) -> dict[str, int]:
    result: dict[str, int] = {}
    openrouter_cache: dict[str, dict[str, int]] = {}

    for p in personas:
        if p.context_length:
            result[p.name] = p.context_length
        elif p.provider == "anthropic":
            result[p.name] = _anthropic_context(p.model)
        elif p.provider == "zai":
            result[p.name] = ZAIAdapter.resolve_context(p.model)
        elif p.provider == "gemini":
            result[p.name] = GeminiAdapter.resolve_context(p.model)
        elif p.provider == "openai" and p.base_url and "openrouter.ai" in p.base_url:
            catalog = openrouter_cache.get(p.base_url)
            if catalog is None:
                catalog = _fetch_openrouter(p.base_url)
                openrouter_cache[p.base_url] = catalog
            result[p.name] = catalog.get(p.model, DEFAULT_CONTEXT)
        else:
            result[p.name] = DEFAULT_CONTEXT

    return result
