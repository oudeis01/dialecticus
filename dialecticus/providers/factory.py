"""Build one adapter per persona, reusing clients that share a configuration."""

from __future__ import annotations

import os

from ..persona import Persona
from .anthropic_adapter import AnthropicAdapter
from .base import ProviderAdapter
from .openai_adapter import OpenAIAdapter


def build_adapters(personas: list[Persona]) -> dict[str, ProviderAdapter]:
    adapters: dict[str, ProviderAdapter] = {}
    cache: dict[tuple, ProviderAdapter] = {}

    for p in personas:
        if p.provider == "anthropic":
            key = ("anthropic",)
            cache.setdefault(key, AnthropicAdapter())
        elif p.provider == "openai":
            api_key = os.environ.get(p.api_key_env) if p.api_key_env else None
            key = ("openai", p.base_url, p.api_key_env)
            cache.setdefault(key, OpenAIAdapter(base_url=p.base_url, api_key=api_key))
        else:
            raise ValueError(f"unknown provider for persona {p.name!r}: {p.provider!r}")
        adapters[p.name] = cache[key]

    return adapters
