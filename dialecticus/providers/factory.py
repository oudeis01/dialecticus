"""Build one adapter per persona, reusing clients that share a configuration."""

from __future__ import annotations

import os

from ..filetools import FileSandbox
from ..persona import Persona
from .anthropic_adapter import AnthropicAdapter
from .base import ProviderAdapter
from .openai_adapter import OpenAIAdapter
from .zai_adapter import ZAIAdapter


def build_adapters(
    personas: list[Persona],
    sandbox: FileSandbox | None = None,
    max_tool_rounds: int = 8,
) -> dict[str, ProviderAdapter]:
    adapters: dict[str, ProviderAdapter] = {}
    cache: dict[tuple, ProviderAdapter] = {}

    for p in personas:
        if p.provider == "anthropic":
            key = ("anthropic",)
            cache.setdefault(
                key,
                AnthropicAdapter(sandbox=sandbox, max_tool_rounds=max_tool_rounds),
            )
        elif p.provider == "openai":
            api_key = os.environ.get(p.api_key_env) if p.api_key_env else None
            key = ("openai", p.base_url, p.api_key_env)
            cache.setdefault(
                key,
                OpenAIAdapter(
                    base_url=p.base_url,
                    api_key=api_key,
                    sandbox=sandbox,
                    max_tool_rounds=max_tool_rounds,
                ),
            )
        elif p.provider == "zai":
            api_key = os.environ.get(p.api_key_env) if p.api_key_env else None
            key = ("zai", p.base_url, p.api_key_env)
            cache.setdefault(
                key,
                ZAIAdapter(
                    base_url=p.base_url,
                    api_key=api_key,
                    sandbox=sandbox,
                    max_tool_rounds=max_tool_rounds,
                ),
            )
        else:
            raise ValueError(f"unknown provider for persona {p.name!r}: {p.provider!r}")
        adapters[p.name] = cache[key]

    return adapters
