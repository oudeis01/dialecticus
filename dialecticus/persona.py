"""A conversation participant: a model plus the identity and scope it speaks under."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Persona:
    name: str
    provider: str  # "anthropic" | "openai"
    model: str
    system_prompt: str  # identity + topic + scope, assembled in config.py
    # Cap on the model's reply length. None means "no cap": the OpenAI adapter
    # omits the parameter so the model generates up to the context limit. (The
    # Anthropic API requires a cap, so its adapter falls back to a default.)
    max_tokens: int | None = 1024
    # OpenAI-compatible only: where to point the client and which env var holds the key.
    base_url: str | None = None
    api_key_env: str | None = None
    # Optional override for the model's context window (tokens). When unset it is
    # resolved from the provider; see context.resolve_context_lengths.
    context_length: int | None = None
