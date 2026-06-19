"""Load a conversation definition from YAML and assemble each persona's prompt."""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from .persona import Persona

SYSTEM_TEMPLATE = """{identity}

You are taking part in an ongoing conversation with one or more other AI participants.
Topic of discussion: {topic}
Stay within this scope: {scope}

Speak in your own voice. Respond directly to what the others have said, keep each
turn focused and conversational, and do not narrate stage directions or pretend to
take actions. You are {name}."""


@dataclass
class Conversation:
    personas: list[Persona]
    kickoff: str
    max_turns: int
    show_thinking: bool


def load(path: str) -> Conversation:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    topic = data.get("topic", "")
    scope = data.get("scope", "")

    personas: list[Persona] = []
    for entry in data["personas"]:
        system = SYSTEM_TEMPLATE.format(
            identity=entry["identity"],
            topic=topic,
            scope=scope,
            name=entry["name"],
        )
        # Absent key -> a safe default. An explicit 0, null, or negative value
        # means "no cap" (None), which the OpenAI adapter renders by omitting the
        # parameter entirely.
        raw_max = entry.get("max_tokens", 1024)
        max_tokens = None if raw_max is None or raw_max <= 0 else raw_max
        personas.append(
            Persona(
                name=entry["name"],
                provider=entry["provider"],
                model=entry["model"],
                system_prompt=system,
                max_tokens=max_tokens,
                base_url=entry.get("base_url"),
                api_key_env=entry.get("api_key_env"),
                context_length=entry.get("context_length"),
            )
        )

    return Conversation(
        personas=personas,
        kickoff=data.get("kickoff") or topic,
        max_turns=data.get("max_turns", 6),
        show_thinking=data.get("show_thinking", True),
    )
