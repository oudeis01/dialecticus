"""Local session persistence: record every run and reconstruct it for resume.

A run is captured losslessly as a single JSON file (thinking, text, errors,
moderator injections, usage, timestamps, and enough persona config to rebuild
the conversation without the original YAML). API keys are never stored: only the
*name* of the env var that holds each key is kept.

`export.py` derives human/portable views (Markdown, filtered JSON) from this
canonical record; `resume` reads it back to continue a conversation.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone

from .config import Conversation
from .engine import Turn
from .events import (
    Injected,
    TextDelta,
    ThinkingDelta,
    TurnComplete,
    TurnError,
    TurnStarted,
)
from .persona import Persona

SESSIONS_DIR = "sessions"
SCHEMA_VERSION = 1


def _slug(text: str, n: int = 40) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return text[:n].strip("-") or "session"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Recorder:
    """Consume the engine event stream and persist it to a JSON file.

    The file is rewritten atomically after every finalized turn, so a crash or
    interrupt mid-run still leaves a valid record of everything up to that point.
    """

    def __init__(
        self,
        conv: Conversation,
        path: str | None = None,
        seed_turns: list[dict] | None = None,
    ) -> None:
        if path is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join(SESSIONS_DIR, f"{stamp}-{_slug(conv.kickoff)}.json")
        self.path = path
        self.record: dict = {
            "version": SCHEMA_VERSION,
            "created_at": _now(),
            "updated_at": _now(),
            "kickoff": conv.kickoff,
            "max_turns": conv.max_turns,
            "show_thinking": conv.show_thinking,
            "personas": [asdict(p) for p in conv.personas],
            # When resuming, carry the prior turns forward so the new file is a
            # complete, re-resumable record rather than just the tail.
            "turns": list(seed_turns or []),
        }
        self._cur: dict | None = None
        self.flush()

    def feed(self, event) -> None:
        if isinstance(event, Injected):
            self.record["turns"].append(
                {
                    "speaker": event.speaker,
                    "kind": "moderator",
                    "thinking": "",
                    "text": event.text,
                    "error": None,
                    "usage": None,
                    "at": _now(),
                }
            )
            self.flush()
        elif isinstance(event, TurnStarted):
            self._cur = {
                "speaker": event.speaker,
                "kind": "model",
                "thinking": "",
                "text": "",
                "error": None,
                "usage": None,
                "at": _now(),
            }
        elif isinstance(event, ThinkingDelta):
            if self._cur is not None:
                self._cur["thinking"] += event.text
        elif isinstance(event, TextDelta):
            if self._cur is not None:
                self._cur["text"] += event.text
        elif isinstance(event, TurnError):
            if self._cur is not None:
                self._cur["error"] = event.message
                self._commit()
        elif isinstance(event, TurnComplete):
            if self._cur is not None:
                if event.usage is not None:
                    self._cur["usage"] = {
                        "input_tokens": event.usage.input_tokens,
                        "output_tokens": event.usage.output_tokens,
                    }
                self._commit()

    def _commit(self) -> None:
        if self._cur is not None:
            self.record["turns"].append(self._cur)
            self._cur = None
            self.flush()

    def flush(self) -> None:
        self.record["updated_at"] = _now()
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.record, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)  # atomic: never leave a half-written file
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def load_session(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def conversation_from_record(record: dict) -> Conversation:
    """Rebuild a Conversation from a saved session (no YAML needed)."""
    personas = [Persona(**p) for p in record["personas"]]
    return Conversation(
        personas=personas,
        kickoff=record["kickoff"],
        max_turns=record.get("max_turns", 6),
        show_thinking=record.get("show_thinking", True),
    )


def transcript_from_record(record: dict) -> tuple[list[Turn], int]:
    """Reconstruct the engine transcript and the next turn index.

    Returns (transcript, start_turn). start_turn counts *every* model turn,
    including ones that errored, because the live engine advances its round-robin
    index on error too; transcript itself mirrors what the live run stored
    (errored empty turns recorded nothing; empty non-error turns recorded a
    placeholder).
    """
    transcript: list[Turn] = []
    start_turn = 0
    for t in record["turns"]:
        if t["kind"] == "model":
            start_turn += 1
            text = (t.get("text") or "").strip()
            if not text:
                if t.get("error"):
                    continue  # errored turn stored nothing in the live transcript
                text = "(no response)"
            transcript.append(Turn(t["speaker"], text))
        else:
            transcript.append(Turn(t["speaker"], t.get("text") or ""))
    return transcript, start_turn
