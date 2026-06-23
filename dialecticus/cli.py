"""Console runner: drive the engine and render its event stream live.

This is the throwaway viewer for the MVP. The Textual TUI will later subscribe to
the exact same engine event stream, so nothing here is load-bearing for the design.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from . import config
from .engine import Engine
from .events import (
    RetryNotice,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnComplete,
    TurnError,
    TurnStarted,
    format_usage,
)
from .persona import Persona
from .providers.factory import build_adapters

_PALETTE = ["\033[36m", "\033[33m", "\033[35m", "\033[32m"]  # cyan, yellow, magenta, green
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_RESET = "\033[0m"


class Renderer:
    def __init__(self, personas: list[Persona]) -> None:
        self.color = {p.name: _PALETTE[i % len(_PALETTE)] for i, p in enumerate(personas)}
        self.in_thinking = False

    def handle(self, event) -> None:
        color = self.color.get(getattr(event, "speaker", ""), "")

        if isinstance(event, TurnStarted):
            self.in_thinking = False
            sys.stdout.write(f"\n\n{color}{_BOLD}{'─' * 60}\n{event.speaker}{_RESET}\n")
        elif isinstance(event, ThinkingDelta):
            if not self.in_thinking:
                sys.stdout.write(f"{_DIM}[thinking] ")
                self.in_thinking = True
            sys.stdout.write(f"{_DIM}{event.text}")
        elif isinstance(event, TextDelta):
            if self.in_thinking:
                sys.stdout.write(f"{_RESET}\n")
                self.in_thinking = False
            sys.stdout.write(f"{color}{event.text}{_RESET}")
        elif isinstance(event, RetryNotice):
            if self.in_thinking:
                sys.stdout.write(_RESET)
                self.in_thinking = False
            sys.stdout.write(
                f"\n{_DIM}⟳ {event.reason}; retrying in {event.delay:.0f}s "
                f"(attempt {event.attempt}){_RESET}\n"
            )
        elif isinstance(event, ToolCall):
            if self.in_thinking:
                sys.stdout.write(_RESET)
                self.in_thinking = False
            from .filetools import format_call

            sys.stdout.write(
                f"\n{_DIM}⚙ {event.tool}({format_call(event.tool, event.arguments)}){_RESET}\n"
            )
        elif isinstance(event, ToolResult):
            mark = "↳" if event.ok else "✗"
            sys.stdout.write(f"{_DIM}  {mark} {event.summary}{_RESET}\n")
        elif isinstance(event, TurnError):
            if self.in_thinking:
                sys.stdout.write(_RESET)
                self.in_thinking = False
            sys.stdout.write(f"\n{_RED}✗ {event.message}{_RESET}\n")
        elif isinstance(event, TurnComplete):
            if self.in_thinking:
                sys.stdout.write(_RESET)
                self.in_thinking = False
            summary = format_usage(event.usage)
            if summary:
                sys.stdout.write(f"{_DIM}  ◷ {summary}{_RESET}\n")

        sys.stdout.flush()


async def _run_plain(conv, initial_transcript=None, start_turn=0, seed_turns=None) -> None:
    from .context import resolve_context_lengths
    from .filetools import FileSandbox
    from .session import Recorder

    sandbox = FileSandbox(conv.workspace) if conv.workspace else None
    adapters = build_adapters(
        conv.personas, sandbox=sandbox, max_tool_rounds=conv.max_tool_rounds
    )
    engine = Engine(
        conv.personas,
        adapters,
        conv.kickoff,
        max_turns=conv.max_turns,
        show_thinking=conv.show_thinking,
        context_lengths=resolve_context_lengths(conv.personas),
        initial_transcript=initial_transcript,
        start_turn=start_turn,
    )
    recorder = Recorder(conv, seed_turns=seed_turns)
    renderer = Renderer(conv.personas)
    async for event in engine.run():
        recorder.feed(event)
        renderer.handle(event)
    sys.stdout.write("\n")
    sys.stdout.write(f"{_DIM}saved → {recorder.path}{_RESET}\n")


def _load_dotenv() -> None:
    # Load API keys from a local .env if present, so they need not be exported.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # python-dotenv optional; env vars still work without it


def _check_workspace(conv) -> None:
    """Fail early with a clear message if file_access points nowhere valid."""
    if conv.workspace and not os.path.isdir(conv.workspace):
        sys.stderr.write(f"error: file_access directory not found: {conv.workspace}\n")
        sys.exit(1)


def _cmd_run(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="dialecticus")
    parser.add_argument("config", help="path to a personas YAML file")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="stream to the console instead of the interactive TUI",
    )
    args = parser.parse_args(argv)
    conv = config.load(args.config)
    _check_workspace(conv)

    if args.plain:
        try:
            asyncio.run(_run_plain(conv))
        except KeyboardInterrupt:
            sys.stdout.write(f"{_RESET}\n[interrupted]\n")
        return

    from . import tui

    tui.run(conv)


def _cmd_export(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="dialecticus export")
    parser.add_argument("session", help="path to a saved session JSON file")
    parser.add_argument(
        "--format", "-f", choices=["md", "markdown", "json"], default="md"
    )
    parser.add_argument(
        "--thinking", action="store_true", help="include the models' reasoning"
    )
    parser.add_argument(
        "--out", "-o", help="write to this file instead of stdout"
    )
    args = parser.parse_args(argv)

    from .export import export_session
    from .session import load_session

    text = export_session(load_session(args.session), args.format, args.thinking)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        sys.stdout.write(f"wrote {args.out}\n")
    else:
        sys.stdout.write(text)


def _cmd_resume(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="dialecticus resume")
    parser.add_argument("session", help="path to a saved session JSON file")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="stream to the console instead of the interactive TUI",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help="additional turns to run (default: the session's max_turns)",
    )
    args = parser.parse_args(argv)

    from .session import (
        conversation_from_record,
        load_session,
        transcript_from_record,
    )

    record = load_session(args.session)
    conv = conversation_from_record(record)
    _check_workspace(conv)
    transcript, start_turn = transcript_from_record(record)
    additional = args.turns if args.turns is not None else conv.max_turns
    conv.max_turns = start_turn + additional  # budget = already done + more

    if args.plain:
        try:
            asyncio.run(
                _run_plain(
                    conv,
                    initial_transcript=transcript,
                    start_turn=start_turn,
                    seed_turns=record["turns"],
                )
            )
        except KeyboardInterrupt:
            sys.stdout.write(f"{_RESET}\n[interrupted]\n")
        return

    from . import tui

    tui.run(
        conv,
        initial_transcript=transcript,
        start_turn=start_turn,
        seed_turns=record["turns"],
    )


def main() -> None:
    _load_dotenv()
    argv = sys.argv[1:]
    if argv and argv[0] == "export":
        _cmd_export(argv[1:])
    elif argv and argv[0] == "resume":
        _cmd_resume(argv[1:])
    else:
        _cmd_run(argv)


if __name__ == "__main__":
    main()
