"""Textual TUI: watch the conversation stream and intervene at turn boundaries.

It subscribes to the very same engine event stream the console renderer uses, so
the engine stays UI-agnostic. The TUI adds the controls: pause/resume, a
single-step mode, live moderator injection, and a thinking toggle.
"""

from __future__ import annotations

import os

from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Grid, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, Static

from .config import Conversation
from .context import resolve_context_lengths
from .engine import Engine, Turn
from .events import (
    Injected,
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
from .export import export_session
from .filetools import format_call
from .persona import Persona
from .providers.factory import build_adapters
from .session import Recorder

_PALETTE = ["cyan", "yellow", "magenta", "green", "blue", "red"]


class TurnView(Static):
    """One speaker's turn, rendered as ordered segments.

    Thinking, text, and tool activity are kept in the order they actually arrived
    so a file read shows up inline at the point in the reply where the model made
    it, instead of jumping up to the speaker header. Each segment renders as its
    own line block under the colored header.
    """

    # Segment kinds: "thinking", "text" (both accumulate), "tool", "notice".
    def __init__(self, speaker: str, color: str, show_thinking: bool) -> None:
        super().__init__()
        self.speaker = speaker
        self.color = color
        self.show_thinking = show_thinking
        self._segments: list[list] = []  # each: [kind, content]
        self._error: str | None = None
        self._done = False

    def _accumulate(self, kind: str, text: str) -> None:
        """Append to the last segment if it is the same kind, else open a new one."""
        if self._segments and self._segments[-1][0] == kind:
            self._segments[-1][1] += text
        else:
            self._segments.append([kind, text])

    def append_thinking(self, text: str) -> None:
        self._accumulate("thinking", text)
        self._rebuild()

    def append_text(self, text: str) -> None:
        self._accumulate("text", text)
        self._rebuild()

    def add_tool(self, text: str) -> None:
        self._segments.append(["tool", text])
        self._rebuild()

    def add_notice(self, text: str) -> None:
        self._segments.append(["notice", text])
        self._rebuild()

    def set_error(self, text: str) -> None:
        self._error = text
        self._rebuild()

    def finalize(self) -> None:
        """Mark the turn done so text segments re-render as Markdown."""
        self._done = True
        self._rebuild()

    def _rebuild(self) -> None:
        head = Text()
        head.append(f"● {self.speaker}", style=f"bold {self.color}")
        parts: list = [head]
        for kind, content in self._segments:
            if not content:
                continue
            if kind == "thinking":
                if self.show_thinking:
                    parts.append(Text(content, style="dim italic"))
            elif kind == "text":
                # While streaming, show plain colored text (cheap, no half-parsed
                # Markdown); once the turn is done, re-render as Markdown.
                if self._done and content.strip():
                    parts.append(Markdown(content))
                else:
                    parts.append(Text(content, style=self.color))
            elif kind == "tool":
                parts.append(Text(content, style="dim cyan"))
            elif kind == "notice":
                parts.append(Text(f"⟳ {content}", style="dim yellow"))

        if self._error is not None:
            parts.append(Text(f"✗ {self._error}", style="bold red"))

        self.update(Group(*parts))


class InjectedView(Static):
    def __init__(self, text: str) -> None:
        out = Text()
        out.append("» moderator: ", style="bold bright_white")
        out.append(text, style="italic bright_white")
        super().__init__(out)


class IntroView(Static):
    """A header panel shown once at the top: how it runs, who is talking, and the
    kickoff, so the rules and the full prompts are visible before the first turn."""

    def __init__(self, conv: Conversation, color_map: dict[str, str]) -> None:
        body = Text()
        body.append("How this runs\n", style="bold")
        body.append(
            "· Participants speak in turn (round-robin) until the turn limit.\n",
            style="dim",
        )
        body.append("· Starts in STEP mode: press ", style="dim")
        body.append("n", style="bold")
        body.append(" for the next turn, ", style="dim")
        body.append("s", style="bold")
        body.append(" to switch step ↔ auto (continuous).\n", style="dim")
        body.append(
            "· space pause/resume · i inject · t thinking · e export · q quit\n",
            style="dim",
        )
        if conv.workspace:
            body.append(
                "· Read-only file tools available: list_files, read_file, search.\n",
                style="dim",
            )

        body.append("\nParticipants\n", style="bold")
        for p in conv.personas:
            color = color_map.get(p.name, "white")
            body.append(f"● {p.name} ", style=f"bold {color}")
            body.append(f"({p.model})\n", style="dim")
            body.append(p.system_prompt.strip() + "\n\n", style="dim italic")

        body.append("Kickoff\n", style="bold")
        body.append(conv.kickoff.strip(), style="italic")
        super().__init__(
            Panel(body, border_style="dim", title="dialecticus", title_align="left")
        )


class ExportModal(ModalScreen[str | None]):
    """Pick an export format; dismisses with a choice id or None on cancel."""

    CSS = """
    ExportModal { align: center middle; }
    #export-dialog {
        grid-size: 1; grid-gutter: 1; padding: 1 2;
        width: 44; height: auto; border: thick $accent; background: $surface;
    }
    #export-dialog Button { width: 100%; }
    #export-title { text-align: center; width: 100%; padding-bottom: 1; }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    _CHOICES = [
        ("md", "Markdown"),
        ("mdt", "Markdown + thinking"),
        ("json", "JSON"),
        ("jsont", "JSON + thinking"),
    ]

    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Export this conversation", id="export-title"),
            *(Button(label, id=cid) for cid, label in self._CHOICES),
            Button("Cancel", id="cancel", variant="error"),
            id="export-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None if event.button.id == "cancel" else event.button.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class DialecticusApp(App):
    CSS = """
    #transcript { height: 1fr; padding: 0 1; }
    #status { height: 1; color: $text-muted; padding: 0 1; }
    TurnView { margin: 1 0 0 0; }
    InjectedView { margin: 1 0 0 0; }
    IntroView { margin: 0 0 1 0; }
    """

    BINDINGS = [
        ("space", "toggle_pause", "Pause/Resume"),
        ("s", "toggle_step", "Step mode"),
        ("n", "next_turn", "Next turn"),
        ("i", "focus_inject", "Inject"),
        ("t", "toggle_thinking", "Thinking"),
        ("e", "export", "Export"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        conv: Conversation,
        initial_transcript: list[Turn] | None = None,
        start_turn: int = 0,
        seed_turns: list[dict] | None = None,
    ) -> None:
        super().__init__()
        self.conv = conv
        from .filetools import FileSandbox

        sandbox = FileSandbox(conv.workspace) if conv.workspace else None
        self.engine = Engine(
            conv.personas,
            build_adapters(conv.personas, sandbox=sandbox),
            conv.kickoff,
            max_turns=conv.max_turns,
            show_thinking=conv.show_thinking,
            context_lengths=resolve_context_lengths(conv.personas),
            initial_transcript=initial_transcript,
            start_turn=start_turn,
        )
        # Start held in step mode so the viewer can read the intro and drive the
        # pace; `n` advances a turn, `s` switches to continuous (auto) mode.
        self.engine.set_step_mode(True)
        # Every run is recorded losslessly; a resume seeds the new file with the
        # prior turns so it stays complete and re-resumable.
        self.recorder = Recorder(conv, seed_turns=seed_turns)
        self._seed_turns = seed_turns or []
        self._color = {
            p.name: _PALETTE[i % len(_PALETTE)] for i, p in enumerate(conv.personas)
        }
        self._current: TurnView | None = None
        self._finished = False
        self._spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_i = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="transcript")
        yield Static(id="status")
        yield Input(placeholder="press i to inject a moderator message…", id="inject")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "dialecticus"
        scroll = self.query_one("#transcript", VerticalScroll)
        scroll.can_focus = True
        self.set_focus(scroll)
        scroll.mount(IntroView(self.conv, self._color))
        self.call_after_refresh(self._render_seed)
        self._update_status()
        self.run_worker(self._drive(), exclusive=True)
        # Drive the activity spinner; it only animates while a turn is in flight.
        self.set_interval(1 / 12, self._tick_spinner)

    def _render_seed(self) -> None:
        """Show prior turns when resuming, already finalized as Markdown."""
        if not self._seed_turns:
            return
        scroll = self.query_one("#transcript", VerticalScroll)
        for t in self._seed_turns:
            if t["kind"] == "moderator":
                scroll.mount(InjectedView(t.get("text", "")))
                continue
            view = TurnView(
                t["speaker"], self._color.get(t["speaker"], "white"), self.engine.show_thinking
            )
            # The record stores thinking/text/tools separately (interleave order
            # is not preserved), so a resumed turn shows them in a fixed order.
            if t.get("thinking"):
                view._segments.append(["thinking", t["thinking"]])
            if t.get("text"):
                view._segments.append(["text", t["text"]])
            for call in t.get("tools", []) or []:
                shown = format_call(call.get("tool", ""), call.get("arguments") or {})
                view._segments.append(["tool", f"⚙ {call.get('tool', '?')}({shown})"])
                result = call.get("result") or {}
                if result:
                    mark = "↳" if result.get("ok") else "✗"
                    view._segments.append(["tool", f"   {mark} {result.get('summary', '')}"])
            if t.get("error"):
                view._error = t["error"]
            view._done = True
            view._rebuild()
            scroll.mount(view)
        scroll.scroll_end(animate=False)

    # --- engine driver ---------------------------------------------------

    async def _drive(self) -> None:
        try:
            async for event in self.engine.run():
                self.recorder.feed(event)
                await self._handle(event)
        finally:
            self._finished = True
            self._current = None
            self._update_status()

    async def _handle(self, event) -> None:
        scroll = self.query_one("#transcript", VerticalScroll)
        # Auto-follow only if the user is already pinned to the bottom. The
        # moment they scroll up to read history, we stop yanking them down; when
        # they scroll back to the bottom, following resumes on its own.
        follow = scroll.scroll_offset.y >= scroll.max_scroll_y - 2

        if isinstance(event, Injected):
            await scroll.mount(InjectedView(event.text))
        elif isinstance(event, TurnStarted):
            view = TurnView(
                event.speaker,
                self._color.get(event.speaker, "white"),
                self.engine.show_thinking,
            )
            await scroll.mount(view)
            self._current = view
            self._update_status()
        elif isinstance(event, ThinkingDelta):
            if self._current is not None:
                self._current.append_thinking(event.text)
        elif isinstance(event, TextDelta):
            if self._current is not None:
                self._current.append_text(event.text)
        elif isinstance(event, ToolCall):
            if self._current is not None:
                shown = format_call(event.tool, event.arguments)
                self._current.add_tool(f"⚙ {event.tool}({shown})")
        elif isinstance(event, ToolResult):
            if self._current is not None:
                mark = "↳" if event.ok else "✗"
                self._current.add_tool(f"   {mark} {event.summary}")
        elif isinstance(event, RetryNotice):
            if self._current is not None:
                self._current.add_notice(
                    f"{event.reason}; retrying in {event.delay:.0f}s "
                    f"(attempt {event.attempt})"
                )
        elif isinstance(event, TurnError):
            if self._current is not None:
                self._current.set_error(event.message)
                self._current.finalize()
            self._current = None
            self._update_status()
        elif isinstance(event, TurnComplete):
            if self._current is not None:
                summary = format_usage(event.usage)
                if summary:
                    self._current.add_notice(summary)
                self._current.finalize()
            self._current = None
            self._update_status()

        if follow:
            scroll.scroll_end(animate=False)

    # --- status bar ------------------------------------------------------

    def _spinning(self) -> bool:
        """A turn is in flight (streaming or waiting on a retry)."""
        return self._current is not None and not self._finished

    def _tick_spinner(self) -> None:
        if self._spinning():
            self._spinner_i = (self._spinner_i + 1) % len(self._spinner_frames)
            self._update_status()

    def _update_status(self) -> None:
        from textual.css.query import NoMatches

        try:
            status = self.query_one("#status", Static)
        except NoMatches:
            return  # widgets gone (app shutting down); nothing to update

        if self._finished:
            mode = "ended"
        elif self.engine.step_mode:
            mode = "step"
        elif self.engine.is_paused():
            mode = "paused"
        else:
            mode = "running"
        thinking = "on" if self.engine.show_thinking else "off"
        turns = sum(1 for t in self.engine.transcript if t.speaker != self.engine.moderator_name)
        spin = f"{self._spinner_frames[self._spinner_i]} " if self._spinning() else ""
        text = f"{spin}{mode} · turn {turns}/{self.engine.max_turns} · thinking {thinking}"
        if not self._finished:
            text += " · n=next, s=auto" if self.engine.step_mode else " · s=step"
        status.update(text)

    # --- actions ---------------------------------------------------------

    def action_toggle_pause(self) -> None:
        if self.engine.step_mode or self._finished:
            return
        if self.engine.is_paused():
            self.engine.resume()
        else:
            self.engine.pause()
        self._update_status()

    def action_toggle_step(self) -> None:
        self.engine.toggle_step_mode()
        self._update_status()

    def action_next_turn(self) -> None:
        if self.engine.step_mode:
            self.engine.step()

    def action_focus_inject(self) -> None:
        self.set_focus(self.query_one("#inject", Input))

    def action_toggle_thinking(self) -> None:
        self.engine.show_thinking = not self.engine.show_thinking
        self._update_status()

    def action_export(self) -> None:
        self.push_screen(ExportModal(), self._do_export)

    def _do_export(self, choice: str | None) -> None:
        if not choice:
            return
        fmt = "json" if choice.startswith("json") else "md"
        thinking = choice.endswith("t")
        base = os.path.splitext(self.recorder.path)[0]
        suffix = ".thinking" if thinking else ""
        out = f"{base}{suffix}.{'json' if fmt == 'json' else 'md'}"
        try:
            text = export_session(self.recorder.record, fmt, thinking)
            with open(out, "w", encoding="utf-8") as f:
                f.write(text)
            self.notify(f"exported → {out}")
        except Exception as exc:  # surface, don't crash the session
            self.notify(f"export failed: {exc}", severity="error")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.engine.inject(event.value)
        inp = self.query_one("#inject", Input)
        inp.value = ""
        self.set_focus(self.query_one("#transcript", VerticalScroll))

    def on_key(self, event) -> None:
        if event.key == "escape" and self.focused is self.query_one("#inject", Input):
            self.set_focus(self.query_one("#transcript", VerticalScroll))

    def action_quit(self) -> None:
        self.engine.stop()
        self.exit()


def run(
    conv: Conversation,
    initial_transcript: list[Turn] | None = None,
    start_turn: int = 0,
    seed_turns: list[dict] | None = None,
) -> None:
    DialecticusApp(
        conv,
        initial_transcript=initial_transcript,
        start_turn=start_turn,
        seed_turns=seed_turns,
    ).run()
