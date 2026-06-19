"""Derive portable views from a saved session record.

Two formats (Markdown, JSON), each with or without the models' reasoning. The
input is the canonical dict produced by session.Recorder.
"""

from __future__ import annotations

import json

FORMATS = ("md", "markdown", "json")


def export_session(record: dict, fmt: str = "md", include_thinking: bool = False) -> str:
    if fmt == "json":
        return _to_json(record, include_thinking)
    if fmt in ("md", "markdown"):
        return _to_markdown(record, include_thinking)
    raise ValueError(f"unknown export format: {fmt!r} (use one of {FORMATS})")


def _to_json(record: dict, include_thinking: bool) -> str:
    turns = []
    for t in record["turns"]:
        item: dict = {"speaker": t["speaker"], "kind": t["kind"], "text": t.get("text", "")}
        if t.get("error"):
            item["error"] = t["error"]
        if include_thinking and t.get("thinking", "").strip():
            item["thinking"] = t["thinking"]
        turns.append(item)
    out = {
        "kickoff": record.get("kickoff"),
        "created_at": record.get("created_at"),
        "personas": [
            {"name": p["name"], "provider": p["provider"], "model": p["model"]}
            for p in record.get("personas", [])
        ],
        "turns": turns,
    }
    return json.dumps(out, ensure_ascii=False, indent=2)


def _to_markdown(record: dict, include_thinking: bool) -> str:
    lines: list[str] = ["# dialecticus conversation", ""]
    if record.get("kickoff"):
        lines += [f"> {record['kickoff']}", ""]
    if record.get("personas"):
        models = ", ".join(f"{p['name']} ({p['model']})" for p in record["personas"])
        lines += [f"*Participants: {models}*", ""]

    for t in record["turns"]:
        if t["kind"] == "moderator":
            lines += [f"**» {t['speaker']} (moderator):** {t.get('text', '')}", ""]
            continue
        lines += [f"## {t['speaker']}", ""]
        if include_thinking and t.get("thinking", "").strip():
            for ln in t["thinking"].strip().splitlines():
                lines.append(f"> {ln}")
            lines.append("")
        for call in t.get("tools", []) or []:
            from .filetools import format_call

            shown = format_call(call.get("tool", ""), call.get("arguments") or {})
            result = call.get("result") or {}
            summary = f" → {result.get('summary')}" if result.get("summary") else ""
            lines.append(f"- `{call.get('tool', '?')}({shown})`{summary}")
        if t.get("tools"):
            lines.append("")
        if t.get("text", "").strip():
            lines += [t["text"].strip(), ""]
        if t.get("error"):
            lines += [f"**✗ {t['error']}**", ""]

    return "\n".join(lines).rstrip() + "\n"
