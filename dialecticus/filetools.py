"""Read-only file access for personas, sandboxed to a single directory.

When a conversation sets `file_access.directory`, every persona is given three
tools, `list_files`, `read_file`, and `search`, scoped to that one directory.
Access is read-only and confined to the directory: paths that try to escape it
(via `..` or a symlink that points outside) are refused. Nothing here ever writes.

The toolset mirrors the locate-then-read workflow: `search` finds the lines that
matter across the whole directory, then `read_file` pages a precise line range
(every line numbered) so a large document is read in slices rather than whole.

The tool *schema* is provider-neutral; `anthropic_tools()` / `openai_tools()`
translate it into each SDK's expected shape, the same way the adapters translate
streaming events.
"""

from __future__ import annotations

import os

# Caps so a huge file or a sprawling tree can never blow the context budget.
MAX_READ_BYTES = 64 * 1024       # max bytes returned by a single read_file call
MAX_READ_LINES = 400             # default line span when read_file gets no limit
MAX_FILE_BYTES = 8 * 1024 * 1024  # max bytes scanned from any one file
MAX_LIST_ENTRIES = 1000
MAX_SEARCH_MATCHES = 100         # cap on search hits returned at once
MAX_MATCH_CHARS = 200            # trim each search hit line to keep results compact


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class FileSandbox:
    """A directory exposed to personas for read-only access only."""

    def __init__(self, root: str) -> None:
        # realpath resolves symlinks on the root itself, so the containment
        # check below compares like with like.
        self.root = os.path.realpath(root)
        if not os.path.isdir(self.root):
            raise ValueError(f"file_access directory not found: {root}")

    def _resolve(self, rel_path: str) -> str:
        """Resolve a requested path and confirm it stays inside the root."""
        candidate = os.path.realpath(os.path.join(self.root, rel_path))
        # After resolving symlinks and `..`, the real path must be the root or
        # live beneath it. The os.sep guard stops a sibling like `/root-evil`
        # from matching `/root` by prefix.
        if candidate != self.root and not candidate.startswith(self.root + os.sep):
            raise ValueError("path escapes the allowed directory")
        return candidate

    def _read_lines(self, full_path: str) -> list[str]:
        """Read a file as UTF-8 text and split into lines, scan-capped."""
        with open(full_path, "rb") as f:
            data = f.read(MAX_FILE_BYTES)
        return data.decode("utf-8", errors="replace").splitlines()

    def list_files(self) -> str:
        entries: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames.sort()  # deterministic ordering
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, self.root)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    size = 0
                entries.append(f"{rel} ({size} bytes)")
                if len(entries) >= MAX_LIST_ENTRIES:
                    entries.append("… (listing truncated)")
                    return "\n".join(entries)
        return "\n".join(entries) if entries else "(directory is empty)"

    def read_file(
        self, rel_path: str, offset: int | None = None, limit: int | None = None
    ) -> str:
        """Read a line range of a file, every line prefixed with its number.

        `offset` is a 1-based start line (default 1); `limit` is how many lines to
        return (default MAX_READ_LINES). The returned text is also capped at
        MAX_READ_BYTES, and a trailing note tells the model how to continue paging.
        """
        path = self._resolve(rel_path)
        if not os.path.isfile(path):
            raise ValueError(f"not a file: {rel_path}")
        lines = self._read_lines(path)
        total = len(lines)
        if total == 0:
            return "(file is empty)"

        start = 1 if offset is None else max(1, offset)
        span = MAX_READ_LINES if limit is None else max(1, limit)
        if start > total:
            return f"(offset {start} is past end of file; it has {total} line(s))"
        end = min(total, start + span - 1)

        out: list[str] = []
        used = 0
        byte_capped = False
        i = start
        while i <= end:
            rendered = f"{i:>6}  {lines[i - 1]}"
            cost = len(rendered.encode("utf-8")) + 1
            if out and used + cost > MAX_READ_BYTES:
                byte_capped = True
                end = i - 1
                break
            out.append(rendered)
            used += cost
            i += 1

        body = "\n".join(out)
        if byte_capped:
            body += f"\n(stopped at {MAX_READ_BYTES} bytes; continue with offset={end + 1})"
        elif end < total:
            body += f"\n({total - end} more line(s) below; continue with offset={end + 1})"
        return body

    def search(self, pattern: str, rel_path: str | None = None) -> str:
        """Case-insensitive literal search; returns `file:line: text` hits.

        Searches every file in the directory by default; pass `rel_path` to scan a
        single file. Use the line numbers to read around a hit with read_file.
        """
        needle = pattern.lower()
        if rel_path:
            target = self._resolve(rel_path)
            if not os.path.isfile(target):
                raise ValueError(f"not a file: {rel_path}")
            targets = [target]
        else:
            targets = []
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames.sort()
                for name in sorted(filenames):
                    targets.append(os.path.join(dirpath, name))

        matches: list[str] = []
        capped = False
        for full in targets:
            rel = os.path.relpath(full, self.root)
            try:
                lines = self._read_lines(full)
            except OSError:
                continue
            for lineno, line in enumerate(lines, 1):
                if needle in line.lower():
                    snippet = line.strip()
                    if len(snippet) > MAX_MATCH_CHARS:
                        snippet = snippet[:MAX_MATCH_CHARS] + "…"
                    matches.append(f"{rel}:{lineno}: {snippet}")
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        capped = True
                        break
            if capped:
                break

        if not matches:
            return f"(no matches for {pattern!r})"
        if capped:
            matches.append(
                f"(stopped at {MAX_SEARCH_MATCHES} matches; narrow the pattern or pass a path)"
            )
        return "\n".join(matches)

    def execute(self, name: str, arguments: dict) -> str:
        """Run a tool by name, always returning a string.

        Errors (unknown tool, bad path, escape attempt) come back as an
        `error: ...` string rather than raising, so the adapter can feed the
        message straight back to the model as the tool result.
        """
        try:
            if name == "list_files":
                return self.list_files()
            if name == "read_file":
                path = arguments.get("path")
                if not isinstance(path, str) or not path:
                    return "error: read_file requires a 'path' string"
                return self.read_file(
                    path, _as_int(arguments.get("offset")), _as_int(arguments.get("limit"))
                )
            if name == "search":
                pattern = arguments.get("pattern")
                if not isinstance(pattern, str) or not pattern:
                    return "error: search requires a 'pattern' string"
                scope = arguments.get("path")
                return self.search(
                    pattern, scope if isinstance(scope, str) and scope else None
                )
            return f"error: unknown tool {name!r}"
        except Exception as exc:  # surfaced to the model, never crashes the turn
            return f"error: {exc}"


def format_call(name: str, arguments: dict | None) -> str:
    """A short, human-readable rendering of a tool call's arguments for the UI."""
    arguments = arguments or {}
    if name == "read_file":
        path = arguments.get("path", "")
        offset = arguments.get("offset")
        limit = arguments.get("limit")
        if offset or limit:
            return f"{path}:{offset or 1}+{limit or MAX_READ_LINES}"
        return str(path)
    if name == "search":
        pattern = arguments.get("pattern", "")
        scope = arguments.get("path")
        return f'"{pattern}"' + (f" in {scope}" if scope else "")
    if name == "list_files":
        return ""
    return str(arguments.get("path", ""))


def summarize_result(name: str, arguments: dict, output: str) -> tuple[bool, str]:
    """A short, log-friendly summary of a tool result for the UI/recorder."""
    if output.startswith("error:"):
        return False, output[:120]
    if name == "read_file":
        path = arguments.get("path", "?")
        offset = arguments.get("offset")
        loc = f"{path}:{offset}" if offset else path
        return True, f"{loc} · {len(output)} chars"
    if name == "search":
        pattern = arguments.get("pattern", "?")
        if output.startswith("(no matches"):
            return True, f"'{pattern}' · 0 hits"
        n = sum(1 for ln in output.splitlines() if ln and not ln.startswith("("))
        return True, f"'{pattern}' · {n} hit(s)"
    if name == "list_files":
        n = sum(1 for ln in output.splitlines() if ln and not ln.startswith("("))
        return True, f"{n} file(s)"
    return True, "ok"


# --- provider-neutral tool schema -----------------------------------------

TOOL_SPECS = [
    {
        "name": "list_files",
        "description": (
            "List every file you may read, with byte sizes. These files live in "
            "a shared read-only directory provided for this conversation. Call "
            "this first to discover what is available before reading anything."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file from the shared read-only directory, returned "
            "with every line numbered. Pass a relative path exactly as shown by "
            "list_files. By default it returns the first lines of the file; for a "
            "long file, page through it with 'offset' (1-based start line) and "
            "'limit' (number of lines), and a trailing note tells you the next "
            "offset to continue from. To jump straight to the relevant part, use "
            "'search' first and read around the line numbers it reports. You "
            "cannot write, modify, or read anything outside this directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path of the file, as shown by list_files.",
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start reading from (default 1).",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Number of lines to return (default {MAX_READ_LINES}).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "search",
        "description": (
            "Find a literal, case-insensitive string across the shared read-only "
            "directory. Returns matching lines as 'file:line: text'. By default it "
            "searches every file; pass 'path' to restrict to one file. Use the "
            "reported line numbers with read_file's offset/limit to read the "
            "surrounding context. This is the fast way to locate where a term, "
            "author, or claim appears without reading whole documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Literal text to find (case-insensitive).",
                },
                "path": {
                    "type": "string",
                    "description": "Optional: restrict the search to this one file.",
                },
            },
            "required": ["pattern"],
        },
    },
]


def anthropic_tools() -> list[dict]:
    """TOOL_SPECS in the Anthropic Messages `tools` shape."""
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": s["parameters"],
        }
        for s in TOOL_SPECS
    ]


def openai_tools() -> list[dict]:
    """TOOL_SPECS in the OpenAI Chat Completions `tools` shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        }
        for s in TOOL_SPECS
    ]
