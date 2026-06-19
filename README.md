# dialecticus

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Textual](https://img.shields.io/badge/Textual-TUI-important)
![Anthropic](https://img.shields.io/badge/Anthropic-API-8A2BE2)
![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-brightgreen)
![Prompt caching](https://img.shields.io/badge/Prompt_Caching-Anthropic-8A2BE2)
![GitHub last commit](https://img.shields.io/github/last-commit/oudeis01/dialecticus)
![License](https://img.shields.io/badge/license-MIT-green)

Watch different language models converse with each other within an identity and
topic scope you set.



https://github.com/user-attachments/assets/9bb82c05-5e6a-4ef3-abed-d4af03ec7c3f



A normalized streaming engine, two provider adapters, and a Textual TUI you can
intervene in (pause / step / inject / toggle thinking). The console renderer is
still there behind `--plain` and reads the exact same engine event stream.

## Architecture

```
config.yaml ─▶ Engine (turn loop, shared transcript, intervention gate)
                  │  builds each persona's view of the conversation
                  ▼
            ProviderAdapter        ─┬─ AnthropicAdapter   (native Messages API)
            (normalized events)     └─ OpenAIAdapter      (any OpenAI-compatible endpoint)
                  │
                  ▼  TurnStarted | ThinkingDelta | TextDelta | TurnComplete | Injected
            Textual TUI  (or console renderer with --plain)
```

Pause, single-step, moderator injection, and stop are applied at turn boundaries,
so intervening never tears a streaming reply in half.

The two adapters are the whole story for provider coverage: native Anthropic for
Claude, and one OpenAI-compatible adapter that reaches everything else (OpenAI,
OpenRouter's catalog of model makers, local servers) by changing `base_url`.

## Install

```sh
pip install -e .
```

## Run

Set the keys your personas need, then point at a YAML file:

Keys come from the environment, or from a local `.env` (loaded automatically):

```sh
cp .env.example .env        # then fill in OPENROUTER_API_KEY / OPENCODE_API_KEY / ANTHROPIC_API_KEY
dialecticus personas.openrouter-free.yaml          # interactive TUI, free models
dialecticus personas.openrouter-free.yaml --plain  # plain console stream
dialecticus personas.zen-free.yaml                 # free models via OpenCode Zen
```

`personas.openrouter-free.yaml` pairs two free OpenRouter models from different
developers (Meta Llama vs Alibaba Qwen) and only needs `OPENROUTER_API_KEY`.

`personas.zen-free.yaml` does the same through [OpenCode Zen](https://opencode.ai/zen)
(DeepSeek vs Qwen) and only needs `OPENCODE_API_KEY`. Zen's free tier tends to be
more generous on rate limits.

### Adding another OpenAI-compatible gateway

The OpenAI adapter reaches any Chat Completions endpoint, so a new gateway is just
config, not code: set `base_url` and `api_key_env` on each persona. OpenCode Zen is
wired up this way (`base_url: https://opencode.ai/zen/v1`). One caveat for Zen:
only its DeepSeek / Qwen / MiniMax / GLM / Kimi / Grok models use
`/chat/completions`; its GPT models use `/responses` and Claude models use
`/messages`, which this adapter does not drive. Its `/models` list also omits
context windows, so Zen personas fall back to the default budget unless you set
`context_length:` yourself.

## File access (read-only)

Personas can be granted read-only access to a directory of files. Point a
conversation at one directory and every persona gets three tools, `list_files`,
`read_file`, and `search`, scoped to that directory:

```yaml
file_access:
  directory: ./shared      # relative paths resolve against the config file
```

When this is set, the engine hands both adapters the tools and runs the
tool-call loop inside a turn: the model can list the directory, locate the lines
it cares about, read those line ranges, and then answer using what it read, all
within the same turn. Tool activity shows up dimmed inline, at the point in the
reply where the model made the call (`⚙ search("perspectivism")`), and is saved
into the session record and Markdown export.

The toolset mirrors a locate-then-read workflow:

- `list_files` lists every readable file with byte sizes.
- `search(pattern, path?)` does a literal, case-insensitive search and returns
  matching lines as `file:line: text`. It scans the whole directory by default;
  pass `path` to restrict it to one file.
- `read_file(path, offset?, limit?)` returns the file with every line numbered.
  `offset` (1-based start line) and `limit` (line count) page through a long
  file, and a trailing note reports the next offset to continue from. Reading
  around the line numbers `search` reports is the fast path into a large document.

Access is read-only and confined to the directory:

- There is no write/edit/delete tool; the adapters never open a file for writing.
- Paths that try to escape the directory, via `..` or a symlink pointing
  outside, are refused. Absolute paths are resolved against the directory, not
  the filesystem root.
- A single read returns at most 64 KB (and 400 lines by default), a listing at
  most 1000 entries, and a search at most 100 matches, so a large tree cannot
  blow the context budget.

Without a `file_access` block, no tools are offered and behaviour is unchanged.

## TUI controls

| Key     | Action                                                      |
| ------- | ---------------------------------------------------------- |
| `space` | pause / resume (continuous mode)                           |
| `s`     | toggle single-step mode                                    |
| `n`     | advance one turn (in step mode)                            |
| `i`     | focus the input to inject a moderator message; `enter` sends, `esc` cancels |
| `t`     | toggle thinking display (applies to the next turn onward)  |
| `q`     | quit                                                       |

The session opens with an intro panel (the participants' full system prompts and
the kickoff) and **starts in step mode**: press `n` to advance one turn, or `s`
to switch to continuous (auto) mode. The status line shows the mode (`running` /
`paused` / `step` / `ended`), the turn count against `max_turns`, whether
thinking is on, and the key to switch modes.

## Context budgeting

Models have very different context windows (free OpenRouter models alone span
32k..1M), and the transcript grows every turn. So before each turn the engine
trims to fit *that speaker's* model:

- Each persona's window is resolved once at startup: a YAML `context_length`
  override wins, else OpenRouter's live `/models` catalog, else a small map of
  known Anthropic windows, else a conservative default.
- The budget is `context_length * 0.75 - max_tokens` (leaving room for output).
- Token counts are a heuristic estimate (~4 chars/token) with that safety margin,
  so no per-turn token-counting calls are needed.
- The system prompt and the kickoff are always kept; the oldest turns are dropped
  first, and the most recent turn is always kept.

Set `context_length:` on a persona to override the resolved window.

## Rate limits and errors

Free models in particular get rate-limited often, so a 429 no longer takes the
whole session down:

- A rate limit (HTTP 429) is retried automatically. The engine honours the
  provider's `Retry-After` header, or OpenRouter's nested `retry_after_seconds`,
  and otherwise backs off ~20s, capped at 60s. The TUI shows a `⟳ retrying in Ns`
  notice on the speaker's turn while it waits; `q` aborts a pending retry.
- After `max_retries` (default 5) the turn is given up on. A failed turn shows a
  red `✗` error line, records nothing in the transcript, and the loop moves on to
  the next speaker instead of crashing.
- Non-rate-limit errors (auth, bad request, …) are shown the same way but are not
  retried.

`max_retries` and `max_retry_delay` are constructor arguments on `Engine`.

## Notes

- `show_thinking` streams reasoning **where the provider exposes it**. Anthropic
  models stream a summarized chain of thought; DeepSeek-style models stream
  `reasoning_content`; OpenAI's o-series does not expose raw reasoning at all.
- Turn order is round-robin over `personas`; the loop stops at `max_turns`.
- `max_tokens` caps a reply. Setting it to `0` or `null` (or omitting it with no
  value) means **no cap**: the OpenAI adapter drops the parameter so the model can
  finish its reply instead of being truncated mid-sentence (or mid-reasoning).
  Output is then bounded only by the model's context window. The Anthropic API
  requires a cap, so an uncapped Anthropic persona falls back to 4096. If the key
  is absent entirely, a persona defaults to 1024.
