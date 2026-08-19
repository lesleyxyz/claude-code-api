# Claude Code API Gateway

OpenAI-compatible API gateway for Claude Code CLI.
This is a fork based on codingworkflow's claude-code-api with the following additional functionalities:
- Support for json_schema
- Support for function tools
- Support for multi-turn conversation history, including agent tool loops
- Native tool calling via the Claude Agent SDK engine (default; `ENGINE=cli` for the old path)
- Support for `/v1/responses` API
- Support for reasoning/effort levels using both OpenAI/Anthropic enums
- Daily docker builds for vulnerabilities at `ghcr.io/lesleyxyz/claude-code-api:latest`
- Latest Anthropic models

This project is a wrapper around claude-code CLI such that it does not violate Anthropic's Terms of Service.

## What You Get

- OpenAI-style endpoints (`/v1/chat/completions`, `/v1/models`, sessions/projects APIs)
- Streaming and non-streaming chat completions
- Claude model aliases and fallback behavior
- Optional `model` field: if omitted, CLI default model is used

## Limitations

These follow from wrapping the Claude Code CLI, which is a coding agent rather
than a completions endpoint:

- **Conversation history is replayed, not resumed.** The CLI takes a single
  prompt, so the whole message array is rendered into it on every request and
  the client stays the source of truth (as OpenAI clients expect). Multi-turn
  requests therefore cost more input tokens than a native chat endpoint, and
  very long conversations are truncated oldest-first. Controlled by
  `CONVERSATION_HISTORY`; set it to `off` for the old last-message-only
  behaviour. A single-message request is unaffected either way.
- **`tools` are emulated, not native.** The CLI has no caller-supplied tools, so
  they are described in the system prompt and the reply is constrained with
  `--json-schema`. Note that `tool_calls[].function.arguments` must be a JSON
  *string*, as in the OpenAI schema; sending an object gets a 422.
- **No token-level streaming.** SSE chunks track whole assistant messages, so a
  single-turn answer arrives as one chunk once it is finished. The CLI's
  `--include-partial-messages` would allow finer deltas but is not wired up.
- **Sampling parameters are ignored.** `temperature`, `top_p`, `max_tokens`,
  `stop` and friends are accepted for compatibility; the CLI exposes no way to
  pass them through.
- **Claude's built-in tools run on the host.** The CLI can read files and run
  commands in the project directory, so a prompt can reach the filesystem of the
  machine running the gateway. Treat prompts as untrusted input and isolate the
  container accordingly.

## Quick Start (Linux/macOS)

```bash
git clone https://github.com/codingworkflow/claude-code-api
cd claude-code-api
make install
make start
```

Server URLs:
- API: `http://localhost:8000`
- OpenAPI docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Quick Start (Windows)

Use the provided wrappers:

```bat
make.bat install
start.bat
```

Notes:
- `start.bat` starts the API in dev mode.
- `make.bat` provides common project commands.
- Claude Code CLI support on Windows may require WSL depending on your local setup.

## Supported Models

Model config is in `claude_code_api/config/models.json`.
Override with `CLAUDE_CODE_API_MODELS_PATH`.

- `claude-opus-4-6-20260205`
- `claude-opus-4-5-20251101`
- `claude-sonnet-4-5-20250929`
- `claude-haiku-4-5-20251001`

Alias/fallback behavior:
- `model` is optional in `/v1/chat/completions`.
- `opus`, `claude-opus-latest`, `claude-opus-4-6` resolve to Opus 4.6.
- If Opus 4.6 is rejected at runtime, gateway retries once with latest configured Opus 4.5.
- If all attempted models are rejected, API returns `400` with `error.code = model_not_supported`.

## API Usage

Chat completion:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

List models:

```bash
curl http://localhost:8000/v1/models
```

Streaming:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }'
```

Tool calling:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Classify this invoice"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "classify",
        "description": "Classify a document",
        "parameters": {
          "type": "object",
          "properties": {"title": {"type": "string"}},
          "required": ["title"]
        }
      }
    }],
    "tool_choice": "required"
  }'
```

The Claude CLI has no notion of caller-supplied tools, so `tools` is emulated on
top of its `--json-schema` support: the declared tools are described in the
system prompt, output is constrained to a `{content, tool_calls}` envelope, and
the validated JSON is unpacked into standard OpenAI `tool_calls`. `tool_choice`
accepts `none`, `auto` (default), `required`/`any`, or a named function, and
`parallel_tool_calls: false` caps the response at one call. Claude's own
built-in tools are never surfaced as `tool_calls`.

Structured output:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Classify this invoice"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {"name": "doc", "schema": {"type": "object"}}
    }
  }'
```

The schema is passed to the CLI's `--json-schema` flag and the validated JSON
arrives in `message.content`.

The two are independent, as in the OpenAI API: `tools` constrains the calls,
`response_format` constrains the content message. Send both and the caller's
schema becomes the schema of the envelope's `content` slot, so the model can
either call a tool or answer in the requested shape. Each schema keeps its own
`$defs` where it declared them - local `$ref`s (including `#` root recursion)
are rebased onto the embedding site - so two schemas that define the same name
cannot collide. The exception is `tool_choice: "required"`, which leaves no
content message to constrain: `response_format` is then unreachable and the
gateway logs a warning.

## Configuration

Common settings are in `claude_code_api/core/config.py`. Every field is
settable as an environment variable of the same name (case-insensitive).

- `claude_binary_path`
- `project_root`
- `database_url`
- `require_auth`

Conversation history:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONVERSATION_HISTORY` | `flatten` | `off` sends only the last user message. `flatten` renders the whole array into the prompt. `resume` is reserved for reusing the CLI session. |
| `CONVERSATION_HISTORY_MAX_CHARS` | `200000` | Cap on the rendered history. Oldest messages are dropped first and the prompt says so; the newest turn is never truncated. `0` disables the cap. |

Engine:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENGINE` | `sdk` | `sdk` drives the Claude Agent SDK in-process, with native tool calling. `cli` spawns `claude -p` per request and parses stream-json, emulating tools with a JSON envelope. |

The `sdk` engine is the default. It registers the caller's `tools` with Claude
as real in-process MCP tools instead of emulating them, which removes the
envelope entirely:

- a tool call arrives as a genuine tool use, so the model cannot fail to find a
  tool that is actually in its toolset;
- each tool keeps its own JSON Schema, instead of collapsing to
  `additionalProperties: true` once there is more than one tool;
- tool schemas no longer sit in the system prompt (10 n8n-sized tools cost about
  56,000 characters there on the `cli` engine);
- `tools` and `response_format` become independent channels, so a request can
  get a tool call on one turn and a schema-conformant answer on the next. On the
  `cli` engine the two compete for the same envelope and `response_format` is
  dropped when a tool call is forced.

Claude's own built-in tools are switched off on this engine, so only the
caller's tools can run. Streaming, `/v1/responses`, `response_format`,
conversation history and reasoning effort all work on both engines.

Authentication follows the SDK: `ANTHROPIC_API_KEY` when set, otherwise the
signed-in Claude Code session. Note that Anthropic's Agent SDK documentation
states that third-party developers may not offer claude.ai login for their
products without prior approval, and directs them to API key authentication.

Large prompts:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PROMPT_FILE_DIR` | OS temp dir + `claude-code-api-prompts` | Scratch directory for per-request system prompt files. Keep it off any mounted volume: the files are short-lived and can hold sensitive text. |
| `PROMPT_FILE_MAX_AGE_MINUTES` | `60` | Age at which an orphaned prompt file is swept at startup. Files are normally deleted when their process ends; this only catches hard crashes. |

The conversation prompt is written to the CLI on stdin and the system prompt is
passed via `--system-prompt-file`, so neither is limited by the operating
system's command-line size (32 KB on Windows) nor visible in the process table.

## Bug Reports & Support

For `claude-code-api` and `ai-code-fusion`, use this discussion thread for bug reports and troubleshooting:

- https://github.com/codingworkflow/ai-code-fusion/discussions/151

## Developer Docs

For engineering workflows and internal commands:
- `docs/dev.md`

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
