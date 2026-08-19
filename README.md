# Claude Code API Gateway

OpenAI-compatible API gateway for Claude Code CLI.
This is a fork based on codingworkflow's claude-code-api with the following additional functionalities:
- Support for json_schema
- Support for function tools
- Support for multi-turn conversation history, including agent tool loops
- Native tool calling via the Claude Agent SDK
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

These follow from wrapping the Claude Code CLI/Agent SDK, which is a coding agent rather
than a completions endpoint:

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
| `CONVERSATION_HISTORY` | `flatten` | `off` sends only the last user message. `flatten` renders the whole array into the prompt every turn. `resume` continues the Claude session and sends only the new messages. |
| `CONVERSATION_HISTORY_MAX_CHARS` | `200000` | Cap on the rendered history. Oldest messages are dropped first and the prompt says so; the newest turn is never truncated. `0` disables the cap. |

### The Claude Agent SDK engine

The server runs entirely on the Claude Agent SDK, in-process - there is no
`claude -p` subprocess and no CLI-emulation path. The caller's `tools` are
registered with Claude as real in-process MCP tools rather than emulated with
a JSON envelope:

- a tool call arrives as a genuine tool use, so the model cannot fail to find a
  tool that is actually in its toolset;
- each tool keeps its own JSON Schema, instead of collapsing to
  `additionalProperties: true` once there is more than one tool;
- tool schemas never sit in the system prompt, so a large toolset costs nothing
  there;
- `tools` and `response_format` are independent channels, so a request can get
  a tool call on one turn and a schema-conformant answer on the next.

Claude's own built-in tools are switched off, so only the caller's tools can
run. Streaming, `/v1/responses`, `response_format`, conversation history and
reasoning effort are all supported.

`tool_choice` has no SDK equivalent - there is no way to oblige the model to
reach for a tool it was offered - so a demand for one is stated in the system
prompt instead, and a turn that ends without the call is retried in the same
session:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SDK_REQUIRED_TOOL_ATTEMPTS` | `2` | Extra follow-up messages a session gets, in the same conversation, when `tool_choice` demanded a call and the model answered without making it. `0` disables retrying. The final attempt is released to the caller either way - as a tool call if it made one, otherwise as whatever the model said - rather than the caller getting an empty reply. |
| `SDK_ISOLATE_SETTINGS` | `true` | Whether the engine runs isolated from this host's own Claude Code config: no `~/.claude` or project `CLAUDE.md`, no `settings.json` (which can otherwise silently override the effort a request asked for), and no MCP servers beyond the client tools this engine registers itself. Set to `false` for local debugging against your own Claude Code setup. |

Authentication follows the SDK: `ANTHROPIC_API_KEY` when set, otherwise the
signed-in Claude Code session. Note that Anthropic's Agent SDK documentation
states that third-party developers may not offer claude.ai login for their
products without prior approval, and directs them to API key authentication.

### Resuming instead of replaying

`CONVERSATION_HISTORY=resume` continues the conversation Claude already holds
rather than rebuilding it, so a long chat costs one message per turn instead of
the whole transcript.

OpenAI clients carry no session id, so the conversation is identified by
content: each turn is fingerprinted and the resulting chain locates the session
that already holds this exact prefix. Resuming is abandoned, and the transcript
replayed instead, whenever that cannot be proven - the client edited, trimmed or
reordered history, the system prompt changed, or the session is simply unknown.
Every such decision is logged with a reason.

If Claude no longer has the session, the SDK fails loudly with
`No conversation found with session ID`; the gateway catches that, drops its
record and retries with the full transcript, so the answer is still correct.


## Bug Reports & Support

For `claude-code-api` and `ai-code-fusion`, use this discussion thread for bug reports and troubleshooting:

- https://github.com/codingworkflow/ai-code-fusion/discussions/151

## Developer Docs

For engineering workflows and internal commands:
- `docs/dev.md`

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
