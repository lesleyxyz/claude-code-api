# Claude Code API Gateway

OpenAI-compatible API gateway for Claude Code CLI.
This is a fork based on codingworkflow's claude-code-api with the following additional functionalities:
- Support for json_schema
- Support for function tools
- Support for `/v1/responses` API
- Daily docker builds for vulnerabilities at `ghcr.io/lesleyxyz/claude-code-api:latest`
- Latest Anthropic models

This project is a wrapper around claude-code CLI such that it does not violate Anthropic's Terms of Service.

## What You Get

- OpenAI-style endpoints (`/v1/chat/completions`, `/v1/models`, sessions/projects APIs)
- Streaming and non-streaming chat completions
- Claude model aliases and fallback behavior
- Optional `model` field: if omitted, CLI default model is used

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

Known limitation: only the last user message reaches the CLI, so a tool *result*
posted back as a `role: "tool"` message is dropped. One-shot tool calling works;
multi-turn agent loops that feed results back do not.

## Configuration

Common settings are in `claude_code_api/core/config.py`:
- `claude_binary_path`
- `project_root`
- `database_url`
- `require_auth`

## Bug Reports & Support

For `claude-code-api` and `ai-code-fusion`, use this discussion thread for bug reports and troubleshooting:

- https://github.com/codingworkflow/ai-code-fusion/discussions/151

## Developer Docs

For engineering workflows and internal commands:
- `docs/dev.md`

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
