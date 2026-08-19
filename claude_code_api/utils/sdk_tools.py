"""Expose OpenAI `tools` to the Agent SDK as real, in-process MCP tools.

This replaces the `utils.tools` envelope hack for the SDK engine. Instead of
describing the caller's tools in the system prompt and constraining the reply
with `--json-schema`, each one becomes a genuine SDK tool. The model then emits
a normal `ToolUseBlock`, which removes three failure modes at once:

* it cannot "not find" a tool that is actually in its toolset, so the
  `No such tool available` prose reply becomes unrepresentable;
* argument shapes are validated per tool, rather than collapsing to
  `additionalProperties: true` as soon as there is more than one tool;
* SDK tool schemas are deferred by tool search, so a large tool set no longer
  costs tens of thousands of system-prompt tokens on every request.

The client still executes the tools. A handler here never computes anything: it
records the call, signals the session, and waits to be abandoned when the turn
is torn down. The recorded call goes back over HTTP as `tool_calls`.
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import structlog

logger = structlog.get_logger()

# MCP namespaces every tool as mcp__{server}__{tool}. Keeping the server name
# short and fixed makes the round trip back to the client's own name trivial.
CLIENT_TOOL_SERVER = "client"
_PREFIX = f"mcp__{CLIENT_TOOL_SERVER}__"

# Returned to Claude if a handler is ever allowed to complete. In normal
# operation the turn is torn down first and the handler is cancelled.
_DEFERRED_RESULT = (
    "This tool runs on the API client. The call has been handed back to it."
)


def qualified_name(tool_name: str) -> str:
    """The MCP name Claude will use for a client tool."""
    return f"{_PREFIX}{tool_name}"


def local_name(name: str) -> str:
    """The client's own tool name, given whatever Claude called it.

    Names that are not ours are returned unchanged, so a built-in slipping
    through is visible in logs rather than silently rewritten.
    """
    return name[len(_PREFIX) :] if name.startswith(_PREFIX) else name


def is_client_tool(name: str) -> bool:
    return name.startswith(_PREFIX)


def _schema_for(tool: Any) -> Dict[str, Any]:
    """The JSON Schema for a tool's arguments.

    The Python `@tool` decorator accepts a full JSON Schema dict, so the
    caller's schema is passed through as-is - draft-07 keywords, nested
    objects, enums and all - rather than being flattened.
    """
    function = getattr(tool, "function", None)
    parameters = getattr(function, "parameters", None) if function else None
    if isinstance(parameters, dict) and parameters:
        return parameters
    # A tool with no parameters still needs a valid object schema.
    return {"type": "object", "properties": {}}


def _describe(tool: Any) -> str:
    function = getattr(tool, "function", None)
    description = getattr(function, "description", None) if function else None
    return description or ""


def _tool_name(tool: Any) -> Optional[str]:
    function = getattr(tool, "function", None)
    name = getattr(function, "name", None) if function else None
    return name or None


def build_client_tool_server(
    tools: Sequence[Any],
    on_call: Callable[[str, str, Dict[str, Any]], Awaitable[None]],
) -> Tuple[Any, List[str], List[str]]:
    """Build an in-process MCP server exposing the caller's tools.

    `on_call(local_name, qualified_name, arguments)` is awaited when Claude
    calls one. It is expected never to return: the session tears the turn down
    once it has the call, and the handler is cancelled with it.

    Returns (server, allowed_tools, names). `names` holds the client-facing
    names in declaration order.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool as sdk_tool

    handlers = []
    names: List[str] = []

    for definition in tools:
        name = _tool_name(definition)
        if not name:
            logger.warning("Skipping a tool with no function name")
            continue

        # Bind the name per iteration; a closure over the loop variable would
        # make every handler report the last tool.
        def _make_handler(tool_name: str):
            async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
                await on_call(tool_name, qualified_name(tool_name), dict(args or {}))
                # Only reached if the session chose not to tear the turn down.
                return {
                    "content": [{"type": "text", "text": _DEFERRED_RESULT}],
                    "is_error": True,
                }

            return handler

        handlers.append(
            sdk_tool(name, _describe(definition), _schema_for(definition))(
                _make_handler(name)
            )
        )
        names.append(name)

    if not handlers:
        return None, [], []

    server = create_sdk_mcp_server(
        name=CLIENT_TOOL_SERVER, version="1.0.0", tools=handlers
    )
    # A wildcard keeps the allow-list short and stable regardless of tool count.
    return server, [f"{_PREFIX}*"], names
