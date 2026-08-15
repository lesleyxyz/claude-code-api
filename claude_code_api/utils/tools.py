"""Bridge OpenAI `tools` / `tool_choice` onto the Claude CLI's --json-schema flag.

The Claude CLI has no concept of caller-supplied tools: it only knows its own
built-in toolset. What it does support is schema-constrained output
(``--json-schema``), which is enough to emulate tool calling faithfully.

The trick is to constrain the CLI to an *envelope* schema::

    {"content": ..., "tool_calls": [{"name": ..., "arguments": {...}}]}

and to describe the caller's tools in the system prompt. The validated JSON
that comes back is then unpacked into standard OpenAI `tool_calls`, so clients
such as llama_index / paperless-ngx see exactly what they expect.

The two OpenAI channels stay independent, as they are upstream: `tools`
constrains the entries in `tool_calls`, while `response_format` constrains the
`content` message. When a request carries both, the caller's schema simply
becomes the schema of the envelope's `content` slot.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger()

# tool_choice values that mean "you must call a tool".
FORCED_TOOL_CHOICES = {"required", "any"}
# tool_choice values that mean "no tools at all".
DISABLED_TOOL_CHOICES = {"none"}

# Envelope key holding the calls. Deliberately the same name the client sees on
# the way out: it is the name the model knows best for a list of
# {name, arguments}, and it keeps one name across code, logs and fixtures.
TOOL_CALLS_KEY = "tool_calls"

_GENERIC_ARGUMENTS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
}

_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_DEFS_KEYS = ("$defs", "definitions")


@dataclass
class BridgedTool:
    """A caller-supplied tool definition, normalized."""

    name: str
    description: Optional[str]
    parameters: Dict[str, Any]


@dataclass
class ToolBridge:
    """Everything needed to emulate one request's tool calling."""

    tools: List[BridgedTool]
    allowed_names: List[str]
    force_tool_call: bool
    allow_parallel: bool
    # The caller's response_format schema, when they asked for one. It becomes
    # the schema of the envelope's `content` slot.
    content_schema: Optional[Dict[str, Any]] = None
    schema: Dict[str, Any] = field(default_factory=dict)
    instructions: str = ""

    @property
    def single_tool(self) -> Optional[BridgedTool]:
        if len(self.allowed_names) != 1:
            return None
        name = self.allowed_names[0]
        return next((tool for tool in self.tools if tool.name == name), None)

    def convert_result(
        self, result_text: Optional[str]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Unpack the CLI's validated JSON result into (content, tool_calls)."""
        return convert_envelope(result_text, self)


def _normalize_tool_choice(tool_choice: Any) -> Tuple[str, Optional[str]]:
    """Return (mode, forced_name) where mode is 'none' | 'auto' | 'required'."""
    if tool_choice is None:
        return "auto", None

    if isinstance(tool_choice, str):
        value = tool_choice.strip().lower()
        if value in DISABLED_TOOL_CHOICES:
            return "none", None
        if value in FORCED_TOOL_CHOICES:
            return "required", None
        return "auto", None

    # Pydantic ToolChoice model or a raw dict from a lenient client.
    function = getattr(tool_choice, "function", None)
    if function is None and isinstance(tool_choice, dict):
        function = tool_choice.get("function")

    name = getattr(function, "name", None)
    if name is None and isinstance(function, dict):
        name = function.get("name")

    if isinstance(name, str) and name:
        return "required", name

    return "auto", None


def _normalize_tools(tools: Any) -> List[BridgedTool]:
    normalized: List[BridgedTool] = []
    for tool in tools or []:
        function = getattr(tool, "function", None)
        if function is None and isinstance(tool, dict):
            function = tool.get("function")
        if function is None:
            continue

        name = getattr(function, "name", None)
        description = getattr(function, "description", None)
        parameters = getattr(function, "parameters", None)
        if isinstance(function, dict):
            name = function.get("name")
            description = function.get("description")
            parameters = function.get("parameters")

        if not isinstance(name, str) or not name:
            continue
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}

        normalized.append(
            BridgedTool(name=name, description=description, parameters=parameters)
        )
    return normalized


def _rewrite_refs(node: Any, prefix: str) -> Any:
    """Repoint root-relative `$ref`s at their namespaced definitions."""
    if isinstance(node, dict):
        rewritten: Dict[str, Any] = {}
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                for pointer in ("#/$defs/", "#/definitions/"):
                    if value.startswith(pointer):
                        value = f"#/$defs/{prefix}{value[len(pointer):]}"
                        break
            else:
                value = _rewrite_refs(value, prefix)
            rewritten[key] = value
        return rewritten
    if isinstance(node, list):
        return [_rewrite_refs(item, prefix) for item in node]
    return node


def _extract_defs(schema: Dict[str, Any], prefix: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Hoist `$defs`/`definitions` out of a sub-schema, under a namespace.

    Nested pydantic models produce root-relative `$ref`s (``#/$defs/Foo``). Once
    the schema is embedded inside the envelope those refs only resolve if the
    definitions live at the document root. The tool schema and the caller's
    `response_format` schema are hoisted side by side, so each keeps its own
    prefix: two unrelated models that both define a `Foo` must not silently
    collapse into one definition.
    """
    inner = dict(schema)
    collected: Dict[str, Any] = {}
    for key in _DEFS_KEYS:
        value = inner.pop(key, None)
        if isinstance(value, dict):
            collected.update(value)

    if not collected:
        return inner, {}

    namespaced = {
        f"{prefix}{name}": _rewrite_refs(body, prefix)
        for name, body in collected.items()
    }
    return _rewrite_refs(inner, prefix), namespaced


def _content_schema(bridge: ToolBridge) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Schema for the envelope's `content` slot, plus any definitions to hoist."""
    if bridge.content_schema is None:
        return {
            "type": "string",
            "description": "Plain-text answer, used only when no tool applies.",
        }, {}

    inner, defs = _extract_defs(bridge.content_schema, "content_")
    inner.setdefault(
        "description",
        "Answer matching the response format the client requested. Used only "
        "when no tool applies.",
    )
    return inner, defs


def _build_schema(bridge: ToolBridge) -> Dict[str, Any]:
    single = bridge.single_tool
    root_defs: Dict[str, Any] = {}

    if single is not None:
        arguments_schema, tool_defs = _extract_defs(single.parameters, "tool_")
        root_defs.update(tool_defs)
    else:
        # With several tools in play the argument shape depends on which tool
        # the model picks, so it is described in the prompt instead.
        arguments_schema = dict(_GENERIC_ARGUMENTS_SCHEMA)

    call_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": list(bridge.allowed_names),
                "description": "Name of the tool being called.",
            },
            "arguments": arguments_schema,
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }

    tool_calls_schema: Dict[str, Any] = {
        "type": "array",
        "description": "Tool calls to hand back to the client for execution.",
        "items": call_schema,
    }
    if bridge.force_tool_call:
        tool_calls_schema["minItems"] = 1
    if not bridge.allow_parallel:
        tool_calls_schema["maxItems"] = 1

    properties: Dict[str, Any] = {TOOL_CALLS_KEY: tool_calls_schema}
    required = [TOOL_CALLS_KEY]

    if not bridge.force_tool_call:
        # Only leave room for an answer when the client did not demand a call.
        content_schema, content_defs = _content_schema(bridge)
        root_defs.update(content_defs)
        properties["content"] = content_schema
        required = []

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    if root_defs:
        schema["$defs"] = root_defs
    return schema


def _build_instructions(bridge: ToolBridge) -> str:
    lines = [
        "## Client-supplied tools",
        "",
        "The API client has declared the tools below, and they are the only "
        "tools available for this request. You call them through the "
        f'"{TOOL_CALLS_KEY}" field of your response. You cannot execute them '
        "yourself - you can only emit calls, which the client will run. Do not "
        "use your own built-in tools (file, search, or otherwise) to satisfy "
        "this request, and do not look for these tools in your own toolset: "
        "they are not there. Answer from the conversation alone.",
        "",
    ]

    for tool in bridge.tools:
        if tool.name not in bridge.allowed_names:
            continue
        lines.append(f"### {tool.name}")
        if tool.description:
            lines.append(tool.description)
        lines.append("Arguments JSON Schema:")
        lines.append(json.dumps(tool.parameters, ensure_ascii=False))
        lines.append("")

    lines.append("## Response format")
    lines.append("")
    lines.append(
        "Reply with a single JSON object matching the required output schema. "
        f'Each entry in "{TOOL_CALLS_KEY}" needs a "name" (one of: '
        f"{', '.join(bridge.allowed_names)}) and an \"arguments\" object that "
        "validates against that tool's schema above."
    )
    if bridge.force_tool_call:
        lines.append(
            "You MUST emit at least one tool call. Do not answer in prose, and "
            "do not explain that a tool is unavailable."
        )
    elif bridge.content_schema is not None:
        lines.append(
            f'If no tool applies, leave "{TOOL_CALLS_KEY}" empty and put your '
            'answer in "content", matching the schema the client requested for '
            "it:"
        )
        lines.append(json.dumps(bridge.content_schema, ensure_ascii=False))
    else:
        lines.append(
            f'If no tool applies, leave "{TOOL_CALLS_KEY}" empty and put your '
            'answer in "content".'
        )
    if not bridge.allow_parallel:
        lines.append("Emit at most one tool call.")

    return "\n".join(lines)


def build_tool_bridge(
    request: Any, content_schema: Optional[Dict[str, Any]] = None
) -> Optional[ToolBridge]:
    """Build a ToolBridge for a chat request, or None if tools are not in play.

    `content_schema` is the caller's `response_format` schema, if any. With no
    tools in play there is no envelope and it is passed to the CLI verbatim, so
    this returns None and the caller keeps using it as-is.
    """
    tools = _normalize_tools(getattr(request, "tools", None))
    if not tools:
        return None

    mode, forced_name = _normalize_tool_choice(getattr(request, "tool_choice", None))
    if mode == "none":
        return None

    names = [tool.name for tool in tools]
    if forced_name is not None:
        if forced_name not in names:
            logger.warning(
                "tool_choice names an undeclared tool; falling back to all tools",
                tool_choice=forced_name,
                declared=names,
            )
        else:
            names = [forced_name]

    parallel = getattr(request, "parallel_tool_calls", None)
    bridge = ToolBridge(
        tools=tools,
        allowed_names=names,
        force_tool_call=mode == "required",
        allow_parallel=parallel is not False,
        content_schema=content_schema,
    )
    bridge.schema = _build_schema(bridge)
    bridge.instructions = _build_instructions(bridge)
    return bridge


def _loads_envelope(result_text: str) -> Optional[Any]:
    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        pass

    # The CLI validates against the schema, but a stray markdown fence or a
    # preamble sentence is cheap to recover from.
    match = _FENCED_JSON.search(result_text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = result_text.find("{")
    end = result_text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(result_text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _tool_call_payload(name: str, arguments: Any) -> Dict[str, Any]:
    if not isinstance(arguments, (dict, list)):
        arguments = {} if arguments is None else {"value": arguments}
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments, separators=(",", ":"), ensure_ascii=False
            ),
        },
    }


def convert_envelope(
    result_text: Optional[str], bridge: ToolBridge
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Turn the CLI's validated JSON into (content, tool_calls)."""
    if not result_text or not result_text.strip():
        return None, []

    payload = _loads_envelope(result_text)
    if payload is None:
        logger.warning("Tool envelope was not valid JSON; passing through as content")
        return result_text.strip(), []

    if not isinstance(payload, dict):
        return result_text.strip(), []

    raw_calls = payload.get(TOOL_CALLS_KEY)

    if raw_calls is None and bridge.force_tool_call:
        # The model skipped the envelope and returned the arguments directly.
        single = bridge.single_tool
        if single is not None:
            logger.info("Recovered bare arguments object as a tool call")
            return None, [_tool_call_payload(single.name, payload)]

    content = payload.get("content")
    if isinstance(content, str):
        content = content.strip() or None
    elif content is not None:
        # A structured `content` (the caller's response_format) has to reach the
        # client as JSON text, exactly as the schema-only path delivers it.
        content = json.dumps(content, separators=(",", ":"), ensure_ascii=False)

    tool_calls: List[Dict[str, Any]] = []
    for raw_call in raw_calls or []:
        if not isinstance(raw_call, dict):
            continue
        name = raw_call.get("name")
        if not isinstance(name, str) or not name:
            continue
        tool_calls.append(_tool_call_payload(name, raw_call.get("arguments")))

    if not tool_calls and content is None:
        # Nothing usable in the envelope - surface the raw payload rather than
        # returning an empty message.
        return result_text.strip(), []

    return content, tool_calls
