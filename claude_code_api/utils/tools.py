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

# Where each caller sub-schema ends up inside the envelope. Local `$ref`s are
# rebased onto these pointers, so a sub-schema keeps referring to itself rather
# than to the envelope root.
_CONTENT_POINTER = "#/properties/content"
_ARGUMENTS_POINTER = f"#/properties/{TOOL_CALLS_KEY}/items/properties/arguments"

# Meaningless once a sub-schema is embedded, and actively harmful: `$id` would
# re-base every local ref onto a different document.
_EMBED_STRIPPED_KEYS = ("$id", "$schema")


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


def _rebase_refs(node: Any, base: str) -> Any:
    """Repoint local `$ref`s at the sub-schema's new home inside the envelope."""
    if isinstance(node, dict):
        rebased: Dict[str, Any] = {}
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                if value == "#":
                    # OpenAI's documented root recursion.
                    value = base
                elif value.startswith("#/"):
                    value = f"{base}{value[1:]}"
                # `#anchor` refs are resolved by anchor, not by location, so
                # they survive the move untouched.
            else:
                value = _rebase_refs(value, base)
            rebased[key] = value
        return rebased
    if isinstance(node, list):
        return [_rebase_refs(item, base) for item in node]
    return node


def _embed(schema: Dict[str, Any], base: str) -> Dict[str, Any]:
    """Prepare a caller sub-schema to sit at `base` inside the envelope.

    A caller schema is written as its own document: `$defs` at the top and refs
    like ``#/$defs/Foo`` pointing at them. Embedded as-is, those refs would
    resolve against the envelope root and hit nothing. Rebasing them onto the
    embedding site keeps every definition where the caller put it - no hoisting,
    no renaming, and no way for two schemas to collide over a shared name.
    """
    embedded = _rebase_refs(schema, base)
    if not isinstance(embedded, dict):
        return embedded
    for key in _EMBED_STRIPPED_KEYS:
        embedded.pop(key, None)
    return embedded


def _content_schema(bridge: ToolBridge) -> Dict[str, Any]:
    """Schema for the envelope's `content` slot."""
    if bridge.content_schema is None:
        return {
            "type": "string",
            "description": "Plain-text answer, used only when no tool applies.",
        }

    inner = _embed(bridge.content_schema, _CONTENT_POINTER)
    inner.setdefault(
        "description",
        "Answer matching the response format the client requested. Used only "
        "when no tool applies.",
    )
    return inner


def _build_schema(bridge: ToolBridge) -> Dict[str, Any]:
    single = bridge.single_tool

    if single is not None:
        arguments_schema = _embed(single.parameters, _ARGUMENTS_POINTER)
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
        properties["content"] = _content_schema(bridge)
        required = []

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
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


def _normalize_arguments(arguments: Any) -> Dict[str, Any]:
    """Coerce whatever the model produced into an arguments object.

    OpenAI puts tool arguments on the wire as a JSON *string*, so a model may
    well hand one back that way. Clients do `json.loads(arguments)` and expect a
    mapping, so anything that is not an object has to become one.
    """
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"value": arguments}
        arguments = parsed

    if isinstance(arguments, dict):
        return arguments
    if arguments is None:
        return {}
    return {"value": arguments}


def _tool_call_payload(name: str, arguments: Any) -> Dict[str, Any]:
    arguments = _normalize_arguments(arguments)
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
    # A forced envelope has no `content` slot, so only the calls key identifies
    # one - otherwise a tool that declares its own `content` argument would be
    # mistaken for an envelope.
    is_envelope = TOOL_CALLS_KEY in payload or (
        "content" in payload and not bridge.force_tool_call
    )

    if not is_envelope and bridge.force_tool_call:
        recovered = _recover_bare_arguments(payload, bridge)
        if recovered is not None:
            return None, [recovered]

    content = _envelope_content(payload, bridge)

    tool_calls: List[Dict[str, Any]] = []
    for raw_call in raw_calls or []:
        parsed = _parse_call(raw_call)
        if parsed is not None:
            tool_calls.append(parsed)

    if not tool_calls and content is None and not is_envelope:
        # Not an envelope at all - surface the raw payload rather than an empty
        # message. A genuine envelope that is simply empty stays empty: echoing
        # its JSON back as the assistant's answer would be worse than nothing.
        return result_text.strip(), []

    return content, tool_calls


def _envelope_content(payload: Dict[str, Any], bridge: ToolBridge) -> Optional[str]:
    """The envelope's `content` slot, as the text the client will receive."""
    content = payload.get("content")
    if content is None:
        return None

    if isinstance(content, str) and bridge.content_schema is None:
        return content.strip() or None

    # Under a caller `response_format` the content is a JSON value, and has to
    # reach the client as JSON text - quoting included, exactly as the
    # schema-only path delivers it.
    return json.dumps(content, separators=(",", ":"), ensure_ascii=False)


def _parse_call(raw_call: Any) -> Optional[Dict[str, Any]]:
    """One entry of the envelope's tool call list, in either shape."""
    if not isinstance(raw_call, dict):
        return None

    # The model may reproduce OpenAI's own nested wire format instead of the
    # flat {name, arguments} the schema asks for.
    function = raw_call.get("function")
    if isinstance(function, dict):
        raw_call = function

    name = raw_call.get("name")
    if not isinstance(name, str) or not name:
        return None
    return _tool_call_payload(name, raw_call.get("arguments"))


def _recover_bare_arguments(
    payload: Dict[str, Any], bridge: ToolBridge
) -> Optional[Dict[str, Any]]:
    """Treat a bare object as the arguments of the single forced tool.

    Only when it actually looks like those arguments: a refusal or an error
    payload that happens to be JSON must not become a fabricated tool call.
    """
    single = bridge.single_tool
    if single is None or not payload:
        return None

    declared = single.parameters.get("properties")
    if isinstance(declared, dict) and declared:
        if not any(key in declared for key in payload):
            logger.warning(
                "Discarding non-envelope JSON that does not match the tool's "
                "arguments",
                tool=single.name,
            )
            return None

    logger.info("Recovered bare arguments object as a tool call", tool=single.name)
    return _tool_call_payload(single.name, payload)
