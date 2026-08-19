"""Chat completions API endpoint - OpenAI compatible."""

import hashlib
import json
import re
from types import SimpleNamespace
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from claude_code_api.core.claude_manager import (
    ClaudeCommandTooLongError,
    ClaudeModelNotSupportedError,
    ClaudeSessionConflictError,
    create_project_directory,
)
from claude_code_api.core.config import settings
from claude_code_api.core.session_manager import SessionManager
from claude_code_api.models.claude import get_default_model, validate_claude_model
from claude_code_api.models.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    ResponsesCreateRequest,
    ResponsesResponse,
    ToolChoice,
    ToolChoiceFunction,
    ToolDefinition,
    ToolFunction,
)
from claude_code_api.utils.parser import (
    ClaudeOutputParser,
    OpenAIConverter,
    estimate_tokens,
    normalize_claude_message,
)
from claude_code_api.utils.streaming import (
    create_non_streaming_response,
    create_sse_response,
)
from claude_code_api.utils.effort import normalize_reasoning_effort
from claude_code_api.utils.engine import ENGINE_SDK
from claude_code_api.utils.sdk_prompt import resolve_system_prompt
from claude_code_api.utils.tool_choice import required_tool_names
from claude_code_api.utils.ledger import (
    MODE_RESUME,
    conversation_fingerprints,
    fingerprint_system,
    is_session_missing_error,
    plan_turn,
)
from claude_code_api.utils.history import tool_call_names as history_tool_call_names
from claude_code_api.utils.history import (
    NoConversationTurnError,
    current_turn_text,
    render_prompt,
)
from claude_code_api.utils.time import utc_timestamp
from claude_code_api.utils.tools import ToolBridge, build_tool_bridge

logger = structlog.get_logger()
router = APIRouter()

CHAT_COMPLETION_RESPONSES = {
    200: {
        "description": "Chat completion response (JSON when stream=false, SSE when stream=true).",
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/ChatCompletionResponse"}
            },
            "text/event-stream": {
                "schema": {"$ref": "#/components/schemas/ChatCompletionChunk"}
            },
        },
    },
    400: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}

RESPONSES_API_RESPONSES = {
    200: {
        "model": ResponsesResponse,
        "description": "Responses API response (JSON when stream=false, SSE when stream=true).",
        "content": {
            "text/event-stream": {"schema": {"type": "string"}},
        },
    },
    400: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}

RESPONSE_TEXT_BLOCK_TYPES = {"input_text", "output_text", "text"}
RESPONSE_INPUT_ROLES = {"system", "user", "assistant", "tool"}


def _http_error(
    status_code: int, message: str, error_type: str, code: str
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"error": {"message": message, "type": error_type, "code": code}},
    )


def _input_error(message: str, code: str = "invalid_input") -> HTTPException:
    return _http_error(
        status.HTTP_400_BAD_REQUEST,
        message,
        "invalid_request_error",
        code,
    )


async def _log_raw_request(req: Request) -> None:
    raw_body = await req.body()
    content_type = req.headers.get("content-type", "unknown")
    sensitive_headers = {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "x-auth-token",
    }
    sanitized_headers = {}
    for key, value in req.headers.items():
        if key.lower() in sensitive_headers:
            sanitized_headers[key] = "<redacted>"
        else:
            sanitized_headers[key] = value
    body_hash = hashlib.sha256(raw_body).hexdigest() if raw_body else None
    logger.info(
        "Raw request received",
        content_type=content_type,
        body_size=len(raw_body),
        user_agent=sanitized_headers.get("user-agent", "unknown"),
        headers=sanitized_headers,
        body_hash=body_hash or "empty",
    )


def _extract_json_schema(request: ChatCompletionRequest) -> Optional[Dict[str, Any]]:
    response_format = request.response_format
    if not response_format or response_format.type != "json_schema":
        return None
    if not response_format.json_schema:
        raise _http_error(
            status.HTTP_400_BAD_REQUEST,
            "response_format.type is 'json_schema' but no json_schema was provided.",
            "invalid_request_error",
            "missing_json_schema",
        )
    return response_format.json_schema.schema_


def _resolve_reasoning_effort(request: ChatCompletionRequest) -> Optional[str]:
    """Map `reasoning_effort` onto a CLI --effort level, or None if unset.

    The CLI only warns about an unknown level and then runs at its default, so
    this is the only place a bad value can be reported back to the caller.
    """
    try:
        return normalize_reasoning_effort(request.reasoning_effort)
    except ValueError as e:
        raise _input_error(str(e), "invalid_reasoning_effort") from e


def _merge_system_prompt(system_prompt: Optional[str], addition: str) -> str:
    """Append bridged-tool instructions to whatever system prompt the caller sent."""
    if not system_prompt:
        return addition
    return f"{system_prompt}\n\n{addition}"


def _apply_tool_bridge(
    request: ChatCompletionRequest,
    json_schema: Optional[Dict[str, Any]],
    system_prompt: Optional[str],
) -> Tuple[Optional[ToolBridge], Optional[Dict[str, Any]], Optional[str]]:
    """Fold `tools`/`tool_choice` into the CLI's schema + system prompt.

    Returns the bridge (None when tools are not in play) alongside the schema and
    system prompt to actually run with. With no tools the caller's
    `response_format` schema is passed through untouched; with tools it becomes
    the schema of the envelope's `content` slot, so both OpenAI channels stay
    available in the same request.
    """
    if settings.engine == ENGINE_SDK:
        # The SDK engine gives Claude the caller's tools for real, so the
        # envelope emulation is not just unnecessary here - running it would
        # constrain the reply to a JSON envelope the tools no longer need.
        #
        # What the envelope also carried, though, was `tool_choice`, and a real
        # toolset has no equivalent of it: the SDK offers tools, it cannot
        # oblige the model to reach for one. Saying so in the system prompt is
        # how the demand survives, and `SdkSession` enforces it by asking again
        # when a turn ends without the call.
        required = required_tool_names(request)
        if required:
            logger.info(
                "Requiring a client tool call on the SDK engine",
                required_tools=required,
                tool_choice=request.tool_choice,
            )
        return None, json_schema, resolve_system_prompt(system_prompt, required)

    bridge = build_tool_bridge(request, content_schema=json_schema)
    if bridge is None:
        if request.tools:
            # The caller declared tools but the bridge was skipped, so nothing
            # tells the CLI they exist. The model then emits a native tool_use,
            # the CLI answers "No such tool available", and the complaint ends
            # up as prose. Loud on purpose: it is silent otherwise.
            logger.warning(
                "Tools were declared but the bridge was not applied; the model "
                "has no channel to call them",
                tool_count=len(request.tools),
                tool_choice=request.tool_choice,
                reason=(
                    "tool_choice disables tools"
                    if request.tool_choice == "none"
                    else "no usable tool definitions"
                ),
            )
        return None, json_schema, system_prompt

    if json_schema is not None and bridge.force_tool_call:
        logger.warning(
            "response_format.json_schema cannot be honoured because tool_choice "
            "forces a tool call, which leaves no content message to constrain",
            tool_names=bridge.allowed_names,
        )

    logger.info(
        "Bridging client tools onto --json-schema",
        tool_names=bridge.allowed_names,
        forced=bridge.force_tool_call,
        parallel=bridge.allow_parallel,
        has_content_schema=json_schema is not None,
    )
    return (
        bridge,
        bridge.schema,
        _merge_system_prompt(system_prompt, bridge.instructions),
    )


# One event name for every resume-vs-new outcome, so `grep "Conversation
# continuity"` shows the decision and its reason for every turn - including the
# turns that never get as far as considering a resume.
_CONTINUITY_EVENT = "Conversation continuity"


def _plan_conversation(
    request: ChatCompletionRequest,
    session_manager: SessionManager,
    system_prompt: Optional[str],
) -> Tuple[Any, Optional[str]]:
    """Decide how to run this turn, and which session it belongs to.

    Returns (plan, session id to reuse). The session id is None when nothing
    resumable was found, in which case the caller creates one as before.

    Resuming is only attempted on the SDK engine: the CLI engine has no way to
    hand a client-executed tool result back into a running conversation.
    """
    resume_enabled = (
        settings.engine == ENGINE_SDK and settings.conversation_history == MODE_RESUME
    )
    if not resume_enabled:
        logger.info(
            _CONTINUITY_EVENT,
            decision="new",
            reason=(
                f"resume is off (engine={settings.engine}, "
                f"history={settings.conversation_history})"
            ),
        )
        return None, None

    fingerprints = conversation_fingerprints(request.messages)
    matched = session_manager.find_resumable(fingerprints)
    if matched is None:
        logger.info(
            _CONTINUITY_EVENT,
            decision="new",
            reason="no live session matches this conversation",
            message_count=len(request.messages),
        )
        return None, None

    matched_session_id, _already_sent = matched
    session_info = session_manager.active_sessions.get(matched_session_id)
    sdk_session_id = session_info.cli_session_id if session_info else None

    plan = plan_turn(
        request.messages,
        session_manager.get_ledger(matched_session_id),
        sdk_session_id,
        system_prompt,
        MODE_RESUME,
        max_chars=settings.conversation_history_max_chars,
    )
    if not plan.resuming:
        logger.info(
            _CONTINUITY_EVENT,
            decision="new",
            reason=plan.reason,
            session_id=matched_session_id,
            sdk_session_id=sdk_session_id,
        )
        return None, None

    logger.info(
        _CONTINUITY_EVENT,
        decision="resume",
        reason=plan.reason,
        session_id=matched_session_id,
        sdk_session_id=sdk_session_id,
        new_message_count=len(plan.consumed),
    )
    return plan, matched_session_id


def _assistant_echo(response: Dict[str, Any]) -> Optional[Any]:
    """The assistant message as the client will send it back next turn.

    Recorded so the following request's history lines up with what the session
    was actually told; without it every turn after the first would diverge.
    """
    choices = response.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    return SimpleNamespace(
        role="assistant",
        content=message.get("content"),
        tool_calls=message.get("tool_calls"),
        tool_call_id=None,
        name=None,
    )


def _suppress_internal_tools(
    request: ChatCompletionRequest, json_schema: Optional[Dict[str, Any]] = None
) -> bool:
    """Whether tool_use blocks in the stream should be hidden from the client.

    On the CLI engine they are Claude's own built-ins - an implementation
    detail the caller never declared. They are hidden whenever the caller sent
    tools (the emulated call arrives separately in the envelope) or asked for
    structured output (the validated payload is the whole answer).

    On the SDK engine the caller's tools ARE registered with Claude, so a
    tool_use block is exactly what the client asked for and must pass through -
    including alongside a response_format, which is an independent channel
    there rather than a competing one.
    """
    if settings.engine == ENGINE_SDK:
        return False
    return bool(request.tools) or json_schema is not None


def _extract_prompts(request: ChatCompletionRequest) -> Tuple[str, str]:
    """Render the request into the (prompt, system prompt) pair the CLI takes.

    System messages stay in --system-prompt-file rather than being folded into the
    transcript: they are instructions, not conversation, and they need to stay
    adjacent to the tool-bridge block `_merge_system_prompt` appends.
    """
    if not request.messages:
        raise _http_error(
            status.HTTP_400_BAD_REQUEST,
            "At least one message is required",
            "invalid_request_error",
            "missing_messages",
        )

    system_texts = [
        text
        for msg in request.messages
        if msg.role == "system"
        for text in (msg.get_text_content().strip(),)
        if text
    ]
    system_prompt = "\n\n".join(system_texts) if system_texts else request.system_prompt

    # Read the settings lazily: the tests mutate the singleton in place.
    try:
        user_prompt = render_prompt(
            request.messages,
            mode=settings.conversation_history,
            max_chars=settings.conversation_history_max_chars,
        )
    except NoConversationTurnError as e:
        raise _http_error(
            status.HTTP_400_BAD_REQUEST,
            str(e),
            "invalid_request_error",
            "missing_user_message",
        ) from e

    return user_prompt, system_prompt


def _coerce_response_content_block(block: Any, location: str) -> str:
    if isinstance(block, str):
        return block

    if not isinstance(block, dict):
        raise _input_error(
            f"Unsupported content block at {location}: expected an object or string.",
            "unsupported_input_block",
        )

    block_type = block.get("type")
    if block_type in RESPONSE_TEXT_BLOCK_TYPES:
        if "text" not in block:
            raise _input_error(
                f"Text content block at {location} is missing the 'text' field.",
                "invalid_input_block",
            )
        return str(block["text"])

    if block_type is None:
        if "text" in block:
            return str(block["text"])
        if "content" in block:
            return str(block["content"])

    raise _input_error(
        f"Unsupported content block type at {location}: {block_type!r}.",
        "unsupported_input_block",
    )


def _coerce_response_content(content: Any, location: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            _coerce_response_content_block(block, f"{location}.content[{index}]")
            for index, block in enumerate(content)
        ]
        return "\n".join(part for part in text_parts if part)
    if isinstance(content, dict):
        return _coerce_response_content_block(content, f"{location}.content")

    raise _input_error(
        f"Unsupported content at {location}: expected a string or content block array.",
        "unsupported_input_content",
    )


def _coerce_response_role(role: Any, location: str) -> str:
    if not isinstance(role, str) or not role:
        raise _input_error(
            f"Message at {location} is missing a valid 'role'.",
            "invalid_input_message",
        )

    if role == "developer":
        return "system"

    if role not in RESPONSE_INPUT_ROLES:
        raise _input_error(
            f"Unsupported message role at {location}: {role!r}.",
            "unsupported_input_role",
        )

    return role


def _responses_input_to_chat_messages(input_value: Any) -> List[Dict[str, Any]]:
    if isinstance(input_value, str):
        return [{"role": "user", "content": input_value}]

    if not isinstance(input_value, list):
        raise _input_error(
            "The 'input' field must be a string or an array of message objects.",
            "invalid_input",
        )

    messages: List[Dict[str, Any]] = []
    for index, item in enumerate(input_value):
        location = f"input[{index}]"
        if not isinstance(item, dict):
            raise _input_error(
                f"Message at {location} must be an object.",
                "invalid_input_message",
            )

        item_type = item.get("type")

        if item_type == "function_call":
            # The model's own previous tool call. Chat Completions carries it as
            # an assistant message, and parallel calls share one message, so a
            # run of these items folds into the message the first one opened.
            call = _responses_function_call_to_chat(item, location)
            if messages and messages[-1].get("_from_function_call"):
                messages[-1]["tool_calls"].append(call)
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [call],
                        "_from_function_call": True,
                    }
                )
            continue

        if item_type == "function_call_output":
            messages.append(_responses_function_output_to_chat(item, location))
            continue

        if item_type not in (None, "message"):
            raise _input_error(
                f"Unsupported input item type at {location}: {item_type!r}.",
                "unsupported_input_item",
            )

        role = _coerce_response_role(item.get("role"), location)
        content = _coerce_response_content(item.get("content"), location)
        message: Dict[str, Any] = {"role": role, "content": content}

        for optional_field in ("name", "tool_call_id", "tool_calls"):
            if optional_field in item:
                message[optional_field] = item[optional_field]

        messages.append(message)

    for message in messages:
        message.pop("_from_function_call", None)

    return messages


def _responses_required_field(item: Dict[str, Any], field: str, location: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value:
        raise _input_error(
            f"Item at {location} is missing a valid {field!r}.",
            "invalid_input_item",
        )
    return value


def _responses_function_call_to_chat(
    item: Dict[str, Any], location: str
) -> Dict[str, Any]:
    """One `function_call` input item as a Chat Completions tool call."""
    arguments = item.get("arguments")
    return {
        "id": _responses_required_field(item, "call_id", location),
        "type": "function",
        "function": {
            "name": _responses_required_field(item, "name", location),
            # Both APIs carry arguments as a JSON string, so it is passed
            # through rather than parsed and re-encoded.
            "arguments": arguments if isinstance(arguments, str) else "{}",
        },
    }


def _responses_function_output_to_chat(
    item: Dict[str, Any], location: str
) -> Dict[str, Any]:
    """One `function_call_output` input item as a Chat Completions tool message."""
    output = item.get("output")
    return {
        "role": "tool",
        "tool_call_id": _responses_required_field(item, "call_id", location),
        "content": output if isinstance(output, str) else json.dumps(output),
    }


def _responses_tools_to_chat_tools(
    tools: Optional[List[Any]],
) -> Optional[List[ToolDefinition]]:
    """Flattened Responses tools as the nested Chat Completions equivalents."""
    if not tools:
        return None

    converted = [
        ToolDefinition(
            function=ToolFunction(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
        )
        for tool in tools
    ]
    return converted or None


def _responses_tool_choice_to_chat(
    tool_choice: Any,
) -> Optional[Union[str, ToolChoice]]:
    """A Responses `tool_choice` as the Chat Completions equivalent.

    The strings are identical in both APIs; only the named form differs, having
    no `function` wrapper there.
    """
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice

    name = getattr(tool_choice, "name", None)
    if isinstance(name, str) and name:
        return ToolChoice(function=ToolChoiceFunction(name=name))
    return None


def _responses_request_to_chat_request(
    request: ResponsesCreateRequest, stream: bool = False
) -> ChatCompletionRequest:
    messages = _responses_input_to_chat_messages(request.input)
    system_prompt = request.instructions if request.instructions else None

    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_output_tokens,
        stream=stream,
        project_id=request.project_id,
        session_id=request.session_id,
        system_prompt=system_prompt,
        reasoning_effort=request.reasoning.effort if request.reasoning else None,
        tools=_responses_tools_to_chat_tools(request.tools),
        tool_choice=_responses_tool_choice_to_chat(request.tool_choice),
        parallel_tool_calls=request.parallel_tool_calls,
    )


def _extract_chat_response_text(chat_response: Dict[str, Any]) -> str:
    choices = chat_response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""

    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def _extract_chat_response_tool_calls(
    chat_response: Dict[str, Any],
) -> List[Dict[str, Any]]:
    choices = chat_response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return []

    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        return []

    calls = message.get("tool_calls") or []
    return [call for call in calls if isinstance(call, dict)]


def _responses_message_item(message_id: str, output_text: str) -> Dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": output_text,
                "annotations": [],
            }
        ],
    }


def _responses_function_call_item(
    call: Dict[str, Any], item_id: Optional[str] = None
) -> Dict[str, Any]:
    """One Chat Completions tool call as a Responses `function_call` item.

    `call_id` keeps the chat call's own id, so the id the client quotes back in
    a `function_call_output` item is the one the tool message needs.
    """
    function = call.get("function") or {}
    return {
        "id": item_id or f"fc_{uuid.uuid4().hex}",
        "type": "function_call",
        "status": "completed",
        "call_id": call.get("id") or f"call_{uuid.uuid4().hex}",
        "name": function.get("name") or "",
        "arguments": function.get("arguments") or "{}",
    }


def _responses_output_items(
    message_id: str,
    output_text: str,
    tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """The `output` array, shared by the streaming and non-streaming paths.

    A message item is emitted whenever there is text, and for a turn with no
    tool calls even when the text is empty - that empty message is what a
    text-only client expects to find. A turn that only calls tools carries just
    the calls, which is what the Responses API does.
    """
    items: List[Dict[str, Any]] = []
    if output_text or not tool_calls:
        items.append(_responses_message_item(message_id, output_text))
    items.extend(_responses_function_call_item(call) for call in tool_calls)
    return items


def _responses_usage_from_chat(chat_response: Dict[str, Any]) -> Dict[str, Any]:
    usage = chat_response.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}

    return {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _chat_response_to_responses_response(
    request: ResponsesCreateRequest, chat_response: Dict[str, Any]
) -> Dict[str, Any]:
    created_at = chat_response.get("created") or utc_timestamp()
    completed_at = utc_timestamp()
    output_text = _extract_chat_response_text(chat_response)
    tool_calls = _extract_chat_response_tool_calls(chat_response)

    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "completed_at": completed_at,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": request.max_output_tokens,
        "model": chat_response.get("model") or request.model,
        "output": _responses_output_items(
            f"msg_{uuid.uuid4().hex}", output_text, tool_calls
        ),
        "output_text": output_text,
        "usage": _responses_usage_from_chat(chat_response),
    }


def _responses_stream_event(event_type: str, data: Dict[str, Any]) -> str:
    payload = {"type": event_type, **data}
    json_data = json.dumps(payload, separators=(",", ":"))
    return f"event: {event_type}\ndata: {json_data}\n\n"


def _responses_stream_error(message: str) -> str:
    return _responses_stream_event(
        "response.failed",
        {
            "response": {
                "id": f"resp_{uuid.uuid4().hex}",
                "object": "response",
                "created_at": utc_timestamp(),
                "status": "failed",
                "error": {
                    "message": message,
                    "type": "server_error",
                    "code": "stream_error",
                },
            }
        },
    )


async def _iter_sse_events(body_iterator: Any) -> AsyncGenerator[str, None]:
    buffer = ""
    async for chunk in body_iterator:
        if isinstance(chunk, bytes):
            buffer += chunk.decode("utf-8")
        else:
            buffer += str(chunk)

        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            if raw_event.strip():
                yield raw_event

    if buffer.strip():
        yield buffer


def _sse_data(raw_event: str) -> Optional[str]:
    data_lines = []
    for line in raw_event.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None
    return "\n".join(data_lines)


def _responses_completed_payload(
    response_id: str,
    message_id: str,
    created_at: int,
    completed_at: int,
    request: ResponsesCreateRequest,
    model: str,
    output_text: str,
    output_items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "completed_at": completed_at,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": request.max_output_tokens,
        "model": model,
        "output": (
            output_items
            if output_items is not None
            else _responses_output_items(message_id, output_text, [])
        ),
        "output_text": output_text,
        "usage": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
    }


async def _create_responses_sse_from_chat_stream(
    chat_stream_response: StreamingResponse,
    request: ResponsesCreateRequest,
) -> AsyncGenerator[str, None]:
    response_id = f"resp_{uuid.uuid4().hex}"
    message_id = f"msg_{uuid.uuid4().hex}"
    created_at = utc_timestamp()
    model = request.model
    output_parts: List[str] = []
    content_started = False
    message_index: Optional[int] = None
    # Tool calls by the index the chat stream gave them, so a call delivered in
    # fragments accumulates rather than opening a second item.
    calls: Dict[Any, Dict[str, Any]] = {}
    # Completed items paired with the output index they were announced at, so
    # the terminal payload reports them in that order.
    indexed_items: List[Tuple[int, Dict[str, Any]]] = []
    next_index = 0

    yield _responses_stream_event(
        "response.created",
        {
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "model": model,
            }
        },
    )

    try:
        async for raw_event in _iter_sse_events(chat_stream_response.body_iterator):
            payload = _sse_data(raw_event)
            if payload is None:
                continue
            if payload == "[DONE]":
                break

            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if "error" in chunk:
                yield _responses_stream_event(
                    "response.failed", {"response": {"id": response_id, **chunk}}
                )
                return

            model = chunk.get("model") or model
            choices = chunk.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}

            for raw_call in delta.get("tool_calls") or []:
                if not isinstance(raw_call, dict):
                    continue

                key = raw_call.get("index")
                if key is None:
                    key = f"_unindexed_{len(calls)}"
                function = raw_call.get("function") or {}
                entry = calls.get(key)

                if entry is None:
                    entry = {
                        "item_id": f"fc_{uuid.uuid4().hex}",
                        "output_index": next_index,
                        "call": {
                            "id": raw_call.get("id"),
                            "function": {
                                "name": function.get("name") or "",
                                "arguments": "",
                            },
                        },
                    }
                    next_index += 1
                    calls[key] = entry
                    yield _responses_stream_event(
                        "response.output_item.added",
                        {
                            "output_index": entry["output_index"],
                            "item": {
                                **_responses_function_call_item(
                                    entry["call"], item_id=entry["item_id"]
                                ),
                                "status": "in_progress",
                                "arguments": "",
                            },
                        },
                    )
                elif function.get("name"):
                    entry["call"]["function"]["name"] = function["name"]

                argument_delta = function.get("arguments")
                if argument_delta:
                    entry["call"]["function"]["arguments"] += str(argument_delta)
                    yield _responses_stream_event(
                        "response.function_call_arguments.delta",
                        {
                            "item_id": entry["item_id"],
                            "output_index": entry["output_index"],
                            "delta": str(argument_delta),
                        },
                    )

            text_delta = delta.get("content")
            if not text_delta:
                continue

            if not content_started:
                content_started = True
                message_index = next_index
                next_index += 1
                yield _responses_stream_event(
                    "response.output_item.added",
                    {
                        "output_index": message_index,
                        "item": {
                            "id": message_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                )
                yield _responses_stream_event(
                    "response.content_part.added",
                    {
                        "item_id": message_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        },
                    },
                )

            output_parts.append(str(text_delta))
            yield _responses_stream_event(
                "response.output_text.delta",
                {
                    "item_id": message_id,
                    "output_index": message_index,
                    "content_index": 0,
                    "delta": str(text_delta),
                },
            )

        output_text = "".join(output_parts)

        for entry in calls.values():
            item = _responses_function_call_item(
                entry["call"], item_id=entry["item_id"]
            )
            yield _responses_stream_event(
                "response.function_call_arguments.done",
                {
                    "item_id": entry["item_id"],
                    "output_index": entry["output_index"],
                    "arguments": item["arguments"],
                },
            )
            yield _responses_stream_event(
                "response.output_item.done",
                {"output_index": entry["output_index"], "item": item},
            )
            indexed_items.append((entry["output_index"], item))

        # A turn with no text and no tool calls still owes the client an item,
        # so the empty message the text-only path always produced is kept.
        if content_started or not calls:
            if not content_started:
                message_index = next_index
                next_index += 1
                yield _responses_stream_event(
                    "response.output_item.added",
                    {
                        "output_index": message_index,
                        "item": {
                            "id": message_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                )
                yield _responses_stream_event(
                    "response.content_part.added",
                    {
                        "item_id": message_id,
                        "output_index": message_index,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        },
                    },
                )

            yield _responses_stream_event(
                "response.output_text.done",
                {
                    "item_id": message_id,
                    "output_index": message_index,
                    "content_index": 0,
                    "text": output_text,
                },
            )
            yield _responses_stream_event(
                "response.content_part.done",
                {
                    "item_id": message_id,
                    "output_index": message_index,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": output_text,
                        "annotations": [],
                    },
                },
            )
            message_item = _responses_message_item(message_id, output_text)
            yield _responses_stream_event(
                "response.output_item.done",
                {"output_index": message_index, "item": message_item},
            )
            indexed_items.append((message_index, message_item))

        output_items = [
            item for _, item in sorted(indexed_items, key=lambda pair: pair[0])
        ]

        completed_at = utc_timestamp()
        yield _responses_stream_event(
            "response.completed",
            {
                "response": _responses_completed_payload(
                    response_id=response_id,
                    message_id=message_id,
                    created_at=created_at,
                    completed_at=completed_at,
                    request=request,
                    model=model,
                    output_text=output_text,
                    output_items=output_items,
                )
            },
        )
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error("Responses streaming error", error=str(e), exc_info=True)
        yield _responses_stream_error("Stream error")


async def _resolve_session(
    session_manager: SessionManager,
    request: ChatCompletionRequest,
    project_id: str,
    claude_model: Optional[str],
    system_prompt: Optional[str],
) -> str:
    if request.session_id:
        session_id = request.session_id
        session_info = await session_manager.get_session(session_id)
        if not session_info:
            raise _http_error(
                status.HTTP_404_NOT_FOUND,
                f"Session {session_id} not found",
                "invalid_request_error",
                "session_not_found",
            )
        return session_id
    return await session_manager.create_session(
        project_id=project_id, model=claude_model, system_prompt=system_prompt
    )


async def _collect_non_streaming_response(
    claude_process,
    session_manager: SessionManager,
    session_id: str,
    model: str,
    project_id: str,
    prefer_result_content: bool = False,
    tool_bridge: Optional[ToolBridge] = None,
    suppress_internal_tools: bool = False,
) -> Dict[str, Any]:
    messages, parser = await _gather_claude_messages(claude_process)
    _log_message_summary(messages)

    usage_summary = OpenAIConverter.calculate_usage(parser)
    await _update_session_usage(
        session_manager, session_id, usage_summary, parser.total_cost
    )

    response = _build_non_streaming_response(
        messages,
        session_id,
        model,
        usage_summary,
        project_id,
        prefer_result_content=prefer_result_content,
        tool_bridge=tool_bridge,
        suppress_internal_tools=suppress_internal_tools,
    )
    _log_response_payload(response)
    return response


async def _gather_claude_messages(claude_process) -> Tuple[list, ClaudeOutputParser]:
    messages = []
    parser = ClaudeOutputParser()
    async for claude_message in claude_process.get_output():
        _log_claude_message(claude_message)
        messages.append(claude_message)
        normalized = normalize_claude_message(claude_message)
        if not normalized:
            continue
        parser.parse_message(normalized)
        if parser.is_final_message(normalized):
            break
    return messages, parser


def _log_claude_message(claude_message: Any) -> None:
    logger.info(
        "Received Claude message",
        message_type=(
            claude_message.get("type")
            if isinstance(claude_message, dict)
            else type(claude_message).__name__
        ),
        message_keys=(
            list(claude_message.keys()) if isinstance(claude_message, dict) else []
        ),
        has_assistant_content=bool(
            isinstance(claude_message, dict)
            and claude_message.get("type") == "assistant"
            and claude_message.get("message", {}).get("content")
        ),
        message_preview=str(claude_message)[:200] if claude_message else "None",
    )


def _log_message_summary(messages: list) -> None:
    logger.info(
        "Claude messages collected",
        total_messages=len(messages),
        message_types=[
            msg.get("type") if isinstance(msg, dict) else type(msg).__name__
            for msg in messages
        ],
    )


async def _update_session_usage(
    session_manager: SessionManager,
    session_id: str,
    usage_summary: Dict[str, Any],
    total_cost: float,
) -> None:
    await session_manager.update_session(
        session_id=session_id,
        tokens_used=usage_summary.get("total_tokens", 0),
        cost=total_cost,
    )


def _build_non_streaming_response(
    messages: list,
    session_id: str,
    model: str,
    usage_summary: Dict[str, Any],
    project_id: str,
    prefer_result_content: bool = False,
    tool_bridge: Optional[ToolBridge] = None,
    suppress_internal_tools: bool = False,
) -> Dict[str, Any]:
    response = create_non_streaming_response(
        messages=messages,
        session_id=session_id,
        model=model,
        usage=usage_summary,
        prefer_result_content=prefer_result_content,
        tool_bridge=tool_bridge,
        suppress_internal_tools=suppress_internal_tools,
    )
    response["project_id"] = project_id
    return response


# Prose the model produces after the CLI answers a native tool_use with
# "No such tool available". Under tool_choice="auto" such a reply is a valid
# envelope, so it reaches the client as a normal answer unless it is spotted.
_TOOL_REFUSAL_PATTERN = re.compile(
    r"no such tool available|tool[- ]access issue|"
    r"tools? (?:are|is) not .{0,20}available",
    re.IGNORECASE,
)


def _warn_if_tools_went_unused(
    request: ChatCompletionRequest, response: Dict[str, Any]
) -> None:
    """Flag a turn where tools were offered but the model answered in prose."""
    if not request.tools:
        return

    choices = response.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    if message.get("tool_calls"):
        return

    content = message.get("content") or ""
    if not _TOOL_REFUSAL_PATTERN.search(content):
        return

    logger.warning(
        "Model reported tools as unavailable instead of calling them; retrying "
        'with tool_choice="required" would make the prose path unrepresentable',
        tool_count=len(request.tools),
        tool_choice=request.tool_choice,
        content_preview=content[:200],
    )


def _log_response_payload(response: Dict[str, Any]) -> None:
    choices = response.get("choices") or []
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None

    logger.info(
        "Returning chat completion response",
        response_id=response.get("id"),
        choices_count=len(choices),
        has_choices_0=bool(choices),
        choices_0_keys=(
            list(first_choice.keys()) if isinstance(first_choice, dict) else []
        ),
        message_keys=list(message.keys()) if isinstance(message, dict) else [],
        content_length=len(content or ""),
        full_response_keys=list(response.keys()),
        response_size=len(str(response)),
    )


@router.post(
    "/responses",
    responses=RESPONSES_API_RESPONSES,
)
async def create_response(request: ResponsesCreateRequest, req: Request) -> Any:
    """Create a minimal OpenAI Responses API response."""
    logger.info(
        "Responses API request validated",
        model=request.model,
        stream=request.stream,
        max_output_tokens=request.max_output_tokens,
        project_id=request.project_id,
        session_id=request.session_id,
    )

    chat_request = _responses_request_to_chat_request(
        request, stream=bool(request.stream)
    )
    chat_response = await create_chat_completion(chat_request, req)

    if request.stream:
        if not isinstance(chat_response, StreamingResponse):
            raise _http_error(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "Unexpected chat completion streaming response type.",
                "internal_error",
                "unexpected_response_type",
            )
        return StreamingResponse(
            _create_responses_sse_from_chat_stream(chat_response, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if hasattr(chat_response, "model_dump"):
        chat_response = chat_response.model_dump()

    if not isinstance(chat_response, dict):
        raise _http_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Unexpected chat completion response type.",
            "internal_error",
            "unexpected_response_type",
        )

    return _chat_response_to_responses_response(request, chat_response)


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    responses=CHAT_COMPLETION_RESPONSES,
)
async def create_chat_completion(request: ChatCompletionRequest, req: Request) -> Any:
    """Create a chat completion, compatible with OpenAI API."""

    # Log raw request for debugging
    try:
        await _log_raw_request(req)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process request", error=str(e))
        raise _http_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal server error",
            "internal_error",
            "internal_error",
        )

    # Get managers from app state
    session_manager: SessionManager = req.app.state.session_manager
    claude_manager = req.app.state.claude_manager

    # Extract client info for logging
    client_id = getattr(req.state, "client_id", "anonymous")

    logger.info(
        "Chat completion request validated",
        client_id=client_id,
        model=request.model,
        messages_count=len(request.messages),
        stream=request.stream,
        project_id=request.project_id,
        session_id=request.session_id,
        reasoning_effort=request.reasoning_effort,
    )

    try:
        requested_model = (request.model or "").strip() or None
        # Normalize only when user explicitly requested a model.
        claude_model = (
            validate_claude_model(requested_model) if requested_model else None
        )
        response_model = claude_model or get_default_model()

        user_prompt, system_prompt = _extract_prompts(request)
        json_schema = _extract_json_schema(request)
        # Validate before any project directory or session row is created, so a
        # rejected request leaves nothing behind.
        effort = _resolve_reasoning_effort(request)
        tool_bridge, json_schema, system_prompt = _apply_tool_bridge(
            request, json_schema, system_prompt
        )

        # Handle project context
        project_id = request.project_id or f"default-{client_id}"
        project_path = create_project_directory(project_id)

        # Continue an existing Claude session when this conversation is one
        # we have already been part of, so only the new messages are sent.
        turn_plan, resumable_session_id = _plan_conversation(
            request, session_manager, system_prompt
        )
        if turn_plan is not None:
            user_prompt = turn_plan.prompt

        # Handle session management
        if resumable_session_id is not None and not request.session_id:
            session_id = resumable_session_id
        else:
            session_id = await _resolve_session(
                session_manager=session_manager,
                request=request,
                project_id=project_id,
                claude_model=claude_model,
                system_prompt=system_prompt,
            )

        # Start Claude Code process
        try:

            def _register_cli_session(cli_session_id: str):
                session_manager.register_cli_session(session_id, cli_session_id)

            engine_kwargs = {}
            if settings.engine == ENGINE_SDK and request.tools:
                engine_kwargs["client_tools"] = request.tools
                required_tools = required_tool_names(request)
                if required_tools:
                    engine_kwargs["required_tool_names"] = required_tools
            resuming_turn = turn_plan is not None and turn_plan.resuming
            if resuming_turn:
                engine_kwargs["resume"] = turn_plan.resume_session_id

            # Says outright which of the two happened, next to the prompt size
            # it implies: a resumed turn sends only the new messages, a new one
            # sends the whole transcript.
            logger.info(
                "Running turn",
                conversation="resumed" if resuming_turn else "new",
                engine=settings.engine,
                session_id=session_id,
                resume_session_id=(
                    turn_plan.resume_session_id if resuming_turn else None
                ),
                prompt_chars=len(user_prompt),
                client_tool_count=len(request.tools or []),
            )

            async def _start(prompt: str, **extra):
                return await claude_manager.create_session(
                    session_id=session_id,
                    project_path=project_path,
                    prompt=prompt,
                    model=claude_model,
                    system_prompt=system_prompt,
                    on_cli_session_id=_register_cli_session,
                    json_schema=json_schema,
                    effort=effort,
                    **extra,
                )

            try:
                claude_process = await _start(user_prompt, **engine_kwargs)
            except Exception as e:
                if not (
                    turn_plan is not None
                    and turn_plan.resuming
                    and is_session_missing_error(e)
                ):
                    raise
                # Claude no longer has the session we recorded - it may have
                # been pruned, or the gateway may be talking to a different
                # host than the one that created it. Replay the transcript
                # instead; the ledger is wrong, so drop it.
                logger.warning(
                    "Resume failed, replaying the full conversation",
                    session_id=session_id,
                    error=str(e),
                )
                session_manager.discard_ledger(session_id)
                turn_plan = None
                resuming_turn = False
                engine_kwargs.pop("resume", None)
                user_prompt, _ = _extract_prompts(request)
                logger.info(
                    "Running turn",
                    conversation="new",
                    engine=settings.engine,
                    session_id=session_id,
                    resume_session_id=None,
                    prompt_chars=len(user_prompt),
                    client_tool_count=len(request.tools or []),
                    reason="resume was rejected, replaying the transcript",
                )
                claude_process = await _start(user_prompt, **engine_kwargs)
        except ClaudeSessionConflictError as e:
            logger.warning(
                "Session already has an active Claude process",
                session_id=session_id,
                error=str(e),
            )
            raise _http_error(
                status.HTTP_409_CONFLICT,
                "The session is currently busy with another process.",
                "invalid_request_error",
                "session_busy",
            ) from e
        except ClaudeCommandTooLongError as e:
            logger.warning(
                "Request too large for the operating system command line",
                session_id=session_id,
                error=str(e),
            )
            raise _http_error(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                str(e),
                "invalid_request_error",
                "request_too_large",
            ) from e
        except ClaudeModelNotSupportedError as e:
            logger.warning(
                "Claude rejected requested model",
                session_id=session_id,
                model=claude_model,
                error=str(e),
            )
            raise _http_error(
                status.HTTP_400_BAD_REQUEST,
                "The requested model is not supported.",
                "invalid_request_error",
                "model_not_supported",
            ) from e
        except Exception as e:
            logger.error(
                "Failed to create Claude session", session_id=session_id, error=str(e)
            )
            raise _http_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"Failed to start Claude Code: {str(e)}",
                "service_unavailable",
                "claude_unavailable",
            )

        # Use Claude's actual session ID
        api_session_id = session_id

        # Update session with user message
        await session_manager.update_session(
            session_id=api_session_id,
            # What the user said, not the whole transcript that was rendered
            # around it. Tokens still track what the CLI was charged for.
            message_content=current_turn_text(request.messages),
            role="user",
            tokens_used=estimate_tokens(user_prompt),
        )

        # Handle streaming vs non-streaming
        if request.stream:
            return StreamingResponse(
                create_sse_response(
                    api_session_id,
                    response_model,
                    claude_process,
                    prefer_result_content=json_schema is not None,
                    tool_bridge=tool_bridge,
                    hold_text_for_tool_calls=bool(request.tools),
                    suppress_internal_tools=_suppress_internal_tools(
                        request, json_schema
                    ),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Session-ID": api_session_id,
                    "X-Project-ID": project_id,
                },
            )

        def _record_turn(response: Dict[str, Any]) -> None:
            """Note what this session has now been told, once the turn worked."""
            if settings.engine != ENGINE_SDK:
                return
            if settings.conversation_history != MODE_RESUME:
                return

            consumed = (
                list(turn_plan.consumed)
                if turn_plan is not None
                else [m for m in request.messages if m.role != "system"]
            )
            echo = _assistant_echo(response)
            if echo is not None:
                consumed.append(echo)
            session_manager.record_turn(
                session_id,
                consumed,
                history_tool_call_names(request.messages),
                system_fingerprint=fingerprint_system(system_prompt),
            )

        completion = await _collect_non_streaming_response(
            claude_process=claude_process,
            session_manager=session_manager,
            session_id=api_session_id,
            model=response_model,
            project_id=project_id,
            prefer_result_content=json_schema is not None,
            tool_bridge=tool_bridge,
            suppress_internal_tools=_suppress_internal_tools(request, json_schema),
        )
        _warn_if_tools_went_unused(request, completion)
        _record_turn(completion)
        return completion

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            "Unexpected error in chat completion",
            client_id=client_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "Internal server error",
                    "type": "internal_error",
                    "code": "unexpected_error",
                }
            },
        )


@router.get("/chat/completions/{session_id}/status")
async def get_completion_status(session_id: str, req: Request) -> Dict[str, Any]:
    """Get status of a chat completion session."""

    session_manager: SessionManager = req.app.state.session_manager
    claude_manager = req.app.state.claude_manager

    # Get session info
    session_info = await session_manager.get_session(session_id)
    if not session_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"Session {session_id} not found",
                    "type": "not_found",
                    "code": "session_not_found",
                }
            },
        )

    # Get Claude process status
    claude_process = claude_manager.get_session(session_id)
    is_running = claude_process is not None and claude_process.is_running

    return {
        "session_id": session_id,
        "project_id": session_info.project_id,
        "model": session_info.model,
        "is_running": is_running,
        "created_at": session_info.created_at.isoformat(),
        "updated_at": session_info.updated_at.isoformat(),
        "total_tokens": session_info.total_tokens,
        "total_cost": session_info.total_cost,
        "message_count": session_info.message_count,
    }


@router.post("/chat/completions/debug")
async def debug_chat_completion(req: Request) -> Dict[str, Any]:
    """Debug endpoint to test request validation."""
    try:
        raw_body = await req.body()
        headers = dict(req.headers)

        logger.info(
            "Debug request",
            content_type=headers.get("content-type"),
            body_size=len(raw_body),
            headers=headers,
            raw_body=raw_body.decode() if raw_body else "empty",
        )

        if raw_body:
            json_data = json.loads(raw_body.decode())

            # Try validation
            try:
                request = ChatCompletionRequest(**json_data)
                return {
                    "status": "success",
                    "message": "Request validation passed",
                    "parsed_data": {
                        "model": request.model,
                        "messages_count": len(request.messages),
                        "stream": request.stream,
                    },
                }
            except ValidationError as e:
                return {
                    "status": "validation_error",
                    "message": str(e),
                    "errors": e.errors(),
                    "raw_data": json_data,
                }

        return {"status": "no_body"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.delete("/chat/completions/{session_id}")
async def stop_completion(session_id: str, req: Request) -> Dict[str, str]:
    """Stop a running chat completion session."""

    session_manager: SessionManager = req.app.state.session_manager
    claude_manager = req.app.state.claude_manager

    # Stop Claude process
    await claude_manager.stop_session(session_id)

    # End session
    await session_manager.end_session(session_id)

    logger.info("Chat completion stopped", session_id=session_id)

    return {"session_id": session_id, "status": "stopped"}
