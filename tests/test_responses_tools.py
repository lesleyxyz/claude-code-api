"""Tool calling on the Responses API route.

`/v1/responses` is a translation layer over `/v1/chat/completions`: the request
is converted into a `ChatCompletionRequest`, the ordinary chat pipeline runs, and
the chat response is converted back. Tools were the one thing the translation
dropped in both directions, so these tests pin all three crossings:

* flat Responses tool definitions and `tool_choice` becoming their nested chat
  equivalents on the way in,
* `function_call` / `function_call_output` input items becoming the assistant
  and tool messages the chat format carries them as, and
* chat `tool_calls` becoming `function_call` output items on the way out -
  through the same builders for the streaming and non-streaming paths, so the
  two cannot drift.
"""

import json

import pytest
from fastapi import HTTPException

from claude_code_api.api.chat import (
    _chat_response_to_responses_response,
    _create_responses_sse_from_chat_stream,
    _responses_completed_payload,
    _responses_function_call_item,
    _responses_input_to_chat_messages,
    _responses_output_items,
    _responses_request_to_chat_request,
    _responses_tool_choice_to_chat,
    _responses_tools_to_chat_tools,
    _sse_data,
)
from claude_code_api.models.openai import (
    ResponsesCreateRequest,
    ResponsesToolChoiceFunction,
    ResponsesToolFunction,
)

DICE_SCHEMA = {
    "type": "object",
    "$schema": "http://json-schema.org/draft-07/schema#",
    "properties": {"input": {"type": "string"}},
    "additionalProperties": False,
}


def dice_tool(name="roll_dice"):
    return ResponsesToolFunction(
        name=name,
        description="Rolls dice.",
        parameters=DICE_SCHEMA,
        strict=False,
    )


def chat_tool_call(name="roll_dice", arguments='{"input":"1d20+3"}', call_id="toolu_1"):
    """A tool call in the shape the chat response carries."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def chat_response(content=None, tool_calls=None, model="claude-sonnet-5"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# --------------------------------------------------------------------------
# Gap 1: tools and tool_choice reaching the chat request
# --------------------------------------------------------------------------


class TestToolsIntoChatRequest:
    def test_a_flat_tool_becomes_a_nested_chat_tool(self):
        tools = _responses_tools_to_chat_tools([dice_tool()])
        assert len(tools) == 1
        assert tools[0].type == "function"
        assert tools[0].function.name == "roll_dice"
        assert tools[0].function.description == "Rolls dice."

    def test_the_parameter_schema_is_passed_through_untouched(self):
        # Draft-07 keywords and `$schema` survive, because the schema is handed
        # to the SDK tool verbatim.
        tools = _responses_tools_to_chat_tools([dice_tool()])
        assert tools[0].function.parameters == DICE_SCHEMA

    def test_no_tools_stays_none(self):
        assert _responses_tools_to_chat_tools(None) is None
        assert _responses_tools_to_chat_tools([]) is None

    @pytest.mark.parametrize("choice", ["auto", "none", "required"])
    def test_string_tool_choices_pass_through(self, choice):
        assert _responses_tool_choice_to_chat(choice) == choice

    def test_a_named_choice_gains_the_function_wrapper(self):
        # The Responses API flattens the selector; the chat API nests it.
        converted = _responses_tool_choice_to_chat(
            ResponsesToolChoiceFunction(name="roll_dice")
        )
        assert converted.type == "function"
        assert converted.function.name == "roll_dice"

    def test_absent_tool_choice_stays_none(self):
        assert _responses_tool_choice_to_chat(None) is None

    def test_the_request_carries_tools_tool_choice_and_parallel(self):
        request = ResponsesCreateRequest(
            model="claude-sonnet-5",
            input="Roll 1d20",
            tools=[dice_tool()],
            tool_choice="required",
            parallel_tool_calls=True,
        )
        chat_request = _responses_request_to_chat_request(request)

        assert [tool.function.name for tool in chat_request.tools] == ["roll_dice"]
        assert chat_request.tool_choice == "required"
        assert chat_request.parallel_tool_calls is True

    def test_a_request_without_tools_is_unchanged(self):
        request = ResponsesCreateRequest(model="claude-sonnet-5", input="hello")
        chat_request = _responses_request_to_chat_request(request)

        assert chat_request.tools is None
        assert chat_request.tool_choice is None
        assert chat_request.parallel_tool_calls is None


# --------------------------------------------------------------------------
# Gap 3: the tool result round trip on input
# --------------------------------------------------------------------------


class TestToolItemsOnInput:
    def test_a_function_call_item_becomes_an_assistant_tool_call(self):
        messages = _responses_input_to_chat_messages(
            [
                {"role": "user", "type": "message", "content": "Roll 1d20"},
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "toolu_1",
                    "name": "roll_dice",
                    "arguments": '{"input":"1d20"}',
                },
            ]
        )

        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] is None
        call = messages[1]["tool_calls"][0]
        # The chat tool_call id is the Responses call_id, which is what the
        # matching function_call_output quotes back.
        assert call["id"] == "toolu_1"
        assert call["function"] == {
            "name": "roll_dice",
            "arguments": '{"input":"1d20"}',
        }

    def test_parallel_calls_share_one_assistant_message(self):
        messages = _responses_input_to_chat_messages(
            [
                {"role": "user", "type": "message", "content": "Roll twice"},
                {
                    "type": "function_call",
                    "call_id": "a",
                    "name": "roll_dice",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "b",
                    "name": "roll_dice",
                    "arguments": "{}",
                },
            ]
        )

        assert len(messages) == 2
        assert [c["id"] for c in messages[1]["tool_calls"]] == ["a", "b"]

    def test_a_call_after_a_tool_result_opens_a_new_message(self):
        messages = _responses_input_to_chat_messages(
            [
                {
                    "type": "function_call",
                    "call_id": "a",
                    "name": "roll_dice",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "a", "output": "3"},
                {
                    "type": "function_call",
                    "call_id": "b",
                    "name": "roll_dice",
                    "arguments": "{}",
                },
            ]
        )

        assert [m["role"] for m in messages] == ["assistant", "tool", "assistant"]
        assert [c["id"] for c in messages[2]["tool_calls"]] == ["b"]

    def test_a_function_call_output_becomes_a_tool_message(self):
        messages = _responses_input_to_chat_messages(
            [
                {
                    "type": "function_call_output",
                    "call_id": "toolu_1",
                    "output": "total 22",
                }
            ]
        )

        assert messages == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "total 22"}
        ]

    def test_a_structured_output_is_json_encoded(self):
        # Clients are free to send an object; the chat format wants text.
        messages = _responses_input_to_chat_messages(
            [
                {
                    "type": "function_call_output",
                    "call_id": "toolu_1",
                    "output": {"total": 22},
                }
            ]
        )
        assert json.loads(messages[0]["content"]) == {"total": 22}

    def test_the_merge_marker_never_leaks_into_a_message(self):
        messages = _responses_input_to_chat_messages(
            [
                {
                    "type": "function_call",
                    "call_id": "a",
                    "name": "roll_dice",
                    "arguments": "{}",
                }
            ]
        )
        assert "_from_function_call" not in messages[0]

    def test_missing_arguments_default_to_an_empty_object(self):
        messages = _responses_input_to_chat_messages(
            [{"type": "function_call", "call_id": "a", "name": "roll_dice"}]
        )
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    @pytest.mark.parametrize(
        "item",
        [
            {"type": "function_call", "name": "roll_dice", "arguments": "{}"},
            {"type": "function_call", "call_id": "a", "arguments": "{}"},
            {"type": "function_call_output", "output": "3"},
        ],
    )
    def test_an_incomplete_tool_item_is_rejected(self, item):
        with pytest.raises(HTTPException) as excinfo:
            _responses_input_to_chat_messages([item])
        assert excinfo.value.status_code == 400

    def test_an_unknown_item_type_is_still_rejected(self):
        # The allowlist stays closed: only the item types that can be
        # represented in the chat format are accepted.
        with pytest.raises(HTTPException) as excinfo:
            _responses_input_to_chat_messages([{"type": "computer_call"}])
        assert excinfo.value.status_code == 400


# --------------------------------------------------------------------------
# Gap 2: tool calls reaching the client as output items
# --------------------------------------------------------------------------


class TestToolCallsOnOutput:
    def test_a_chat_tool_call_becomes_a_function_call_item(self):
        item = _responses_function_call_item(chat_tool_call())

        assert item["type"] == "function_call"
        assert item["status"] == "completed"
        assert item["name"] == "roll_dice"
        assert item["arguments"] == '{"input":"1d20+3"}'
        # call_id is the chat id, so the client's function_call_output lines up
        # with the tool message the next request needs.
        assert item["call_id"] == "toolu_1"
        assert item["id"].startswith("fc_")

    def test_a_text_only_turn_yields_one_message_item(self):
        items = _responses_output_items("msg_1", "hello", [])
        assert [i["type"] for i in items] == ["message"]
        assert items[0]["content"][0]["text"] == "hello"

    def test_an_empty_text_turn_still_yields_a_message_item(self):
        # Text-only clients expect an item to read, even when it is empty.
        items = _responses_output_items("msg_1", "", [])
        assert [i["type"] for i in items] == ["message"]

    def test_a_tool_only_turn_yields_no_empty_message(self):
        items = _responses_output_items("msg_1", "", [chat_tool_call()])
        assert [i["type"] for i in items] == ["function_call"]

    def test_text_and_tool_calls_yield_the_message_first(self):
        items = _responses_output_items(
            "msg_1",
            "rolling now",
            [chat_tool_call(), chat_tool_call(call_id="toolu_2")],
        )
        assert [i["type"] for i in items] == [
            "message",
            "function_call",
            "function_call",
        ]

    def test_the_response_body_carries_the_call(self):
        request = ResponsesCreateRequest(model="claude-sonnet-5", input="Roll 1d20")
        response = _chat_response_to_responses_response(
            request, chat_response(content=None, tool_calls=[chat_tool_call()])
        )

        assert response["status"] == "completed"
        assert [i["type"] for i in response["output"]] == ["function_call"]
        assert response["output"][0]["name"] == "roll_dice"
        assert response["output_text"] == ""

    def test_a_text_response_body_is_unchanged(self):
        request = ResponsesCreateRequest(model="claude-sonnet-5", input="hi")
        response = _chat_response_to_responses_response(
            request, chat_response(content="hello there")
        )

        assert [i["type"] for i in response["output"]] == ["message"]
        assert response["output_text"] == "hello there"

    def test_the_completed_payload_defaults_to_a_message(self):
        # Called without output_items, as the text-only path used to.
        request = ResponsesCreateRequest(model="claude-sonnet-5", input="hi")
        payload = _responses_completed_payload(
            response_id="resp_1",
            message_id="msg_1",
            created_at=1,
            completed_at=2,
            request=request,
            model="claude-sonnet-5",
            output_text="hello",
        )
        assert [i["type"] for i in payload["output"]] == ["message"]

    def test_the_completed_payload_reports_given_items(self):
        request = ResponsesCreateRequest(model="claude-sonnet-5", input="hi")
        items = _responses_output_items("msg_1", "", [chat_tool_call()])
        payload = _responses_completed_payload(
            response_id="resp_1",
            message_id="msg_1",
            created_at=1,
            completed_at=2,
            request=request,
            model="claude-sonnet-5",
            output_text="",
            output_items=items,
        )
        assert payload["output"] == items


# --------------------------------------------------------------------------
# The streamed path, over the same builders
# --------------------------------------------------------------------------


def chat_sse(delta, finish_reason=None):
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "claude-sonnet-5",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


class FakeChatStream:
    """Stands in for the StreamingResponse the chat route returns."""

    def __init__(self, raw_events):
        self._raw = list(raw_events)

    @property
    def body_iterator(self):
        async def gen():
            for raw in self._raw:
                yield raw

        return gen()


async def collect(raw_events):
    request = ResponsesCreateRequest(
        model="claude-sonnet-5", input="Roll 1d20", stream=True
    )
    events = []
    async for raw in _create_responses_sse_from_chat_stream(
        FakeChatStream(raw_events), request
    ):
        data = _sse_data(raw)
        if data is None or data == "[DONE]":
            continue
        events.append(json.loads(data))
    return events


def types_of(events):
    return [event["type"] for event in events]


def completed(events):
    return next(e for e in events if e["type"] == "response.completed")["response"]


class TestStreamedToolCalls:
    @pytest.mark.asyncio
    async def test_text_only_stream_is_unchanged(self):
        events = await collect(
            [
                chat_sse({"content": "hel"}),
                chat_sse({"content": "lo"}),
                "data: [DONE]\n\n",
            ]
        )

        assert types_of(events) == [
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        assert completed(events)["output_text"] == "hello"

    @pytest.mark.asyncio
    async def test_a_tool_call_streams_argument_events(self):
        events = await collect(
            [
                chat_sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "toolu_1",
                                "type": "function",
                                "function": {
                                    "name": "roll_dice",
                                    "arguments": '{"input":"2d6"}',
                                },
                            }
                        ]
                    }
                ),
                "data: [DONE]\n\n",
            ]
        )

        assert types_of(events) == [
            "response.created",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
        # No phantom message item on a turn that only called a tool.
        assert [i["type"] for i in completed(events)["output"]] == ["function_call"]
        call = completed(events)["output"][0]
        assert call["name"] == "roll_dice"
        assert call["arguments"] == '{"input":"2d6"}'
        assert call["call_id"] == "toolu_1"

    @pytest.mark.asyncio
    async def test_the_item_id_is_stable_across_its_events(self):
        events = await collect(
            [
                chat_sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "toolu_1",
                                "function": {"name": "roll_dice", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                "data: [DONE]\n\n",
            ]
        )

        added = next(e for e in events if e["type"] == "response.output_item.added")
        done = next(e for e in events if e["type"] == "response.output_item.done")
        deltas = [
            e
            for e in events
            if e["type"].startswith("response.function_call_arguments")
        ]
        assert added["item"]["id"] == done["item"]["id"]
        assert {e["item_id"] for e in deltas} == {added["item"]["id"]}
        assert added["item"]["status"] == "in_progress"
        assert done["item"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_fragmented_arguments_accumulate_into_one_item(self):
        # The engine currently sends a call in one chunk; this keeps the path
        # correct if it ever fragments them the way OpenAI does.
        events = await collect(
            [
                chat_sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "toolu_1",
                                "function": {"name": "roll_dice", "arguments": '{"in'},
                            }
                        ]
                    }
                ),
                chat_sse(
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": 'put":"2d6"}'}}
                        ]
                    }
                ),
                "data: [DONE]\n\n",
            ]
        )

        assert types_of(events).count("response.output_item.added") == 1
        assert completed(events)["output"][0]["arguments"] == '{"input":"2d6"}'

    @pytest.mark.asyncio
    async def test_text_then_a_call_reports_both_in_order(self):
        events = await collect(
            [
                chat_sse({"content": "rolling"}),
                chat_sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "toolu_1",
                                "function": {"name": "roll_dice", "arguments": "{}"},
                            }
                        ]
                    }
                ),
                "data: [DONE]\n\n",
            ]
        )

        output = completed(events)["output"]
        assert [i["type"] for i in output] == ["message", "function_call"]
        assert completed(events)["output_text"] == "rolling"

    @pytest.mark.asyncio
    async def test_parallel_calls_become_two_items(self):
        events = await collect(
            [
                chat_sse(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "toolu_1",
                                "function": {"name": "roll_dice", "arguments": "{}"},
                            },
                            {
                                "index": 1,
                                "id": "toolu_2",
                                "function": {"name": "roll_dice", "arguments": "{}"},
                            },
                        ]
                    }
                ),
                "data: [DONE]\n\n",
            ]
        )

        output = completed(events)["output"]
        assert [i["call_id"] for i in output] == ["toolu_1", "toolu_2"]

    @pytest.mark.asyncio
    async def test_an_empty_stream_still_reports_a_message(self):
        events = await collect(["data: [DONE]\n\n"])

        assert [i["type"] for i in completed(events)["output"]] == ["message"]
        assert completed(events)["output_text"] == ""
