"""A turn that calls a tool answers with the call, not with narration.

Claude often prefixes a tool call with a line of commentary - "I'll roll the
dice for you" - in the same assistant message as the call. OpenAI's models do
not: they return `content: null` alongside `tool_calls`. Clients are built for
that shape, and an agent that reads the text first takes the narration for the
final answer and never runs the tool.

Both API routes inherit the rule from the chat layer, so it is pinned here once
for the chat response and once for the Responses translation on top of it.
"""

import json

import pytest

from claude_code_api.api.chat import _chat_response_to_responses_response
from claude_code_api.models.openai import ResponsesCreateRequest
from claude_code_api.utils.parser import ClaudeOutputParser
from claude_code_api.utils.streaming import (
    OpenAIStreamConverter,
    create_non_streaming_response,
    normalize_claude_message,
)


def assistant(*blocks, session_id="sess-1"):
    """One assistant message in the stream-json shape both engines produce."""
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": list(blocks)},
        "session_id": session_id,
        "model": "claude-sonnet-5",
    }


def text(value):
    return {"type": "text", "text": value}


def tool_use(name="roll_dice", arguments=None, block_id="toolu_1"):
    return {
        "type": "tool_use",
        "id": block_id,
        "name": name,
        "input": {"input": "1d20+3"} if arguments is None else arguments,
    }


def result(session_id="sess-1"):
    return {
        "type": "result",
        "subtype": "success",
        "result": None,
        "session_id": session_id,
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "cost_usd": 0.0001,
        "duration_ms": 100,
        "num_turns": 1,
        "error": None,
    }


def non_streaming(*payloads):
    messages = [normalize_claude_message(p) for p in payloads]
    parser = ClaudeOutputParser()
    for message in messages:
        parser.parse_message(message)
    return create_non_streaming_response(
        messages=messages, session_id="sess-1", model="claude-sonnet-5"
    )


class TestNonStreaming:
    def test_narration_beside_a_call_is_dropped(self):
        response = non_streaming(
            assistant(text("I'll roll the dice for you."), tool_use()), result()
        )
        message = response["choices"][0]["message"]

        assert message["content"] is None
        assert [c["function"]["name"] for c in message["tool_calls"]] == ["roll_dice"]
        assert response["choices"][0]["finish_reason"] == "tool_calls"

    def test_narration_in_an_earlier_message_is_dropped_too(self):
        # Claude sometimes splits the two across messages; the non-streaming
        # path sees the whole turn, so it can still drop the text.
        response = non_streaming(
            assistant(text("Let me roll for that.")),
            assistant(tool_use()),
            result(),
        )
        assert response["choices"][0]["message"]["content"] is None

    def test_a_call_without_narration_is_unchanged(self):
        response = non_streaming(assistant(tool_use()), result())
        message = response["choices"][0]["message"]

        assert message["content"] is None
        assert len(message["tool_calls"]) == 1

    def test_a_text_only_turn_keeps_its_text(self):
        response = non_streaming(assistant(text("The capital is Paris.")), result())
        message = response["choices"][0]["message"]

        assert message["content"] == "The capital is Paris."
        assert "tool_calls" not in message
        assert response["choices"][0]["finish_reason"] == "stop"

    def test_parallel_calls_still_drop_the_narration_once(self):
        response = non_streaming(
            assistant(
                text("Rolling both."),
                tool_use(block_id="toolu_1"),
                tool_use(block_id="toolu_2"),
            ),
            result(),
        )
        message = response["choices"][0]["message"]

        assert message["content"] is None
        assert len(message["tool_calls"]) == 2


class TestStreaming:
    def converter(self):
        return OpenAIStreamConverter(model="claude-sonnet-5", session_id="sess-1")

    def deltas(self, chunks):
        out = []
        for chunk in chunks:
            for line in chunk.splitlines():
                if not line.startswith("data: "):
                    continue
                body = line[6:].strip()
                if body == "[DONE]":
                    continue
                out.append(json.loads(body)["choices"][0]["delta"])
        return out

    def test_narration_is_never_streamed_when_the_message_calls_a_tool(self):
        message = normalize_claude_message(
            assistant(text("I'll roll the dice for you."), tool_use())
        )
        chunks, saw_text, saw_tool_calls = self.converter()._assistant_chunks(message)

        # A stream cannot retract text, so the decision is made before emitting.
        assert saw_text is False
        assert saw_tool_calls is True
        deltas = self.deltas(chunks)
        assert all("content" not in delta for delta in deltas)
        assert [
            call["function"]["name"] for delta in deltas for call in delta["tool_calls"]
        ] == ["roll_dice"]

    def test_a_text_only_message_still_streams_its_text(self):
        message = normalize_claude_message(assistant(text("Paris.")))
        chunks, saw_text, saw_tool_calls = self.converter()._assistant_chunks(message)

        assert saw_text is True
        assert saw_tool_calls is False
        assert self.deltas(chunks)[0]["content"] == "Paris."

    def test_a_call_only_message_streams_only_the_call(self):
        message = normalize_claude_message(assistant(tool_use()))
        chunks, saw_text, saw_tool_calls = self.converter()._assistant_chunks(message)

        assert (saw_text, saw_tool_calls) == (False, True)
        assert len(self.deltas(chunks)) == 1

    def test_suppressed_internal_tools_also_suppress_nothing_else(self):
        # With internal tools hidden there is no call to shadow the text, so the
        # text must survive: this is the CLI engine's built-in-tool case.
        converter = OpenAIStreamConverter(
            model="claude-sonnet-5", session_id="sess-1", suppress_internal_tools=True
        )
        message = normalize_claude_message(
            assistant(text("Reading the file now."), tool_use(name="Read"))
        )
        chunks, saw_text, saw_tool_calls = converter._assistant_chunks(message)

        assert saw_text is True
        assert saw_tool_calls is False
        assert self.deltas(chunks)[0]["content"] == "Reading the file now."


class TestResponsesRouteInheritsIt:
    def test_the_message_item_disappears_with_the_narration(self):
        chat = non_streaming(
            assistant(text("I'll roll the dice for you."), tool_use()), result()
        )
        request = ResponsesCreateRequest(model="claude-sonnet-5", input="roll")
        response = _chat_response_to_responses_response(request, chat)

        # Only the call is left, which is what an agent runtime looks for.
        assert [item["type"] for item in response["output"]] == ["function_call"]
        assert response["output_text"] == ""
        assert response["output"][0]["name"] == "roll_dice"

    def test_a_text_turn_still_has_its_message_item(self):
        chat = non_streaming(assistant(text("Paris.")), result())
        request = ResponsesCreateRequest(model="claude-sonnet-5", input="capital?")
        response = _chat_response_to_responses_response(request, chat)

        assert [item["type"] for item in response["output"]] == ["message"]
        assert response["output_text"] == "Paris."


class FakeProcess:
    """Yields scripted stream-json payloads, like ClaudeProcess/SdkSession do."""

    def __init__(self, payloads):
        self._payloads = list(payloads)

    async def get_output(self):
        for payload in self._payloads:
            yield payload


async def stream_deltas(payloads, hold_text_for_tool_calls):
    converter = OpenAIStreamConverter(
        model="claude-sonnet-5",
        session_id="sess-1",
        hold_text_for_tool_calls=hold_text_for_tool_calls,
    )
    deltas, finish_reasons = [], []
    async for chunk in converter.convert_stream(FakeProcess(payloads)):
        for line in chunk.splitlines():
            if not line.startswith("data: "):
                continue
            body = line[6:].strip()
            if body == "[DONE]":
                continue
            choice = json.loads(body)["choices"][0]
            deltas.append(choice["delta"])
            if choice.get("finish_reason"):
                finish_reasons.append(choice["finish_reason"])
    return deltas, finish_reasons


class TestStreamedNarrationDeferral:
    """Narration and the call it precedes can arrive in separate messages.

    A stream cannot retract text, so when the caller declared tools the text is
    held until the turn declares itself. It costs nothing in practice: this
    engine emits one text chunk per assistant message, never per token.
    """

    @pytest.mark.asyncio
    async def test_narration_in_an_earlier_message_is_dropped(self):
        deltas, finish_reasons = await stream_deltas(
            [assistant(text("Let me roll for that.")), assistant(tool_use()), result()],
            hold_text_for_tool_calls=True,
        )

        assert not any(delta.get("content") for delta in deltas)
        assert any(delta.get("tool_calls") for delta in deltas)
        assert finish_reasons == ["tool_calls"]

    @pytest.mark.asyncio
    async def test_held_text_is_released_when_no_call_arrives(self):
        deltas, finish_reasons = await stream_deltas(
            [assistant(text("The capital is Paris.")), result()],
            hold_text_for_tool_calls=True,
        )

        assert [d["content"] for d in deltas if d.get("content")] == [
            "The capital is Paris."
        ]
        assert finish_reasons == ["stop"]

    @pytest.mark.asyncio
    async def test_held_text_is_released_before_the_finish_chunk(self):
        # A client that stops reading at finish_reason must still have seen it.
        deltas, _ = await stream_deltas(
            [assistant(text("Paris.")), result()], hold_text_for_tool_calls=True
        )
        content_at = next(i for i, d in enumerate(deltas) if d.get("content") == "Paris.")
        assert content_at < len(deltas) - 1

    @pytest.mark.asyncio
    async def test_without_the_flag_text_goes_out_immediately(self):
        # The default for callers that declared no tools: nothing to wait for.
        deltas, _ = await stream_deltas(
            [assistant(text("Let me roll for that.")), assistant(tool_use()), result()],
            hold_text_for_tool_calls=False,
        )
        assert [d["content"] for d in deltas if d.get("content")] == [
            "Let me roll for that."
        ]
