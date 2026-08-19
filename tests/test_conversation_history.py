"""Tests for conversation history flattening.

The headline case is `TestAgentLoop`: an OpenAI client that feeds a tool result
back used to get the same tool call again forever, because every message but
the last user one was dropped before reaching the engine.
"""

import json
from types import SimpleNamespace

import pytest

from claude_code_api.api.chat import _extract_prompts
from claude_code_api.models.openai import ChatCompletionRequest
from claude_code_api.utils.history import (
    CURRENT_TURN_OPEN,
    NoConversationTurnError,
    current_turn_text,
    render_prompt,
)
from tests.model_utils import get_test_model_id


def msg(role, content=None, tool_calls=None, tool_call_id=None, name=None):
    return SimpleNamespace(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        name=name,
    )


def call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def current_window(prompt):
    """The slice the fixture matcher looks at."""
    marker = prompt.rfind(CURRENT_TURN_OPEN)
    return prompt[marker:] if marker != -1 else prompt


class TestRenderPrompt:
    def test_single_user_message_is_unchanged(self):
        """The common shape must cost nothing and behave exactly as before."""
        assert render_prompt([msg("user", "Hi")]) == "Hi"

    def test_single_user_message_keeps_multimodal_extraction(self):
        blocks = [
            {"type": "text", "text": "line one"},
            {"type": "text", "text": "line two"},
        ]
        assert render_prompt([msg("user", blocks)]) == "line one\nline two"

    def test_multi_turn_keeps_order_and_does_not_duplicate(self):
        prompt = render_prompt(
            [msg("user", "one"), msg("assistant", "two"), msg("user", "three")]
        )
        assert prompt.index("one") < prompt.index("two") < prompt.index("three")
        assert prompt.count("three") == 1
        assert "three" in current_window(prompt)

    def test_system_messages_never_enter_the_transcript(self):
        prompt = render_prompt(
            [
                msg("system", "SYSTEM MARKER"),
                msg("user", "a"),
                msg("assistant", "b"),
                msg("user", "c"),
            ]
        )
        assert "SYSTEM MARKER" not in prompt

    def test_assistant_tool_calls_are_rendered(self):
        prompt = render_prompt(
            [
                msg("user", "weather?"),
                msg(
                    "assistant",
                    None,
                    tool_calls=[call("call_1", "get_weather", '{"city": "Paris"}')],
                ),
                msg("user", "well?"),
            ]
        )
        assert 'id="call_1"' in prompt
        assert 'name="get_weather"' in prompt
        # Arguments are re-serialised compactly, as utils.tools emits them.
        assert '{"city":"Paris"}' in prompt

    def test_unparseable_arguments_are_passed_through(self):
        prompt = render_prompt(
            [
                msg("user", "go"),
                msg("assistant", None, tool_calls=[call("call_1", "t", "not json")]),
                msg("user", "again"),
            ]
        )
        assert "not json" in prompt

    def test_tool_result_name_is_resolved_from_the_call(self):
        """OpenAI deprecated `name` on tool messages, so clients omit it."""
        prompt = render_prompt(
            [
                msg("user", "weather?"),
                msg(
                    "assistant", None, tool_calls=[call("call_1", "get_weather", "{}")]
                ),
                msg("tool", '{"temp_c": 18}', tool_call_id="call_1"),
            ]
        )
        assert 'name="get_weather"' in current_window(prompt)
        assert 'tool_call_id="call_1"' in current_window(prompt)

    def test_agent_loop_puts_the_tool_result_in_the_current_turn(self):
        prompt = render_prompt(
            [
                msg("user", "weather?"),
                msg(
                    "assistant", None, tool_calls=[call("call_1", "get_weather", "{}")]
                ),
                msg("tool", '{"temp_c": 18}', tool_call_id="call_1"),
            ]
        )
        window = current_window(prompt)
        assert "temp_c" in window
        # The assistant turn is history, not the current turn.
        assert "<tool_call" not in window
        assert "do not emit a call again" in prompt

    def test_parallel_tool_results_are_all_present(self):
        prompt = render_prompt(
            [
                msg("user", "two things"),
                msg(
                    "assistant",
                    None,
                    tool_calls=[call("call_1", "a", "{}"), call("call_2", "b", "{}")],
                ),
                msg("tool", "first", tool_call_id="call_1"),
                msg("tool", "second", tool_call_id="call_2"),
            ]
        )
        window = current_window(prompt)
        assert "first" in window and "second" in window
        assert 'tool_call_id="call_1"' in window
        assert 'tool_call_id="call_2"' in window

    def test_consecutive_user_messages_are_both_kept(self):
        """Today the first of these is silently dropped."""
        prompt = render_prompt([msg("user", "first"), msg("user", "second")])
        assert "first" in prompt and "second" in prompt

    def test_empty_assistant_message_is_skipped(self):
        prompt = render_prompt(
            [msg("user", "one"), msg("assistant", None), msg("user", "two")]
        )
        assert 'role="assistant"' not in prompt

    def test_empty_tool_result_is_kept(self):
        """An empty result means "no matches", not "nothing happened"."""
        prompt = render_prompt(
            [
                msg("user", "search"),
                msg("assistant", None, tool_calls=[call("call_1", "search", "{}")]),
                msg("tool", "", tool_call_id="call_1"),
            ]
        )
        assert 'tool_call_id="call_1"' in current_window(prompt)

    def test_quotes_in_attributes_are_neutralised(self):
        prompt = render_prompt(
            [
                msg("user", "go"),
                msg("assistant", None, tool_calls=[call('a"b', "t", "{}")]),
                msg("user", "again"),
            ]
        )
        assert 'id="a\'b"' in prompt

    def test_no_user_message_raises(self):
        with pytest.raises(NoConversationTurnError):
            render_prompt([msg("system", "only system")])
        with pytest.raises(NoConversationTurnError):
            render_prompt([msg("system", "s"), msg("assistant", "a")])


class TestTruncation:
    def _long_history(self, turns=12, filler=400):
        messages = []
        for index in range(turns):
            messages.append(msg("user", f"old-{index} " + "x" * filler))
            messages.append(msg("assistant", f"reply-{index} " + "y" * filler))
        messages.append(msg("user", "the newest question"))
        return messages

    def test_oldest_messages_are_dropped_with_a_marker(self):
        prompt = render_prompt(self._long_history(), max_chars=3000)
        assert "<truncated>" in prompt
        assert "old-0" not in prompt
        assert "the newest question" in current_window(prompt)

    def test_unlimited_when_cap_is_zero(self):
        prompt = render_prompt(self._long_history(), max_chars=0)
        assert "<truncated>" not in prompt
        assert "old-0" in prompt

    def test_orphaned_tool_results_are_swept(self):
        """A result whose call was truncated away only confuses the model."""
        # Sized so the drop boundary falls between the call and its result:
        # the bulky assistant turn goes, the small result would survive.
        messages = [
            msg("assistant", "x" * 2000, tool_calls=[call("call_old", "gone", "{}")]),
            msg("tool", "ORPHANED RESULT", tool_call_id="call_old"),
            msg("assistant", "kept reply"),
            msg("user", "newest"),
        ]
        prompt = render_prompt(messages, max_chars=500)
        assert "x" * 2000 not in prompt
        assert "kept reply" in prompt
        assert "ORPHANED RESULT" not in prompt

    def test_current_turn_is_never_truncated(self):
        huge = "z" * 5000
        prompt = render_prompt(
            [msg("user", "old"), msg("assistant", "reply"), msg("user", huge)],
            max_chars=100,
        )
        assert huge in prompt


class TestCurrentTurnText:
    def test_returns_the_newest_user_text(self):
        messages = [msg("user", "one"), msg("assistant", "two"), msg("user", "three")]
        assert current_turn_text(messages) == "three"

    def test_returns_empty_without_a_user_message(self):
        assert current_turn_text([msg("assistant", "a")]) == ""


class TestExtractPrompts:
    """First tests this function has ever had."""

    def _request(self, messages, **kwargs):
        return ChatCompletionRequest(
            model=get_test_model_id(), messages=messages, **kwargs
        )

    def test_all_system_messages_are_joined(self):
        request = self._request(
            [
                {"role": "system", "content": "first rule"},
                {"role": "system", "content": "second rule"},
                {"role": "user", "content": "go"},
            ]
        )
        _, system_prompt = _extract_prompts(request)
        assert system_prompt == "first rule\n\nsecond rule"

    def test_falls_back_to_the_extension_field(self):
        request = self._request(
            [{"role": "user", "content": "go"}], system_prompt="from the field"
        )
        _, system_prompt = _extract_prompts(request)
        assert system_prompt == "from the field"

    def test_empty_messages_rejected(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            _extract_prompts(self._request([]))
        assert exc.value.detail["error"]["code"] == "missing_messages"

    def test_missing_user_message_rejected(self):
        from fastapi import HTTPException

        request = self._request(
            [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]
        )
        with pytest.raises(HTTPException) as exc:
            _extract_prompts(request)
        assert exc.value.detail["error"]["code"] == "missing_user_message"


class TestEndToEnd:
    def _post(self, client, messages, **kwargs):
        payload = {"model": get_test_model_id(), "messages": messages, "stream": False}
        payload.update(kwargs)
        return client.post("/v1/chat/completions", json=payload)

    def test_single_turn_prompt_reaches_the_cli_unchanged(
        self, test_client, sdk_prompt
    ):
        """Zero tax on the common shape."""
        response = self._post(test_client, [{"role": "user", "content": "Hi"}])
        assert response.status_code == 200
        assert sdk_prompt()[0] == "Hi"

    def test_multi_turn_history_reaches_the_cli(self, test_client, sdk_prompt):
        response = self._post(
            test_client,
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "hello again"},
            ],
        )
        assert response.status_code == 200
        prompt = sdk_prompt()[0]
        assert "first question" in prompt
        assert "first answer" in prompt
        assert "hello again" in prompt

    def test_streaming_carries_history(self, test_client, sdk_prompt):
        response = self._post(
            test_client,
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "hello again"},
            ],
            stream=True,
        )
        assert response.status_code == 200
        assert "first question" in sdk_prompt()[0]

    def test_responses_api_carries_history(self, test_client, sdk_prompt):
        """The Responses endpoint funnels into the same code path."""
        response = test_client.post(
            "/v1/responses",
            json={
                "model": get_test_model_id(),
                "input": [
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": "hello again"},
                ],
            },
        )
        assert response.status_code == 200
        prompt = sdk_prompt()[0]
        assert "first question" in prompt and "hello again" in prompt


class TestAgentLoop:
    """The reason this feature exists.

    Before history, turn 2 dropped the assistant call and the tool result, so
    the model re-emitted the same tool call instead of answering.
    """

    WEATHER_TOOL = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }

    def test_tool_result_produces_an_answer_not_another_call(
        self, test_client, sdk_prompt
    ):
        first = test_client.post(
            "/v1/chat/completions",
            json={
                "model": get_test_model_id(),
                "messages": [
                    {"role": "user", "content": "What is the weather in Paris?"}
                ],
                "tools": [self.WEATHER_TOOL],
                "tool_choice": "required",
                "stream": False,
            },
        )
        assert first.status_code == 200
        first_choice = first.json()["choices"][0]
        assert first_choice["finish_reason"] == "tool_calls"
        tool_call = first_choice["message"]["tool_calls"][0]
        call_id = tool_call["id"]
        assert tool_call["function"]["name"] == "get_weather"

        second = test_client.post(
            "/v1/chat/completions",
            json={
                "model": get_test_model_id(),
                "messages": [
                    {"role": "user", "content": "What is the weather in Paris?"},
                    {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps({"temp_c": 18, "conditions": "cloudy"}),
                    },
                ],
                "tools": [self.WEATHER_TOOL],
                "tool_choice": "auto",
                "stream": False,
            },
        )
        assert second.status_code == 200

        # The tool result, the call it answers and the tool name all reached the CLI.
        second_prompt = sdk_prompt()[1]
        assert "temp_c" in second_prompt
        assert call_id in second_prompt
        assert "get_weather" in second_prompt

        # And the model answered instead of calling the same tool again.
        second_choice = second.json()["choices"][0]
        assert second_choice["finish_reason"] == "stop"
        assert not second_choice["message"].get("tool_calls")
        assert "18" in second_choice["message"]["content"]
