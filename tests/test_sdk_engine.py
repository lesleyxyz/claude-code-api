"""Unit tests for the Claude Agent SDK engine.

The adapter's whole job is to hand downstream code the same dicts the CLI's
stream-json produces, so the parser, the SSE converter and the tool bridge need
no knowledge of which engine ran. These tests pin that contract without
touching the network: SDK message objects are constructed directly.
"""

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    UserMessage,
)

from claude_code_api.core.config import Settings
from claude_code_api.core.sdk_session import sdk_message_to_stream_dict
from claude_code_api.utils.engine import ENGINE_CLI, ENGINE_SDK, normalize_engine
from claude_code_api.utils.parser import ClaudeOutputParser, normalize_claude_message
from claude_code_api.utils.streaming import create_non_streaming_response


def assistant(*blocks, **kwargs):
    kwargs.setdefault("model", "claude-haiku-4-5-20251001")
    kwargs.setdefault("parent_tool_use_id", None)
    kwargs.setdefault("error", None)
    kwargs.setdefault("usage", None)
    kwargs.setdefault("message_id", None)
    kwargs.setdefault("stop_reason", None)
    kwargs.setdefault("session_id", "sess-abc")
    kwargs.setdefault("uuid", None)
    return AssistantMessage(content=list(blocks), **kwargs)


def result(**kwargs):
    defaults = dict(
        subtype="success",
        duration_ms=1200,
        duration_api_ms=1000,
        is_error=False,
        num_turns=1,
        session_id="sess-abc",
        stop_reason=None,
        total_cost_usd=0.00004,
        usage={"input_tokens": 10, "output_tokens": 5},
        result="done",
        structured_output=None,
        model_usage=None,
        permission_denials=None,
        deferred_tool_use=None,
        errors=None,
        api_error_status=None,
        uuid=None,
        terminal_reason=None,
        origin=None,
    )
    defaults.update(kwargs)
    return ResultMessage(**defaults)


class TestNormalizeEngine:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, ENGINE_CLI),
            ("", ENGINE_CLI),
            ("  ", ENGINE_CLI),
            ("CLI", ENGINE_CLI),
            (" sdk ", ENGINE_SDK),
        ],
    )
    def test_accepted_values(self, value, expected):
        assert normalize_engine(value) == expected

    def test_unknown_value_is_rejected(self):
        with pytest.raises(ValueError, match="Supported values"):
            normalize_engine("grpc")

    def test_default_is_the_cli_engine(self):
        """The SDK engine is opt-in; the CLI path stays the default."""
        assert Settings().engine == ENGINE_CLI

    def test_settings_reject_a_bad_engine(self):
        with pytest.raises(Exception):
            Settings(engine="bogus")


class TestMessageTranslation:
    def test_text_block_becomes_cli_shaped_assistant_message(self):
        payload = sdk_message_to_stream_dict(assistant(TextBlock(text="hello")))

        assert payload["type"] == "assistant"
        assert payload["message"]["role"] == "assistant"
        assert payload["message"]["content"] == [{"type": "text", "text": "hello"}]
        assert payload["session_id"] == "sess-abc"

    def test_tool_use_block_is_preserved_with_id_and_input(self):
        """Native tool calls are what the SDK engine exists to deliver."""
        block = ToolUseBlock(
            id="toolu_1", name="mcp__gw__search_nodes", input={"query": "slack"}
        )
        payload = sdk_message_to_stream_dict(assistant(block))

        assert payload["message"]["content"] == [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "mcp__gw__search_nodes",
                "input": {"query": "slack"},
            }
        ]

    def test_thinking_is_carried_but_not_treated_as_text(self):
        payload = sdk_message_to_stream_dict(
            assistant(ThinkingBlock(thinking="pondering", signature="sig"))
        )
        message = normalize_claude_message(payload)

        assert payload["message"]["content"][0]["type"] == "thinking"
        # Thinking must not leak into the OpenAI content field.
        assert ClaudeOutputParser().extract_text_content(message) == ""

    def test_result_maps_cost_to_the_name_the_parser_reads(self):
        payload = sdk_message_to_stream_dict(result())

        assert payload["type"] == "result"
        assert payload["result"] == "done"
        # The SDK calls it total_cost_usd; the parser reads cost_usd.
        assert payload["cost_usd"] == 0.00004
        assert payload["usage"] == {"input_tokens": 10, "output_tokens": 5}

    def test_result_errors_are_surfaced(self):
        payload = sdk_message_to_stream_dict(
            result(is_error=True, errors=["boom", "again"], result=None)
        )
        assert payload["error"] == "boom; again"

    def test_successful_result_has_no_error(self):
        assert sdk_message_to_stream_dict(result())["error"] is None

    def test_system_message_keeps_its_data(self):
        payload = sdk_message_to_stream_dict(
            SystemMessage(subtype="init", data={"session_id": "s1", "model": "m"})
        )

        assert payload["type"] == "system"
        assert payload["subtype"] == "init"
        assert payload["session_id"] == "s1"

    def test_user_message_string_content(self):
        payload = sdk_message_to_stream_dict(
            UserMessage(
                content="hi",
                uuid=None,
                parent_tool_use_id=None,
                tool_use_result=None,
                origin=None,
            )
        )
        assert payload["message"]["content"] == [{"type": "text", "text": "hi"}]

    def test_unknown_message_is_skipped_not_fatal(self):
        """The SDK adds message types over time; unknown ones must not crash."""

        class SomethingNew:
            pass

        assert sdk_message_to_stream_dict(SomethingNew()) is None


class TestDownstreamCompatibility:
    """The contract that keeps the two engines swappable."""

    def test_translated_messages_build_an_openai_response(self):
        messages = [
            sdk_message_to_stream_dict(
                SystemMessage(subtype="init", data={"session_id": "sess-abc"})
            ),
            sdk_message_to_stream_dict(assistant(TextBlock(text="the answer"))),
            sdk_message_to_stream_dict(result(result="the answer")),
        ]

        response = create_non_streaming_response(
            messages=messages, session_id="sess-abc", model="claude-haiku-4-5-20251001"
        )
        choice = response["choices"][0]

        assert choice["message"]["content"] == "the answer"
        assert choice["finish_reason"] == "stop"

    def test_native_tool_call_becomes_openai_tool_calls(self):
        """No envelope, no --json-schema: the call arrives as a real block."""
        messages = [
            sdk_message_to_stream_dict(
                assistant(
                    ToolUseBlock(
                        id="toolu_1", name="get_weather", input={"city": "Paris"}
                    )
                )
            ),
            sdk_message_to_stream_dict(result(result="")),
        ]

        response = create_non_streaming_response(
            messages=messages, session_id="sess-abc", model="claude-haiku-4-5-20251001"
        )
        choice = response["choices"][0]

        assert choice["finish_reason"] == "tool_calls"
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "get_weather"
        assert call["function"]["arguments"] == '{"city":"Paris"}'

    def test_final_message_is_detected(self):
        message = normalize_claude_message(sdk_message_to_stream_dict(result()))
        assert ClaudeOutputParser().is_final_message(message) is True


class TestClientToolNaming:
    """MCP namespacing is an implementation detail; clients see their own names."""

    def test_qualified_and_local_round_trip(self):
        from claude_code_api.utils.sdk_tools import local_name, qualified_name

        assert qualified_name("search_nodes") == "mcp__client__search_nodes"
        assert local_name("mcp__client__search_nodes") == "search_nodes"

    def test_hyphenated_names_survive(self):
        """n8n declares tools like data-tables and n8n-docs."""
        from claude_code_api.utils.sdk_tools import local_name, qualified_name

        for name in ("data-tables", "n8n-docs", "verify-built-workflow"):
            assert local_name(qualified_name(name)) == name

    def test_foreign_names_are_left_alone(self):
        from claude_code_api.utils.sdk_tools import is_client_tool, local_name

        assert is_client_tool("Bash") is False
        assert local_name("Bash") == "Bash"


class TestClientToolServer:
    def _tool(self, name, params=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            type="function",
            function=SimpleNamespace(
                name=name, description=f"does {name}", parameters=params
            ),
        )

    async def _noop(self, *_args):
        return None

    def test_builds_a_server_and_wildcard_allowlist(self):
        from claude_code_api.utils.sdk_tools import build_client_tool_server

        server, allowed, names = build_client_tool_server(
            [self._tool("a"), self._tool("b")], self._noop
        )

        assert server is not None
        assert allowed == ["mcp__client__*"]
        assert names == ["a", "b"]

    def test_no_tools_yields_no_server(self):
        from claude_code_api.utils.sdk_tools import build_client_tool_server

        server, allowed, names = build_client_tool_server([], self._noop)
        assert server is None and allowed == [] and names == []

    def test_full_json_schema_is_passed_through(self):
        """Unlike the envelope bridge, per-tool schemas stay intact."""
        from claude_code_api.utils.sdk_tools import _schema_for

        schema = {
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {"action": {"type": "string", "enum": ["list", "get"]}},
            "required": ["action"],
            "additionalProperties": False,
        }
        assert _schema_for(self._tool("workflows", schema)) is schema

    def test_missing_parameters_become_a_valid_object_schema(self):
        from claude_code_api.utils.sdk_tools import _schema_for

        assert _schema_for(self._tool("noargs", None)) == {
            "type": "object",
            "properties": {},
        }


class TestInterception:
    """A client tool call ends the turn and travels back as tool_calls."""

    def _session(self):
        from claude_code_api.core.sdk_session import SdkSession

        return SdkSession(session_id="s", project_path=".")

    def _assistant_payload(self, tool_name):
        return {
            "type": "assistant",
            "session_id": "sess-1",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": tool_name,
                        "input": {"query": "slack"},
                    }
                ],
            },
        }

    def test_client_tool_is_detected_and_renamed(self):
        session = self._session()
        payload = self._assistant_payload("mcp__client__search_nodes")

        assert session._rewrite_client_tool_names(payload) is True
        assert payload["message"]["content"][0]["name"] == "search_nodes"

    def test_builtin_tool_is_not_treated_as_a_client_call(self):
        session = self._session()
        payload = self._assistant_payload("Bash")

        assert session._rewrite_client_tool_names(payload) is False
        assert payload["message"]["content"][0]["name"] == "Bash"

    def test_text_only_message_is_not_an_interception(self):
        session = self._session()
        payload = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "x"}],
            },
        }
        assert session._rewrite_client_tool_names(payload) is False

    def test_synthetic_result_is_clean_not_an_error(self):
        """The turn ended because the client has work to do, not because it failed."""
        session = self._session()
        result = session._tool_call_result(self._assistant_payload("mcp__client__x"))

        assert result["type"] == "result"
        assert result["subtype"] == "tool_calls"
        assert result["error"] is None

    def test_intercepted_turn_produces_openai_tool_calls(self):
        from claude_code_api.utils.streaming import create_non_streaming_response

        session = self._session()
        payload = self._assistant_payload("mcp__client__search_nodes")
        session._rewrite_client_tool_names(payload)

        response = create_non_streaming_response(
            messages=[payload, session._tool_call_result(payload)],
            session_id="s",
            model="claude-haiku-4-5-20251001",
        )
        choice = response["choices"][0]

        assert choice["finish_reason"] == "tool_calls"
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "search_nodes"
        assert call["function"]["arguments"] == '{"query":"slack"}'


class TestToolSuppressionPerEngine:
    """Regression: the same flag must mean opposite things per engine.

    `suppress_internal_tools` hides tool_use blocks. On the CLI engine those are
    Claude's own built-ins and must be hidden; on the SDK engine they are the
    caller's registered tools and must pass through. Getting this wrong silently
    strips every tool_call and returns finish_reason "stop" instead.
    """

    def _request(self, tools=None):
        from claude_code_api.models.openai import ChatCompletionRequest

        payload = {"messages": [{"role": "user", "content": "go"}]}
        if tools is not None:
            payload["tools"] = tools
        return ChatCompletionRequest(**payload)

    @property
    def _tool(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_nodes",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def test_cli_engine_hides_builtin_tool_uses(self, monkeypatch):
        from claude_code_api.api.chat import _suppress_internal_tools
        from claude_code_api.core.config import settings

        monkeypatch.setattr(settings, "engine", ENGINE_CLI)
        assert _suppress_internal_tools(self._request(self._tool)) is True

    def test_sdk_engine_lets_client_tool_uses_through(self, monkeypatch):
        from claude_code_api.api.chat import _suppress_internal_tools
        from claude_code_api.core.config import settings

        monkeypatch.setattr(settings, "engine", ENGINE_SDK)
        assert _suppress_internal_tools(self._request(self._tool)) is False

    def test_no_tools_means_nothing_to_suppress(self, monkeypatch):
        from claude_code_api.api.chat import _suppress_internal_tools
        from claude_code_api.core.config import settings

        monkeypatch.setattr(settings, "engine", ENGINE_CLI)
        assert _suppress_internal_tools(self._request()) is False


class TestBridgeIsSkippedOnSdkEngine:
    def test_envelope_bridge_is_not_applied(self, monkeypatch):
        """Native tools make the envelope emulation unnecessary and harmful."""
        from claude_code_api.api.chat import _apply_tool_bridge
        from claude_code_api.core.config import settings
        from claude_code_api.models.openai import ChatCompletionRequest

        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "go"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "t",
                        "description": "d",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        monkeypatch.setattr(settings, "engine", ENGINE_SDK)
        bridge, schema, system_prompt = _apply_tool_bridge(request, None, "sys")

        assert bridge is None
        assert schema is None
        assert system_prompt == "sys"

    def test_cli_engine_still_applies_the_bridge(self, monkeypatch):
        from claude_code_api.api.chat import _apply_tool_bridge
        from claude_code_api.core.config import settings
        from claude_code_api.models.openai import ChatCompletionRequest

        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "go"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "t",
                        "description": "d",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        monkeypatch.setattr(settings, "engine", ENGINE_CLI)
        bridge, schema, system_prompt = _apply_tool_bridge(request, None, "sys")

        assert bridge is not None
        assert schema is not None
        assert "t" in system_prompt
