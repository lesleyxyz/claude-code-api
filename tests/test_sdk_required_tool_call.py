"""Tests for honouring `tool_choice` on the SDK engine.

The SDK engine registers the caller's tools with Claude for real, which is why
it does not use the CLI's envelope bridge - but the bridge was also what carried
`tool_choice`, and a real toolset has no equivalent: the SDK can offer a tool, it
cannot oblige the model to reach for one. Two things close that gap, and both are
pinned here:

* the demand is stated in the system prompt (`utils.sdk_prompt`), which is also
  the only system prompt the model gets when the caller sent none, and
* a turn that ends without the call is discarded and asked again in the same
  session (`SdkSession._pump`), so prose cannot be delivered as the answer while
  an attempt remains.

The `_pump` tests drive stand-in message classes rather than the SDK's own. The
adapter dispatches on the class *name*, so these exercise the real code path,
and the translation of genuine SDK objects is already pinned by
`test_sdk_engine.py`.
"""

from typing import Any, Dict, List, Optional

import pytest

from claude_code_api.core.sdk_session import SdkSession
from claude_code_api.models.openai import (
    ChatCompletionRequest,
    ChatMessage,
    ToolChoice,
    ToolChoiceFunction,
    ToolDefinition,
    ToolFunction,
)
from claude_code_api.utils.parser import normalize_claude_message
from claude_code_api.utils.sdk_prompt import (
    DEFAULT_SYSTEM_PROMPT,
    required_tool_instruction,
    required_tool_nudge,
    resolve_system_prompt,
)
from claude_code_api.utils.sdk_tools import qualified_name
from claude_code_api.utils.streaming import create_non_streaming_response
from claude_code_api.utils.tool_choice import required_tool_names

CLASSIFIER = "DocumentClassifierSchema"


def request(tools=None, tool_choice=None, content="classify this"):
    return ChatCompletionRequest(
        model="claude-haiku-4-5",
        messages=[ChatMessage(role="user", content=content)],
        tools=tools,
        tool_choice=tool_choice,
    )


def tool(name, description="Schema for document classification suggestions."):
    return ToolDefinition(
        function=ToolFunction(
            name=name,
            description=description,
            parameters={"type": "object", "properties": {"title": {"type": "string"}}},
        )
    )


def named_choice(name):
    return ToolChoice(function=ToolChoiceFunction(name=name))


# --------------------------------------------------------------------------
# Which tools the caller actually demanded
# --------------------------------------------------------------------------


class TestRequiredToolNames:
    def test_required_demands_every_declared_tool(self):
        req = request(tools=[tool(CLASSIFIER), tool("Other")], tool_choice="required")
        assert required_tool_names(req) == [CLASSIFIER, "Other"]

    def test_any_is_a_synonym_for_required(self):
        req = request(tools=[tool(CLASSIFIER)], tool_choice="any")
        assert required_tool_names(req) == [CLASSIFIER]

    def test_a_named_choice_narrows_the_demand(self):
        req = request(
            tools=[tool(CLASSIFIER), tool("Other")],
            tool_choice=named_choice("Other"),
        )
        assert required_tool_names(req) == ["Other"]

    def test_auto_demands_nothing(self):
        req = request(tools=[tool(CLASSIFIER)], tool_choice="auto")
        assert required_tool_names(req) == []

    def test_an_absent_tool_choice_demands_nothing(self):
        assert required_tool_names(request(tools=[tool(CLASSIFIER)])) == []

    def test_none_demands_nothing(self):
        req = request(tools=[tool(CLASSIFIER)], tool_choice="none")
        assert required_tool_names(req) == []

    def test_required_without_tools_demands_nothing(self):
        # Nothing to call, so nothing to insist on: the demand is unsatisfiable
        # and must not send the engine round the retry loop.
        assert required_tool_names(request(tool_choice="required")) == []

    def test_a_named_tool_that_was_never_declared_demands_nothing(self):
        req = request(tools=[tool(CLASSIFIER)], tool_choice=named_choice("Ghost"))
        assert required_tool_names(req) == []


# --------------------------------------------------------------------------
# The system prompt the model ends up with
# --------------------------------------------------------------------------


class TestResolveSystemPrompt:
    def test_no_caller_prompt_falls_back_to_the_default(self):
        # The SDK sends `--system-prompt ""` for None, so without this the model
        # runs with no instructions at all.
        assert resolve_system_prompt(None) == DEFAULT_SYSTEM_PROMPT

    def test_a_blank_caller_prompt_falls_back_too(self):
        assert resolve_system_prompt("   \n  ") == DEFAULT_SYSTEM_PROMPT

    def test_the_callers_prompt_is_kept_verbatim(self):
        caller = "You are an AI assistant integrated into Paperless-ngx."
        assert resolve_system_prompt(caller) == caller

    def test_a_demand_is_appended_to_the_callers_prompt(self):
        caller = "You are an AI assistant integrated into Paperless-ngx."
        resolved = resolve_system_prompt(caller, [CLASSIFIER])
        assert resolved.startswith(caller)
        assert f"`{CLASSIFIER}`" in resolved
        assert "must answer" in resolved

    def test_a_demand_is_appended_to_the_default_prompt(self):
        resolved = resolve_system_prompt(None, [CLASSIFIER])
        assert resolved.startswith(DEFAULT_SYSTEM_PROMPT)
        assert f"`{CLASSIFIER}`" in resolved

    def test_no_demand_adds_no_instruction(self):
        assert resolve_system_prompt(None, []) == DEFAULT_SYSTEM_PROMPT

    def test_several_tools_are_all_named(self):
        instruction = required_tool_instruction([CLASSIFIER, "Other", "Third"])
        assert f"`{CLASSIFIER}`" in instruction
        assert "`Other`" in instruction
        assert "or `Third`" in instruction

    def test_the_instruction_covers_unreadable_input(self):
        # The prose replies seen in practice are refusals about poor OCR, so the
        # instruction has to pre-empt exactly that.
        instruction = required_tool_instruction([CLASSIFIER])
        assert "unreadable" in instruction

    def test_no_demand_means_no_nudge(self):
        assert required_tool_nudge([]) == ""

    def test_the_nudge_names_the_tool_and_says_the_reply_was_dropped(self):
        nudge = required_tool_nudge([CLASSIFIER])
        assert f"`{CLASSIFIER}`" in nudge
        assert "discarded" in nudge


# --------------------------------------------------------------------------
# Stand-ins for the SDK's message objects
# --------------------------------------------------------------------------


class TextBlock:
    def __init__(self, text: str):
        self.text = text


class ToolUseBlock:
    def __init__(self, id: str, name: str, input: Dict[str, Any]):
        self.id = id
        self.name = name
        self.input = input


class AssistantMessage:
    def __init__(self, content: List[Any], session_id: str = "sess-1", usage=None):
        self.content = content
        self.session_id = session_id
        self.model = "claude-haiku-4-5"
        self.usage = usage


class ResultMessage:
    def __init__(self, result: Optional[str] = "done", session_id: str = "sess-1"):
        self.subtype = "success"
        self.result = result
        self.session_id = session_id
        self.usage = {"input_tokens": 3979, "output_tokens": 200}
        self.total_cost_usd = 0.001
        self.duration_ms = 1200
        self.num_turns = 1
        self.is_error = False
        self.errors = None


def prose_turn(text="The receipt is too garbled to classify reliably."):
    return [AssistantMessage([TextBlock(text)]), ResultMessage(result=text)]


def tool_turn(name=CLASSIFIER, arguments=None):
    arguments = {"title": "INNO receipt"} if arguments is None else arguments
    return [
        AssistantMessage(
            [ToolUseBlock(id="toolu_1", name=qualified_name(name), input=arguments)]
        ),
        # Never reached: the adapter ends the turn on the tool call itself.
        ResultMessage(),
    ]


class FakeClient:
    """Serves one scripted turn per `query()`, as ClaudeSDKClient does.

    `receive_response()` stops after that turn's ResultMessage, so a retry has to
    call `query()` again to get anything more - which is exactly the behaviour
    being tested.
    """

    def __init__(self, turns: List[List[Any]]):
        self.turns = list(turns)
        self.nudges: List[str] = []
        self.served = 0

    async def query(self, prompt: str, session_id: str = "default") -> None:
        self.nudges.append(prompt)

    async def receive_response(self):
        turn = self.turns[self.served] if self.served < len(self.turns) else []
        self.served += 1
        for message in turn:
            yield message


async def run_pump(turns, required=None, attempts=2, monkeypatch=None):
    """Drive `_pump` over scripted turns and return (payloads, client)."""
    from claude_code_api.core import sdk_session as module

    monkeypatch.setattr(
        module.settings, "sdk_required_tool_attempts", attempts, raising=False
    )

    session = SdkSession(session_id="api-sess", project_path=".")
    session._required_tool_names = list(required or [])
    client = FakeClient(turns)
    session._client = client

    await session._pump()

    payloads = []
    while not session.output_queue.empty():
        payload = session.output_queue.get_nowait()
        if payload is None:
            break
        payloads.append(payload)
    return payloads, client, session


def tool_calls_of(payloads):
    """The tool calls the client would be handed, by name."""
    names = []
    for payload in payloads:
        if payload.get("type") != "assistant":
            continue
        for block in payload.get("message", {}).get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                names.append(block["name"])
    return names


def texts_of(payloads):
    out = []
    for payload in payloads:
        if payload.get("type") != "assistant":
            continue
        for block in payload.get("message", {}).get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(block["text"])
    return out


# --------------------------------------------------------------------------
# The retry itself
# --------------------------------------------------------------------------


class TestRequiredToolRetry:
    @pytest.mark.asyncio
    async def test_prose_is_retried_and_the_tool_call_wins(self, monkeypatch):
        payloads, client, _ = await run_pump(
            [prose_turn(), tool_turn()],
            required=[CLASSIFIER],
            monkeypatch=monkeypatch,
        )

        assert tool_calls_of(payloads) == [CLASSIFIER]
        # The discarded turn must not reach the client as part of the answer.
        assert texts_of(payloads) == []
        assert len(client.nudges) == 1
        assert CLASSIFIER in client.nudges[0]

    @pytest.mark.asyncio
    async def test_a_first_turn_tool_call_is_not_nudged(self, monkeypatch):
        payloads, client, _ = await run_pump(
            [tool_turn()], required=[CLASSIFIER], monkeypatch=monkeypatch
        )

        assert tool_calls_of(payloads) == [CLASSIFIER]
        assert client.nudges == []
        assert client.served == 1

    @pytest.mark.asyncio
    async def test_retries_are_bounded_and_the_last_reply_is_released(
        self, monkeypatch
    ):
        payloads, client, _ = await run_pump(
            [prose_turn("first"), prose_turn("second"), prose_turn("third")],
            required=[CLASSIFIER],
            attempts=2,
            monkeypatch=monkeypatch,
        )

        assert len(client.nudges) == 2
        # Only the final attempt is delivered - an answer the client can see,
        # rather than an empty reply.
        assert texts_of(payloads) == ["third"]
        assert tool_calls_of(payloads) == []

    @pytest.mark.asyncio
    async def test_attempts_can_be_disabled(self, monkeypatch):
        payloads, client, _ = await run_pump(
            [prose_turn("only"), tool_turn()],
            required=[CLASSIFIER],
            attempts=0,
            monkeypatch=monkeypatch,
        )

        assert client.nudges == []
        assert texts_of(payloads) == ["only"]

    @pytest.mark.asyncio
    async def test_without_a_demand_prose_passes_straight_through(self, monkeypatch):
        payloads, client, _ = await run_pump(
            [prose_turn("here you go"), tool_turn()],
            required=[],
            monkeypatch=monkeypatch,
        )

        assert client.nudges == []
        assert texts_of(payloads) == ["here you go"]
        # The second turn was never requested.
        assert client.served == 1

    @pytest.mark.asyncio
    async def test_the_result_terminator_survives_a_retry(self, monkeypatch):
        payloads, _, _ = await run_pump(
            [prose_turn(), tool_turn()],
            required=[CLASSIFIER],
            monkeypatch=monkeypatch,
        )

        results = [p for p in payloads if p.get("type") == "result"]
        assert len(results) == 1
        assert results[0]["subtype"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_claudes_session_id_is_recorded_from_a_discarded_turn(
        self, monkeypatch
    ):
        seen = []
        payloads, _, session = await run_pump(
            [prose_turn(), tool_turn()],
            required=[CLASSIFIER],
            monkeypatch=monkeypatch,
        )
        seen.append(session.cli_session_id)

        # Resuming depends on this id, and the turn that carried it was dropped.
        assert seen == ["sess-1"]

    @pytest.mark.asyncio
    async def test_a_retried_call_reaches_the_client_as_openai_tool_calls(
        self, monkeypatch
    ):
        payloads, _, _ = await run_pump(
            [prose_turn(), tool_turn()],
            required=[CLASSIFIER],
            monkeypatch=monkeypatch,
        )

        messages = [normalize_claude_message(p) for p in payloads]
        response = create_non_streaming_response(
            messages=messages,
            session_id="sess-1",
            model="claude-haiku-4-5",
        )

        message = response["choices"][0]["message"]
        assert response["choices"][0]["finish_reason"] == "tool_calls"
        assert [call["function"]["name"] for call in message["tool_calls"]] == [
            CLASSIFIER
        ]
