"""Tests for OpenAI `reasoning_effort` mapped onto the CLI's --effort flag."""

import types

import pytest
from fastapi import HTTPException

from claude_code_api.api.chat import _resolve_reasoning_effort
from claude_code_api.core import claude_manager as cm
from claude_code_api.models.openai import ChatCompletionRequest
from claude_code_api.utils.effort import normalize_reasoning_effort
from tests.model_utils import get_test_model_id
from tests.test_claude_manager_unit import flag_value, spawn_recorder


def _request(**overrides):
    payload = {
        "model": get_test_model_id(),
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }
    payload.update(overrides)
    return payload


class TestEffortMapping:
    """The request value -> CLI level table."""

    @pytest.mark.parametrize(
        "requested,expected",
        [
            ("none", "low"),
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "xhigh"),
            ("max", "max"),
            ("High", "high"),
            ("  medium  ", "medium"),
            ("MAX", "max"),
        ],
    )
    def test_supported_values(self, requested, expected):
        assert normalize_reasoning_effort(requested) == expected

    @pytest.mark.parametrize("blank", [None, "", "   "])
    def test_blank_means_no_flag(self, blank):
        assert normalize_reasoning_effort(blank) is None

    @pytest.mark.parametrize("bad", ["bogus", "ultracode", "highest", "1"])
    def test_unknown_values_raise(self, bad):
        with pytest.raises(ValueError) as excinfo:
            normalize_reasoning_effort(bad)
        assert "Supported values" in str(excinfo.value)

    def test_endpoint_helper_maps_the_alias(self):
        request = ChatCompletionRequest(**_request(reasoning_effort="minimal"))
        assert _resolve_reasoning_effort(request) == "low"

    def test_endpoint_helper_raises_openai_shaped_400(self):
        request = ChatCompletionRequest(**_request(reasoning_effort="bogus"))

        with pytest.raises(HTTPException) as excinfo:
            _resolve_reasoning_effort(request)

        error = excinfo.value.detail["error"]
        assert excinfo.value.status_code == 400
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "invalid_reasoning_effort"
        assert "bogus" in error["message"]


class TestEffortReachesTheCli:
    """Flag construction, without spawning anything."""

    @pytest.mark.asyncio
    async def test_effort_is_appended_to_argv(self, monkeypatch):
        process = cm.ClaudeProcess(session_id="sess", project_path="/tmp")
        calls = spawn_recorder(monkeypatch)

        try:
            assert await process.start(prompt="hi", model="m", effort="high") is True
        finally:
            await process.stop()

        argv = calls["argv"]
        assert flag_value(argv, "--effort") == "high"
        assert argv.index("--effort") < argv.index("--output-format")

    @pytest.mark.asyncio
    async def test_no_effort_means_no_flag(self, monkeypatch):
        process = cm.ClaudeProcess(session_id="sess", project_path="/tmp")
        calls = spawn_recorder(monkeypatch)

        try:
            assert await process.start(prompt="hi") is True
        finally:
            await process.stop()

        assert "--effort" not in calls["argv"]

    @pytest.mark.asyncio
    async def test_empty_effort_leaves_no_dangling_flag(self, monkeypatch):
        """A blank value must not swallow the next flag as its argument."""
        process = cm.ClaudeProcess(session_id="sess", project_path="/tmp")
        calls = spawn_recorder(monkeypatch)

        try:
            assert await process.start(prompt="hi", effort="") is True
        finally:
            await process.stop()

        assert "--effort" not in calls["argv"]

    @pytest.mark.asyncio
    async def test_effort_survives_model_fallback(self, monkeypatch, tmp_path):
        manager = cm.ClaudeManager()
        attempts = []

        monkeypatch.setattr(
            cm,
            "get_available_models",
            lambda: [
                types.SimpleNamespace(id="claude-opus-4-5-20251101"),
                types.SimpleNamespace(id="claude-opus-4-6-20260205"),
            ],
        )

        async def fake_start(self, prompt, model=None, system_prompt=None, **kwargs):
            attempts.append((model, kwargs.get("effort")))
            if model == "claude-opus-4-6-20260205":
                self.last_error = "invalid model: claude-opus-4-6-20260205"
                self.is_running = False
                return False
            self.is_running = True
            return True

        monkeypatch.setattr(cm.ClaudeProcess, "start", fake_start)

        await manager.create_session(
            session_id="sess-effort-fallback",
            project_path=str(tmp_path),
            prompt="prompt",
            model="claude-opus-4-6-20260205",
            effort="xhigh",
        )

        assert attempts == [
            ("claude-opus-4-6-20260205", "xhigh"),
            ("claude-opus-4-5-20251101", "xhigh"),
        ]

        await manager.cleanup_all()


class TestEffortEndToEnd:
    """Through the HTTP layer, asserting on what the CLI actually received."""

    def test_effort_reaches_the_cli(self, test_client, cli_argv):
        response = test_client.post(
            "/v1/chat/completions", json=_request(reasoning_effort="high")
        )
        assert response.status_code == 200

        runs = cli_argv()
        assert len(runs) == 1
        assert flag_value(runs[0], "--effort") == "high"

    def test_alias_is_mapped_before_reaching_the_cli(self, test_client, cli_argv):
        response = test_client.post(
            "/v1/chat/completions", json=_request(reasoning_effort="minimal")
        )
        assert response.status_code == 200
        assert flag_value(cli_argv()[0], "--effort") == "low"

    def test_claude_only_level_is_accepted(self, test_client, cli_argv):
        response = test_client.post(
            "/v1/chat/completions", json=_request(reasoning_effort="max")
        )
        assert response.status_code == 200
        assert flag_value(cli_argv()[0], "--effort") == "max"

    def test_omitting_the_field_changes_nothing(self, test_client, cli_argv):
        response = test_client.post("/v1/chat/completions", json=_request())
        assert response.status_code == 200
        assert "--effort" not in cli_argv()[0]

    def test_bad_value_is_a_400_and_never_spawns_the_cli(self, test_client, cli_argv):
        """The CLI would accept this with only a warning, so we must not ask it."""
        response = test_client.post(
            "/v1/chat/completions", json=_request(reasoning_effort="bogus")
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "invalid_reasoning_effort"
        assert cli_argv() == []

    def test_streaming_still_carries_the_flag(self, test_client, cli_argv):
        response = test_client.post(
            "/v1/chat/completions",
            json=_request(reasoning_effort="high", stream=True),
        )
        assert response.status_code == 200
        assert "[DONE]" in response.text
        assert flag_value(cli_argv()[0], "--effort") == "high"

    def test_responses_api_forwards_reasoning_effort(self, test_client, cli_argv):
        response = test_client.post(
            "/v1/responses",
            json={
                "model": get_test_model_id(),
                "input": "Hi",
                "reasoning": {"effort": "high"},
            },
        )
        assert response.status_code == 200
        assert flag_value(cli_argv()[0], "--effort") == "high"

    def test_responses_api_rejects_a_bad_value(self, test_client, cli_argv):
        response = test_client.post(
            "/v1/responses",
            json={
                "model": get_test_model_id(),
                "input": "Hi",
                "reasoning": {"effort": "bogus"},
            },
        )
        assert response.status_code == 400
        assert cli_argv() == []
