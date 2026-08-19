"""Tests for OpenAI `reasoning_effort` mapped onto the engine's effort level."""

import pytest
from fastapi import HTTPException

from claude_code_api.api.chat import _resolve_reasoning_effort
from claude_code_api.models.openai import ChatCompletionRequest
from claude_code_api.utils.effort import normalize_reasoning_effort
from tests.model_utils import get_test_model_id


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


class TestEffortEndToEnd:
    """Through the HTTP layer, asserting on what the engine actually received."""

    def test_effort_reaches_the_engine(self, test_client, sdk_options):
        response = test_client.post(
            "/v1/chat/completions", json=_request(reasoning_effort="high")
        )
        assert response.status_code == 200

        runs = sdk_options()
        assert len(runs) == 1
        assert runs[0].effort == "high"

    def test_alias_is_mapped_before_reaching_the_engine(self, test_client, sdk_options):
        response = test_client.post(
            "/v1/chat/completions", json=_request(reasoning_effort="minimal")
        )
        assert response.status_code == 200
        assert sdk_options()[0].effort == "low"

    def test_claude_only_level_is_accepted(self, test_client, sdk_options):
        response = test_client.post(
            "/v1/chat/completions", json=_request(reasoning_effort="max")
        )
        assert response.status_code == 200
        assert sdk_options()[0].effort == "max"

    def test_omitting_the_field_changes_nothing(self, test_client, sdk_options):
        response = test_client.post("/v1/chat/completions", json=_request())
        assert response.status_code == 200
        assert sdk_options()[0].effort is None

    def test_bad_value_is_a_400_and_never_reaches_the_engine(
        self, test_client, sdk_options
    ):
        """The engine would accept this with only a warning, so we must not ask it."""
        response = test_client.post(
            "/v1/chat/completions", json=_request(reasoning_effort="bogus")
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "invalid_reasoning_effort"
        assert sdk_options() == []

    def test_streaming_still_carries_the_level(self, test_client, sdk_options):
        response = test_client.post(
            "/v1/chat/completions",
            json=_request(reasoning_effort="high", stream=True),
        )
        assert response.status_code == 200
        assert "[DONE]" in response.text
        assert sdk_options()[0].effort == "high"

    def test_responses_api_forwards_reasoning_effort(self, test_client, sdk_options):
        response = test_client.post(
            "/v1/responses",
            json={
                "model": get_test_model_id(),
                "input": "Hi",
                "reasoning": {"effort": "high"},
            },
        )
        assert response.status_code == 200
        assert sdk_options()[0].effort == "high"

    def test_responses_api_rejects_a_bad_value(self, test_client, sdk_options):
        response = test_client.post(
            "/v1/responses",
            json={
                "model": get_test_model_id(),
                "input": "Hi",
                "reasoning": {"effort": "bogus"},
            },
        )
        assert response.status_code == 400
        assert sdk_options() == []
