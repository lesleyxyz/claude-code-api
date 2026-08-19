"""Pytest configuration and fixtures.

The deterministic suite runs against a fake `ClaudeSDKClient` installed over
`claude_agent_sdk`, so no test spawns the Claude binary or reaches the network.
The fake sits *above* the SDK's transport: it yields real SDK message objects,
so `SdkSession`'s translation, interception and retry machinery all run for
real. Set CLAUDE_CODE_API_USE_REAL_CLAUDE=1 to run against the real thing.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from claude_code_api.core.config import settings
from claude_code_api.utils.history import CURRENT_TURN_OPEN

# Now import the app and configuration
from claude_code_api.main import app
from tests.model_utils import get_test_model_id

PROJECT_ROOT = Path(__file__).parent.parent

# What each fake session records, exposed through the `sdk_prompt` and
# `sdk_options` fixtures. Nothing asserts on the real SDK's wire protocol, so
# this is the only way to check a request-level field actually reaches the
# engine.
SDK_PROMPTS: List[str] = []
SDK_OPTIONS: List[Any] = []


def _use_real_claude() -> bool:
    return os.environ.get("CLAUDE_CODE_API_USE_REAL_CLAUDE") == "1"


# ---------------------------------------------------------------------------
# Scripted turns. The router mirrors the old fixture index: substring rules
# matched against the current-turn window of the prompt, LAST match wins -
# transcript scaffolding would otherwise steer selection.
# ---------------------------------------------------------------------------


def _text_turn(text: str, session_id: str):
    def build():
        from claude_agent_sdk import AssistantMessage, TextBlock

        return [
            _assistant(AssistantMessage, [TextBlock(text=text)], session_id),
            _result(session_id),
        ]

    return build


def _tool_turn(tool_name: str, arguments: Dict[str, Any], session_id: str):
    """A turn that calls one of the client's tools (MCP-qualified)."""

    def build():
        from claude_agent_sdk import AssistantMessage, ToolUseBlock

        block = ToolUseBlock(
            id=f"toolu_{session_id}",
            name=f"mcp__client__{tool_name}",
            input=arguments,
        )
        return [
            _assistant(AssistantMessage, [block], session_id),
            _result(session_id),
        ]

    return build


def _assistant(cls, blocks, session_id):
    return cls(
        content=blocks,
        model="claude-haiku-4-5-20251001",
        parent_tool_use_id=None,
        error=None,
        usage={"input_tokens": 12, "output_tokens": 8},
        message_id=None,
        stop_reason=None,
        session_id=session_id,
        uuid=None,
    )


def _result(session_id):
    from claude_agent_sdk import ResultMessage

    return ResultMessage(
        subtype="success",
        duration_ms=1200,
        duration_api_ms=1000,
        is_error=False,
        num_turns=1,
        session_id=session_id,
        stop_reason=None,
        total_cost_usd=0.00002,
        usage={"input_tokens": 12, "output_tokens": 8},
        result="ok",
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


# (substring matches, turn builder) - last match wins, like the old index.json.
SDK_TURN_RULES = [
    (
        ("mapping test", "session map"),
        _text_turn("Mapping acknowledged.", "sess_map_1"),
    ),
    (
        ("list files", "list the files", "use a tool"),
        _tool_turn("list_files", {"path": "."}, "sess_tool_1"),
    ),
    (("hi", "hello"), _text_turn("Hello! How can I help today?", "sess_simple_1")),
    (
        ("weather in paris",),
        _tool_turn("get_weather", {"city": "Paris"}, "sess_weather_1"),
    ),
    (
        ("temp_c",),
        _text_turn("It is 18°C and cloudy in Paris.", "sess_weather_1"),
    ),
]
DEFAULT_TURN = _text_turn("Hello! How can I help today?", "sess_simple_1")


def _resolve_turn(prompt: str):
    marker = prompt.rfind(CURRENT_TURN_OPEN)
    window = (prompt[marker:] if marker != -1 else prompt).lower()
    turn = DEFAULT_TURN
    for matches, builder in SDK_TURN_RULES:
        if any(match in window for match in matches):
            turn = builder
    return turn


class FakeSDKClient:
    """Stands in for `claude_agent_sdk.ClaudeSDKClient`.

    Speaks the client's contract - `query()` then `receive_response()` until a
    ResultMessage - and records every prompt and options object it is handed.
    """

    def __init__(self, options=None):
        self.options = options
        self._pending: List[Any] = []
        SDK_OPTIONS.append(options)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def query(self, prompt: str, session_id: str = "default") -> None:
        SDK_PROMPTS.append(prompt)
        self._pending = _resolve_turn(prompt)()

    async def receive_response(self):
        pending, self._pending = self._pending, []
        for message in pending:
            yield message


@pytest.fixture(scope="session", autouse=True)
def _fake_sdk_client():
    """Install the fake over the real SDK for the whole deterministic suite.

    `SdkSession.start` imports `ClaudeSDKClient` from `claude_agent_sdk` at
    call time, so patching the attribute on the package is sufficient - and
    the only thing that works.
    """
    if _use_real_claude():
        yield None
        return

    import claude_agent_sdk

    real = claude_agent_sdk.ClaudeSDKClient
    claude_agent_sdk.ClaudeSDKClient = FakeSDKClient
    try:
        yield FakeSDKClient
    finally:
        claude_agent_sdk.ClaudeSDKClient = real


def _create_stub_claude_binary(temp_dir: str) -> str:
    """A binary that only answers `--version`.

    The fake intercepts everything above the SDK transport, so nothing ever
    runs this beyond the startup availability gate - which does spawn it.
    """
    runner_path = Path(temp_dir) / "claude_stub.py"
    runner_path.write_text(
        "#!/usr/bin/env python3\nprint('Claude Code 1.0.0')\n", encoding="utf-8"
    )
    os.chmod(runner_path, 0o755)

    if os.name == "nt":
        launcher_path = Path(temp_dir) / "claude.cmd"
        launcher_path.write_text(
            f'@echo off\r\n"{sys.executable}" "{runner_path}" %*\r\n',
            encoding="utf-8",
        )
        return str(launcher_path)

    launcher_path = Path(temp_dir) / "claude"
    launcher_path.write_text(
        f'#!/usr/bin/env sh\nexec "{sys.executable}" "{runner_path}" "$@"\n',
        encoding="utf-8",
    )
    os.chmod(launcher_path, 0o755)
    return str(launcher_path)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before all tests."""
    # Create temporary directory for testing
    test_root = PROJECT_ROOT / "dist" / "tests"
    test_root.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="claude_api_test_", dir=str(test_root))

    # Store original settings
    original_settings = {
        "project_root": getattr(settings, "project_root", None),
        "require_auth": getattr(settings, "require_auth", False),
        "claude_binary_path": getattr(settings, "claude_binary_path", "claude"),
        "database_url": getattr(settings, "database_url", "sqlite:///./test.db"),
        "debug": getattr(settings, "debug", False),
        "session_map_path": getattr(settings, "session_map_path", None),
    }

    # Set test settings
    settings.project_root = os.path.join(temp_dir, "projects")
    settings.require_auth = False

    if not _use_real_claude():
        settings.claude_binary_path = _create_stub_claude_binary(temp_dir)
    else:
        # Ensure the real binary is available when requested
        if not shutil.which(settings.claude_binary_path) and not os.path.exists(
            settings.claude_binary_path
        ):
            raise RuntimeError(
                f"CLAUDE_CODE_API_USE_REAL_CLAUDE=1 but binary not found at {settings.claude_binary_path}"
            )

    settings.database_url = f"sqlite:///{temp_dir}/test.db"
    settings.debug = True
    settings.session_map_path = os.path.join(temp_dir, "session_map.json")

    # Create directories
    os.makedirs(settings.project_root, exist_ok=True)

    yield temp_dir

    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Cleanup warning: {e}")

    # Restore original settings (if they existed)
    for key, value in original_settings.items():
        if value is not None:
            setattr(settings, key, value)


@pytest.fixture
def sdk_prompt(setup_test_environment):
    """Return a callable giving the prompt of each engine turn in this test."""
    if _use_real_claude():
        pytest.skip("prompt recording requires the fake SDK client")

    start = len(SDK_PROMPTS)
    return lambda: SDK_PROMPTS[start:]


@pytest.fixture
def sdk_options(setup_test_environment):
    """Return a callable giving the ClaudeAgentOptions of each engine session."""
    if _use_real_claude():
        pytest.skip("options recording requires the fake SDK client")

    start = len(SDK_OPTIONS)
    return lambda: SDK_OPTIONS[start:]


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_test_client():
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_chat_request():
    """Sample chat completion request."""
    return {
        "model": get_test_model_id(),
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }


@pytest.fixture
def sample_streaming_request():
    """Sample streaming chat completion request."""
    return {
        "model": get_test_model_id(),
        "messages": [{"role": "user", "content": "Tell me a joke"}],
        "stream": True,
    }


@pytest.fixture
def sample_project_request():
    """Sample project creation request."""
    return {"name": "Test Project", "description": "A test project"}


@pytest.fixture
def sample_session_request():
    """Sample session creation request."""
    return {
        "project_id": "test-project",
        "title": "Test Session",
        "model": get_test_model_id(),
    }


# Configure pytest
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test names/paths
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)

        # Mark slow tests
        if any(
            keyword in item.name.lower()
            for keyword in ["concurrent", "performance", "large"]
        ):
            item.add_marker(pytest.mark.slow)
