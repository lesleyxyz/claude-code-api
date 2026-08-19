"""Pytest configuration and fixtures."""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from claude_code_api.core.config import settings
from claude_code_api.utils.engine import ENGINE_CLI
from claude_code_api.utils.history import CURRENT_TURN_OPEN

# Now import the app and configuration
from claude_code_api.main import app
from tests.model_utils import get_test_model_id

PROJECT_ROOT = Path(__file__).parent.parent

# The mock CLI appends one JSON array per invocation here; see the `cli_argv`
# fixture. Nothing asserts on the real CLI's argv, so this is the only way to
# check that a request-level flag actually reaches the binary.
CLI_ARGV_LOG_NAME = "claude_argv.jsonl"
CLI_PROMPT_LOG_NAME = "claude_prompt.jsonl"


def _serialize_fixture_rules(fixture_rules, fixtures_dir: Path):
    serialized_rules = []
    for rule in fixture_rules:
        matches = [str(match).lower() for match in rule.get("match", []) if match]
        fixture_file = rule.get("file")
        if not fixture_file or not matches:
            continue
        serialized_rules.append(
            {
                "matches": matches,
                "fixture_path": str(fixtures_dir / fixture_file),
            }
        )
    return serialized_rules


def _create_mock_claude_binary(
    temp_dir: str,
    default_fixture: Path,
    fixture_rules,
    fixtures_dir: Path,
    argv_log: Path,
    prompt_log: Path,
) -> str:
    """Create a mock Claude CLI launcher that works on POSIX and Windows."""
    serialized_rules = _serialize_fixture_rules(fixture_rules, fixtures_dir)
    runner_path = Path(temp_dir) / "claude_mock.py"
    runner_code = "\n".join(
        [
            "#!/usr/bin/env python3",
            "import json",
            "import sys",
            "",
            f"DEFAULT_FIXTURE = {str(default_fixture)!r}",
            f"FIXTURE_RULES = {serialized_rules!r}",
            f"ARGV_LOG = {str(argv_log)!r}",
            f"PROMPT_LOG = {str(prompt_log)!r}",
            f"CURRENT_TURN_OPEN = {CURRENT_TURN_OPEN!r}",
            "",
            "def _record(args):",
            "    try:",
            "        with open(ARGV_LOG, 'a', encoding='utf-8') as handle:",
            "            handle.write(json.dumps(args) + '\\n')",
            "    except OSError:",
            "        pass",
            "",
            "def _extract_prompt(args):",
            "    for idx, value in enumerate(args):",
            "        if value == '-p':",
            "            nxt = args[idx + 1] if idx + 1 < len(args) else None",
            "            if nxt is not None and not nxt.startswith('-'):",
            "                return nxt",
            "            return sys.stdin.read()",
            "    return ''",
            "",
            "def _record_prompt(prompt):",
            "    try:",
            "        with open(PROMPT_LOG, 'a', encoding='utf-8') as handle:",
            "            handle.write(json.dumps(prompt) + '\\n')",
            "    except OSError:",
            "        pass",
            "",
            "def _resolve_fixture(prompt):",
            "    # Match against the current turn only. Rules are substrings and",
            "    # the LAST match wins, so transcript scaffolding would otherwise",
            "    # steer selection: the 'hi' rule alone is hit by 'this',",
            "    # 'which' and 'history'. Keep this window narrow.",
            "    marker = prompt.rfind(CURRENT_TURN_OPEN)",
            "    window = prompt[marker:] if marker != -1 else prompt",
            "    window_lower = window.lower()",
            "    fixture_path = DEFAULT_FIXTURE",
            "    for rule in FIXTURE_RULES:",
            "        if any(match in window_lower for match in rule['matches']):",
            "            fixture_path = rule['fixture_path']",
            "    return fixture_path",
            "",
            "def main():",
            "    args = sys.argv[1:]",
            "    if args and args[0] == '--version':",
            "        print('Claude Code 1.0.0')",
            "        return 0",
            "    _record(args)",
            "    prompt = _extract_prompt(args)",
            "    _record_prompt(prompt)",
            "    fixture_path = _resolve_fixture(prompt)",
            "    with open(fixture_path, 'r', encoding='utf-8') as handle:",
            "        sys.stdout.write(handle.read())",
            "    return 0",
            "",
            "if __name__ == '__main__':",
            "    raise SystemExit(main())",
            "",
        ]
    )
    runner_path.write_text(runner_code, encoding="utf-8")
    os.chmod(runner_path, 0o755)

    if os.name == "nt":
        launcher_path = Path(temp_dir) / "claude.cmd"
        launcher_code = f'@echo off\r\n"{sys.executable}" "{runner_path}" %*\r\n'
        launcher_path.write_text(launcher_code, encoding="utf-8")
        return str(launcher_path)

    launcher_path = Path(temp_dir) / "claude"
    launcher_code = "\n".join(
        [
            "#!/usr/bin/env sh",
            f'exec "{sys.executable}" "{runner_path}" "$@"',
            "",
        ]
    )
    launcher_path.write_text(launcher_code, encoding="utf-8")
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
        "engine": getattr(settings, "engine", None),
    }

    # Set test settings
    settings.project_root = os.path.join(temp_dir, "projects")
    settings.require_auth = False
    # Pin the CLI engine for the deterministic suite. The mock binary below
    # speaks the CLI's argv/stdin protocol, not the stream-json control
    # protocol the SDK uses, so leaving this on the default would quietly
    # send the whole suite to the real Claude. The SDK engine is covered by
    # tests/test_sdk_engine.py, which builds SDK objects directly.
    settings.engine = ENGINE_CLI

    # Prefer deterministic fixtures unless explicitly using real Claude
    use_real_claude = os.environ.get("CLAUDE_CODE_API_USE_REAL_CLAUDE") == "1"
    if not use_real_claude:
        fixtures_dir = Path(__file__).parent / "fixtures"
        index_path = fixtures_dir / "index.json"
        default_fixture = fixtures_dir / "claude_stream_simple.jsonl"

        fixture_rules = []
        if index_path.exists():
            try:
                fixture_rules = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Failed to parse fixture index: {exc}") from exc

        settings.claude_binary_path = _create_mock_claude_binary(
            temp_dir=temp_dir,
            default_fixture=default_fixture,
            fixture_rules=fixture_rules,
            fixtures_dir=fixtures_dir,
            argv_log=Path(temp_dir) / CLI_ARGV_LOG_NAME,
            prompt_log=Path(temp_dir) / CLI_PROMPT_LOG_NAME,
        )
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
def cli_argv(setup_test_environment):
    """Return a callable giving the argv of each mock-CLI run in this test."""
    if os.environ.get("CLAUDE_CODE_API_USE_REAL_CLAUDE") == "1":
        pytest.skip("argv recording requires the fixture CLI")

    log_path = Path(setup_test_environment) / CLI_ARGV_LOG_NAME
    log_path.write_text("", encoding="utf-8")  # isolate this test

    def _read():
        if not log_path.exists():
            return []
        return [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return _read


@pytest.fixture
def cli_prompt(setup_test_environment):
    """Return a callable giving the prompt each mock-CLI run received.

    The prompt travels on stdin now, so it is no longer visible in `cli_argv`.
    """
    if os.environ.get("CLAUDE_CODE_API_USE_REAL_CLAUDE") == "1":
        pytest.skip("prompt recording requires the fixture CLI")

    log_path = Path(setup_test_environment) / CLI_PROMPT_LOG_NAME
    log_path.write_text("", encoding="utf-8")  # isolate this test

    def _read():
        if not log_path.exists():
            return []
        return [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return _read


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
