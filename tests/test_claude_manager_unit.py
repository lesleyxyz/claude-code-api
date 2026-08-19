"""Unit tests for Claude manager helpers."""

import asyncio
import errno
import os
import time
import types
from pathlib import Path

import pytest

from claude_code_api.core import claude_manager as cm
from claude_code_api.core.config import settings


def test_create_and_cleanup_project_directory(tmp_path):
    original_root = settings.project_root
    try:
        settings.project_root = str(tmp_path)
        project_path = cm.create_project_directory("proj1")
        assert os.path.isdir(project_path)
        cm.cleanup_project_directory(project_path)
        assert not os.path.exists(project_path)
    finally:
        settings.project_root = original_root


def test_validate_claude_binary(monkeypatch):
    class Result:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(*_args, **_kwargs):
        return Result(0)

    monkeypatch.setattr(cm.subprocess, "run", fake_run)
    assert cm.validate_claude_binary() is True

    def fake_run_fail(*_args, **_kwargs):
        raise OSError("nope")

    monkeypatch.setattr(cm.subprocess, "run", fake_run_fail)
    assert cm.validate_claude_binary() is False


def test_decode_output_line():
    process = cm.ClaudeProcess(session_id="sess", project_path="/tmp")
    data = process._decode_output_line(b'{"type":"assistant"}\n')
    assert data["type"] == "assistant"

    data = process._decode_output_line(b'data: {"type":"assistant"}\n')
    assert data["type"] == "assistant"

    data = process._decode_output_line(b"not-json\n")
    assert data["type"] == "text"


def spawn_recorder(monkeypatch):
    """Record the argv and kwargs `ClaudeProcess.start` would exec.

    Nothing is spawned, so this is the cheap way to pin how the CLI command
    line is built. Returns a dict with "argv" and "kwargs".
    """
    calls = {"argv": [], "kwargs": {}, "stdin": None}

    class EmptyStream:
        async def readline(self):
            return b""

    class FakeStdin:
        """Records what start() hands the CLI on stdin."""

        def __init__(self):
            self.buffer = b""
            self.closed = False
            self.drained = 0

        def write(self, data):
            self.buffer += data

        async def drain(self):
            self.drained += 1

        def close(self):
            self.closed = True

    class FakeProcess:
        stdout = EmptyStream()
        stderr = EmptyStream()
        returncode = None

        def __init__(self):
            self.stdin = FakeStdin()

        def terminate(self):
            pass

        async def wait(self):
            return 0

    async def fake_create_subprocess_exec(*args, **kwargs):
        calls["argv"] = list(args)
        calls["kwargs"].update(kwargs)
        process = FakeProcess()
        calls["stdin"] = process.stdin
        return process

    async def fake_verify_startup(self):
        return True

    monkeypatch.setattr(
        cm.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )
    monkeypatch.setattr(cm.ClaudeProcess, "_verify_startup", fake_verify_startup)
    return calls


def flag_value(argv, flag):
    """The value following `flag` in argv, or None when the flag is absent."""
    return argv[argv.index(flag) + 1] if flag in argv else None


@pytest.mark.asyncio
async def test_prompt_is_written_to_stdin_and_closed(monkeypatch):
    """The close is what ends the prompt: the CLI reads stdin until EOF."""
    process = cm.ClaudeProcess(session_id="sess", project_path="/tmp")
    calls = spawn_recorder(monkeypatch)

    try:
        assert await process.start(prompt="hello") is True
        assert calls["kwargs"]["stdin"] == asyncio.subprocess.PIPE
        assert calls["stdin"].buffer.decode("utf-8") == "hello"
        assert calls["stdin"].closed is True
    finally:
        await process.stop()


@pytest.mark.asyncio
async def test_prompt_never_appears_in_argv(monkeypatch):
    """A long conversation must not be able to overflow the command line."""
    process = cm.ClaudeProcess(session_id="sess", project_path="/tmp")
    calls = spawn_recorder(monkeypatch)
    prompt = "a very long conversation transcript"

    try:
        assert await process.start(prompt=prompt) is True
        argv = calls["argv"]
        assert prompt not in argv
        # -p is a bare flag now; nothing follows it but another flag.
        following = argv[argv.index("-p") + 1 :]
        assert not following or following[0].startswith("-")
    finally:
        await process.stop()


def test_redaction_keeps_secret_values_out_of_logs():
    """Regression: -p must not be treated as carrying a value.

    When it was, it consumed the *next flag name* as its value, and that
    flag's own value was then logged in the clear.
    """
    safe = cm._redact_command(
        [
            "claude",
            "-p",
            "--system-prompt-file",
            "SECRET SYSTEM PROMPT",
            "--json-schema",
            '{"secret": true}',
            "--model",
            "claude-sonnet-4-5-20250929",
        ]
    )

    assert "SECRET SYSTEM PROMPT" not in safe
    assert '{"secret": true}' not in safe
    # Flag names survive, so the log still shows the shape of the command.
    assert "--system-prompt-file" in safe
    assert "--json-schema" in safe
    # Non-secret values are untouched.
    assert "claude-sonnet-4-5-20250929" in safe


@pytest.fixture
def prompt_dir(tmp_path, monkeypatch):
    """Isolate the system-prompt scratch directory for a test."""
    directory = tmp_path / "prompts"
    monkeypatch.setattr(settings, "prompt_file_dir", str(directory))
    return directory


@pytest.mark.asyncio
async def test_oversized_command_line_raises_rather_than_returning_false(
    monkeypatch, prompt_dir
):
    """Too-large is deterministic and caller-caused, so it propagates.

    Flattening it to False would send it through the model-fallback loop,
    which would fail identically against every model.
    """
    if os.name != "nt":
        pytest.skip("the pre-flight guard is Windows-only by design")

    process = cm.ClaudeProcess(session_id="sess", project_path="/tmp")
    spawn_recorder(monkeypatch)

    with pytest.raises(cm.ClaudeCommandTooLongError):
        await process.start(
            prompt="hi", system_prompt="s", json_schema={"d": "x" * 40000}
        )

    # The prompt file must not survive the failure.
    assert list(prompt_dir.iterdir()) == []


def test_command_line_is_measured_in_utf16_units_not_bytes():
    """Regression: measuring UTF-8 bytes rejected valid non-ASCII commands.

    Windows counts UTF-16 code units. 12k CJK characters cost 12k units but
    36k UTF-8 bytes, so a byte-based cap refused a command line the OS would
    have run without complaint.
    """
    cjk = "中" * 12000
    cmd = ["claude", "-p", "--json-schema", cjk]

    assert cm.command_line_units(cmd) < 13000
    assert len(cjk.encode("utf-8")) == 36000
    assert cm.oversize_command_reason(cmd) is None


def test_command_line_accounts_for_windows_quoting():
    """Regression: a raw sum of argument lengths under-counts badly.

    CPython joins argv with list2cmdline on Windows and escapes every quote;
    JSON is quote-dense, so the real command line runs far over the sum of
    the parts and an unquoted measure lets through commands that then fail.
    """
    payload = '{"a":"b","c":"d"}' * 1600
    cmd = ["claude", "-p", "--json-schema", payload]

    raw = sum(len(part) for part in cmd)
    assert cm.command_line_units(cmd) > raw * 1.2


def test_guard_is_windows_only():
    """POSIX gets an exact answer from execve; a pre-check can only misfire."""
    huge = ["claude", "-p", "--json-schema", "x" * 500_000]

    if os.name == "nt":
        assert cm.oversize_command_reason(huge) is not None
    else:
        assert cm.oversize_command_reason(huge) is None


def test_os_errors_are_recognised_as_too_long():
    too_big = OSError()
    too_big.errno = errno.E2BIG
    assert cm._is_command_too_long_error(too_big) is True

    # Windows reports this as a FileNotFoundError that never mentions argv.
    windows = FileNotFoundError()
    windows.winerror = 206
    assert cm._is_command_too_long_error(windows) is True

    unrelated = OSError()
    unrelated.errno = errno.ENOENT
    assert cm._is_command_too_long_error(unrelated) is False


@pytest.mark.asyncio
async def test_spawn_e2big_is_translated(monkeypatch, prompt_dir):
    """On POSIX nothing is pre-checked, so the OS error is the only signal."""
    process = cm.ClaudeProcess(session_id="sess", project_path="/tmp")

    async def boom(*_args, **_kwargs):
        error = OSError("Argument list too long")
        error.errno = errno.E2BIG
        raise error

    monkeypatch.setattr(cm.asyncio, "create_subprocess_exec", boom)

    with pytest.raises(cm.ClaudeCommandTooLongError):
        await process.start(prompt="hi", system_prompt="s")

    assert list(prompt_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_unrelated_oserror_still_returns_false(monkeypatch, prompt_dir):
    """Only E2BIG/206 is reinterpreted; everything else keeps its behaviour."""
    process = cm.ClaudeProcess(session_id="sess", project_path="/tmp")

    async def boom(*_args, **_kwargs):
        error = OSError("no such binary")
        error.errno = errno.ENOENT
        raise error

    monkeypatch.setattr(cm.asyncio, "create_subprocess_exec", boom)

    assert await process.start(prompt="hi") is False


@pytest.mark.asyncio
async def test_system_prompt_is_passed_as_a_file(monkeypatch, prompt_dir):
    """With tools in play the system prompt is the biggest argument there is."""
    process = cm.ClaudeProcess(session_id="sess", project_path="/tmp")
    calls = spawn_recorder(monkeypatch)
    system_prompt = "SECRET SYSTEM PROMPT " * 50

    try:
        assert await process.start(prompt="hi", system_prompt=system_prompt) is True
        argv = calls["argv"]

        assert "--system-prompt" not in argv
        path = flag_value(argv, "--system-prompt-file")
        assert path is not None
        assert system_prompt not in argv
        # Scratch lives outside the project tree, so no Docker volume keeps it.
        assert Path(path).parent == prompt_dir
        assert Path(path).read_text(encoding="utf-8") == system_prompt
    finally:
        await process.stop()


@pytest.mark.asyncio
async def test_system_prompt_file_is_removed_when_the_process_stops(
    monkeypatch, prompt_dir
):
    process = cm.ClaudeProcess(session_id="sess", project_path="/tmp")
    calls = spawn_recorder(monkeypatch)

    assert await process.start(prompt="hi", system_prompt="secret") is True
    path = Path(flag_value(calls["argv"], "--system-prompt-file"))

    await process.stop()

    assert not path.exists()
    assert list(prompt_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_no_prompt_file_without_a_system_prompt(monkeypatch, prompt_dir):
    process = cm.ClaudeProcess(session_id="sess", project_path="/tmp")
    calls = spawn_recorder(monkeypatch)

    try:
        assert await process.start(prompt="hi") is True
        assert "--system-prompt-file" not in calls["argv"]
    finally:
        await process.stop()


def test_prompt_file_names_are_unique(prompt_dir):
    """Concurrent sessions must not collide on a shared scratch directory."""
    first = cm.write_system_prompt_file("one")
    second = cm.write_system_prompt_file("two")

    try:
        assert first != second
        assert Path(first).read_text(encoding="utf-8") == "one"
        assert Path(second).read_text(encoding="utf-8") == "two"
    finally:
        cm.remove_system_prompt_file(first)
        cm.remove_system_prompt_file(second)

    assert list(prompt_dir.iterdir()) == []


def test_remove_prompt_file_tolerates_a_missing_file(prompt_dir):
    path = cm.write_system_prompt_file("gone")
    cm.remove_system_prompt_file(path)
    cm.remove_system_prompt_file(path)  # must not raise
    cm.remove_system_prompt_file(None)


def test_sweep_removes_only_stale_prompt_files(prompt_dir, monkeypatch):
    """Files are normally removed with their process; this catches crashes."""
    monkeypatch.setattr(settings, "prompt_file_max_age_minutes", 60)
    stale = Path(cm.write_system_prompt_file("orphaned by a crash"))
    fresh = Path(cm.write_system_prompt_file("still in use"))
    unrelated = prompt_dir / "not-ours.txt"
    unrelated.write_text("leave me alone", encoding="utf-8")

    old = time.time() - 2 * 60 * 60
    os.utime(stale, (old, old))

    try:
        assert cm.sweep_stale_prompt_files() == 1
        assert not stale.exists()
        assert fresh.exists()
        assert unrelated.exists()
    finally:
        cm.remove_system_prompt_file(str(fresh))


def test_sweep_is_safe_when_the_directory_is_absent(prompt_dir):
    assert not prompt_dir.exists()
    assert cm.sweep_stale_prompt_files() == 0


@pytest.mark.asyncio
async def test_create_session_rejects_duplicate_active_session(monkeypatch, tmp_path):
    manager = cm.ClaudeManager()

    async def fake_start(self, prompt, model=None, system_prompt=None, **_kwargs):
        self.is_running = True
        return True

    monkeypatch.setattr(cm.ClaudeProcess, "start", fake_start)

    await manager.create_session(
        session_id="sess-dup",
        project_path=str(tmp_path),
        prompt="first prompt",
        model="claude-sonnet-4-5-20250929",
    )

    with pytest.raises(cm.ClaudeSessionConflictError):
        await manager.create_session(
            session_id="sess-dup",
            project_path=str(tmp_path),
            prompt="second prompt",
            model="claude-sonnet-4-5-20250929",
        )

    await manager.cleanup_all()


@pytest.mark.asyncio
async def test_create_session_replaces_stale_process(monkeypatch, tmp_path):
    manager = cm.ClaudeManager()

    async def fake_start(self, prompt, model=None, system_prompt=None, **_kwargs):
        self.is_running = True
        return True

    monkeypatch.setattr(cm.ClaudeProcess, "start", fake_start)

    first_process = await manager.create_session(
        session_id="sess-stale",
        project_path=str(tmp_path),
        prompt="first prompt",
        model="claude-sonnet-4-5-20250929",
    )
    first_process.is_running = False

    second_process = await manager.create_session(
        session_id="sess-stale",
        project_path=str(tmp_path),
        prompt="second prompt",
        model="claude-sonnet-4-5-20250929",
    )

    assert second_process is not first_process
    assert manager.get_session("sess-stale") is second_process

    await manager.cleanup_all()


@pytest.mark.asyncio
async def test_create_session_retries_opus_45_when_opus_46_rejected(
    monkeypatch, tmp_path
):
    manager = cm.ClaudeManager()
    attempted_models = []

    monkeypatch.setattr(
        cm,
        "get_available_models",
        lambda: [
            types.SimpleNamespace(id="claude-opus-4-5-20251101"),
            types.SimpleNamespace(id="claude-opus-4-6-20260205"),
        ],
    )

    async def fake_start(self, prompt, model=None, system_prompt=None, **_kwargs):
        attempted_models.append(model)
        if model == "claude-opus-4-6-20260205":
            self.last_error = "invalid model: claude-opus-4-6-20260205"
            self.is_running = False
            return False
        self.is_running = True
        return True

    monkeypatch.setattr(cm.ClaudeProcess, "start", fake_start)

    process = await manager.create_session(
        session_id="sess-fallback",
        project_path=str(tmp_path),
        prompt="prompt",
        model="claude-opus-4-6-20260205",
    )

    assert process is not None
    assert attempted_models == [
        "claude-opus-4-6-20260205",
        "claude-opus-4-5-20251101",
    ]

    await manager.cleanup_all()


@pytest.mark.asyncio
async def test_create_session_raises_when_model_rejected_without_fallback(
    monkeypatch, tmp_path
):
    manager = cm.ClaudeManager()

    monkeypatch.setattr(
        cm,
        "get_available_models",
        lambda: [types.SimpleNamespace(id="claude-sonnet-4-5-20250929")],
    )

    async def fake_start(self, prompt, model=None, system_prompt=None, **_kwargs):
        self.last_error = "unsupported model"
        self.is_running = False
        return False

    monkeypatch.setattr(cm.ClaudeProcess, "start", fake_start)

    with pytest.raises(cm.ClaudeModelNotSupportedError):
        await manager.create_session(
            session_id="sess-model-error",
            project_path=str(tmp_path),
            prompt="prompt",
            model="claude-opus-4-6-20260205",
        )

    await manager.cleanup_all()


@pytest.mark.asyncio
async def test_create_session_raises_for_non_model_start_failure_without_fallback(
    monkeypatch, tmp_path
):
    manager = cm.ClaudeManager()
    attempted_models = []

    monkeypatch.setattr(
        cm,
        "get_available_models",
        lambda: [types.SimpleNamespace(id="claude-opus-4-5-20251101")],
    )

    async def fake_start(self, prompt, model=None, system_prompt=None, **_kwargs):
        attempted_models.append(model)
        self.last_error = "failed to spawn process"
        self.is_running = False
        return False

    monkeypatch.setattr(cm.ClaudeProcess, "start", fake_start)

    with pytest.raises(cm.ClaudeProcessStartError):
        await manager.create_session(
            session_id="sess-process-error",
            project_path=str(tmp_path),
            prompt="prompt",
            model="claude-opus-4-6-20260205",
        )

    assert attempted_models == ["claude-opus-4-6-20260205"]
    await manager.cleanup_all()


@pytest.mark.asyncio
async def test_create_session_without_model_does_not_force_model_flag(
    monkeypatch, tmp_path
):
    manager = cm.ClaudeManager()
    attempted_models = []

    async def fake_start(self, prompt, model=None, system_prompt=None, **_kwargs):
        attempted_models.append(model)
        self.is_running = True
        return True

    monkeypatch.setattr(cm.ClaudeProcess, "start", fake_start)

    await manager.create_session(
        session_id="sess-no-model",
        project_path=str(tmp_path),
        prompt="prompt",
        model=None,
    )

    assert attempted_models == [None]

    await manager.cleanup_all()
