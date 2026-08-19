"""Claude Agent SDK engine, an alternative to spawning the CLI per request.

The CLI engine (`core.claude_manager`) shells out to `claude -p` and parses
stream-json off stdout. That works, but it forces two workarounds this module
exists to remove:

* Client tools have to be *emulated*. The CLI has no caller-supplied tools, so
  `utils.tools` describes them in the system prompt and constrains the reply to
  a JSON envelope with `--json-schema`. The SDK takes real tools through an
  in-process MCP server, so a tool call arrives as a genuine `ToolUseBlock`.
* The CLI's own toolset is always present, so the model can go looking for a
  client tool among its built-ins, fail, and report "No such tool available".
  `ClaudeAgentOptions(tools=[])` removes every built-in, leaving only the
  client's.

To keep the blast radius small, SDK messages are translated into exactly the
dicts the CLI's stream-json produces, so `utils.parser`, `utils.streaming` and
the tool bridge keep working unchanged and the two engines stay swappable.
"""

import asyncio
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence

import structlog

from claude_code_api.models.claude import get_default_model

from claude_code_api.utils.engine import (  # noqa: F401  (re-exported)
    ENGINE_CLI,
    ENGINE_SDK,
    ENGINES,
    normalize_engine,
)

from claude_code_api.utils.sdk_tools import (
    CLIENT_TOOL_SERVER,
    build_client_tool_server,
    is_client_tool,
    local_name,
)

from .config import settings

logger = structlog.get_logger()


def _content_block_to_dict(block: Any) -> Optional[Dict[str, Any]]:
    """Translate one SDK content block into a CLI stream-json content block."""
    kind = type(block).__name__

    if kind == "TextBlock":
        return {"type": "text", "text": block.text}

    if kind == "ThinkingBlock":
        # Not part of the OpenAI response shape, but carried through so the
        # parser sees the same structure the CLI emits.
        return {"type": "thinking", "thinking": block.thinking}

    if kind in ("ToolUseBlock", "ServerToolUseBlock"):
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": getattr(block, "input", {}) or {},
        }

    if kind in ("ToolResultBlock", "ServerToolResultBlock"):
        return {
            "type": "tool_result",
            "tool_use_id": getattr(block, "tool_use_id", ""),
            "content": getattr(block, "content", ""),
            "is_error": bool(getattr(block, "is_error", False)),
        }

    logger.debug("Unrecognised SDK content block", block_type=kind)
    return None


def sdk_message_to_stream_dict(message: Any) -> Optional[Dict[str, Any]]:
    """Translate an SDK message into the CLI's stream-json dict shape.

    Returning the CLI's shape rather than a new one is deliberate: every
    consumer downstream (`ClaudeMessage`, the parser, the SSE converter, the
    tool bridge) already understands it, so neither engine needs special cases.
    """
    kind = type(message).__name__

    if kind == "SystemMessage":
        payload: Dict[str, Any] = {"type": "system"}
        payload.update(getattr(message, "data", None) or {})
        payload["subtype"] = getattr(message, "subtype", None)
        return payload

    if kind == "AssistantMessage":
        blocks = [
            block
            for block in (
                _content_block_to_dict(b) for b in getattr(message, "content", []) or []
            )
            if block is not None
        ]
        return {
            "type": "assistant",
            "message": {"role": "assistant", "content": blocks},
            "session_id": getattr(message, "session_id", None),
            "model": getattr(message, "model", None),
            "usage": getattr(message, "usage", None),
        }

    if kind == "UserMessage":
        content = getattr(message, "content", None)
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        else:
            blocks = [
                block
                for block in (_content_block_to_dict(b) for b in content or [])
                if block is not None
            ]
        return {"type": "user", "message": {"role": "user", "content": blocks}}

    if kind == "ResultMessage":
        return {
            "type": "result",
            "subtype": getattr(message, "subtype", None),
            "result": getattr(message, "result", None),
            "session_id": getattr(message, "session_id", None),
            "usage": getattr(message, "usage", None),
            # The CLI calls this cost_usd; keep that name for the parser.
            "cost_usd": getattr(message, "total_cost_usd", None),
            "duration_ms": getattr(message, "duration_ms", None),
            "num_turns": getattr(message, "num_turns", None),
            "error": (
                "; ".join(getattr(message, "errors", None) or [])
                if getattr(message, "is_error", False)
                else None
            ),
        }

    logger.debug("Unrecognised SDK message", message_type=kind)
    return None


class SdkSession:
    """One Agent SDK conversation, shaped like `ClaudeProcess`.

    The public surface deliberately mirrors the CLI engine - `get_output()`,
    `stop()`, `is_running`, `cli_session_id` - so `api.chat` can hold either
    without knowing which it has.
    """

    def __init__(
        self,
        session_id: str,
        project_path: str,
        on_cli_session_id: Optional[Callable[[str], None]] = None,
        on_end: Optional[Callable[["SdkSession"], None]] = None,
    ):
        self.session_id = session_id
        self.cli_session_id: Optional[str] = None
        self.project_path = project_path
        self.is_running = False
        self.last_error: Optional[str] = None
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self._on_cli_session_id = on_cli_session_id
        self._on_end = on_end
        self._client: Any = None
        self._pump_task: Optional[asyncio.Task] = None
        # Set once Claude calls a client tool, so the turn can be ended
        # and the call handed back over HTTP.
        self.intercepted_tool_calls: List[Dict[str, Any]] = []
        self._abandon = asyncio.Event()
        self._tool_server: Any = None
        self._allowed_tools: List[str] = []

    async def _record_tool_call(
        self, name: str, qualified: str, arguments: Dict[str, Any]
    ) -> None:
        """Capture a client tool call, then wait to be torn down.

        The client owns execution, so there is no result to compute here. The
        handler parks until `stop()` abandons it; returning instead would let
        Claude carry on with a fabricated result.
        """
        logger.info(
            "Intercepted client tool call",
            session_id=self.session_id,
            tool=name,
        )
        self.intercepted_tool_calls.append({"name": name, "arguments": arguments})
        await self._abandon.wait()

    def _build_options(
        self,
        model: Optional[str],
        system_prompt: Optional[str],
        effort: Optional[str],
    ) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions

        options: Dict[str, Any] = {
            # Claude's own tools stay off. This engine exists to serve an
            # OpenAI-compatible API, where the only tools that may run are the
            # client's, and leaving the built-ins visible is what lets the
            # model hunt for a client tool among them and then report it
            # missing.
            "tools": [],
            "cwd": self.project_path,
        }
        if model:
            options["model"] = model
        if system_prompt:
            options["system_prompt"] = system_prompt
        if effort:
            options["effort"] = effort
        if self._tool_server is not None:
            options["mcp_servers"] = {CLIENT_TOOL_SERVER: self._tool_server}
            options["allowed_tools"] = self._allowed_tools

        return ClaudeAgentOptions(**options)

    async def start(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        effort: Optional[str] = None,
        client_tools: Optional[Sequence[Any]] = None,
    ) -> bool:
        """Open an SDK session and begin draining it into `output_queue`."""
        self.last_error = None

        if client_tools:
            self._tool_server, self._allowed_tools, names = build_client_tool_server(
                client_tools, self._record_tool_call
            )
            logger.info(
                "Exposing client tools to the SDK natively",
                session_id=self.session_id,
                tool_names=names,
            )

        # json_schema is accepted for signature parity with the CLI engine.
        # Structured output moves to the SDK's own channel in a later stage.
        if json_schema is not None:
            logger.debug(
                "SDK engine ignoring json_schema for now",
                session_id=self.session_id,
            )

        try:
            from claude_agent_sdk import ClaudeSDKClient

            options = self._build_options(model, system_prompt, effort)
            logger.info(
                "Starting Claude SDK session",
                session_id=self.session_id,
                project_path=self.project_path,
                model=model or get_default_model(),
                effort=effort or "<sdk-default>",
            )

            self._client = ClaudeSDKClient(options=options)
            await self._client.__aenter__()
            await self._client.query(prompt)

            self.is_running = True
            self._pump_task = asyncio.create_task(self._pump())
            return True

        except Exception as e:
            self.last_error = str(e)
            await self.stop()
            logger.error(
                "Failed to start Claude SDK session",
                session_id=self.session_id,
                error=str(e),
            )
            return False

    async def _pump(self) -> None:
        """Translate SDK messages onto the queue until the turn ends."""
        try:
            async for message in self._client.receive_response():
                payload = sdk_message_to_stream_dict(message)
                if payload is None:
                    continue

                if self._rewrite_client_tool_names(payload):
                    # Claude asked the client to run a tool. The client owns
                    # execution, so the turn ends here and the call travels
                    # back over HTTP as OpenAI tool_calls.
                    await self.output_queue.put(payload)
                    await self.output_queue.put(self._tool_call_result(payload))
                    break

                cli_session_id = payload.get("session_id")
                if cli_session_id and not self.cli_session_id:
                    self.cli_session_id = cli_session_id
                    logger.info("Extracted SDK session ID", session_id=cli_session_id)
                    if self._on_cli_session_id:
                        self._on_cli_session_id(cli_session_id)

                await self.output_queue.put(payload)
        except Exception as e:
            self.last_error = str(e)
            logger.error(
                "Error reading SDK output", session_id=self.session_id, error=str(e)
            )
        finally:
            await self.output_queue.put(None)
            self.is_running = False
            logger.info("Claude SDK session output ended", session_id=self.session_id)
            if self._on_end:
                self._on_end(self)

    def _rewrite_client_tool_names(self, payload: Dict[str, Any]) -> bool:
        """Map mcp__client__foo back to foo. True if a client tool was called.

        The client declared `foo` and expects `foo` back; the MCP namespace is
        an implementation detail of how it reached Claude.
        """
        if payload.get("type") != "assistant":
            return False

        found = False
        for block in payload.get("message", {}).get("content", []) or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name", "")
            if is_client_tool(name):
                block["name"] = local_name(name)
                found = True
        return found

    def _tool_call_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """A synthetic terminator for an intercepted turn.

        The SDK would emit `error_during_execution` once the turn is torn down,
        which is not what happened: the turn ended because the client has work
        to do. Downstream reads this as a clean end and the tool_use blocks
        already queued decide the finish_reason.
        """
        return {
            "type": "result",
            "subtype": "tool_calls",
            "result": None,
            "session_id": payload.get("session_id") or self.cli_session_id,
            "usage": payload.get("usage"),
            "cost_usd": None,
            "duration_ms": None,
            "num_turns": None,
            "error": None,
        }

    async def get_output(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield translated messages, matching `ClaudeProcess.get_output`."""
        while True:
            try:
                output = await asyncio.wait_for(
                    self.output_queue.get(),
                    timeout=settings.streaming_timeout_seconds,
                )
                if output is None:
                    break
                yield output
            except asyncio.TimeoutError:
                logger.warning("SDK output timeout", session_id=self.session_id)
                break
            except Exception as e:
                logger.error(
                    "Error getting SDK output",
                    session_id=self.session_id,
                    error=str(e),
                )
                break

    async def stop(self) -> None:
        """Close the SDK session and cancel the pump."""
        self.is_running = False
        # Release any handler parked in _record_tool_call first: closing the
        # client while a tool is still in flight can otherwise block.
        self._abandon.set()

        if self._pump_task and not self._pump_task.done():
            self._pump_task.cancel()
        self._pump_task = None

        client, self._client = self._client, None
        if client is not None:
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(
                    "Error closing SDK session",
                    session_id=self.session_id,
                    error=str(e),
                )

        logger.info("Claude SDK session stopped", session_id=self.session_id)


class SdkManager:
    """Manages SDK sessions, mirroring `ClaudeManager`'s consumed surface."""

    def __init__(self) -> None:
        self.sessions: Dict[str, SdkSession] = {}
        self._lock = asyncio.Lock()

    async def get_version(self) -> str:
        from claude_agent_sdk import __version__ as sdk_version

        return f"Claude Agent SDK {sdk_version}"

    async def create_session(
        self,
        session_id: str,
        project_path: str,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        on_cli_session_id: Optional[Callable[[str], None]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        effort: Optional[str] = None,
        client_tools: Optional[Sequence[Any]] = None,
    ) -> SdkSession:
        from .claude_manager import (
            ClaudeProcessStartError,
            ClaudeSessionConflictError,
        )

        async with self._lock:
            existing = self.sessions.get(session_id)
            if existing and existing.is_running:
                raise ClaudeSessionConflictError(
                    f"Session {session_id} already has an active SDK session"
                )
            if existing:
                await existing.stop()
                self.sessions.pop(session_id, None)

            session = SdkSession(
                session_id=session_id,
                project_path=project_path,
                on_cli_session_id=on_cli_session_id,
                on_end=self._cleanup,
            )
            started = await session.start(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                json_schema=json_schema,
                effort=effort,
                client_tools=client_tools,
            )
            if not started:
                raise ClaudeProcessStartError(
                    session.last_error or "Failed to start Claude SDK session"
                )

            self.sessions[session_id] = session
            return session

    def _cleanup(self, session: SdkSession) -> None:
        self.sessions.pop(session.session_id, None)

    def get_session(self, session_id: str) -> Optional[SdkSession]:
        return self.sessions.get(session_id)

    async def stop_session(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session:
            await session.stop()

    async def cleanup_all(self) -> None:
        for session in list(self.sessions.values()):
            await session.stop()
        self.sessions.clear()

    def get_active_sessions(self) -> List[str]:
        return list(self.sessions)
