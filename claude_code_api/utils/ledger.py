"""Decide whether a request can continue an existing Claude session.

OpenAI clients are stateless: they resend the whole conversation every request.
The `flatten` strategy takes that literally and rebuilds a transcript each time,
which is correct but pays for the entire history on every turn.

The Agent SDK can do better. `resume=<session id>` continues the conversation
Claude already has, so only the *new* messages need sending. Verified against
SDK 0.2.140: a resumed session keeps its context, and it accepts the result of a
tool call whose turn was torn down - the model uses the result rather than
re-issuing the call.

The risk with resuming is answering confidently from a session that does not
hold the conversation the client thinks it does. Two things guard against it:

* the client's messages are matched against a ledger of what this session has
  already been told, and any mismatch falls back to `flatten`;
* the ledger is held in memory only, never persisted. If the gateway restarts,
  the ledger is gone and every request falls back to `flatten` - correct, just
  costlier. A *persisted* ledger paired with a lost Claude session would be the
  dangerous combination: a delta sent into a session with no history behind it,
  answered confidently from nothing.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import structlog

from claude_code_api.utils.history import (
    render_prompt,
    tool_call_names,
)

logger = structlog.get_logger()

MODE_RESUME = "resume"
MODE_FLATTEN = "flatten"


@dataclass
class SessionLedger:
    """What a Claude session has already been told, for this API session."""

    fingerprints: List[str] = field(default_factory=list)
    # Tool names by call id, so a returned result can be labelled for the model.
    tool_call_names: Dict[str, str] = field(default_factory=dict)
    # The system prompt in force when the session started. Claude's cannot be
    # retracted mid-session, so a change means the session is unusable.
    system_fingerprint: Optional[str] = None

    def extend(self, messages: Sequence[Any], names: Dict[str, str]) -> None:
        self.fingerprints.extend(fingerprint_message(m) for m in messages)
        self.tool_call_names.update(names)


@dataclass
class TurnPlan:
    """How to run one request."""

    mode: str
    prompt: str
    resume_session_id: Optional[str]
    reason: str
    # Messages this turn accounts for, appended to the ledger once it succeeds.
    consumed: List[Any] = field(default_factory=list)

    @property
    def resuming(self) -> bool:
        return self.mode == MODE_RESUME and self.resume_session_id is not None


def _role(message: Any) -> str:
    role = getattr(message, "role", None)
    if role is None and isinstance(message, dict):
        role = message.get("role")
    return str(role or "")


def _field(message: Any, name: str) -> Any:
    value = getattr(message, name, None)
    if value is None and isinstance(message, dict):
        value = message.get(name)
    return value


def _text_of(message: Any) -> str:
    getter = getattr(message, "get_text_content", None)
    if callable(getter):
        return getter() or ""
    content = _field(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif not isinstance(item, dict):
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _canonical_tool_calls(message: Any) -> List[List[str]]:
    """Tool calls in a form that survives a client's serializer.

    Arguments are reparsed and re-emitted with sorted keys, because a client
    that round-trips our response will not reproduce our byte ordering.
    """
    canonical = []
    for call in _field(message, "tool_calls") or []:
        function = _field(call, "function")
        if function is None:
            continue
        raw = _field(function, "arguments")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                pass
        try:
            arguments = json.dumps(raw or {}, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            arguments = str(raw)
        canonical.append(
            [
                str(_field(call, "id") or ""),
                str(_field(function, "name") or ""),
                arguments,
            ]
        )
    return canonical


def fingerprint_message(message: Any) -> str:
    """A stable identity for one message.

    Deliberately lossy. We fingerprint a message we produced, then have to
    recognise it after the client's serializer has been through it - which drops
    `content: null`, adds `refusal: null`, reorders keys and reformats argument
    JSON. Only role, visible text, tool calls and any tool_call_id are used.
    """
    payload = [
        _role(message),
        _text_of(message).strip(),
        _canonical_tool_calls(message),
        str(_field(message, "tool_call_id") or ""),
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def fingerprint_system(system_prompt: Optional[str]) -> str:
    text = (system_prompt or "").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def chain_hash(fingerprints: Sequence[str]) -> str:
    """One identity for a conversation prefix.

    Stateless OpenAI clients do not carry a session id, so a resumable session
    has to be found by content instead. Hashing the fingerprint chain gives a
    key for "the conversation up to here", which the next request will extend.
    """
    digest = hashlib.sha256()
    for fingerprint in fingerprints:
        digest.update(fingerprint.encode("ascii"))
        digest.update(b"|")
    return digest.hexdigest()[:32]


def match_prefix(
    index: Dict[str, str], fingerprints: Sequence[str]
) -> Optional[Tuple[str, int]]:
    """Find the session that already holds the longest prefix of this request.

    Returns (session id, number of messages that session was already sent), or
    None. Longest wins so a conversation keeps extending the same session
    rather than restarting from an earlier fork of itself.
    """
    best: Optional[Tuple[str, int]] = None
    digest = hashlib.sha256()

    for count, fingerprint in enumerate(fingerprints, start=1):
        digest.update(fingerprint.encode("ascii"))
        digest.update(b"|")
        session_id = index.get(digest.hexdigest()[:32])
        if session_id is not None:
            best = (session_id, count)
    return best


def conversation_fingerprints(messages: Sequence[Any]) -> List[str]:
    """Fingerprints of the non-system messages, in order."""
    return [fingerprint_message(m) for m in messages if _role(m) != "system"]


def _common_prefix(ledger: Sequence[str], fingerprints: Sequence[str]) -> int:
    length = 0
    for known, incoming in zip(ledger, fingerprints):
        if known != incoming:
            break
        length += 1
    return length


def render_delta(messages: Sequence[Any], names: Dict[str, str]) -> str:
    """Render the new messages as a single prompt for a resumed session.

    Claude already holds everything before these. Tool results are labelled
    with the tool and call id so the model can line them up with the calls it
    made, which is what lets it answer instead of calling again.
    """
    results, others = [], []

    for message in messages:
        role = _role(message)
        text = _text_of(message).strip()
        if role == "tool":
            call_id = str(_field(message, "tool_call_id") or "")
            name = _field(message, "name") or names.get(call_id) or "tool"
            label = f"{name}" + (f" (call {call_id})" if call_id else "")
            results.append(f"### {label}\n{text}")
        elif text:
            others.append(text)

    sections = []
    if results:
        sections.append(
            "Results of the tool calls you just requested:\n\n" + "\n\n".join(results)
        )
    sections.extend(others)
    return "\n\n".join(sections)


def plan_turn(
    messages: Sequence[Any],
    ledger: Optional[SessionLedger],
    sdk_session_id: Optional[str],
    system_prompt: Optional[str],
    max_chars: int = 0,
) -> TurnPlan:
    """Choose between continuing Claude's session and replaying the transcript.

    Every rejection carries a reason, so a deployment that never manages to
    resume can be diagnosed from the logs rather than guessed at.
    """
    conversation = [m for m in messages if _role(m) != "system"]

    def flatten(reason: str) -> TurnPlan:
        return TurnPlan(
            mode=MODE_FLATTEN,
            prompt=render_prompt(messages, max_chars=max_chars),
            resume_session_id=None,
            reason=reason,
            consumed=list(conversation),
        )

    if not sdk_session_id:
        return flatten("no Claude session recorded for this API session")
    if ledger is None or not ledger.fingerprints:
        return flatten("nothing recorded as already sent")

    if ledger.system_fingerprint is not None:
        if ledger.system_fingerprint != fingerprint_system(system_prompt):
            # The session's system prompt was fixed when it started.
            return flatten("system prompt changed since the session started")

    fingerprints = [fingerprint_message(m) for m in conversation]
    prefix = _common_prefix(ledger.fingerprints, fingerprints)

    if prefix < len(ledger.fingerprints):
        # The client trimmed, edited or reordered what we had already sent, so
        # Claude's copy of the conversation is no longer the client's.
        return flatten("client history diverges from what the session was sent")

    delta = conversation[prefix:]
    if not delta:
        return flatten("nothing new to send")
    if any(_role(m) == "assistant" for m in delta):
        # An assistant turn we never produced, or one whose commit was lost.
        return flatten("delta contains an assistant message the session never sent")

    names = dict(ledger.tool_call_names)
    names.update(tool_call_names(conversation))

    prompt = render_delta(delta, names)
    if not prompt.strip():
        return flatten("delta rendered empty")

    return TurnPlan(
        mode=MODE_RESUME,
        prompt=prompt,
        resume_session_id=sdk_session_id,
        reason=f"resuming with {len(delta)} new message(s)",
        consumed=list(delta),
    )


def is_session_missing_error(error: BaseException) -> bool:
    """Whether an SDK failure means the session we tried to resume is gone.

    The SDK surfaces this as a plain result error, so the message is the only
    signal - but it is a loud one, which is what makes resuming safe to attempt.
    """
    text = str(error).lower()
    return (
        "no conversation found" in text or "session id" in text and "not found" in text
    )
