"""The system prompt and nudges the SDK engine runs with.

Two gaps this fills, both created by the SDK engine being *more* native than the
CLI one rather than less:

* With `system_prompt=None` the SDK passes `--system-prompt ""` (verified in
  claude_agent_sdk 0.2.140, `_internal/transport/subprocess_cli.py:568`), so a
  caller who sends no system message leaves the model with no instructions at
  all - nothing saying what it is answering, and nothing saying that a declared
  tool is there to be used.
* `tool_choice` no longer travels through `utils.tools`, because the SDK engine
  does not emulate tool calling. The demand it encoded still has to reach the
  model, and for a real toolset the way to say "you must call this" is to say so.

All pure text, so the wording stays testable without the SDK installed.
"""

from typing import Iterable, Optional, Sequence

DEFAULT_SYSTEM_PROMPT = (
    "You are the model behind an OpenAI-compatible chat completions API. "
    "Answer the request you are given, using only the tools the caller "
    "declared. There is nobody to ask a follow-up question of, no repository to "
    "explore and no work to plan: return the answer itself."
)


def _target(names: Sequence[str]) -> str:
    if len(names) == 1:
        return f"the `{names[0]}` tool"
    listed = ", ".join(f"`{name}`" for name in names[:-1])
    return f"one of these tools: {listed} or `{names[-1]}`"


def required_tool_instruction(names: Sequence[str]) -> str:
    """The sentences that make a prose reply the wrong answer.

    The last one matters more than it looks: the inputs that most often produce
    prose are the ones the model judges too poor to classify - garbled OCR, a
    truncated document - and refusing in text is exactly what the caller cannot
    read.
    """
    if not names:
        return ""
    return (
        f"You must answer this request by calling {_target(names)}. "
        "The caller reads tool calls only, so an explanation, a summary, a "
        "refusal or a clarifying question sent as text is discarded unread. "
        "If the input is incomplete, ambiguous or unreadable, still call the "
        "tool with your best reading of it - that is the answer."
    )


def required_tool_nudge(names: Sequence[str]) -> str:
    """What to send after a turn that ignored the demand.

    Says the reply was discarded, because the model can see its own previous
    turn and would otherwise reasonably treat it as already answered.
    """
    if not names:
        return ""
    return (
        "That reply was discarded: this request is answerable only by calling "
        f"{_target(names)}. Call it now, with your best reading of the input "
        "above. Do not reply with text."
    )


def resolve_system_prompt(
    system_prompt: Optional[str],
    required_names: Iterable[str] = (),
) -> str:
    """The system prompt to run with, given what the caller sent."""
    base = (system_prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
    instruction = required_tool_instruction(list(required_names))
    return f"{base}\n\n{instruction}" if instruction else base
