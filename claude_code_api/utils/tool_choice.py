"""How `tool_choice` is read, for both engines.

A leaf module on purpose. `utils.tools` emulates tool calling for the CLI engine
and the SDK engine registers the caller's tools with Claude for real, but the two
still have to agree on what the caller asked for, so the vocabulary lives here
rather than inside either of them.
"""

from typing import Any, List, Optional, Tuple

CHOICE_AUTO = "auto"
CHOICE_NONE = "none"
CHOICE_REQUIRED = "required"

# tool_choice values that mean "you must call a tool".
FORCED_TOOL_CHOICES = {"required", "any"}
# tool_choice values that mean "no tools at all".
DISABLED_TOOL_CHOICES = {"none"}


def normalize_tool_choice(tool_choice: Any) -> Tuple[str, Optional[str]]:
    """Return (mode, forced_name) where mode is 'none' | 'auto' | 'required'."""
    if tool_choice is None:
        return CHOICE_AUTO, None

    if isinstance(tool_choice, str):
        value = tool_choice.strip().lower()
        if value in DISABLED_TOOL_CHOICES:
            return CHOICE_NONE, None
        if value in FORCED_TOOL_CHOICES:
            return CHOICE_REQUIRED, None
        return CHOICE_AUTO, None

    # Pydantic ToolChoice model or a raw dict from a lenient client.
    function = getattr(tool_choice, "function", None)
    if function is None and isinstance(tool_choice, dict):
        function = tool_choice.get("function")

    name = getattr(function, "name", None)
    if name is None and isinstance(function, dict):
        name = function.get("name")

    if isinstance(name, str) and name:
        return CHOICE_REQUIRED, name

    return CHOICE_AUTO, None


def tool_names(tools: Any) -> List[str]:
    """The caller's tool names, in declaration order."""
    names: List[str] = []
    for tool in tools or []:
        function = getattr(tool, "function", None)
        if function is None and isinstance(tool, dict):
            function = tool.get("function")
        if function is None:
            continue

        name = getattr(function, "name", None)
        if name is None and isinstance(function, dict):
            name = function.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def required_tool_names(request: Any) -> List[str]:
    """The tools the caller demands be called, empty when none are demanded.

    "required" or "any" means the model must call something and any declared
    tool will do; a named `tool_choice` narrows the demand to that one tool.
    Empty for "auto", for "none" and when no tools were declared - in all three
    a reply with no tool call is a legitimate answer.

    A named tool the caller never declared is also empty rather than demanded:
    it was never registered, so no amount of asking could produce it.
    """
    declared = tool_names(getattr(request, "tools", None))
    if not declared:
        return []

    mode, forced_name = normalize_tool_choice(getattr(request, "tool_choice", None))
    if mode != CHOICE_REQUIRED:
        return []
    if forced_name:
        return [forced_name] if forced_name in declared else []
    return declared
