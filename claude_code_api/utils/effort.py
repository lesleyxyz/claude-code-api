"""Map OpenAI's `reasoning_effort` onto the Claude CLI's `--effort` flag.

The two vocabularies almost line up: `low`, `medium` and `high` are shared.
OpenAI additionally has `none` and `minimal`, which have no CLI equivalent and
are mapped down to `low` - the CLI's floor - rather than rejected, so that
strict OpenAI clients keep working. The CLI's own `xhigh` and `max` are accepted
as a Claude-specific extension.

Validation has to happen here rather than being left to the CLI: an unknown
`--effort` value only produces a stderr warning and silently runs at the default
effort, so a typo would otherwise return 200 with quietly wrong behaviour.
"""

from typing import Optional

# Levels the CLI's --effort flag accepts (verified against Claude Code 2.1.220).
CLI_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Request values we accept, mapped onto those levels. Claude-only levels map to
# themselves so callers can reach the full CLI range.
REASONING_EFFORT_MAP = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

REASONING_EFFORT_VALUES = tuple(REASONING_EFFORT_MAP)


def normalize_reasoning_effort(value: Optional[str]) -> Optional[str]:
    """Return the CLI effort level for a request value.

    Returns None when nothing was asked for, so the flag is left off entirely
    and the CLI default applies. Raises ValueError for an unrecognised value.

    The return value always comes from the map, never from the caller's string,
    so the token appended to the CLI argv is one of a handful of literals.
    """
    if value is None:
        return None

    requested = value.strip().lower()
    if not requested:
        return None

    effort = REASONING_EFFORT_MAP.get(requested)
    if effort is None:
        raise ValueError(
            f"Unsupported reasoning_effort {value!r}. Supported values: "
            f"{', '.join(REASONING_EFFORT_VALUES)}."
        )
    return effort
