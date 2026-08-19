"""Which backend serves the API: the Claude Code CLI, or the Agent SDK.

A leaf module on purpose. `core.config` needs the validator and `core.sdk_session`
needs `core.config`, so the names live here to keep that from becoming a cycle -
the same reason `utils.history` holds its own mode enum.
"""

from typing import Optional

ENGINE_CLI = "cli"
ENGINE_SDK = "sdk"
ENGINES = (ENGINE_CLI, ENGINE_SDK)


def normalize_engine(value: Optional[str]) -> str:
    """Return a canonical engine name, defaulting to the SDK.

    The SDK engine is the default because it supports the caller's tools
    natively; the CLI engine has to emulate them with a JSON envelope. Set
    ENGINE=cli to go back to spawning `claude -p` per request.

    Raises ValueError for an unrecognised value so a typo in a server env var
    fails at startup rather than silently selecting the wrong engine.
    """
    if value is None:
        return ENGINE_SDK

    requested = str(value).strip().lower()
    if not requested:
        return ENGINE_SDK

    if requested not in ENGINES:
        supported = ", ".join(ENGINES)
        raise ValueError(
            f"Unsupported engine {value!r}. Supported values: {supported}."
        )
    return requested
