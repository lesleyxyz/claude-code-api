"""Errors raised by the engine, shared with the API layer.

A leaf module so that `api.chat` can catch what `core.sdk_session` raises
without either importing the other's machinery.
"""


class ClaudeManagerError(RuntimeError):
    """Base error for Claude session operations."""


class ClaudeConcurrencyError(ClaudeManagerError):
    """Raised when the concurrent session limit is exceeded."""


class ClaudeProcessStartError(ClaudeManagerError):
    """Raised when a Claude session fails to start."""


class ClaudeSessionConflictError(ClaudeManagerError):
    """Raised when a session already has an active Claude session."""
