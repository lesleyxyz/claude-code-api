"""Project directory management, shared by the chat and projects APIs."""

import os

import structlog

from claude_code_api.core.config import settings
from claude_code_api.core.security import ensure_directory_within_base

logger = structlog.get_logger()


def create_project_directory(project_id: str) -> str:
    """Create project directory."""
    return ensure_directory_within_base(
        project_id,
        settings.project_root,
        allow_subpaths=False,
        sanitize_leaf=True,
    )


def cleanup_project_directory(project_path: str):
    """Clean up project directory."""
    try:
        import shutil

        if os.path.exists(project_path):
            shutil.rmtree(project_path)
            logger.info("Project directory cleaned up", path=project_path)
    except Exception as e:
        logger.error(
            "Failed to cleanup project directory", path=project_path, error=str(e)
        )
