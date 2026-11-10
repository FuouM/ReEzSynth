"""Logging configuration helpers for ReEzSynth command-line entry points."""

from __future__ import annotations

import logging
import os

LOG_LEVEL_ENV = "EZSYNTH_LOG_LEVEL"


def _resolve_level(default: int) -> int:
    name = os.environ.get(LOG_LEVEL_ENV, "").strip().upper()
    if not name:
        return default
    level = logging.getLevelName(name)
    return level if isinstance(level, int) else default


def configure_cli_logging(level: int = logging.INFO) -> None:
    """Configure root logging for CLI use (message-only format, stdout-like output).

    Idempotent: ``logging.basicConfig`` is a no-op once the root logger has handlers.
    The ``EZSYNTH_LOG_LEVEL`` environment variable (e.g. DEBUG, INFO, WARNING)
    overrides ``level`` when set to a valid level name. Library code must never call this.
    """
    logging.basicConfig(level=_resolve_level(level), format="%(message)s")
