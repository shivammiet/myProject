"""
logger.py — Loguru-based logging setup with file and console handlers.

Usage
-----
# In any module:
from ai_project.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Training started")
logger.debug("batch={}", batch)
logger.error("Unexpected error: {}", exc)

# To customise at startup (call once in your entry-point):
from ai_project.utils.logger import setup_logger
setup_logger(level="DEBUG", log_dir="./logs")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _loguru_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
    "<level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} — {message}"
)

_DEFAULT_LEVEL = "INFO"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logger(
    level: str = _DEFAULT_LEVEL,
    log_dir: Optional[str | Path] = None,
    *,
    rotation: str = "10 MB",
    retention: str = "1 week",
    compression: str = "zip",
    colorise: bool = True,
    diagnose: bool = False,
) -> None:
    """
    Configure the global Loguru logger.

    Should be called **once** at application startup (e.g. in your CLI
    entry-point or ``__main__`` block).

    Parameters
    ----------
    level:
        Minimum log level for console output (``DEBUG``, ``INFO``,
        ``WARNING``, ``ERROR``, ``CRITICAL``).
    log_dir:
        If provided, a rotating log file is written to this directory.
    rotation:
        When to rotate the log file (e.g. ``"10 MB"``, ``"1 day"``).
    retention:
        How long to keep old log files (e.g. ``"1 week"``, ``"30 days"``).
    compression:
        Compression format for rotated files (``"zip"``, ``"gz"``).
    colorise:
        Enable ANSI colours in the console sink.
    diagnose:
        Enable Loguru's exception diagnosis (shows variable values on error).
        Disable in production to avoid leaking sensitive data.
    """
    # Remove all existing handlers (safe to call multiple times)
    _loguru_logger.remove()

    # --- Console handler ---
    _loguru_logger.add(
        sys.stderr,
        level=level.upper(),
        format=_DEFAULT_FORMAT,
        colorize=colorise,
        diagnose=diagnose,
        backtrace=True,
    )

    # --- File handler ---
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "app_{time:YYYY-MM-DD}.log"
        _loguru_logger.add(
            str(log_file),
            level=level.upper(),
            format=_FILE_FORMAT,
            rotation=rotation,
            retention=retention,
            compression=compression,
            diagnose=diagnose,
            backtrace=True,
            enqueue=True,  # Thread-safe async writes
        )

    _loguru_logger.debug(
        "Logger configured | level={} | file_sink={}",
        level,
        str(log_dir) if log_dir else "disabled",
    )


def get_logger(name: str):  # type: ignore[return]
    """
    Return a module-level logger bound with the given *name*.

    This is the preferred way to obtain a logger in any module:

    .. code-block:: python

        from ai_project.utils.logger import get_logger
        logger = get_logger(__name__)

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    loguru.Logger
        A context-bound Loguru logger.
    """
    return _loguru_logger.bind(name=name)


# ---------------------------------------------------------------------------
# Default setup (applied when this module is first imported)
# ---------------------------------------------------------------------------
# Provides sensible defaults so the logger works without calling setup_logger.
_loguru_logger.remove()
_loguru_logger.add(
    sys.stderr,
    level=_DEFAULT_LEVEL,
    format=_DEFAULT_FORMAT,
    colorize=True,
    diagnose=False,
    backtrace=True,
)
