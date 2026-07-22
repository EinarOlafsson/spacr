"""
Real Python `logging` setup for the spacr Qt GUI.

Two sinks are wired at app startup:

1. A rotating file at `~/.spacr/logs/spacr-qt.log` (5 MB × 3 backups)
   — captures every DEBUG-and-up record so users can attach a log to
   bug reports without re-running with a special flag.
2. A `QtLogHandler(QObject)` that emits a Qt signal for every
   record; ConsolePanel connects to it and pipes records into the
   merged Console (same stream as pipeline stdout).

Third-party libraries (torch, cellpose, matplotlib, PIL, urllib3)
are pinned at WARNING to keep the console readable during pipelines.

Public API:
    setup_logging(...)   — call once early in launch().
    get_signal_handler() — the QtLogHandler instance (Qt signal
                           `record_ready(str, int)` where int is the
                           logging level).
    log_path()           — absolute path of the rotating log file.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal


# ---------------------------------------------------------------------------
# Where the file log lives
# ---------------------------------------------------------------------------

def log_dir() -> Path:
    root = Path.home() / ".spacr" / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def log_path() -> Path:
    return log_dir() / "spacr-qt.log"


# ---------------------------------------------------------------------------
# Qt-side log handler — bridges Python logging → Qt signal
# ---------------------------------------------------------------------------

class QtLogHandler(QObject, logging.Handler):
    """A logging.Handler that emits every formatted record over a Qt
    signal so QWidget slots (running on the main thread) can display
    them without cross-thread violations.
    """

    record_ready = Signal(str, int)   # (formatted line, levelno)

    def __init__(self, level: int = logging.INFO):
        QObject.__init__(self)
        logging.Handler.__init__(self, level=level)
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:   # noqa: D401
        try:
            text = self.format(record)
            self.record_ready.emit(text + "\n", record.levelno)
        except Exception:
            # Never let a logging failure crash the app
            self.handleError(record)


_SIGNAL_HANDLER: Optional[QtLogHandler] = None
_INITIALISED: bool = False


def get_signal_handler() -> QtLogHandler:
    """Return the shared QtLogHandler. Instantiated on first access."""
    global _SIGNAL_HANDLER
    if _SIGNAL_HANDLER is None:
        _SIGNAL_HANDLER = QtLogHandler()
    return _SIGNAL_HANDLER


# ---------------------------------------------------------------------------
# One-time setup
# ---------------------------------------------------------------------------

# Third-party loggers that spam DEBUG/INFO records we don't need
# during a pipeline run. Everything below WARNING is dropped.
_QUIET_LOGGERS = (
    "PIL",
    "matplotlib",
    "urllib3",
    "asyncio",
    "torch",
    "torchvision",
    "cellpose",
    "tensorflow",
    "botocore",
)


def setup_logging(level: int = logging.INFO,
                    console_level: int = logging.INFO) -> None:
    """Install both the file handler and the Qt signal handler on the
    root logger. Idempotent — safe to call more than once."""
    global _INITIALISED
    if _INITIALISED:
        return
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt_file = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d — %(message)s"
    )
    file_h = logging.handlers.RotatingFileHandler(
        log_path(), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt_file)
    root.addHandler(file_h)

    qt_h = get_signal_handler()
    qt_h.setLevel(console_level)
    root.addHandler(qt_h)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _INITIALISED = True
    logging.getLogger("spacr.qt").info(
        "logging initialised → %s", log_path()
    )


def get_logger(name: str = "spacr.qt") -> logging.Logger:
    """Convenience wrapper — returns a child logger under `spacr`."""
    return logging.getLogger(name)
