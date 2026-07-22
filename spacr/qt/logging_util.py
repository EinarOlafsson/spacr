"""
Qt-side extension of the package-scope logger.

Delegates all file-handler configuration to :mod:`spacr.logging_util`
and adds a :class:`QtLogHandler` that emits every formatted record
over a Qt signal so widgets on the main thread can display them
without cross-thread violations.

Two sinks end up wired at ``spacr-qt`` startup:

1. The rotating file handler at ``~/.spacr/logs/spacr.log``
   (installed by :mod:`spacr.logging_util`).
2. The :class:`QtLogHandler` here — ConsolePanel connects to its
   ``record_ready(str, int)`` signal.

Public API:
    setup_logging(...)   — call once early in ``launch()``.
    get_signal_handler() — the shared QtLogHandler instance.
    log_path()           — absolute path of the rotating log file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

from ..logging_util import (
    log_dir as _package_log_dir,
    log_path as _package_log_path,
    setup_logging as _package_setup_logging,
)


# ---------------------------------------------------------------------------
# Path shims — kept for backwards compatibility with existing callers /
# tests that import log_dir/log_path from spacr.qt.logging_util.
# ---------------------------------------------------------------------------

def log_dir() -> Path:
    """Return the folder where spacr log files live.

    Alias for :func:`spacr.logging_util.log_dir`.
    """
    return _package_log_dir()


def log_path() -> Path:
    """Return the absolute path of the rotating log file.

    Alias for :func:`spacr.logging_util.log_path`.
    """
    return _package_log_path()


# ---------------------------------------------------------------------------
# Qt-side log handler — bridges Python logging → Qt signal
# ---------------------------------------------------------------------------

class QtLogHandler(QObject, logging.Handler):
    """A logging.Handler that emits every formatted record over a Qt
    signal so QWidget slots (running on the main thread) can display
    them without cross-thread violations.

    :ivar record_ready: signal ``(formatted_line, levelno)`` emitted
        once per record.
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
        """Format and re-emit ``record`` over :attr:`record_ready`."""
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

def setup_logging(level: int = logging.INFO,
                    console_level: int = logging.INFO) -> None:
    """Install the file handler + the Qt signal handler on the root
    logger. Idempotent — safe to call more than once.

    :param level: minimum record level for the rotating file handler.
    :param console_level: minimum record level for the Qt signal handler
        (i.e. what ConsolePanel receives).
    """
    global _INITIALISED
    if _INITIALISED:
        return

    # Package-scope file handler — installed once, shared by every
    # spacr subsystem. Explicitly pass log_path() so tests that
    # monkey-patch the Qt-side path are honoured.
    _package_setup_logging(level=level, log_file=log_path())

    # Qt signal handler — only relevant when a QApplication exists.
    qt_h = get_signal_handler()
    qt_h.setLevel(console_level)
    logging.getLogger().addHandler(qt_h)

    _INITIALISED = True
    logging.getLogger("spacr.qt").info(
        "Qt log signal installed → %s", log_path()
    )


def get_logger(name: str = "spacr.qt") -> logging.Logger:
    """Convenience wrapper — returns a child logger under ``spacr.qt``.

    :param name: logger name, defaults to ``"spacr.qt"``.
    """
    return logging.getLogger(name)
