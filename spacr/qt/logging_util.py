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




class _RecordRelay(QObject):
    """Own the Qt signal separately from ``logging.Handler.emit``.

    PySide 6.6 misbinds a signal declared on a QObject/logging.Handler
    multiple-inheritance class: ``signal.emit(text, level)`` resolves back to
    the handler's one-argument ``emit(record)`` method. A plain QObject relay
    avoids that name collision while preserving the public signal instance.
    """

    record_ready = Signal(str, int)


class QtLogHandler(QObject, logging.Handler):
    """A logging.Handler that emits every formatted record over a Qt
    signal so QWidget slots (running on the main thread) can display
    them without cross-thread violations.

    :ivar record_ready: signal ``(formatted_line, levelno)`` emitted
        once per record.

    :param level: the minimum level to relay, as `logging.Handler` takes it.
    """

    def __init__(self, level: int = logging.INFO):
        """Create a logging handler that re-emits records as a Qt signal.

        The signal lives on a small relay object rather than on the handler
        itself, and is re-exported here so the existing
        ``handler.record_ready.connect(...)`` contract is unchanged -- only the
        ``QObject`` that owns it moved.

        :param level: the minimum level to relay.
        """
        QObject.__init__(self)
        logging.Handler.__init__(self, level=level)
        self._record_relay = _RecordRelay(self)
        # The relay exists to deliver records into GUI-thread slots, so it has
        # to LIVE on the GUI thread whichever thread happens to build it.
        # `get_signal_handler` builds the singleton lazily, and one of its
        # callers is `verbose_logger._NotAlreadyShownByTheRootSink.filter`,
        # which runs on whatever thread logged. When a worker's record was the
        # first to ask, the relay was born on that worker: an AutoConnection
        # to a receiver-less slot then queues to a thread with no event loop,
        # and every later record -- the GUI thread's included -- was dropped
        # without an error. Measured in CI run 34961482728 (gw1): the sink
        # built on "Dummy-1", and one Qt warning rendered zero console lines.
        # Pushing from the constructing thread is the direction Qt allows;
        # the relay is a child, so it moves with the handler.
        try:
            from PySide6.QtCore import QCoreApplication

            application = QCoreApplication.instance()
            if (application is not None
                    and self.thread() is not application.thread()):
                self.moveToThread(application.thread())
        except Exception:                                  # noqa: BLE001
            pass
        self.record_ready = self._record_relay.record_ready
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:   # noqa: D401
        """Format and re-emit ``record`` over :attr:`record_ready`.

        Records produced *while* a console panel is mid-write are dropped.
        Every ``ConsolePanel`` in the process subscribes to
        :attr:`record_ready`, so without this a record logged from inside
        ``append_stdout`` — which the function-trace profile hook emits on
        entry to every spaCR function — comes straight back into the same
        widget. ``_StdoutBlock.append`` answers it with a nested
        ``setPlainText``, and the inner call destroys the QTextDocument's
        frames while the outer one is still inside
        ``QTextDocumentPrivate::clear()``: a segfault, gdb'd to
        ``QTextFrame::~QTextFrame``.

        The latch lives in :mod:`spacr.qt.verbose_logger` because that
        module owns the console-target contract; this is the second sink
        that has to honour it. Measured: a 30-file shard still dumped core
        in the same place when only the first sink was guarded.
        """
        try:
            from .verbose_logger import console_write_in_progress
            if console_write_in_progress():
                return
        except Exception:
            pass
        try:
            text = self.format(record)
            self.record_ready.emit(text + "\n", record.levelno)
        except Exception:
            self.handleError(record)


_SIGNAL_HANDLER: Optional[QtLogHandler] = None
_INITIALISED: bool = False


def get_signal_handler() -> QtLogHandler:
    """Return the shared QtLogHandler. Instantiated on first access."""
    global _SIGNAL_HANDLER
    if _SIGNAL_HANDLER is None:
        _SIGNAL_HANDLER = QtLogHandler()
    return _SIGNAL_HANDLER



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

    _package_setup_logging(level=level, log_file=log_path())

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
