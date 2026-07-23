"""
Verbose diagnostic logger for the Qt GUI.

When the user flips the "Verbose logging" preference on, spaCR's Python
loggers are dialled up to DEBUG and every log record is echoed into
whatever ConsolePanel is active. That gives the user a very chatty
stream in the same place they're already looking, which is exactly
what you want when triaging a bug report.

The handler is a lazy module-level singleton so multiple
:func:`apply_verbose_logging` calls (e.g. every time the Preferences
dialog is saved) don't stack up handlers or leak references. Turning
verbose off leaves the handler attached but silent — cheaper than
tearing it down and rebuilding it, and safer for cases where a log
record fires mid-toggle.

Design:

* One :class:`_ConsoleForwarder` handler is added to the root ``spacr``
  logger (and to ``spacr.qt``). Its emit() forwards to whatever
  ConsolePanel is registered via :func:`register_console_target`.
* Registration is a weak reference to avoid keeping a closed screen
  alive. If the target has been garbage-collected the record is
  silently dropped.
* Level and format are set once at first registration; only the
  ``verbose`` gate flips DEBUG ↔ INFO afterwards.
"""
from __future__ import annotations

import logging
import weakref
from typing import Any, Optional


# ---------------------------------------------------------------------------
# The single handler instance
# ---------------------------------------------------------------------------

_console_ref: "Optional[weakref.ReferenceType[Any]]" = None
_handler: "Optional[_ConsoleForwarder]" = None
_ATTACHED_LOGGERS = ("spacr", "spacr.qt", "spacr.pipeline_v2",
                        "spacr.qt.plate_queue", "spacr.qt.hf_download",
                        "spacr.updater")


class _ConsoleForwarder(logging.Handler):
    """Forward every record it sees to the currently-registered ConsolePanel.

    Format: ``[HH:MM:SS] name LEVEL  message``. Keeping the timestamp
    short — the console already scrolls fast when verbose is on.
    """

    def emit(self, record: logging.LogRecord) -> None:
        target = _console_ref() if _console_ref is not None else None
        if target is None:
            return
        try:
            msg = self.format(record)
            # ``append_stdout`` is a slot; safe from the Python logging
            # threading model as long as it's Qt::Auto-connected (it is
            # in ConsolePanel — the emit lives in the main thread).
            append = getattr(target, "append_stdout", None)
            if append is None:
                return
            append(msg + "\n")
        except Exception:
            # Never let a logging failure escape into the app.
            pass


def _ensure_handler() -> _ConsoleForwarder:
    """Attach the single :class:`_ConsoleForwarder` to every spaCR
    logger. Idempotent — safe to call from anywhere."""
    global _handler
    if _handler is None:
        _handler = _ConsoleForwarder()
        _handler.setFormatter(logging.Formatter(
            fmt="[%(asctime)s] %(name)s %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
        ))
    for name in _ATTACHED_LOGGERS:
        logger = logging.getLogger(name)
        if _handler not in logger.handlers:
            logger.addHandler(_handler)
    return _handler


def register_console_target(panel: Any) -> None:
    """Point the verbose logger at ``panel`` (a ConsolePanel).

    The target is stored as a :class:`weakref.ref` so a closed screen
    doesn't keep the panel alive. Any earlier target is replaced.
    """
    global _console_ref
    _ensure_handler()
    _console_ref = weakref.ref(panel)


def apply_verbose_logging(on: bool) -> None:
    """Flip DEBUG ↔ INFO on every attached spaCR logger + the handler.

    The user reaches this via the Preferences dialog. It's idempotent
    and cheap — safe to call on every dialog save.
    """
    handler = _ensure_handler()
    level = logging.DEBUG if on else logging.INFO
    handler.setLevel(level)
    for name in _ATTACHED_LOGGERS:
        logging.getLogger(name).setLevel(level)
    if on:
        # Nudge cellpose's own logger to INFO so its "loaded model X"
        # breadcrumbs come through. We deliberately DO NOT touch
        # torch/PIL/matplotlib: torch's built-in handler writes to a
        # stream that pytest captures + closes, and dialling that
        # logger up produces spurious "I/O operation on closed file"
        # noise. Users can raise those loggers manually if they need
        # to.
        logging.getLogger("cellpose").setLevel(logging.INFO)
