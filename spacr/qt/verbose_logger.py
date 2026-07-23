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

import functools
import logging
import os
import weakref
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# The single handler instance
# ---------------------------------------------------------------------------

_console_ref: "Optional[weakref.ReferenceType[Any]]" = None
_handler: "Optional[_ConsoleForwarder]" = None
_file_handler: "Optional[RotatingFileHandler]" = None
_ATTACHED_LOGGERS = ("spacr", "spacr.qt", "spacr.pipeline_v2",
                        "spacr.qt.plate_queue", "spacr.qt.hf_download",
                        "spacr.updater", "spacr.trace")


# ---------------------------------------------------------------------------
# Log file — always on, so a crash/hang can be diagnosed after the fact
# ---------------------------------------------------------------------------

def log_dir() -> Path:
    """Return ``~/.spacr/logs/`` — created if it doesn't exist.

    Overridable via the ``SPACR_LOG_DIR`` env var so tests can point
    the log at a tmp directory."""
    override = os.environ.get("SPACR_LOG_DIR")
    if override:
        p = Path(override)
    else:
        p = Path.home() / ".spacr" / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def current_log_file() -> Path:
    """Path of today's rotating log file."""
    return log_dir() / f"spacr-{datetime.now().strftime('%Y%m%d')}.log"


def _ensure_file_handler() -> RotatingFileHandler:
    """Attach a rotating file handler to every attached spaCR logger.

    Idempotent. The handler writes to ``~/.spacr/logs/spacr-YYYYMMDD.log``,
    rotates at 5 MB, and keeps 5 backups. Always attached — this is
    NOT gated by the verbose preference so bug reports from users who
    never turned verbose logging on still have a trail we can read.

    Level is INFO by default; verbose mode drops it to DEBUG (same as
    the console forwarder).
    """
    global _file_handler
    if _file_handler is not None:
        # Already attached — but ensure the file target matches today's
        # date (post-midnight the rotate would otherwise keep the old
        # filename).
        return _file_handler
    try:
        handler = RotatingFileHandler(
            str(current_log_file()),
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
    except Exception:
        # If we can't open the file, don't crash — just skip file
        # logging so the app still runs.
        return None                                                       # type: ignore[return-value]
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    handler.setLevel(logging.INFO)
    for name in _ATTACHED_LOGGERS:
        logger = logging.getLogger(name)
        if handler not in logger.handlers:
            logger.addHandler(handler)
        # Ensure records propagate to the root logger's format if any.
        logger.setLevel(min(logger.level or logging.INFO, logging.INFO))
    _file_handler = handler
    return handler


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
    """Flip DEBUG ↔ INFO on every attached spaCR logger + handlers.

    The user reaches this via the Preferences dialog. It's idempotent
    and cheap — safe to call on every dialog save. Also ensures the
    rotating file handler is attached so bug reports always have a
    trail on disk regardless of verbose state.
    """
    handler = _ensure_handler()
    file_handler = _ensure_file_handler()
    level = logging.DEBUG if on else logging.INFO
    handler.setLevel(level)
    if file_handler is not None:
        file_handler.setLevel(level)
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


# ---------------------------------------------------------------------------
# Function-entry / button-press auto-logging
# ---------------------------------------------------------------------------

def is_verbose() -> bool:
    """Cheap runtime check — decorated functions call this on entry so
    they emit NOTHING when verbose mode is off."""
    return _handler is not None and _handler.level == logging.DEBUG


def log_call(fn: Callable) -> Callable:
    """Decorator: log entry + return of ``fn`` when verbose mode is on.

    Zero cost when verbose is off (the wrapper does one attribute
    check and forwards). When on, emits:

        [class.func] args=… kwargs=…
        [class.func] -> return-repr

    Truncates giant reprs to 240 chars so a settings dict with 100
    entries doesn't wreck the console.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_verbose():
            return fn(*args, **kwargs)
        label = _label_for(fn, args)
        logger = logging.getLogger("spacr.trace")
        a_str = _brief(args[1:] if _looks_bound(fn, args) else args)
        k_str = _brief(kwargs) if kwargs else ""
        logger.debug("[%s] args=%s kwargs=%s", label, a_str, k_str)
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            logger.debug("[%s] RAISED %s: %s", label, type(e).__name__, e)
            raise
        logger.debug("[%s] -> %s", label, _brief(result))
        return result
    return wrapper


def log_button_press(button_name: str,
                        context: Optional[dict] = None) -> None:
    """Fire a one-line trace record documenting a UI button press.

    Wire this from Qt slot handlers so the console shows exactly which
    button the user hit, with any relevant context values (e.g. the
    current settings dict on a Run press).
    """
    if not is_verbose():
        return
    logger = logging.getLogger("spacr.trace")
    if context:
        logger.debug("[button:%s] %s", button_name, _brief(context))
    else:
        logger.debug("[button:%s] pressed", button_name)


def _label_for(fn: Callable, args: tuple) -> str:
    """Return "ClassName.method_name" when fn is a bound method, else
    just the function's __qualname__."""
    q = getattr(fn, "__qualname__", fn.__name__)
    return q


def _looks_bound(fn: Callable, args: tuple) -> bool:
    """Rough check for whether the first arg is ``self`` — if so, we
    hide it from the args snapshot."""
    if not args:
        return False
    first = args[0]
    q = getattr(fn, "__qualname__", "")
    if "." not in q:
        return False
    cls_name = q.split(".", 1)[0]
    return type(first).__name__ == cls_name


def _brief(value: Any, max_chars: int = 240) -> str:
    """Return a truncated ``repr(value)`` capped to ``max_chars``."""
    try:
        s = repr(value)
    except Exception:
        s = f"<{type(value).__name__} — repr failed>"
    if len(s) > max_chars:
        s = s[: max_chars - 3] + "…"
    return s
