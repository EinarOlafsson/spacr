"""
Package-scope Python `logging` setup for spacr.

Central configuration for every spacr subsystem — core pipelines,
I/O, measure, utilities, and the Qt GUI all funnel through the same
rotating file handler at ``~/.spacr/logs/spacr.log``.

Two ways to opt in:

- Automatic — the Qt GUI calls :func:`setup_logging` at launch, so
  once you run ``spacr-qt`` the log file is populated for the life
  of the session.
- Manual — for headless scripts and notebooks:

  .. code-block:: python

     from spacr.logging_util import setup_logging, get_logger
     setup_logging()                # once, at program start
     LOG = get_logger(__name__)     # in every module that logs
     LOG.info("started")

The log level can be overridden by ``SPACR_LOG_LEVEL`` in the env
(``DEBUG``, ``INFO``, ``WARNING``, …). :func:`enable_debug` and
:func:`disable_debug` are convenience toggles for interactive use.

Third-party libraries that spam INFO records during a spacr pipeline
(torch, cellpose, matplotlib, PIL, urllib3, botocore, tensorflow,
asyncio) are pinned to WARNING so the log stays useful. Add more to
:data:`QUIET_LOGGERS` if a new dependency starts spamming.

Public API:
    setup_logging(level=INFO, log_file=None) — call once early.
    get_logger(name)                          — module-scoped logger.
    enable_debug()                             — crank spacr.* to DEBUG.
    disable_debug()                            — revert to session level.
    log_dir()                                  — folder holding the log.
    log_path()                                 — absolute log file path.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------

DEFAULT_LOG_FILENAME = "spacr.log"
MAX_BYTES = 5 * 1024 * 1024   # 5 MB per file
BACKUP_COUNT = 3               # → up to ~20 MB total
FILE_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d "
    "— %(message)s"
)
STREAM_FORMAT = "%(levelname)s %(name)s: %(message)s"

#: Third-party loggers that spam INFO records — capped at WARNING.
QUIET_LOGGERS: tuple[str, ...] = (
    "PIL",
    "matplotlib",
    # fontTools.subset logs about FORTY lines for every figure saved -- each
    # glyph name and glyph ID, twice, for MATH then GSUB then glyf, followed
    # by a line per font table. A regression run saves a dozen figures, so
    # thousands of lines of glyph inventory bury the run's own output and the
    # user cannot see what happened. "matplotlib" does not cover this:
    # fontTools is a separate top-level logger that matplotlib calls into.
    "fontTools",
    "urllib3",
    "asyncio",
    "torch",
    "torchvision",
    "cellpose",
    "tensorflow",
    "botocore",
    "numba",
    "h5py",
)

#: The five levels the user can switch on and off, lowest first.
LEVELS: tuple[int, ...] = (
    logging.DEBUG, logging.INFO, logging.WARNING,
    logging.ERROR, logging.CRITICAL,
)

#: Per-level file names, alongside the master :data:`DEFAULT_LOG_FILENAME`.
LEVEL_LOG_FILENAMES: dict = {
    logging.DEBUG: "spacr-debug.log",
    logging.INFO: "spacr-info.log",
    logging.WARNING: "spacr-warning.log",
    logging.ERROR: "spacr-error.log",
    logging.CRITICAL: "spacr-critical.log",
}

#: What a fresh install writes and shows. The file keeps everything from INFO
#: up, so a bug report is useful without being asked for; the console shows
#: only what went wrong, because it is a panel the user is reading while
#: working rather than a transcript.
DEFAULT_FILE_LEVELS: frozenset = frozenset(
    {logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL})
DEFAULT_CONSOLE_LEVELS: frozenset = frozenset(
    {logging.WARNING, logging.ERROR, logging.CRITICAL})


class LevelSetFilter(logging.Filter):
    """Pass only records whose level is in an explicitly enabled set.

    ``setLevel`` is a *threshold*: enabling DEBUG necessarily enables
    everything above it. The preference this serves is a set of independent
    switches, where DEBUG on with INFO off is a legitimate choice, so the
    gate has to be membership rather than comparison.

    The set is mutable in place so a live handler can be re-gated from the
    Preferences dialog without being torn down and rebuilt, which would race
    against any thread logging at that moment.
    """

    def __init__(self, levels: Iterable[int] = ()) -> None:
        super().__init__()
        self.levels = set(levels)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno in self.levels


def normalise_levels(levels: Iterable[int]) -> frozenset:
    """Keep only the five switchable levels, discarding anything else."""
    return frozenset(int(level) for level in levels if int(level) in LEVELS)


def clamp_console_to_file(console: Iterable[int],
                          file_levels: Iterable[int]) -> frozenset:
    """Console levels are a subset of what the file records.

    A level the log file discards cannot reach the console, because the
    console is fed from the same records. Showing a user a line they will
    not find in the log file they are about to attach to a bug report is
    worse than not showing it.
    """
    return normalise_levels(console) & normalise_levels(file_levels)


# Module-level bookkeeping — set once by setup_logging().
_INITIALISED: bool = False
_SESSION_LEVEL: int = logging.INFO
_LOG_PATH: Optional[Path] = None
_FILE_FILTER: Optional[LevelSetFilter] = None
_LEVEL_HANDLERS: dict = {}

# Function-level DEBUG tracing is opt-in.  A profile hook is used instead of
# decorating thousands of functions: it also covers private helpers, class
# methods and functions imported after the preference is enabled.  The hook
# filters by filename before touching logging, so third-party calls are only a
# couple of string comparisons and normal (non-debug) operation pays nothing.
_TRACE_ROOT = os.path.realpath(os.path.dirname(__file__)) + os.sep
_TRACE_THIS_FILE = os.path.realpath(__file__)
_TRACE_ENABLED: bool = False
_TRACE_STATE = threading.local()
_PREVIOUS_SYS_PROFILE = None
_PREVIOUS_THREAD_PROFILE = None

#: Qt virtual-method overrides the trace must never fire on.
#:
#: These are not application logic. Qt calls them once per delivered event,
#: thousands of times a second, and the GUI console is one of the sinks the
#: resulting records go to -- which closes a loop through the event queue:
#: delivering an event logs, logging writes a widget, writing a widget posts
#: a repaint, delivering the repaint logs again. Measured as a Qt shard that
#: made no forward progress at 100% CPU for twenty-five minutes with the GUI
#: thread parked in ``spacr.qt.button_roles.eventFilter -> _trace_profile ->
#: ConsolePanel.append_stdout``, and the same loop is reachable in the shipped
#: app the moment "Verbose logging" is switched on.
#:
#: Excluding them costs nothing worth having. The hook exists to say which
#: spaCR function a run went through, and ``paintEvent`` is not that. It is
#: also what :func:`_trace_profile`'s own contract demands -- a tracing aid
#: must never alter the code it observes, and one that stops event delivery
#: keeping up has altered it beyond recognition.
_TRACE_SKIP_NAMES = frozenset({
    "event", "eventFilter", "customEvent", "childEvent", "timerEvent",
    "paintEvent", "resizeEvent", "moveEvent", "showEvent", "hideEvent",
    "closeEvent", "changeEvent", "enterEvent", "leaveEvent", "focusInEvent",
    "focusOutEvent", "wheelEvent", "mouseMoveEvent", "mousePressEvent",
    "mouseReleaseEvent", "mouseDoubleClickEvent", "hoverMoveEvent",
    "hoverEnterEvent", "hoverLeaveEvent", "keyPressEvent", "keyReleaseEvent",
    "dragEnterEvent", "dragMoveEvent", "dragLeaveEvent", "dropEvent",
    "contextMenuEvent", "viewportEvent", "sizeHint", "minimumSizeHint",
    "heightForWidth",
})


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def log_dir() -> Path:
    """Return the folder where spacr log files live.

    ``SPACR_LOG_DIR`` overrides the default, which is useful for portable
    installs, read-only home directories and test/embedding hosts.

    :returns: the configured directory, otherwise ``~/.spacr/logs``.
    """
    override = os.environ.get("SPACR_LOG_DIR", "").strip()
    root = Path(override).expanduser() if override else (
        Path.home() / ".spacr" / "logs")
    root.mkdir(parents=True, exist_ok=True)
    return root


def log_path() -> Path:
    """Return the absolute path of the rotating log file.

    Uses whatever was passed to :func:`setup_logging` last, or the
    default under :func:`log_dir` when never set.
    """
    return _LOG_PATH if _LOG_PATH is not None else (
        log_dir() / DEFAULT_LOG_FILENAME
    )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(level: Optional[int] = None,
                    log_file: Optional[Path] = None,
                    stream: bool = False,
                    quiet: Iterable[str] = QUIET_LOGGERS) -> Path:
    """Install the rotating file handler on the root logger.

    Idempotent — subsequent calls only re-apply the level, they don't
    stack additional handlers. Honours the ``SPACR_LOG_LEVEL``
    environment variable when ``level`` is not given.

    :param level: minimum record level for the log file. Defaults to
        ``SPACR_LOG_LEVEL`` env var (any of ``DEBUG``/``INFO``/…) or
        :data:`logging.INFO`.
    :param log_file: override for where the file lands. Defaults to
        :func:`log_path`.
    :param stream: also attach a StreamHandler to stderr — handy for
        headless / CI runs where the log file isn't inspected.
    :param quiet: iterable of logger names to pin at WARNING. Defaults
        to :data:`QUIET_LOGGERS`.
    :returns: the resolved log-file path.
    """
    global _INITIALISED, _SESSION_LEVEL, _LOG_PATH

    if level is None:
        level = _env_level()
    _SESSION_LEVEL = level

    resolved_path = Path(log_file) if log_file else log_path()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = resolved_path

    if _INITIALISED:
        logging.getLogger().setLevel(level)
        logging.getLogger("spacr").setLevel(level)
        return resolved_path

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # let each handler cap its own view
    # spacr.* explicitly follows the requested level so enable_debug
    # is the only way records below `level` reach the handlers.
    logging.getLogger("spacr").setLevel(level)

    try:
        file_h = logging.handlers.RotatingFileHandler(
            resolved_path,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
    except OSError as exc:
        # Logging is diagnostic infrastructure; a read-only home directory
        # must not prevent analysis from starting. Keep failures visible on
        # stderr when the requested file cannot be opened.
        sys.stderr.write(
            f"spaCR could not open diagnostic log {resolved_path}: {exc}\n")
        file_h = None
        stream = True
    if file_h is not None:
        # The handler passes everything and the filter decides, so the set of
        # enabled levels can be changed at runtime without rebuilding the
        # handler underneath whatever thread is logging.
        file_h.setLevel(logging.DEBUG)
        file_h.addFilter(_file_filter(_levels_at_or_above(level)))
        file_h.setFormatter(logging.Formatter(FILE_FORMAT))
        root.addHandler(file_h)
        _install_level_handlers(resolved_path, _levels_at_or_above(level))

    if stream:
        stream_h = logging.StreamHandler()
        stream_h.setLevel(level)
        stream_h.setFormatter(logging.Formatter(STREAM_FORMAT))
        root.addHandler(stream_h)

    for name in quiet:
        logging.getLogger(name).setLevel(logging.WARNING)

    _INITIALISED = True
    get_logger("spacr").info("logging initialised → %s", resolved_path)
    return resolved_path


def _levels_at_or_above(level: int) -> frozenset:
    """The switch set equivalent to a classic threshold, for first setup."""
    return frozenset(item for item in LEVELS if item >= level)


def _file_filter(levels: Iterable[int]) -> LevelSetFilter:
    """The one filter shared by the master log file, created on first use."""
    global _FILE_FILTER
    if _FILE_FILTER is None:
        _FILE_FILTER = LevelSetFilter(levels)
    else:
        _FILE_FILTER.levels = set(normalise_levels(levels))
    return _FILE_FILTER


def _install_level_handlers(master_path: Path, levels: Iterable[int]) -> None:
    """Give every level its own file beside the master log.

    One file per level answers "show me only the errors" without grep, and
    the master keeps the interleaved order that makes a sequence of events
    readable. Each is rotated on the same terms as the master.

    A level that is switched off keeps its handler, filtered to nothing,
    rather than being detached: attaching and detaching handlers on a live
    root logger races with any thread that is logging, and an idle handler
    costs one open file.
    """
    enabled = normalise_levels(levels)
    root = logging.getLogger()
    for level in LEVELS:
        handler = _LEVEL_HANDLERS.get(level)
        if handler is None:
            path = master_path.parent / LEVEL_LOG_FILENAMES[level]
            try:
                handler = logging.handlers.RotatingFileHandler(
                    path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
                    encoding="utf-8")
            except OSError as exc:
                sys.stderr.write(
                    f"spaCR could not open {path}: {exc}\n")
                continue
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter(FILE_FORMAT))
            handler.addFilter(LevelSetFilter())
            root.addHandler(handler)
            _LEVEL_HANDLERS[level] = handler
        for existing in handler.filters:
            if isinstance(existing, LevelSetFilter):
                existing.levels = {level} if level in enabled else set()


def apply_level_policy(file_levels: Iterable[int],
                       console_levels: Iterable[int] = ()) -> tuple:
    """Re-gate the live handlers from the Preferences switches.

    :param file_levels: levels written to the log files.
    :param console_levels: levels echoed to the in-app console; silently
        clamped to a subset of ``file_levels``.
    :returns: ``(file_levels, console_levels)`` as actually applied.
    """
    files = normalise_levels(file_levels)
    console = clamp_console_to_file(console_levels, files)

    _file_filter(files)
    if _LOG_PATH is not None:
        _install_level_handlers(_LOG_PATH, files)

    # spacr.* carries its own threshold, which would veto the switches before
    # any handler filter ran. Open it to the lowest level asked for.
    lowest = min(files) if files else logging.CRITICAL
    logging.getLogger("spacr").setLevel(lowest)
    logging.getLogger().setLevel(logging.DEBUG)

    try:
        from .qt.verbose_logger import apply_console_levels
    except Exception:      # Qt is optional; the CLI has no console panel.
        pass
    else:
        apply_console_levels(console)
    return files, console


def _env_level() -> int:
    """Read ``SPACR_LOG_LEVEL`` from env; fall back to INFO."""
    raw = os.environ.get("SPACR_LOG_LEVEL", "").upper().strip()
    if raw and hasattr(logging, raw):
        return getattr(logging, raw)
    return logging.INFO


# ---------------------------------------------------------------------------
# Convenience API for modules and interactive sessions
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a spacr-scoped :class:`logging.Logger`.

    Idiomatic usage from any module:

    .. code-block:: python

       from spacr.logging_util import get_logger
       LOG = get_logger(__name__)

    :param name: logger name — typically ``__name__`` so the log
        stream shows which module the record came from.
    """
    return logging.getLogger(name)


def enable_debug() -> None:
    """Crank every ``spacr.*`` logger to DEBUG.

    Useful when debugging interactively:

    .. code-block:: pycon

       >>> from spacr.logging_util import enable_debug
       >>> enable_debug()

    Third-party loggers listed in :data:`QUIET_LOGGERS` are left at
    WARNING to keep the log readable.
    """
    logging.getLogger("spacr").setLevel(logging.DEBUG)
    for h in logging.getLogger().handlers:
        h.setLevel(logging.DEBUG)
    enable_function_trace()


def disable_debug() -> None:
    """Revert every ``spacr.*`` logger to the level chosen at setup.

    Inverse of :func:`enable_debug`.
    """
    logging.getLogger("spacr").setLevel(_SESSION_LEVEL)
    for h in logging.getLogger().handlers:
        h.setLevel(_SESSION_LEVEL)
    disable_function_trace()


# ---------------------------------------------------------------------------
# Opt-in function/class tracing
# ---------------------------------------------------------------------------

def function_trace_enabled() -> bool:
    """Return whether spaCR function-level DEBUG tracing is active.

    The trace is controlled by :func:`enable_function_trace`,
    :func:`disable_function_trace`, and the GUI's *Verbose logging*
    preference.  It never records arguments or return values, which avoids
    copying large arrays and keeps API keys or filesystem metadata out of the
    diagnostic log.
    """
    return _TRACE_ENABLED


def _trace_profile(frame, event, arg):
    """Profile-hook implementation used by :func:`enable_function_trace`.

    Only Python ``call`` and ``return`` events for files inside the installed
    :mod:`spacr` package are recorded.  The logger implementation itself is
    excluded to prevent recursion, and so are Qt's event-delivery overrides
    (:data:`_TRACE_SKIP_NAMES`) -- tracing those feeds the GUI console from
    inside event delivery, and the console's repaint is another event.
    """
    if event not in {"call", "return"}:
        return
    if frame.f_code.co_name in _TRACE_SKIP_NAMES:
        return
    filename = os.path.realpath(frame.f_code.co_filename)
    if not filename.startswith(_TRACE_ROOT) or filename == _TRACE_THIS_FILE:
        return
    if getattr(_TRACE_STATE, "busy", False):
        return
    _TRACE_STATE.busy = True
    try:
        module = frame.f_globals.get("__name__", "spacr")
        qualname = getattr(
            frame.f_code, "co_qualname", frame.f_code.co_name)
        marker = "→" if event == "call" else "←"
        logging.getLogger("spacr.trace").debug(
            "%s %s.%s", marker, module, qualname)
    except Exception:
        # A tracing aid must never alter the code it observes.
        pass
    finally:
        _TRACE_STATE.busy = False


def enable_function_trace() -> None:
    """Trace every spaCR Python function and method at DEBUG level.

    The hook is installed for the calling thread, all future Python threads,
    and—on Python 3.12+—threads that already exist.  Calls outside the spaCR
    package are ignored.  Repeated calls are idempotent.

    This is intentionally verbose and has measurable overhead, so the GUI
    enables it only while *Verbose logging* is switched on.  Normal operation
    has no profile hook installed.
    """
    global _TRACE_ENABLED, _PREVIOUS_SYS_PROFILE, _PREVIOUS_THREAD_PROFILE
    if _TRACE_ENABLED:
        return
    _PREVIOUS_SYS_PROFILE = sys.getprofile()
    get_thread_profile = getattr(threading, "getprofile", None)
    _PREVIOUS_THREAD_PROFILE = (
        get_thread_profile() if get_thread_profile is not None else None)
    _TRACE_ENABLED = True
    set_all = getattr(threading, "setprofile_all_threads", None)
    if set_all is not None:
        set_all(_trace_profile)
    else:
        sys.setprofile(_trace_profile)
        threading.setprofile(_trace_profile)


def disable_function_trace() -> None:
    """Remove spaCR's function trace and restore prior profile hooks."""
    global _TRACE_ENABLED, _PREVIOUS_SYS_PROFILE, _PREVIOUS_THREAD_PROFILE
    if not _TRACE_ENABLED:
        return
    _TRACE_ENABLED = False
    set_all = getattr(threading, "setprofile_all_threads", None)
    if set_all is not None:
        set_all(_PREVIOUS_THREAD_PROFILE)
        sys.setprofile(_PREVIOUS_SYS_PROFILE)
    else:
        threading.setprofile(_PREVIOUS_THREAD_PROFILE)
        sys.setprofile(_PREVIOUS_SYS_PROFILE)
    _PREVIOUS_SYS_PROFILE = None
    _PREVIOUS_THREAD_PROFILE = None


# ---------------------------------------------------------------------------
# Timing utilities — for benchmarking
# ---------------------------------------------------------------------------
#
# Two shapes users can adopt as they need:
#
# * ``@timed`` decorator — wraps one function; on every call, logs the
#   elapsed wall-clock at INFO. Skips logs faster than SPACR_TIME_THRESHOLD_MS
#   (default 5 ms) so tight inner loops don't drown the log.
#
# * :class:`Timer` context manager — same idea for arbitrary blocks::
#
#     with Timer("cellpose batch"):
#         model.eval(...)     # "cellpose batch took 2.34s"
#
# * :func:`time_module` — one call to wrap every public function on a
#   module with ``@timed``. Use it during ad-hoc profiling; don't
#   leave it on in production code.
#
# All three no-op cheaply when :func:`disable_timing` has been called
# (default: enabled, since the threshold already filters noise).

import functools
import time

TimingCallable = Callable[..., Any]

_TIMING_ENABLED: bool = True
_TIMING_THRESHOLD_MS: int = int(
    os.environ.get("SPACR_TIME_THRESHOLD_MS", "5")
)


def enable_timing() -> None:
    """Turn on the ``@timed`` decorator + :class:`Timer` context manager.

    Enabled by default. Call this after :func:`disable_timing` to
    re-enable timing logs at runtime.
    """
    global _TIMING_ENABLED
    _TIMING_ENABLED = True


def disable_timing() -> None:
    """Turn off all timing logs — the decorators become pass-through."""
    global _TIMING_ENABLED
    _TIMING_ENABLED = False


def set_timing_threshold_ms(ms: int) -> None:
    """Only log timings that exceed ``ms`` milliseconds.

    Defaults to 5 ms (env-overridable via ``SPACR_TIME_THRESHOLD_MS``).
    Setting to 0 logs every call.
    """
    global _TIMING_THRESHOLD_MS
    _TIMING_THRESHOLD_MS = max(0, int(ms))


def timed(fn: Optional[TimingCallable] = None, *,
            name: Optional[str] = None,
            level: int = logging.INFO) -> TimingCallable:
    """Decorator that logs wall-clock time for every call to ``fn``.

    Usable as ``@timed`` or ``@timed(name="…", level=DEBUG)``::

        @timed
        def preprocess_generate_masks(settings):
            ...

        @timed(name="cellpose.batch", level=logging.DEBUG)
        def _run_batch(...):
            ...

    Log line format: ``func_name took 1234.5 ms``. The logger name is
    ``spacr.timing`` unless the wrapped function's module starts with
    ``spacr.``, in which case that module's logger is reused so
    timings interleave with the function's own log records.

    :param fn: the function to wrap (when used as ``@timed``).
    :param name: override the label used in the log line.
    :param level: logging level for the "took Xms" line.
    """
    def _decorate(inner: TimingCallable) -> TimingCallable:
        label = name or f"{inner.__module__}.{inner.__qualname__}"
        mod = inner.__module__
        log_name = mod if mod.startswith("spacr") else "spacr.timing"

        @functools.wraps(inner)
        def _wrapped(*args, **kwargs):
            if not _TIMING_ENABLED:
                return inner(*args, **kwargs)
            log = logging.getLogger(log_name)
            t0 = time.perf_counter()
            try:
                return inner(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if elapsed_ms >= _TIMING_THRESHOLD_MS:
                    log.log(level, "%s took %.1f ms", label, elapsed_ms)

        _wrapped.__wrapped__ = inner   # type: ignore[attr-defined]
        _wrapped.__spacr_timed__ = True   # type: ignore[attr-defined]
        return _wrapped

    if fn is not None:
        return _decorate(fn)
    return _decorate


class Timer:
    """Context manager that logs the elapsed wall-clock for a block.

    Example::

        from spacr.logging_util import Timer

        with Timer("preprocess field"):
            do_work()
        # -> logs "preprocess field took 12.3 ms"

    Nested Timers work fine — each logs its own block independently.

    :param label: human-readable name shown in the log line.
    :param logger: name of the logger to write to (default
        ``"spacr.timing"``).
    :param level: logging level for the line (default INFO).
    :ivar elapsed_ms: filled on exit; None while still running.
    """

    def __init__(self, label: str,
                  logger: str = "spacr.timing",
                  level: int = logging.INFO):
        self.label = label
        self._logger_name = logger
        self._level = level
        self._t0: Optional[float] = None
        self.elapsed_ms: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._t0 is None:
            return
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        if (_TIMING_ENABLED
                and self.elapsed_ms >= _TIMING_THRESHOLD_MS):
            logging.getLogger(self._logger_name).log(
                self._level, "%s took %.1f ms",
                self.label, self.elapsed_ms,
            )


def time_module(module, exclude: tuple = ()) -> int:
    """Wrap every public function in ``module`` with :func:`timed`.

    Idempotent — functions that already carry ``__spacr_timed__`` are
    skipped. Handy during ad-hoc profiling; not recommended for
    production imports because it slows every call by ~1 µs.

    Example::

        import spacr.core
        from spacr.logging_util import time_module
        time_module(spacr.core)
        # Every public function on spacr.core now logs its timing.

    :param module: the module object to wrap.
    :param exclude: iterable of function names to skip.
    :returns: how many functions were wrapped.
    """
    wrapped = 0
    for name in dir(module):
        if name.startswith("_") or name in exclude:
            continue
        obj = getattr(module, name)
        if not callable(obj):
            continue
        if getattr(obj, "__spacr_timed__", False):
            continue
        # Only wrap functions actually defined IN the module
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        setattr(module, name, timed(obj))
        wrapped += 1
    return wrapped
