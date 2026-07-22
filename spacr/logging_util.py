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
from pathlib import Path
from typing import Iterable, Optional

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

# Module-level bookkeeping — set once by setup_logging().
_INITIALISED: bool = False
_SESSION_LEVEL: int = logging.INFO
_LOG_PATH: Optional[Path] = None


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def log_dir() -> Path:
    """Return the folder where spacr log files live.

    :returns: ``~/.spacr/logs`` — created if it does not exist.
    """
    root = Path.home() / ".spacr" / "logs"
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

    file_h = logging.handlers.RotatingFileHandler(
        resolved_path,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_h.setLevel(level)
    file_h.setFormatter(logging.Formatter(FILE_FORMAT))
    root.addHandler(file_h)

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


def disable_debug() -> None:
    """Revert every ``spacr.*`` logger to the level chosen at setup.

    Inverse of :func:`enable_debug`.
    """
    logging.getLogger("spacr").setLevel(_SESSION_LEVEL)
    for h in logging.getLogger().handlers:
        h.setLevel(_SESSION_LEVEL)
