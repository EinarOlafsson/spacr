from __future__ import annotations

import functools
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_LOGGER_NAME = "spacr"


def configure_logger(
    name: str = _LOGGER_NAME,
    log_file_name: str = "spacr.log",
    level: int = logging.INFO,
    stream: bool = False,
) -> logging.Logger:
    """Return a named logger backed by a rotating file handler in the user's home.

    Reuses an already-configured logger with the same name to avoid duplicate handlers.

    :param name: Logger name to fetch or create. Default ``"spacr"``.
    :param log_file_name: File name (placed under ``$HOME``) for rotating log output.
    :param level: Logging level applied to the logger and its handlers.
    :param stream: When True, also attach a stderr stream handler.
    :returns: Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    logger.propagate = False

    log_path = Path.home() / log_file_name
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(file_handler)

    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(
            logging.Formatter("%(levelname)s - %(message)s")
        )
        logger.addHandler(stream_handler)

    return logger


logger = logging.getLogger(_LOGGER_NAME)
logger.addHandler(logging.NullHandler())


def _safe_repr(value: Any, max_length: int = 200) -> str:
    """Return a truncated ``repr`` that never raises."""
    try:
        text = repr(value)
    except Exception:
        text = f"<unreprable {type(value).__name__}>"

    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def log_function_call(func):
    """Decorator that logs call arguments, return value, and exceptions.

    :param func: Callable to wrap.
    :returns: Wrapped callable that emits INFO-level trace entries.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Emit call/return/exception log lines around the wrapped call."""
        active_logger = configure_logger(name=func.__module__)

        args_repr = ", ".join(_safe_repr(arg) for arg in args)
        kwargs_repr = ", ".join(
            f"{key}={_safe_repr(value)}" for key, value in kwargs.items()
        )
        signature = ", ".join(part for part in (args_repr, kwargs_repr) if part)

        active_logger.info("Calling %s(%s)", func.__name__, signature)
        try:
            result = func(*args, **kwargs)
            active_logger.info("%s returned %s", func.__name__, _safe_repr(result))
            return result
        except Exception:
            active_logger.exception("Exception occurred in %s", func.__name__)
            raise

    return wrapper