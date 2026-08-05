"""Cooperative cancellation for long-running spaCR pipelines.

Qt cannot safely kill a Python thread that may be writing a TIFF, committing a
SQLite transaction, or updating several related artifacts.  Instead, the GUI
installs one :class:`CancellationToken` in each worker thread and pipeline code
calls :func:`checkpoint` only at boundaries where the previous unit is fully
durable and the next unit has not started.

The module is intentionally standard-library only.  Core workflows may import
it without importing Qt, torch, or any GUI dependency.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator, Optional

__all__ = [
    "CancellationToken",
    "PipelineCancelled",
    "cancellation_requested",
    "checkpoint",
    "current_token",
    "installed_token",
]


class PipelineCancelled(Exception):
    """Raised at a safe boundary after cancellation has been requested.

    This is a normal control-flow outcome, not a pipeline failure.  Callers
    that catch broad :class:`Exception` values must re-raise it so the worker
    can record the run as ``cancelled`` rather than as a failed item.
    """


class CancellationToken:
    """Thread-safe, idempotent cancellation request shared with one worker.

    :param reason: default user-facing reason raised by :meth:`checkpoint`.
    """

    def __init__(self, reason: str = "cancelled by the user") -> None:
        self._event = threading.Event()
        self._reason = str(reason)
        self._lock = threading.Lock()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._event.is_set()

    @property
    def reason(self) -> str:
        """The first cancellation reason supplied to :meth:`cancel`."""
        with self._lock:
            return self._reason

    def cancel(self, reason: Optional[str] = None) -> bool:
        """Request cancellation and return True only for the first request.

        Repeated Stop clicks are harmless and do not replace the original
        reason, which keeps logs and manifests deterministic.
        """
        with self._lock:
            first = not self._event.is_set()
            if first and reason:
                self._reason = str(reason)
            self._event.set()
            return first

    def checkpoint(self) -> None:
        """Raise :class:`PipelineCancelled` when cancellation was requested."""
        if self._event.is_set():
            raise PipelineCancelled(self.reason)


_LOCAL = threading.local()


def current_token() -> Optional[CancellationToken]:
    """Return the token installed for the calling thread, if any."""
    return getattr(_LOCAL, "token", None)


def cancellation_requested() -> bool:
    """Return whether the calling worker has a pending cancellation request."""
    token = current_token()
    return bool(token is not None and token.cancelled)


def checkpoint() -> None:
    """Stop at this safe boundary when requested; otherwise do nothing.

    Calls outside a managed worker are intentionally no-ops, so the same
    pipeline functions work through the GUI, CLI, notebooks, and tests.
    """
    token = current_token()
    if token is not None:
        token.checkpoint()


@contextmanager
def installed_token(token: CancellationToken) -> Iterator[CancellationToken]:
    """Install ``token`` for this thread and restore any prior token on exit."""
    marker = object()
    previous = getattr(_LOCAL, "token", marker)
    _LOCAL.token = token
    try:
        yield token
    finally:
        if previous is marker:
            try:
                delattr(_LOCAL, "token")
            except AttributeError:
                pass
        else:
            _LOCAL.token = previous
