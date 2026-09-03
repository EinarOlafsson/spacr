"""Failure-isolated multiprocessing bridge for FlowView events."""

from __future__ import annotations

import logging
import math
import operator
import pickle
import queue
import threading
from typing import Callable, Protocol, cast

from .collector import Collector
from .events import (
    EdgeAdded,
    FlowEvent,
    NodeAdded,
    StageCompleted,
    StageFailed,
    StageMetric,
    StageProgress,
    StageStarted,
    StageThumbnail,
)

_LOG = logging.getLogger(__name__)
MAX_EVENT_BYTES = 64 * 1024
_EVENT_TYPES = (
    NodeAdded,
    EdgeAdded,
    StageStarted,
    StageProgress,
    StageMetric,
    StageThumbnail,
    StageCompleted,
    StageFailed,
)


class _QueueSource(Protocol):
    get: Callable[..., object]


class _QueueSink(Protocol):
    put_nowait: Callable[[object], None]


def _positive_byte_limit(value: object) -> int:
    """Return a positive integer byte limit or reject the invalid value."""
    if isinstance(value, bool):
        raise ValueError("max_event_bytes must be a positive integer")
    try:
        limit = operator.index(value)
    except TypeError as exc:
        raise ValueError("max_event_bytes must be a positive integer") from exc
    if limit <= 0:
        raise ValueError("max_event_bytes must be a positive integer")
    return limit


def _finite_seconds(value: object, name: str, *, allow_zero: bool) -> float:
    """Return a finite duration satisfying the caller's zero policy."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number of seconds")
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number of seconds") from exc
    if not math.isfinite(seconds):
        raise ValueError(f"{name} must be finite")
    if seconds < 0 and allow_zero:
        raise ValueError(f"{name} cannot be negative")
    if seconds <= 0 and not allow_zero:
        raise ValueError(f"{name} must be greater than zero")
    return seconds


def is_transport_event(
    value: object,
    *,
    max_event_bytes: int = MAX_EVENT_BYTES,
) -> bool:
    """Return whether a candidate is a bounded public transport event.

    :param value: candidate value to inspect.
    :param max_event_bytes: positive integer ceiling for its highest-protocol
        pickle.
    :returns: ``False`` for the wrong type, a pickle failure, or an oversized
        pickle; otherwise ``True``.
    :raises ValueError: if ``max_event_bytes`` is not a positive integer.
    """

    max_event_bytes = _positive_byte_limit(max_event_bytes)
    if not isinstance(value, _EVENT_TYPES):
        return False
    try:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        return False
    return len(payload) <= max_event_bytes


def put_event_nowait(
    destination: _QueueSink,
    event: object,
    *,
    max_event_bytes: int = MAX_EVENT_BYTES,
) -> bool:
    """Validate and offer one event without waiting for queue capacity.

    :param destination: queue-like sink providing ``put_nowait``.
    :param event: candidate public FlowView event.
    :param max_event_bytes: positive integer pickle-size ceiling.
    :returns: ``True`` only when the event validates and ``put_nowait`` returns
        without raising; otherwise ``False``.
    :raises ValueError: if ``max_event_bytes`` is not a positive integer.
    """

    if not is_transport_event(event, max_event_bytes=max_event_bytes):
        return False
    try:
        destination.put_nowait(event)
    except Exception:
        _LOG.debug("FlowView process-queue emission failed", exc_info=True)
        return False
    return True


class MultiprocessingFeeder:
    """Feed a multiprocessing-compatible queue into an in-process collector.

    The source remains owned by the caller. Stopping does not close or
    intentionally empty it, but one value returned by an in-flight read after
    shutdown can be consumed and discarded rather than emitted.
    """

    def __init__(
        self,
        source: _QueueSource,
        collector: Collector,
        *,
        poll_interval: float = 0.05,
        max_event_bytes: int = MAX_EVENT_BYTES,
    ) -> None:
        """Initialise a stopped queue-to-collector bridge.

        :param source: caller-owned queue-like source whose
            ``get(timeout=...)`` returns candidate events.
        :param collector: in-process collector that receives validated public
            events.
        :param poll_interval: positive finite seconds for each source read and
            fault backoff.
        :param max_event_bytes: positive integer byte ceiling applied when an
            event is re-pickled before forwarding. This limits forwarded data;
            the producer helper must be used to enforce it before enqueue.
        :raises ValueError: if either limit is invalid.
        """
        poll_interval = _finite_seconds(
            poll_interval, "poll_interval", allow_zero=False)
        max_event_bytes = _positive_byte_limit(max_event_bytes)
        self._source = source
        self._collector = collector
        self._poll_interval = poll_interval
        self._max_event_bytes = max_event_bytes
        self._stop_requested = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """Return whether this feeder currently has a live daemon thread."""

        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> "MultiprocessingFeeder":
        """Start one daemon feeder thread if none is alive and return this feeder."""

        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop_requested.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="spacr-flowview-feeder",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self, timeout: float = 1.0) -> bool:
        """Request shutdown and report whether no feeder thread remains.

        :param timeout: non-negative finite seconds to wait for the thread.
        :returns: ``True`` if no thread remains, or ``False`` when the timeout
            expires or the feeder thread itself requests shutdown.
        :raises ValueError: if ``timeout`` is negative or non-finite.
        """

        timeout = _finite_seconds(timeout, "timeout", allow_zero=True)
        self._stop_requested.set()
        with self._state_lock:
            thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout)
        return not thread.is_alive()

    def _run(self) -> None:
        """Poll validated source events into the collector until stopped.

        Source timeouts continue polling; other source and collector failures
        are logged and isolated. A value returned by an in-flight read after
        shutdown is consumed but not emitted, and the thread reference is
        cleared on every exit.
        """
        try:
            while not self._stop_requested.is_set():
                try:
                    value = self._source.get(timeout=self._poll_interval)
                except queue.Empty:
                    continue
                except Exception:
                    _LOG.debug("FlowView process-queue read failed", exc_info=True)
                    self._stop_requested.wait(self._poll_interval)
                    continue

                if self._stop_requested.is_set():
                    break
                if not is_transport_event(
                    value,
                    max_event_bytes=self._max_event_bytes,
                ):
                    _LOG.debug("Discarded an invalid FlowView process-queue value")
                    continue
                try:
                    self._collector.emit(cast(FlowEvent, value))
                except Exception:
                    _LOG.debug("FlowView collector rejected a bridged event", exc_info=True)
        finally:
            with self._state_lock:
                self._thread = None


__all__ = [
    "MAX_EVENT_BYTES",
    "MultiprocessingFeeder",
    "is_transport_event",
    "put_event_nowait",
]
