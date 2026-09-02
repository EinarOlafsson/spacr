"""Failure-isolated multiprocessing bridge for FlowView events."""

from __future__ import annotations

import logging
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


def is_transport_event(
    value: object,
    *,
    max_event_bytes: int = MAX_EVENT_BYTES,
) -> bool:
    """Return whether *value* is a declared, small, picklable FlowView event."""

    if max_event_bytes <= 0:
        raise ValueError("max_event_bytes must be greater than zero")
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
    """Put one validated event without waiting, returning whether it was queued."""

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

    The source remains owned by the caller.  In particular, stopping the
    feeder never closes or drains it, so analysis workers may continue using
    the queue independently of the optional visualisation.

    :param source: the queue events arrive on, typically a
        ``multiprocessing.Queue`` shared with analysis workers. OWNED BY THE
        CALLER -- see above; the feeder only reads from it.
    :param collector: the in-process :class:`~spacr.flowview.collector.Collector`
        the events are forwarded to.
    :param poll_interval: seconds to wait between reads when the source is
        empty. Trades display latency against idle CPU.
    :param max_event_bytes: the largest event accepted from the source. A
        cap rather than a guess: the source may be written to by another
        process, so an oversized payload is a reason to reject the event
        rather than to allocate for it.
    :raises ValueError: when ``poll_interval`` or ``max_event_bytes`` is not
        positive.
    """

    def __init__(
        self,
        source: _QueueSource,
        collector: Collector,
        *,
        poll_interval: float = 0.05,
        max_event_bytes: int = MAX_EVENT_BYTES,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero")
        if max_event_bytes <= 0:
            raise ValueError("max_event_bytes must be greater than zero")
        self._source = source
        self._collector = collector
        self._poll_interval = poll_interval
        self._max_event_bytes = max_event_bytes
        self._stop_requested = threading.Event()
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """Whether the daemon feeder thread is alive."""

        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> "MultiprocessingFeeder":
        """Start the daemon feeder once and return this feeder."""

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
        """Request shutdown and return whether the thread stopped in time."""

        if timeout < 0:
            raise ValueError("timeout cannot be negative")
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
