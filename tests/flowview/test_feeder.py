import multiprocessing
import queue
import subprocess
import sys
import threading
import time

import pytest

from spacr.flowview.collector import Collector
from spacr.flowview.events import (
    EdgeAdded,
    NodeAdded,
    StageCompleted,
    StageFailed,
    StageMetric,
    StageProgress,
    StageStarted,
    StageThumbnail,
)
from spacr.flowview.feeder import (
    MultiprocessingFeeder,
    is_transport_event,
    put_event_nowait,
)
from spacr.flowview.model import Edge, Node, NodeKind, RunGraph


def _collector():
    return Collector(
        RunGraph(
            run_id="feeder-test",
            started_at=1.0,
            nodes={},
            edges=[],
            spacr_version="test",
            settings_digest="digest",
        )
    )


def _wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _events():
    node = Node("stage", "Stage", NodeKind.PROCESS)
    return [
        NodeAdded(node),
        EdgeAdded(Edge("source", "stage")),
        StageStarted(node, 2.0),
        StageProgress("stage", 1, 2),
        StageMetric("stage", "objects", 3),
        StageThumbnail("stage", "cache/stage.png"),
        StageCompleted("stage", 4.0),
        StageFailed("stage", 5.0, "failure"),
    ]


@pytest.mark.parametrize("event", _events())
def test_each_declared_event_is_transport_safe(event):
    assert is_transport_event(event)


def test_transport_validation_rejects_unknown_unpicklable_and_large_values():
    lock = threading.Lock()
    unpicklable = NodeAdded(Node("bad", "Bad", NodeKind.INPUT, params={"lock": lock}))
    large = StageMetric("stage", "large", "x" * 1_000)

    assert is_transport_event(object()) is False
    assert is_transport_event(unpicklable) is False
    assert is_transport_event(large, max_event_bytes=100) is False
    with pytest.raises(ValueError, match="positive integer"):
        is_transport_event(_events()[0], max_event_bytes=0)


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), 1.5, True])
def test_transport_validation_requires_a_positive_integer_limit(limit):
    with pytest.raises(ValueError, match="positive integer"):
        is_transport_event(_events()[0], max_event_bytes=limit)


def test_producer_helper_never_blocks_and_isolates_queue_faults():
    destination = queue.Queue(maxsize=1)
    event = _events()[0]

    assert put_event_nowait(destination, event) is True
    assert destination.get_nowait() is event
    assert put_event_nowait(destination, object()) is False

    destination.put_nowait(event)
    assert put_event_nowait(destination, event) is False


def test_feeder_delivers_events_on_a_daemon_thread_and_is_restartable():
    source = queue.Queue()
    collector = _collector()
    for event in _events():
        source.put_nowait(event)

    feeder = MultiprocessingFeeder(source, collector, poll_interval=0.01)
    assert feeder.running is False
    assert feeder.start() is feeder
    thread = feeder._thread
    assert thread is not None
    assert thread.daemon is True
    assert feeder.start() is feeder
    assert feeder._thread is thread
    assert _wait_for(lambda: collector.pending == len(_events()))
    assert feeder.running is True
    assert feeder.stop() is True
    assert feeder.running is False
    assert feeder.stop() is True

    source.put_nowait(NodeAdded(Node("again", "Again", NodeKind.INPUT)))
    feeder.start()
    assert _wait_for(lambda: collector.pending == len(_events()) + 1)
    assert feeder.stop() is True


def test_real_multiprocessing_queue_is_bridged_without_blocking_puts():
    source = multiprocessing.get_context("spawn").Queue(maxsize=2)
    collector = _collector()
    feeder = MultiprocessingFeeder(source, collector, poll_interval=0.01)
    event = NodeAdded(Node("process", "Process", NodeKind.PROCESS))

    try:
        assert put_event_nowait(source, event) is True
        feeder.start()
        assert _wait_for(lambda: collector.pending == 1)
        assert feeder.stop() is True
    finally:
        feeder.stop()
        source.close()
        source.join_thread()


def test_source_collector_and_unknown_value_faults_are_isolated():
    event = NodeAdded(Node("accepted", "Accepted", NodeKind.INPUT))

    class FaultingSource:
        def __init__(self):
            self.calls = 0

        def get(self, block=True, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary source fault")
            if self.calls == 2:
                return object()
            if self.calls == 3:
                return event
            raise queue.Empty

    class FaultingCollector:
        def __init__(self):
            self.calls = 0

        def emit(self, value):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary collector fault")

    source = FaultingSource()
    collector = FaultingCollector()
    feeder = MultiprocessingFeeder(source, collector, poll_interval=0.005).start()

    assert _wait_for(lambda: collector.calls == 1)
    assert feeder.running is True
    assert feeder.stop() is True


def test_stop_does_not_emit_a_value_released_after_shutdown_request():
    release = threading.Event()
    entered = threading.Event()
    event = _events()[0]

    class SlowSource:
        def get(self, block=True, timeout=None):
            entered.set()
            release.wait()
            return event

    collector = _collector()
    feeder = MultiprocessingFeeder(SlowSource(), collector, poll_interval=0.01).start()
    assert entered.wait(1.0)
    assert feeder.stop(timeout=0) is False
    release.set()
    assert _wait_for(lambda: not feeder.running)
    assert collector.pending == 0


def test_stop_called_by_feeder_thread_requests_shutdown_without_self_join():
    source = queue.Queue()

    class SelfStoppingCollector:
        result = None

        def emit(self, event):
            self.result = feeder.stop()

    collector = SelfStoppingCollector()
    feeder = MultiprocessingFeeder(source, collector, poll_interval=0.01)
    source.put_nowait(_events()[0])
    feeder.start()

    assert _wait_for(lambda: collector.result is not None)
    assert collector.result is False
    assert _wait_for(lambda: not feeder.running)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"poll_interval": 0}, "poll_interval"),
        ({"max_event_bytes": 0}, "max_event_bytes"),
    ],
)
def test_constructor_rejects_invalid_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MultiprocessingFeeder(queue.Queue(), _collector(), **kwargs)


@pytest.mark.parametrize("poll_interval", [float("nan"), float("inf"),
                                            float("-inf"), True])
def test_constructor_rejects_nonfinite_poll_intervals(poll_interval):
    with pytest.raises(ValueError, match="poll_interval"):
        MultiprocessingFeeder(
            queue.Queue(), _collector(), poll_interval=poll_interval)


@pytest.mark.parametrize("max_event_bytes", [float("nan"), float("inf"),
                                              1.5, True])
def test_constructor_rejects_noninteger_byte_limits(max_event_bytes):
    with pytest.raises(ValueError, match="max_event_bytes"):
        MultiprocessingFeeder(
            queue.Queue(), _collector(), max_event_bytes=max_event_bytes)


def test_stop_rejects_negative_timeout():
    feeder = MultiprocessingFeeder(queue.Queue(), _collector())
    with pytest.raises(ValueError, match="cannot be negative"):
        feeder.stop(-1)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"),
                                     float("-inf"), True])
def test_stop_rejects_nonfinite_timeouts(timeout):
    feeder = MultiprocessingFeeder(queue.Queue(), _collector())
    with pytest.raises(ValueError, match="timeout"):
        feeder.stop(timeout)


def test_importing_feeder_does_not_import_qt_in_a_fresh_process():
    command = [
        sys.executable,
        "-c",
        "import sys; import spacr.flowview.feeder; assert 'PySide6' not in sys.modules",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
