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
from spacr.flowview.model import Edge, Node, NodeKind, NodeState, RunGraph


def _graph(nodes=(), edges=()):
    return RunGraph(
        run_id="synthetic",
        started_at=1.0,
        nodes={node.id: node for node in nodes},
        edges=list(edges),
        spacr_version="test",
        settings_digest="digest",
    )


def test_queue_is_bounded_drop_oldest_and_reports_sampling():
    collector = Collector(_graph(), max_queue_size=2)
    first = Node(id="first", label="First", kind=NodeKind.INPUT)
    second = Node(id="second", label="Second", kind=NodeKind.PROCESS)
    third = Node(id="third", label="Third", kind=NodeKind.OUTPUT)

    assert collector.emit(NodeAdded(first)) is True
    assert collector.emit(NodeAdded(second)) is True
    assert collector.emit(NodeAdded(third)) is False
    assert collector.revision == 0
    assert collector.pending == 2
    assert collector.sampled is True
    assert collector.drain(-1) == 0
    assert collector.revision == 0
    assert collector.drain(1) == 1
    assert collector.revision == 1
    assert collector.pending == 1
    assert collector.drain() == 1
    assert collector.revision == 2
    assert set(collector.snapshot().nodes) == {"second", "third"}

    collector.clear_sampled()
    assert collector.sampled is False


def test_queue_size_must_be_positive():
    with pytest.raises(ValueError, match="greater than zero"):
        Collector(_graph(), max_queue_size=0)


def test_every_event_folds_and_snapshots_do_not_share_mappings():
    original = Node(
        id="stage",
        label="Old label",
        kind=NodeKind.PROCESS,
        state=NodeState.FAILED,
        ended_at=0.5,
        metrics={"old": 1},
        params={"kept": True},
        error="old failure",
    )
    collector = Collector(_graph([original]))

    assert collector.fold(NodeAdded(Node("stage", "Ignored", NodeKind.INPUT)))
    assert collector.fold(EdgeAdded(Edge("source", "stage", "files", 3)))
    assert collector.fold(EdgeAdded(Edge("source", "stage", "files", 3)))
    assert collector.fold(
        StageStarted(
            Node(
                "stage",
                "New label",
                NodeKind.OUTPUT,
                params={"added": "yes"},
            ),
            2.0,
        )
    )
    assert collector.fold(StageProgress("stage", 3, 7))
    assert collector.fold(StageMetric("stage", "objects", 42))
    assert collector.fold(StageThumbnail("stage", "cache/node.png"))
    assert collector.fold(StageCompleted("stage", 4.0))

    snapshot = collector.snapshot()
    node = snapshot.nodes["stage"]
    assert node.label == "New label"
    assert node.kind is NodeKind.OUTPUT
    assert node.state is NodeState.DONE
    assert node.started_at == 2.0
    assert node.ended_at == 4.0
    assert node.error is None
    assert node.progress == (3, 7)
    assert node.metrics == {"old": 1, "objects": 42}
    assert node.params == {"kept": True, "added": "yes"}
    assert node.thumbnail == "cache/node.png"
    assert snapshot.edges == [Edge("source", "stage", "files", 3)]

    snapshot.nodes["stage"].metrics["objects"] = -1
    snapshot.nodes["stage"].params["added"] = "mutated"
    fresh = collector.snapshot().nodes["stage"]
    assert fresh.metrics["objects"] == 42
    assert fresh.params["added"] == "yes"


def test_out_of_order_updates_are_ignored_but_start_can_declare_a_node():
    collector = Collector(_graph())

    assert collector.fold(StageProgress("missing", 1, 2)) is False
    assert collector.fold(StageMetric("missing", "count", 1)) is False
    assert collector.fold(StageThumbnail("missing", "missing.png")) is False
    assert collector.fold(StageCompleted("missing", 3.0)) is False
    assert collector.fold(StageFailed("missing", 3.0, "failed")) is False
    assert collector.fold(object()) is False
    assert collector.revision == 0

    observed = Node("observed", "Observed", NodeKind.PROCESS, params={"x": 1})
    assert collector.fold(StageStarted(observed, 5.0)) is True
    assert collector.revision == 1
    node = collector.snapshot().nodes["observed"]
    assert node.state is NodeState.RUNNING
    assert node.started_at == 5.0


def test_an_unknown_failure_does_not_skip_known_descendants():
    child = Node("child", "Child", NodeKind.OUTPUT)
    collector = Collector(_graph([child], [Edge("missing", "child")]))

    assert collector.fold(StageFailed("missing", 3.0, "failed")) is False
    assert collector.revision == 0
    assert collector.snapshot().nodes["child"].state is NodeState.PENDING


def test_failure_marks_every_descendant_skipped_even_when_edges_cycle():
    nodes = [
        Node("a", "A", NodeKind.PROCESS),
        Node("b", "B", NodeKind.PROCESS),
        Node("c", "C", NodeKind.PROCESS),
        Node("d", "D", NodeKind.OUTPUT),
    ]
    edges = [
        Edge("a", "b"),
        Edge("b", "c"),
        Edge("a", "d"),
        Edge("c", "a"),
        Edge("c", "not-declared"),
    ]
    collector = Collector(_graph(nodes, edges))

    assert collector.fold(StageFailed("a", 9.0, "traceback")) is True
    snapshot = collector.snapshot()

    assert snapshot.nodes["a"].state is NodeState.FAILED
    assert snapshot.nodes["a"].error == "traceback"
    for node_id in ("b", "c", "d"):
        assert snapshot.nodes[node_id].state is NodeState.SKIPPED
        assert snapshot.nodes[node_id].ended_at == 9.0


def test_many_producers_never_wait_for_queue_capacity_or_raise():
    collector = Collector(_graph(), max_queue_size=64)
    errors = []
    barrier = threading.Barrier(5)

    def produce(worker: int):
        try:
            barrier.wait()
            for index in range(2_000):
                collector.emit(StageProgress("stage", worker * 2_000 + index, 8_000))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=produce, args=(worker,)) for worker in range(4)]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=3.0)

    assert all(not thread.is_alive() for thread in threads)
    assert time.monotonic() - started < 3.0
    assert errors == []
    assert collector.pending == 64
    assert collector.sampled is True
    assert collector.drain() == 64
