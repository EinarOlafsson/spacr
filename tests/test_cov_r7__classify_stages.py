"""Failure isolation and journal mirroring in the Classify stage clock.

These tests pin the parts of :mod:`spacr.flowview._classify_stages` that only
run when something around it is broken or unusual: a collector that cannot
produce a snapshot, a graph that has not predeclared the eight approved nodes,
a run journal that is open (or absent, or hostile), a logger that itself
raises, and every lifecycle helper called with no stage running.  All of them
exist so that a display or provenance fault can never replace a scientific
result or exception, so each test asserts both the degraded answer and the
healthy one it is degrading from.
"""

from __future__ import annotations

import sys

import pytest

from spacr import flowview
from spacr.flowview import _classify_stages
from spacr.flowview.classify_blueprint import CLASSIFY_NODE_IDS, classify_graph
from spacr.flowview.collector import Collector
from spacr.flowview.model import NodeState, RunGraph


@pytest.fixture(autouse=True)
def _restore_trace_state():
    previous_collector = flowview.get_collector()
    previous_enabled = flowview.is_enabled()
    yield
    flowview.enable(previous_collector)
    if not previous_enabled:
        flowview.disable()


class _BrokenSnapshot(Collector):
    """A collector whose graph cannot be read, as a dead renderer leaves it."""

    def snapshot(self) -> RunGraph:
        raise RuntimeError("snapshot unavailable")


def _empty_graph(run_id: str) -> RunGraph:
    return RunGraph(
        run_id=run_id,
        started_at=1.0,
        nodes={},
        edges=[],
        spacr_version="test",
        settings_digest="0" * 8,
    )


def _blueprint_collector(family: str = "cv") -> Collector:
    settings = {
        "classifier_family": family,
        "model_type": "test_cv",
        "model_type_ml": "random_forest",
    }
    collector = Collector(
        classify_graph(settings, run_id=f"{family}-r7", started_at=1.0)
    )
    flowview.enable(collector)
    return collector


class _HostileLogger:
    """A logger that raises, the way a closed handler stream does."""

    def debug(self, *_args, **_kwargs):
        raise RuntimeError("logging is broken too")


def _break_logging(monkeypatch) -> None:
    monkeypatch.setattr(_classify_stages, "_LOG", _HostileLogger())


def test_emit_reports_a_lost_event_even_when_logging_also_fails(monkeypatch):
    collector = _blueprint_collector()
    started = _classify_stages.StageStarted(
        collector.snapshot().nodes["source"], 1.0
    )
    assert _classify_stages._emit(collector, started) is True

    monkeypatch.setattr(
        collector,
        "emit",
        lambda _event: (_ for _ in ()).throw(RuntimeError("display broke")),
    )
    _break_logging(monkeypatch)
    assert _classify_stages._emit(collector, started) is False

    monkeypatch.undo()
    collector.drain()
    # Only the first, healthy emission ever reached the graph.
    assert collector.snapshot().nodes["source"].state is NodeState.RUNNING


def test_journal_stage_mirrors_a_stage_only_while_a_run_is_open(
    monkeypatch, tmp_path
):
    from spacr import run_journal

    monkeypatch.setattr(run_journal, "runs_root", lambda: tmp_path)

    with run_journal.open_run("classify", {"classifier_family": "ml"}) as run:
        assert (
            _classify_stages._journal_stage(
                "tables",
                label="Measurement tables",
                state="running",
                started_at=10.25,
                metrics={"objects": 48},
            )
            is True
        )
        assert _classify_stages._journal_stage(
            "tables", state="done", ended_at=12.75
        )
        recorded = [dict(stage) for stage in run.stages]

    assert recorded == [
        {
            "id": "tables",
            "label": "Measurement tables",
            "state": "done",
            "started_at": 10.25,
            "ended_at": 12.75,
            "duration_s": 2.5,
            "metrics": {"objects": 48},
        }
    ]

    # Outside the run there is no ``_record_stage`` to reach: the same call
    # is refused rather than inventing a journal for it.
    assert run_journal.current_run() is None
    assert _classify_stages._journal_stage("tables", state="done") is False
    assert [stage["state"] for stage in run.stages] == ["done"]

    # And on the disabled path the journal module is never imported at all,
    # so there is not even a ``current_run`` to ask.
    monkeypatch.delitem(sys.modules, "spacr.run_journal", raising=False)
    assert _classify_stages._journal_stage("tables", state="running") is False
    assert [stage["state"] for stage in run.stages] == ["done"]


def test_journal_stage_refuses_metrics_it_cannot_copy(monkeypatch, tmp_path):
    from spacr import run_journal

    monkeypatch.setattr(run_journal, "runs_root", lambda: tmp_path)
    _break_logging(monkeypatch)

    with run_journal.open_run("classify", {"classifier_family": "cv"}) as run:
        assert _classify_stages._journal_stage(
            "split", state="running", metrics={"train_objects": 4}
        )
        # ``dict(metrics)`` raises before the recorder is ever reached.
        assert (
            _classify_stages._journal_stage("split", metrics=object())  # type: ignore[arg-type]
            is False
        )
        stages = [dict(stage) for stage in run.stages]

    assert len(stages) == 1
    assert stages[0]["id"] == "split"
    assert stages[0]["metrics"] == {"train_objects": 4}


def test_node_falls_back_when_the_graph_cannot_be_read():
    healthy = _blueprint_collector()
    node, declared = _classify_stages._node(healthy, "source")
    assert declared is True
    # ``snapshot`` hands back detached copies, so compare by value; the
    # blueprint node carries the run's ``src`` param the fallback lacks.
    assert node == healthy.snapshot().nodes["source"]
    assert node != _classify_stages._FALLBACK_NODES["source"]

    broken = _BrokenSnapshot(_empty_graph("broken-node"))
    node, declared = _classify_stages._node(broken, "source")
    assert declared is False
    assert node is _classify_stages._FALLBACK_NODES["source"]


def test_stages_missing_from_the_graph_are_declared_before_use():
    # A collector installed by ``trace`` rather than by Classify starts with
    # no nodes at all; the clock has to add each one it touches.
    collector = Collector(_empty_graph("bare-graph"))
    flowview.enable(collector)

    assert _classify_stages._advance("dataset", at=5.0) is True
    collector.drain()
    graph = collector.snapshot()

    assert set(graph.nodes) == {"source", "tables", "dataset"}
    for skipped_id in ("source", "tables"):
        assert graph.nodes[skipped_id].state is NodeState.SKIPPED
        assert graph.nodes[skipped_id].ended_at == 5.0
        assert (
            graph.nodes[skipped_id].label
            == _classify_stages._FALLBACK_NODES[skipped_id].label
        )
    assert graph.nodes["dataset"].state is NodeState.RUNNING
    assert graph.nodes["dataset"].started_at == 5.0

    # With the blueprint already declared, the predeclared labels survive
    # instead of being replaced by the fallbacks used above.
    declared_collector = _blueprint_collector("ml")
    assert _classify_stages._advance("dataset", at=6.0) is True
    declared_collector.drain()
    declared_graph = declared_collector.snapshot()
    assert set(declared_graph.nodes) == set(CLASSIFY_NODE_IDS)
    assert (
        declared_graph.nodes["tables"].label
        != _classify_stages._FALLBACK_NODES["tables"].label
    )


def test_begin_installs_a_classify_graph_when_the_prior_one_is_unreadable():
    broken = _BrokenSnapshot(_empty_graph("broken-begin"))
    flowview.enable(broken)

    assert _classify_stages._begin(
        {"classifier_family": "cv", "src": ["plate1", "plate2"]}, "cv"
    )

    installed = flowview.get_collector()
    assert installed is not broken
    installed.drain()
    graph = installed.snapshot()
    assert tuple(graph.nodes) == CLASSIFY_NODE_IDS
    assert graph.nodes["source"].state is NodeState.RUNNING
    assert graph.nodes["source"].metrics == {"sources": 2}


def test_lifecycle_helpers_are_inert_when_the_collector_cannot_be_reached(
    monkeypatch,
):
    collector = _blueprint_collector()
    assert _classify_stages._begin({"classifier_family": "cv"}, "cv")
    assert _classify_stages._advance("tables", at=2.0)
    assert _classify_stages._metric("objects", 7)

    def unreachable():
        raise RuntimeError("trace state is gone")

    monkeypatch.setattr(_classify_stages, "get_collector", unreachable)
    _break_logging(monkeypatch)

    assert _classify_stages._begin({"classifier_family": "cv"}, "cv") is False
    assert _classify_stages._advance("dataset", at=3.0) is False
    assert _classify_stages._metric("objects", 99) is False
    assert _classify_stages._finish(at=4.0) is False
    assert _classify_stages._fail(LookupError("science"), at=4.0) is False

    monkeypatch.undo()
    collector.drain()
    graph = collector.snapshot()
    # Nothing after the healthy prologue was recorded.
    assert graph.nodes["tables"].state is NodeState.RUNNING
    assert graph.nodes["tables"].metrics == {"objects": 7}
    assert graph.nodes["dataset"].state is NodeState.PENDING


def test_advance_refuses_a_stage_id_outside_the_approved_graph(monkeypatch):
    collector = _blueprint_collector()
    assert _classify_stages._begin({"classifier_family": "cv"}, "cv")
    assert _classify_stages._advance("tables", at=2.0) is True

    _break_logging(monkeypatch)
    assert _classify_stages._advance("not_a_stage", at=3.0) is False

    monkeypatch.undo()
    collector.drain()
    graph = collector.snapshot()
    assert "not_a_stage" not in graph.nodes
    assert graph.nodes["tables"].state is NodeState.RUNNING
    assert graph.nodes["tables"].ended_at is None


def test_finish_and_fail_need_a_running_stage():
    collector = Collector(
        classify_graph(
            {"classifier_family": "cv", "model_type": "test_cv"},
            run_id="terminal-r7",
            started_at=1.0,
        )
    )
    flowview.enable(collector)

    # No run has begun on this collector, so there is no stage to terminate.
    assert _classify_stages._finish(at=2.0) is False
    assert _classify_stages._fail(LookupError("science"), at=2.0) is False
    assert _classify_stages._metric("objects", 3) is False

    assert _classify_stages._begin({"classifier_family": "cv"}, "cv")
    assert _classify_stages._advance("tables", at=2.0)
    assert _classify_stages._metric("objects", 3) is True
    assert _classify_stages._finish(at=3.0) is True

    # A finished run is terminal: nothing further is accepted.
    assert _classify_stages._finish(at=4.0) is False
    assert _classify_stages._fail(LookupError("late"), at=4.0) is False
    assert _classify_stages._metric("objects", 99) is False

    collector.drain()
    graph = collector.snapshot()
    assert graph.nodes["tables"].state is NodeState.DONE
    assert graph.nodes["tables"].ended_at == 3.0
    assert graph.nodes["tables"].metrics == {"objects": 3}
    assert graph.nodes["scores"].state is NodeState.SKIPPED


def test_fail_records_a_placeholder_when_the_traceback_cannot_be_formatted(
    monkeypatch,
):
    collector = _blueprint_collector()
    assert _classify_stages._begin({"classifier_family": "cv"}, "cv")
    assert _classify_stages._advance("tables", at=2.0)

    class _BrokenTraceback:
        @staticmethod
        def format_exception(*_args, **_kwargs):
            raise RuntimeError("frames are gone")

    monkeypatch.setattr(_classify_stages, "traceback", _BrokenTraceback)
    assert _classify_stages._fail(LookupError("science failed"), at=3.0) is True

    monkeypatch.undo()
    collector.drain()
    graph = collector.snapshot()
    assert graph.nodes["tables"].state is NodeState.FAILED
    assert graph.nodes["tables"].error == "LookupError: exception text unavailable"
    assert all(
        graph.nodes[node_id].state is NodeState.SKIPPED
        for node_id in CLASSIFY_NODE_IDS[2:]
    )
