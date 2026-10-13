import os
import subprocess
import sys

import pytest

from spacr import flowview
from spacr.flowview import trace
from spacr.flowview.collector import Collector
from spacr.flowview.model import NodeKind, NodeState, RunGraph


def _collector(max_queue_size=2_000):
    graph = RunGraph(
        run_id="trace-test",
        started_at=1.0,
        nodes={},
        edges=[],
        spacr_version="test",
        settings_digest="digest",
    )
    return Collector(graph, max_queue_size=max_queue_size)


@pytest.fixture(autouse=True)
def _restore_trace_state():
    previous_collector = flowview.get_collector()
    previous_enabled = flowview.is_enabled()
    yield
    flowview.enable(previous_collector)
    if not previous_enabled:
        flowview.disable()


def test_disabled_decorator_is_identical_and_context_is_a_null_object(tmp_path):
    collector = _collector()
    flowview.enable(collector)
    flowview.disable()

    def plain(value):
        return value + 1

    decorated = flowview.stage("Disabled")(plain)
    assert decorated is plain
    assert decorated(2) == 3

    with flowview.stage("Disabled context") as active:
        assert active.node_id is None
        assert active.progress(1, 2) is None
        assert active.metric("objects", 4) is None
        assert active.thumbnail(tmp_path / "preview.png") is None

    sentinel = RuntimeError("body failure")
    with pytest.raises(RuntimeError) as caught:
        with flowview.stage("Disabled failure"):
            raise sentinel
    assert caught.value is sentinel
    assert collector.pending == 0


def test_a_stage_disabled_before_use_stays_a_single_boolean_no_op(monkeypatch):
    flowview.disable()

    def construction_would_be_work():
        raise AssertionError("disabled tracing constructed a stage")

    monkeypatch.setattr(trace, "_StageSpec", construction_would_be_work)
    inactive = flowview.stage(
        "No work",
        kind="not-a-node-kind",
        consumes=iter(("input",)),
        params={"unread": object()},
    )

    def plain():
        return "unchanged"

    assert inactive(plain) is plain
    with inactive as active:
        assert active.node_id is None


def test_a_prepared_stage_can_be_disabled_before_decoration_or_entry():
    flowview.enable(_collector())
    prepared = flowview.stage("Prepared")
    flowview.disable()

    def plain():
        return "unchanged"

    assert prepared(plain) is plain
    with prepared as active:
        assert active.node_id is None


def test_enabled_context_records_artifacts_progress_metrics_and_thumbnail(tmp_path):
    collector = _collector()
    flowview.enable(collector)
    thumbnail = tmp_path / "preview.png"

    with flowview.stage(
        "Train model",
        kind="process",
        consumes=["Dataset"],
        produces=["Weights"],
        params={"epochs": 3},
        node_id="train",
    ) as active:
        assert active.node_id == "train"
        active.progress(2, 3)
        active.metric("loss", 0.25)
        active.thumbnail(thumbnail)

    assert collector.drain() == 11
    graph = collector.snapshot()
    stage_node = graph.nodes["train"]
    input_node = next(node for node in graph.nodes.values() if node.label == "Dataset")
    output_node = next(node for node in graph.nodes.values() if node.label == "Weights")

    assert stage_node.state is NodeState.DONE
    assert stage_node.kind is NodeKind.PROCESS
    assert stage_node.progress == (2, 3)
    assert stage_node.metrics == {"loss": 0.25}
    assert stage_node.thumbnail == os.fspath(thumbnail)
    assert stage_node.params == {"epochs": 3}
    assert input_node.state is NodeState.DONE
    assert output_node.state is NodeState.DONE
    assert {(edge.src, edge.dst) for edge in graph.edges} == {
        (input_node.id, "train"),
        ("train", output_node.id),
    }


def test_enabled_decorator_preserves_metadata_and_honours_later_disable():
    collector = _collector()
    flowview.enable(collector)

    @flowview.stage("Evaluate")
    def evaluate(value=1):
        """Return a measured value."""

        return value * 2

    assert evaluate.__name__ == "evaluate"
    assert evaluate.__doc__ == "Return a measured value."
    assert evaluate(4) == 8
    first_count = collector.pending
    assert first_count == 3

    flowview.disable()
    assert evaluate(5) == 10
    assert collector.pending == first_count


def test_stage_failure_marks_output_skipped_and_propagates_same_exception():
    collector = _collector()
    flowview.enable(collector)
    sentinel = LookupError("original stage failure")

    @flowview.stage("Failing stage", produces=["Never written"])
    def fail():
        raise sentinel

    with pytest.raises(LookupError) as caught:
        fail()
    assert caught.value is sentinel

    collector.drain()
    graph = collector.snapshot()
    failed = next(node for node in graph.nodes.values() if node.label == "Failing stage")
    output = next(node for node in graph.nodes.values() if node.label == "Never written")
    assert failed.state is NodeState.FAILED
    assert "LookupError: original stage failure" in failed.error
    assert output.state is NodeState.SKIPPED


def test_traceback_formatting_fault_cannot_mask_the_stage_exception(monkeypatch):
    collector = _collector()
    flowview.enable(collector)
    sentinel = ValueError("must survive")

    def broken_formatter(*args, **kwargs):
        raise RuntimeError("formatter broke")

    monkeypatch.setattr(trace.traceback, "format_exception", broken_formatter)

    with pytest.raises(ValueError) as caught:
        with flowview.stage("Broken formatter"):
            raise sentinel
    assert caught.value is sentinel

    collector.drain()
    failed = next(iter(collector.snapshot().nodes.values()))
    assert failed.error == "ValueError: exception text was unavailable"


def test_emission_faults_and_bad_thumbnails_never_escape(monkeypatch):
    class BrokenCollector:
        def emit(self, event):
            raise RuntimeError("collector unavailable")

    flowview.enable(BrokenCollector())
    with flowview.stage("Unobservable") as active:
        active.progress(1, 1)
        active.metric("objects", 2)
        active.thumbnail(object())

    replacement = _collector()
    flowview.enable(replacement)
    with flowview.stage("Bad thumbnail") as active:
        active.thumbnail(object())
    assert replacement.drain() == 3


def test_enable_without_replacement_reuses_the_collector_and_environment_is_read():
    collector = _collector()
    flowview.enable(collector)
    flowview.disable()

    assert flowview.enable() is collector
    assert flowview.is_enabled() is True

    original = os.environ.get("SPACR_FLOWVIEW")
    try:
        os.environ["SPACR_FLOWVIEW"] = " YES "
        assert trace._environment_enabled() is True
        os.environ["SPACR_FLOWVIEW"] = "0"
        assert trace._environment_enabled() is False
    finally:
        if original is None:
            os.environ.pop("SPACR_FLOWVIEW", None)
        else:
            os.environ["SPACR_FLOWVIEW"] = original

    fresh = trace._new_collector()
    assert fresh.snapshot().spacr_version


def test_empty_label_gets_a_stable_nonempty_identifier():
    collector = _collector()
    flowview.enable(collector)

    with flowview.stage("") as active:
        assert active.node_id.startswith("stage:stage:")
    collector.drain()
    assert len(collector.snapshot().nodes) == 1


def test_importing_flowview_in_a_fresh_process_does_not_import_pyside6():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; from spacr import flowview; "
            "assert flowview.NodeKind.INPUT.value == 'input'; "
            "assert not any(name == 'PySide6' or name.startswith('PySide6.') "
            "for name in sys.modules)"
        ),
    ]
    environment = dict(os.environ, SPACR_FLOWVIEW="1")
    result = subprocess.run(command, env=environment, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
