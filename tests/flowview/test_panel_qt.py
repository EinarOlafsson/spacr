from __future__ import annotations

import builtins
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from spacr.flowview import trace
from spacr.flowview.collector import Collector
from spacr.flowview.events import (
    EdgeAdded,
    NodeAdded,
    StageFailed,
    StageMetric,
    StageStarted,
)
from spacr.flowview.model import Edge, Node, NodeKind, NodeState, RunGraph

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None

if HAS_QT:
    from PySide6.QtCore import Qt  # noqa: E402
    from PySide6.QtWidgets import QFileDialog, QGraphicsScene  # noqa: E402

    from spacr.flowview import panel  # noqa: E402


def _load_with_qt_blocked(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    real_import = builtins.__import__

    def blocked_import(
        name: str,
        globals_: object = None,
        locals_: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name.startswith("PySide6"):
            raise ImportError("PySide6 deliberately hidden")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    alias = f"{module.__package__}._without_qt_panel"
    spec = importlib.util.spec_from_file_location(alias, module.__file__)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[alias] = loaded
    try:
        spec.loader.exec_module(loaded)
    finally:
        sys.modules.pop(alias, None)
    return loaded


@pytest.mark.skipif(not HAS_QT, reason="normal module path needs PySide6")
def test_panel_module_imports_without_qt_and_names_exact_remediation(monkeypatch):
    blocked = _load_with_qt_blocked(panel, monkeypatch)

    assert blocked.QT_AVAILABLE is False
    assert "inspector_text" not in blocked.__all__
    for widget_class in (blocked.FlowGraphicsView, blocked.FlowViewPanel):
        with pytest.raises(ImportError, match=r"pip install spacr\[flowview\]"):
            widget_class("ignored", keyword="ignored")


def _graph() -> RunGraph:
    nodes = {
        "input": Node(
            "input",
            "Images",
            NodeKind.INPUT,
            params={"folder": "/data"},
        ),
        "model": Node(
            "model",
            "Train model",
            NodeKind.PROCESS,
            started_at=2.0,
            params={"family": "rf"},
        ),
        "scores": Node("scores", "Scores", NodeKind.OUTPUT),
    }
    return RunGraph(
        run_id="live-run",
        started_at=1.0,
        nodes=nodes,
        edges=[Edge("input", "model", "files", 100), Edge("model", "scores")],
        spacr_version="test",
        settings_digest="digest",
    )


@pytest.mark.skipif(not HAS_QT, reason="PySide6 is not installed")
def test_inspector_text_contains_complete_details_and_empty_markers():
    complete = Node(
        "stage",
        "Measured cells",
        "process",
        state="failed",
        started_at=2.0,
        ended_at=5.25,
        progress=(4, 9),
        params={"object": Path("cells")},
        metrics={"count": 42},
        error="traceback line",
    )
    complete_text = panel.inspector_text(complete)
    empty_text = panel.inspector_text(Node("empty", "Pending", "input"))
    half_timed = panel.inspector_text(
        Node("half", "Half timed", "process", started_at=1.0)
    )

    assert "Duration: 3.25" in complete_text
    assert 'Progress: [4, 9]' in complete_text
    assert '"object": "cells"' in complete_text
    assert "traceback line" in complete_text
    assert "Started: —" in empty_text and "Ended: —" in empty_text
    assert "Progress: \"—\"" in empty_text and "Error:\n—" in empty_text
    assert "Duration: —" in half_timed and "Started: 1.0" in half_timed


class _Delta:
    def __init__(self, value: int) -> None:
        self._value = value

    def y(self) -> int:
        return self._value


class _Wheel:
    def __init__(self, value: int) -> None:
        self._delta = _Delta(value)
        self.accepted: bool | None = None

    def angleDelta(self) -> _Delta:  # noqa: N802 - mimics Qt
        return self._delta

    def accept(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.accepted = False


@pytest.mark.skipif(not HAS_QT, reason="PySide6 is not installed")
def test_graphics_view_pans_and_has_bounded_wheel_zoom(qtbot):
    view = panel.FlowGraphicsView(QGraphicsScene())
    qtbot.addWidget(view)

    assert view.zoom == 1.0
    assert view.zoom_by(0) is False
    positive = _Wheel(120)
    view.wheelEvent(positive)
    assert positive.accepted is True and view.zoom > 1.0
    negative = _Wheel(-120)
    view.wheelEvent(negative)
    assert negative.accepted is True and view.zoom == pytest.approx(1.0)
    for _ in range(100):
        view.zoom_by(120)
    assert view.zoom == view.MAX_ZOOM
    at_limit = _Wheel(120)
    view.wheelEvent(at_limit)
    assert at_limit.accepted is False
    for _ in range(200):
        view.zoom_by(-120)
    assert view.zoom == view.MIN_ZOOM
    assert view.dragMode() == view.DragMode.ScrollHandDrag


@pytest.mark.skipif(not HAS_QT, reason="PySide6 is not installed")
def test_panel_drains_updates_without_repainting_unchanged_graph_and_inspects(qtbot):
    collector = Collector(_graph(), max_queue_size=1)
    live = panel.FlowViewPanel(collector, auto_start=False)
    qtbot.addWidget(live)
    live.resize(800, 600)
    live.show()

    assert live.timer.interval() == 50
    assert live.timer.isActive() is False
    assert live.start() is True
    assert live.start() is False
    assert live.stop() is True
    assert live.stop() is False
    snapshot = collector.snapshot
    collector.snapshot = lambda: pytest.fail("idle refresh copied the graph")
    assert live.refresh() is False
    collector.snapshot = snapshot
    original_items = dict(live._node_items)

    collector.emit(StageStarted(_graph().nodes["model"], at=3.0))
    assert live.refresh() is True
    assert live._node_items == original_items
    assert live._edge_items[_graph().edges[1]].source_running is True

    model_item = live._node_items["model"]
    click_position = live.view.mapFromScene(model_item.sceneBoundingRect().center())
    qtbot.mouseClick(
        live.view.viewport(),
        Qt.MouseButton.LeftButton,
        pos=click_position,
    )
    qtbot.wait(1)
    assert live._selected_node_id == "model"
    assert "Parameters:" in live.inspector.toPlainText()
    assert '"family": "rf"' in live.inspector.toPlainText()

    collector.emit(StageMetric("model", "accuracy", 0.91))
    collector.emit(StageFailed("model", at=8.0, error="training failed\ntrace"))
    assert collector.sampled is True
    assert live.refresh() is True
    assert live.sample_note.isVisible() is True
    assert "training failed\ntrace" in live.inspector.toPlainText()
    assert live.graph.nodes["scores"].state is NodeState.SKIPPED

    extra = Node("archive", "Archive", NodeKind.OUTPUT)
    collector.emit(NodeAdded(extra))
    collector.drain()
    collector.fold(EdgeAdded(Edge("scores", "archive", volume=10)))
    previous_model_item = live._node_items["model"]
    assert live.refresh() is True
    assert live._node_items["model"] is not previous_model_item
    assert live._node_items["model"].isSelected() is True

    live.scene.clearSelection()
    qtbot.wait(1)
    assert live._selected_node_id is None
    assert live.inspector.toPlainText() == ""
    live.close()
    assert live.timer.isActive() is False
    assert live.scene.items() == []


@pytest.mark.skipif(not HAS_QT, reason="PySide6 is not installed")
def test_panel_validates_interval_handles_empty_graph_and_auto_starts(qtbot):
    collector = Collector(
        RunGraph("empty", 1.0, {}, [], "test", "digest")
    )
    with pytest.raises(ValueError, match="greater than zero"):
        panel.FlowViewPanel(collector, refresh_interval_ms=0)

    live = panel.FlowViewPanel(collector)
    qtbot.addWidget(live)
    assert live.timer.isActive() is True
    assert live.graph.run_id == "empty"
    assert live.scene.items() == []
    live.close()


@pytest.mark.skipif(not HAS_QT, reason="PySide6 is not installed")
def test_panel_follows_replaced_global_collector_safely(qtbot, monkeypatch):
    previous_collector = trace.get_collector()
    previous_enabled = trace.is_enabled()
    first = Collector(_graph())
    trace.enable(first)
    live = panel.FlowViewPanel(first, auto_start=False)
    qtbot.addWidget(live)

    try:
        second_graph = RunGraph(
            "next-run",
            5.0,
            {"fresh": Node("fresh", "Fresh run", NodeKind.INPUT)},
            [],
            "test",
            "next-digest",
        )
        second = Collector(second_graph)
        trace.enable(second)
        live._selected_node_id = "model"

        assert live.refresh() is True
        assert live.graph.run_id == "next-run"
        assert set(live._node_items) == {"fresh"}
        assert live._selected_node_id is None

        def broken_lookup():
            raise RuntimeError("trace registry unavailable")

        monkeypatch.setattr(panel, "get_collector", broken_lookup)
        assert live.refresh() is False
        assert live.graph.run_id == "next-run"
    finally:
        live.close()
        trace.enable(previous_collector)
        if not previous_enabled:
            trace.disable()


@pytest.mark.skipif(not HAS_QT, reason="PySide6 is not installed")
def test_export_button_uses_static_exporters_and_surfaces_every_outcome(
    tmp_path,
    qtbot,
    monkeypatch,
):
    collector = Collector(_graph())
    target = tmp_path / "live.html"
    live = panel.FlowViewPanel(
        collector,
        export_path_provider=lambda: target,
        auto_start=False,
    )
    qtbot.addWidget(live)

    live.export_button.click()
    assert target.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert live.export_status.text() == "Exported live.html"

    unknown = tmp_path / "figure.unknown"
    live._export_path_provider = lambda: unknown
    assert live._export_current() == unknown
    assert unknown.read_text(encoding="utf-8").startswith("<svg")

    live._export_path_provider = lambda: None
    assert live._export_current() is None
    assert live.export_status.text() == "Export cancelled."

    live._export_path_provider = lambda: tmp_path / "broken.svg"
    real_export = panel.export_graph

    def broken_export(*args, **kwargs):
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr(panel, "export_graph", broken_export)
    assert live._export_current() is None
    assert live.export_status.text() == "Export failed: disk full"

    dialog_target = tmp_path / "dialog.json"
    live._export_path_provider = None
    monkeypatch.setattr(panel, "export_graph", real_export)
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(dialog_target), "JSON (*.json)"),
    )
    assert live._export_current() == dialog_target
    assert dialog_target.read_text(encoding="utf-8") == live.graph.to_json()
    live.close()
