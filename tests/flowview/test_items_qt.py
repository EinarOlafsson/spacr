from __future__ import annotations

import builtins
import importlib.util
import os
import sys
from types import ModuleType

import pytest

from spacr.flowview.layout import NodeLayout, layout_graph
from spacr.flowview.model import Edge, Node, NodeKind, NodeState, RunGraph

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None

if HAS_QT:
    from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
    from PySide6.QtWidgets import QGraphicsScene  # noqa: E402

    from spacr.flowview import items  # noqa: E402


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
    alias = f"{module.__package__}._without_qt_items"
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
def test_items_module_imports_without_qt_and_names_exact_remediation(monkeypatch):
    blocked = _load_with_qt_blocked(items, monkeypatch)

    assert blocked.QT_AVAILABLE is False
    assert "edge_width" not in blocked.__all__
    for item_class in (blocked.NodeItem, blocked.EdgeItem):
        with pytest.raises(ImportError, match=r"pip install spacr\[flowview\]"):
            item_class("ignored", keyword="ignored")


def _graph(thumbnail: str) -> RunGraph:
    nodes = {
        "input": Node(
            "input",
            "Raw fluorescence images",
            NodeKind.INPUT,
            state=NodeState.RUNNING,
            progress=(5, 10),
            metrics={"objects": 1_000, "rate": 12.5, "third": "ok", "hidden": 4},
            thumbnail=thumbnail,
        ),
        "process": Node(
            "process",
            "Measure cells",
            NodeKind.PROCESS,
            state=NodeState.DONE,
            progress=(1, 0),
        ),
        "output": Node(
            "output",
            "Scores",
            NodeKind.OUTPUT,
            state=NodeState.FAILED,
        ),
    }
    return RunGraph(
        run_id="qt-items",
        started_at=1.0,
        nodes=nodes,
        edges=[
            Edge("input", "process", "1,000 objects", 1_000),
            Edge("process", "output", None, None),
        ],
        spacr_version="test",
        settings_digest="digest",
    )


@pytest.mark.skipif(not HAS_QT, reason="PySide6 is not installed")
def test_node_and_edge_items_paint_every_visual_state_and_update(tmp_path, qapp):
    del qapp
    thumbnail = tmp_path / "thumb.png"
    image = QImage(12, 8, QImage.Format.Format_ARGB32)
    image.fill(QColor("#5FA8C7"))
    assert image.save(str(thumbnail))
    graph = _graph(str(thumbnail))
    layout = layout_graph(graph)
    scene = QGraphicsScene()

    edge_items = [
        items.EdgeItem(
            edge,
            layout[edge.src],
            layout[edge.dst],
            source_running=graph.nodes[edge.src].state is NodeState.RUNNING,
        )
        for edge in graph.edges
    ]
    node_items = [
        items.NodeItem(graph.nodes[node_id], layout[node_id])
        for node_id in sorted(graph.nodes)
    ]
    for item in [*edge_items, *node_items]:
        scene.addItem(item)

    canvas = QImage(
        int(layout.width),
        int(layout.height),
        QImage.Format.Format_ARGB32,
    )
    canvas.fill(QColor("#0E1216"))
    painter = QPainter(canvas)
    scene.render(painter)
    painter.end()

    assert node_items[0].node_id == "input"
    assert node_items[0].boundingRect().width() == layout["input"].width
    assert not edge_items[0].boundingRect().isEmpty()
    assert edge_items[0].source_running is True
    assert edge_items[0].set_source_running(True) is False
    assert edge_items[0].set_source_running(False) is True
    assert edge_items[0].source_running is False

    original = graph.nodes["input"]
    input_item = node_items[0]
    assert input_item.update_node(original, layout["input"]) is False
    changed = Node(
        original.id,
        "Updated input",
        original.kind,
        metrics=original.metrics,
        progress=(-2, 10),
    )
    moved = NodeLayout(41.0, 42.0, 225.0, 105.0, 0, 0)
    assert input_item.update_node(changed, moved) is True
    assert input_item.pos().x() == 41.0
    assert input_item.toolTip() == "Updated input — PENDING"

    second_canvas = QImage(600, 300, QImage.Format.Format_ARGB32)
    second_canvas.fill(QColor("#0E1216"))
    second_painter = QPainter(second_canvas)
    scene.render(second_painter)
    second_painter.end()
    assert second_canvas.pixelColor(10, 10).isValid()


@pytest.mark.skipif(not HAS_QT, reason="PySide6 is not installed")
def test_edge_width_is_bounded_and_module_advertises_live_helper():
    assert items.QT_AVAILABLE is True
    assert "edge_width" in items.__all__
    assert items.edge_width(None) == 1.0
    assert items.edge_width(0) == 1.0
    assert 1.0 < items.edge_width(100) < 6.0
    assert items.edge_width(10**20) == 6.0
