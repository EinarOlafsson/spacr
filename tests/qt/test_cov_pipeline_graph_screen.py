"""The Pipeline Graph canvas in the states nothing had put it in.

Written for instruction 60. ``tests/qt/test_pipeline_graph_screen.py`` drives
the screen against a real registry; what it never reached was the canvas with
no graph at all, a click that lands between the boxes, an edge with one end
filtered away, and the screen opened on nothing.

The whole-package ranking listed this module at 51.8% with 160 uncovered
statements. Over its own tests it was 95% with 17 — the same "a module whose
tests are marker-excluded reads as uncovered" trap that file records, one
level down: the ranking's run excluded most of the files that exercise it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt          # noqa: E402
from PySide6.QtGui import QMouseEvent                    # noqa: E402

from spacr import artifacts                              # noqa: E402
from spacr.pipeline_graph import (Edge, Node, PipelineGraph,  # noqa: E402
                                  STATE_CURRENT, STATE_MISSING)
from spacr.qt.screens import pipeline_graph as screen_module  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)


def _node(artifact_id: str, depth: int = 0, state: str = STATE_CURRENT,
          module: str = "mask") -> Node:
    return Node(artifact_id=artifact_id, project="/p", kind="masks",
                role="cell_mask", module=module, path=f"/p/{artifact_id}.npy",
                state=state, depth=depth)


def _graph(*nodes, edges=()) -> PipelineGraph:
    layers: dict = {}
    for node in nodes:
        layers.setdefault(node.depth, []).append(node.artifact_id)
    return PipelineGraph(
        project="/p", nodes=tuple(nodes), edges=tuple(edges),
        layers=tuple(tuple(layers[d]) for d in sorted(layers)))


@pytest.fixture
def canvas(qtbot):
    widget = screen_module.GraphCanvas()
    qtbot.addWidget(widget)
    widget.resize(900, 400)
    return widget


# ---------------------------------------------------------------------------
# A canvas with nothing on it
# ---------------------------------------------------------------------------

def test_a_canvas_with_no_graph_draws_nothing_and_says_so(canvas):
    """Opening the screen before a project is chosen is the ordinary first
    state, not an error, and it must not leave a stale box selected."""
    canvas.set_graph(_graph(_node("a")))
    canvas.select("a")
    assert canvas.selected == "a"
    canvas.set_graph(None)
    assert canvas.graph() is None
    assert canvas.node_rects() == {}
    assert canvas.selected == ""
    assert canvas.minimumSize().width() == 400
    canvas.render(canvas.grab())        # the empty paint really runs


def test_every_box_is_drawn_when_no_filter_was_given(canvas):
    graph = _graph(_node("a"), _node("b", depth=1))
    canvas.set_graph(graph)
    assert canvas.graph() is graph
    assert set(canvas.node_rects()) == {"a", "b"}


def test_a_filter_that_hides_the_selected_box_clears_the_selection(canvas):
    canvas.set_graph(_graph(_node("a"), _node("b", depth=1)))
    canvas.select("b")
    assert canvas.selected == "b"
    canvas.set_graph(_graph(_node("a"), _node("b", depth=1)), visible={"a"})
    assert canvas.selected == ""
    assert set(canvas.node_rects()) == {"a"}


# ---------------------------------------------------------------------------
# Clicking
# ---------------------------------------------------------------------------

def test_clicking_between_the_boxes_clears_the_selection(canvas, qtbot):
    """A click on the background is how a user puts the details panel back;
    it must not leave the previous box ringed."""
    canvas.set_graph(_graph(_node("a")))
    canvas.select("a")
    seen = []
    canvas.node_clicked.connect(seen.append)
    rect = canvas.node_rects()["a"]
    away = QPointF(rect.right() + 40, rect.bottom() + 40)
    canvas.mousePressEvent(QMouseEvent(
        QEvent.MouseButtonPress, away, away, Qt.LeftButton, Qt.LeftButton,
        Qt.NoModifier))
    assert canvas.selected == ""
    assert seen == [""]


def test_clicking_a_box_selects_it_and_names_it(canvas):
    canvas.set_graph(_graph(_node("a")))
    seen = []
    canvas.node_clicked.connect(seen.append)
    centre = QPointF(canvas.node_rects()["a"].center())
    canvas.mousePressEvent(QMouseEvent(
        QEvent.MouseButtonPress, centre, centre, Qt.LeftButton, Qt.LeftButton,
        Qt.NoModifier))
    assert canvas.selected == "a"
    assert seen == ["a"]


def test_asking_which_box_is_under_a_point_outside_them_all_names_none(canvas):
    canvas.set_graph(_graph(_node("a")))
    rect = canvas.node_rects()["a"]
    assert canvas.node_at(rect.right() + 50, rect.bottom() + 50) == ""


# ---------------------------------------------------------------------------
# Edges with one end filtered away
# ---------------------------------------------------------------------------

def test_an_edge_whose_target_is_filtered_away_is_not_drawn(canvas):
    """An arrow into nothing points at empty canvas and reads as a missing
    box rather than a hidden one."""
    graph = _graph(_node("a"), _node("b", depth=1),
                   edges=[Edge(source="a", target="b", kind="masks")])
    canvas.set_graph(graph, visible={"a"})
    canvas.render(canvas.grab())
    assert set(canvas.node_rects()) == {"a"}


def test_an_edge_whose_source_is_filtered_away_is_not_drawn(canvas):
    graph = _graph(_node("a"), _node("b", depth=1),
                   edges=[Edge(source="a", target="b", kind="masks")])
    canvas.set_graph(graph, visible={"b"})
    canvas.render(canvas.grab())
    assert set(canvas.node_rects()) == {"b"}


def test_an_edge_to_an_artifact_the_registry_forgot_is_drawn_dashed(canvas):
    """The fact that something was consumed and then forgotten is exactly
    what makes the target stale; dropping the edge would hide it."""
    graph = _graph(_node("a"), _node("b", depth=1),
                   edges=[Edge(source="a", target="b", kind="masks",
                               dangling=True)])
    canvas.set_graph(graph)
    canvas.render(canvas.grab())
    assert set(canvas.node_rects()) == {"a", "b"}


def test_a_hidden_box_is_not_painted(canvas):
    graph = _graph(_node("a"), _node("b", depth=1, state=STATE_MISSING))
    canvas.set_graph(graph, visible={"a"})
    canvas.render(canvas.grab())
    assert set(canvas.node_rects()) == {"a"}


# ---------------------------------------------------------------------------
# The screen with no project
# ---------------------------------------------------------------------------

def test_opening_the_screen_on_nothing_asks_for_a_project(qtbot):
    """A blank canvas with no message reads as a project that has no
    artifacts, which is a different and much more alarming claim."""
    widget = screen_module.PipelineGraphScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.load_project("   ")
    assert "Choose a spaCR project folder" in widget._verdict.text()
    assert widget._canvas.graph() is None
    assert widget.last_error == ""


def test_typing_a_project_and_pressing_return_loads_it(qtbot, tmp_path):
    root = tmp_path / "plate1"
    (root / "masks").mkdir(parents=True)
    registry = artifacts.Registry(project=str(root))
    path = root / "masks" / "cell.npy"
    path.write_text("x", encoding="utf-8")
    registry.register(module="mask", kind="masks", role="cell_mask",
                      path=str(path), settings={"diameter": 30},
                      run_id="run-1")
    widget = screen_module.PipelineGraphScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._project_edit.setText(str(root))
    widget._on_project_entered()
    assert widget._canvas.graph() is not None
    assert len(widget._canvas.node_rects()) == 1


def test_toggling_a_filter_before_a_project_is_open_draws_nothing(qtbot):
    """The filter boxes exist from the first paint, so they can be clicked
    while there is no graph behind them."""
    widget = screen_module.PipelineGraphScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._filters[STATE_MISSING].setChecked(False)
    assert widget._canvas.graph() is None
    assert widget._canvas.node_rects() == {}


# ---------------------------------------------------------------------------
# The factory the registry calls
# ---------------------------------------------------------------------------

def test_the_registry_factory_builds_the_screen(qtbot):
    """The registry calls it with the app key; the screen takes none."""
    widget = screen_module.make_pipeline_graph_screen(screen_module.APP_KEY)
    qtbot.addWidget(widget)
    assert isinstance(widget, screen_module.PipelineGraphScreen)


# ---------------------------------------------------------------------------
# The two branches that used to be excluded rather than tested
# ---------------------------------------------------------------------------

def test_browsing_for_a_project_loads_the_one_that_was_chosen(qtbot, tmp_path,
                                                                monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    root = tmp_path / "plate1"
    (root / "masks").mkdir(parents=True)
    registry = artifacts.Registry(project=str(root))
    path = root / "masks" / "cell.npy"
    path.write_text("x", encoding="utf-8")
    registry.register(module="mask", kind="masks", role="cell_mask",
                      path=str(path), settings={"diameter": 30},
                      run_id="run-1")
    widget = screen_module.PipelineGraphScreen(threaded=False)
    qtbot.addWidget(widget)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(root)))
    widget._on_browse()
    assert widget._project_edit.text() == str(root)
    assert len(widget._canvas.node_rects()) == 1


def test_cancelling_the_project_browser_leaves_the_open_project_alone(
        qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    widget = screen_module.PipelineGraphScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._project_edit.setText("/kept")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    widget._on_browse()
    assert widget._project_edit.text() == "/kept"


def test_a_layer_naming_an_artifact_the_graph_does_not_hold_is_skipped(canvas):
    """`build_graph` builds the layers FROM the nodes, so it cannot produce
    this -- but a graph handed in by a caller can, and an exception raised
    inside paintEvent is a window that will not redraw at all."""
    graph = PipelineGraph(project="/p", nodes=(_node("a"),),
                          layers=(("a", "ghost"),))
    canvas.set_graph(graph)
    assert set(canvas.node_rects()) == {"a", "ghost"}
    canvas.render(canvas.grab())        # the ghost is skipped, not raised on


def test_a_graph_that_could_not_be_built_says_so_and_clears_the_canvas(qtbot):
    """A screen that kept the PREVIOUS project's graph after a failed read
    would be showing one project's provenance under another's name."""
    widget = screen_module.PipelineGraphScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._on_graph_ready(_graph(_node("a")))
    assert widget.graph() is not None
    widget._on_graph_ready(None)
    assert widget.graph() is None
    assert "could not be built" in widget._verdict.text()
