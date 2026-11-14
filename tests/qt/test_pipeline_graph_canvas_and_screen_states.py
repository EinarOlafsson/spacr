"""Pipeline Graph: the canvas's boxes, clicks and edges, and the screen with no project.

The canvas is a dict of rectangles a test can read, laid out on a fixed grid
(one column per depth).  What it must never do is keep a box selected after
that box stopped being drawn, draw an arrow into a hidden box, or raise in
``paintEvent`` over a layer naming an artifact the graph does not hold.  The
screen, before a project is open or after a failed read, must say so on its
banner rather than show the previous project's graph.

Offscreen; the registry environment variable is cleared for every test.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, QRect, Qt   # noqa: E402
from PySide6.QtGui import QMouseEvent                    # noqa: E402

from spacr import artifacts                              # noqa: E402
from spacr.pipeline_graph import (Edge, Node, PipelineGraph,  # noqa: E402
                                  STATE_CURRENT, STATE_MISSING, STATE_STALE)
from spacr.qt.screens import pipeline_graph as screen_module  # noqa: E402

pytestmark = pytest.mark.qt

#: Where the grid puts depth 0 and depth 1, row 0 (MARGIN 18, 190 x 74 boxes,
#: a 64 px column gap).
RECT_A = QRect(18, 18, 190, 74)
RECT_B = QRect(272, 18, 190, 74)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Never let a developer's real artifact database into these tests."""
    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)


def _node(artifact_id: str, depth: int = 0, state: str = STATE_CURRENT) -> Node:
    return Node(artifact_id=artifact_id, project="/p", kind="masks",
                role="cell_mask", module="mask", path=f"/p/{artifact_id}.npy",
                state=state, depth=depth)


def _graph(*nodes, edges=()) -> PipelineGraph:
    layers: dict = {}
    for node in nodes:
        layers.setdefault(node.depth, []).append(node.artifact_id)
    return PipelineGraph(
        project="/p", nodes=tuple(nodes), edges=tuple(edges),
        layers=tuple(tuple(layers[d]) for d in sorted(layers)))


def _pair(state_b=STATE_CURRENT, **edge):
    return _graph(_node("a"), _node("b", depth=1, state=state_b),
                  edges=[Edge(source="a", target="b", kind="masks", **edge)])


def _press(canvas, x, y):
    point = QPointF(x, y)
    canvas.mousePressEvent(QMouseEvent(
        QEvent.MouseButtonPress, point, point, Qt.LeftButton, Qt.LeftButton,
        Qt.NoModifier))


@pytest.fixture
def canvas(qtbot):
    """A bare canvas, sized so every box of a small graph is on it."""
    widget = screen_module.GraphCanvas()
    qtbot.addWidget(widget)
    widget.resize(900, 400)
    return widget


@pytest.fixture
def screen(qtbot):
    """The whole screen, running its jobs inline."""
    widget = screen_module.PipelineGraphScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_graph_canvas_draws_the_visible_boxes_and_drops_a_hidden_selection(
        subtests, canvas):
    """``set_graph`` lays out, filters and keeps the selection only if drawn."""
    pair = _graph(_node("a"), _node("b", depth=1))
    cases = (
        # case id, graph, visible, selected before, rects, selected after,
        # minimum size
        ("no filter draws every box", pair, None, "", {"a": RECT_A,
                                                       "b": RECT_B},
         "", (479, 200)),
        ("a redraw still showing the box keeps it", pair, {"a", "b"}, "b",
         {"a": RECT_A, "b": RECT_B}, "b", (479, 200)),
        ("a filter hiding the selected box clears it", pair, {"a"}, "b",
         {"a": RECT_A}, "", (479, 200)),
        ("ids the graph does not hold are ignored", pair, {"a", "zz"}, "a",
         {"a": RECT_A}, "a", (479, 200)),
        ("one box needs only the minimum", _graph(_node("a")), None, "a",
         {"a": RECT_A}, "a", (400, 200)),
        ("no graph draws nothing", None, None, "a", {}, "", (400, 200)),
    )
    for case_id, graph, visible, before, rects, after, minimum in cases:
        with subtests.test(msg=case_id):
            if before:
                canvas.set_graph(pair)
                canvas.select(before)
                assert canvas.selected == before
            canvas.set_graph(graph, visible=visible)
            assert canvas.graph() is graph
            assert canvas.node_rects() == rects
            assert canvas.selected == after
            assert (canvas.minimumSize().width(),
                    canvas.minimumSize().height()) == minimum
            assert not canvas.grab().isNull()      # the paint really runs


def test_graph_canvas_click_selects_the_box_under_it_or_none(subtests, canvas):
    """A click on a box names it; a click on the background clears and says so."""
    canvas.set_graph(_graph(_node("a"), _node("b", depth=1)), visible={"a"})
    seen = []
    canvas.node_clicked.connect(seen.append)
    cases = (
        ("centre of a", RECT_A.center().x(), RECT_A.center().y(), "a", ["a"]),
        ("between the boxes", RECT_A.right() + 40, RECT_A.bottom() + 40, "",
         ["a", ""]),
        ("on the hidden box b", RECT_B.center().x(), RECT_B.center().y(), "",
         ["a", "", ""]),
        ("a's top-left corner", 18, 18, "a", ["a", "", "", "a"]),
    )
    for case_id, x, y, selected, emitted in cases:
        with subtests.test(msg=case_id):
            assert canvas.node_at(x, y) == selected
            _press(canvas, x, y)
            assert canvas.selected == selected
            assert seen == emitted

    with subtests.test(msg="select() of an undrawn id clears"):
        canvas.select("b")
        assert canvas.selected == ""
        assert seen[-1] == ""


def test_graph_canvas_paints_only_edges_whose_ends_are_both_drawn(subtests,
                                                                  canvas):
    """Filtered ends drop the arrow, a dangling edge is dashed, a ghost is skipped."""
    plain = _pair()
    pictures = {}
    cases = (
        ("both ends drawn", plain, None, {"a": RECT_A, "b": RECT_B}),
        ("target filtered away", plain, {"a"}, {"a": RECT_A}),
        ("source filtered away", plain, {"b"}, {"b": RECT_B}),
        ("a forgotten source is dashed", _pair(dangling=True), None,
         {"a": RECT_A, "b": RECT_B}),
        ("a hidden missing box", _pair(state_b=STATE_MISSING), {"a"},
         {"a": RECT_A}),
        ("a stale box", _pair(state_b=STATE_STALE), None,
         {"a": RECT_A, "b": RECT_B}),
        ("a layer naming an artifact the graph lacks",
         PipelineGraph(project="/p", nodes=(_node("a"),),
                       layers=(("a", "ghost"),)), None,
         {"a": RECT_A, "ghost": QRect(18, 114, 190, 74)}),
    )
    for case_id, graph, visible, rects in cases:
        with subtests.test(msg=case_id):
            canvas.set_graph(graph, visible=visible)
            assert canvas.node_rects() == rects
            pictures[case_id] = canvas.grab().toImage()   # must not raise

    with subtests.test(msg="what is drawn differs where the state differs"):
        assert pictures["a forgotten source is dashed"] != \
            pictures["both ends drawn"]
        assert pictures["a stale box"] != pictures["both ends drawn"]
        assert pictures["target filtered away"] == \
            pictures["a hidden missing box"]


def test_pipeline_graph_screen_before_a_project_and_after_a_failed_read(
        subtests, screen, tmp_path):
    """No project, no graph or a failed build each leave an exact banner."""
    with subtests.test(msg="a filter toggled before a project draws nothing"):
        screen._filters[STATE_MISSING].setChecked(False)
        assert screen._canvas.graph() is None
        assert screen._canvas.node_rects() == {}
        assert screen._verdict.text() == (
            "Choose a spaCR project folder to draw what it has produced.")
        assert screen._verdict.property("problem") == "false"
        screen._filters[STATE_MISSING].setChecked(True)

    with subtests.test(msg="copying before a project is open"):
        assert screen.graph() is None
        screen._on_copy_dot()
        assert screen._verdict.text() == "There is no graph to copy yet."
        assert screen._verdict.property("problem") == "true"

    with subtests.test(msg="typing a project and pressing return loads it"):
        root = tmp_path / "plate1"
        (root / "masks").mkdir(parents=True)
        registry = artifacts.Registry(project=str(root))
        path = root / "masks" / "cell.npy"
        path.write_text("x", encoding="utf-8")
        registry.register(module="mask", kind="masks", role="cell_mask",
                          path=str(path), settings={"diameter": 30},
                          run_id="run-1")
        loaded = []
        screen.graph_loaded.connect(loaded.append)
        screen._project_edit.setText(f"  {root}  ")
        screen._on_project_entered()
        assert screen._project_edit.text() == str(root)
        assert screen.last_error == ""
        graph = screen._canvas.graph()
        assert loaded == [graph]
        assert screen.graph() is graph
        assert list(screen._canvas.node_rects().values()) == [RECT_A]
        assert screen._verdict.property("problem") == "false"

    with subtests.test(msg="a graph that could not be built"):
        screen._on_graph_ready(_graph(_node("a")))
        assert screen.graph() is not None
        screen._on_graph_ready(None)
        assert screen.graph() is None
        assert screen._verdict.text() == "The graph could not be built."
        assert screen._verdict.property("problem") == "true"
