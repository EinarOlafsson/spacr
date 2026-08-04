"""The Pipeline Graph screen, driven against a real registry on disk.

The screen is a view over :mod:`spacr.pipeline_graph`, so what is tested here
is the view: that the boxes land where the layers say they should, that a
stale artifact is drawn differently from a missing one, that the filters
actually remove boxes and their edges, and that clicking a box produces the
provenance a user came for.

Every test runs ``threaded=False`` so the graph is built by the time the call
returns. Both paths run the same code and emit the same signals.

Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PySide6")

from spacr import artifacts                                    # noqa: E402
from spacr.pipeline_graph import (STATE_CURRENT, STATE_MISSING,  # noqa: E402
                                  STATE_STALE)
from spacr.qt.screens import pipeline_graph as screen_module    # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """No test may inherit a shared-registry override from the environment."""
    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)


def _write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture()
def project(tmp_path):
    """A project whose mask step was re-run after measure and classify."""
    root = tmp_path / "plate1"
    root.mkdir()
    registry = artifacts.Registry(project=str(root))
    masks = registry.register(
        module="mask", kind="masks", role="cell_mask",
        path=_write(root / "masks" / "cell.npy"),
        settings={"diameter": 30}, run_id="run-mask-1")
    time.sleep(0.002)
    db = registry.register(
        module="measure", kind="measurements-db", role="merged",
        path=_write(root / "measurements" / "measurements.db"),
        settings={"channels": [0, 1]}, inputs=[masks], run_id="run-measure-1")
    time.sleep(0.002)
    registry.register(
        module="classify", kind="predictions", role="predictions",
        path=_write(root / "results" / "predictions.csv"),
        settings={"epochs": 5}, inputs=[db], run_id="run-classify-1")
    time.sleep(0.002)
    registry.register(
        module="mask", kind="masks", role="cell_mask",
        path=_write(root / "masks" / "cell.npy", "re-run"),
        settings={"diameter": 45}, run_id="run-mask-2")
    return str(root)


@pytest.fixture()
def screen(qtbot, project):
    """The screen, opened on the project, running inline."""
    widget = screen_module.PipelineGraphScreen(project=project, threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_the_screen_registers_itself_through_the_seam():
    """No row in ``app.py``; the import is what puts it in the registry."""
    from spacr.qt.app import APPS, SECTION_EXPLORE, registered_factory

    row = next((r for r in APPS if r[0] == screen_module.APP_KEY), None)
    assert row is not None, "importing the module did not register the app"
    assert row[3] == SECTION_EXPLORE
    assert registered_factory(screen_module.APP_KEY) is (
        screen_module.make_pipeline_graph_screen)
    assert screen_module.register() is False, "register() is not idempotent"


def test_the_screen_answers_spacr_run_with_a_sentence():
    from spacr import cli

    note = cli.INTERACTIVE_ONLY.get(screen_module.APP_KEY, "")
    assert len(note) >= 40, "a GUI-only app owes the CLI an explanation"
    assert "build_graph" in note, "the note must name the headless call"


def test_the_screen_styles_itself_through_the_theme_seam(qapp):
    from spacr.qt.theme import stylesheet, widget_qss_names

    assert "PipelineGraphCanvasArea" in widget_qss_names()
    assert "QFrame#PipelineGraphBanner" in stylesheet()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def test_the_layout_puts_one_column_per_layer(screen):
    graph = screen.graph()
    rects = screen_module.layout_rects(graph)

    assert len(rects) == len(graph.nodes)
    columns = {}
    for node in graph.nodes:
        columns.setdefault(node.depth, set()).add(rects[node.artifact_id].x())
    for depth, xs in columns.items():
        assert len(xs) == 1, f"layer {depth} was drawn at {xs}"
    ordered = [min(columns[d]) for d in sorted(columns)]
    assert ordered == sorted(ordered), "columns must increase with depth"


def test_boxes_in_one_column_do_not_overlap(screen):
    rects = screen_module.layout_rects(screen.graph())
    by_column = {}
    for rect in rects.values():
        by_column.setdefault(rect.x(), []).append(rect)
    for column in by_column.values():
        column.sort(key=lambda r: r.y())
        for first, second in zip(column, column[1:]):
            assert first.bottom() < second.top(), "two boxes overlap"


def test_the_canvas_is_sized_to_hold_the_whole_graph(screen):
    width, height = screen_module.canvas_size(screen.graph())
    rects = screen_module.layout_rects(screen.graph())

    assert width >= max(r.right() for r in rects.values())
    assert height >= max(r.bottom() for r in rects.values())


def test_an_empty_graph_still_has_a_canvas_size():
    from spacr.pipeline_graph import PipelineGraph

    assert screen_module.canvas_size(PipelineGraph()) == (400, 200)
    assert screen_module.layout_rects(PipelineGraph()) == {}


# ---------------------------------------------------------------------------
# Drawing the verdict
# ---------------------------------------------------------------------------

def test_the_graph_loads_and_the_banner_reports_the_stale_count(screen):
    graph = screen.graph()

    assert graph is not None
    assert len(graph) == 4
    states = {n.module: n.state for n in graph.nodes if n.module != "mask"}
    assert states["measure"] == STATE_STALE
    assert states["classify"] == STATE_STALE
    assert "stale" in screen._verdict.text()
    assert screen._verdict.property("problem") == "true"


def test_a_project_with_no_registry_is_reported_not_raised(qtbot, tmp_path):
    widget = screen_module.PipelineGraphScreen(
        project=str(tmp_path / "never-run"), threaded=False)
    qtbot.addWidget(widget)

    assert widget.last_error == ""
    assert widget.graph() is not None
    assert len(widget.graph()) == 0
    assert "registry" in widget._verdict.text().lower()
    assert "mask" in widget._declared.text(), (
        "an empty project must still show the pipeline it is about to run")


def test_a_missing_file_is_drawn_differently_from_a_stale_one(qtbot, project):
    os.remove(os.path.join(project, "results", "predictions.csv"))
    widget = screen_module.PipelineGraphScreen(project=project, threaded=False)
    qtbot.addWidget(widget)

    states = {n.module: n.state for n in widget.graph().nodes}
    assert states["classify"] == STATE_MISSING
    assert states["measure"] == STATE_STALE
    fills = screen_module.STATE_COLOURS
    assert fills[STATE_MISSING] != fills[STATE_STALE] != fills[STATE_CURRENT]


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------

def test_clicking_a_box_selects_it_and_fills_the_detail_pane(screen, qtbot):
    graph = screen.graph()
    node = next(n for n in graph.nodes if n.module == "measure")
    rect = screen._canvas.node_rects()[node.artifact_id]

    with qtbot.waitSignal(screen._canvas.node_clicked, timeout=1000) as caught:
        screen._canvas.select(node.artifact_id)

    assert caught.args == [node.artifact_id]
    assert screen._canvas.selected == node.artifact_id
    assert screen._canvas.node_at(rect.center().x(),
                                 rect.center().y()) == node.artifact_id
    text = screen._details.toPlainText()
    assert "run-measure-1" in text
    assert "settings digest" in text
    assert "Re-running this would invalidate" in text
    assert "classify" in text


def test_clicking_the_background_clears_the_selection(screen):
    node = screen.graph().nodes[0]
    screen._canvas.select(node.artifact_id)

    screen._canvas.select("")

    assert screen._canvas.selected == ""
    assert screen._details.toPlainText().startswith("Click a box")


def test_the_detail_block_names_the_reasons_a_node_is_stale(screen):
    node = next(n for n in screen.graph().nodes if n.module == "measure")

    text = screen.describe(node.artifact_id)

    assert "Why it is flagged:" in text
    assert "re-produced by mask" in text
    assert "Made from:" in text


def test_describe_is_empty_for_an_unknown_id(screen):
    assert screen.describe("no-such-id") == ""
    assert screen.describe("") == ""


def test_the_state_filters_remove_boxes(screen):
    everything = len(screen._canvas.node_rects())

    screen._filters[STATE_STALE].setChecked(False)

    remaining = screen._canvas.node_rects()
    assert len(remaining) < everything
    stale_ids = {n.artifact_id for n in screen.graph().nodes
                 if n.state == STATE_STALE}
    assert not (set(remaining) & stale_ids)


def test_hiding_every_state_leaves_an_empty_canvas_not_a_crash(screen, qapp):
    for box in screen._filters.values():
        box.setChecked(False)

    assert screen._canvas.node_rects() == {}
    screen.resize(1200, 720)
    screen.show()
    assert not screen.grab().isNull()


def test_hiding_a_selected_node_drops_the_selection(screen):
    node = next(n for n in screen.graph().nodes if n.state == STATE_STALE)
    screen._canvas.select(node.artifact_id)

    screen._filters[STATE_STALE].setChecked(False)

    assert screen._canvas.selected == ""


# ---------------------------------------------------------------------------
# Painting and export
# ---------------------------------------------------------------------------

def test_the_screen_renders_at_the_window_size(screen, qt_theme_applied):
    screen.resize(1200, 720)
    screen.show()

    frame = screen.grab()

    assert not frame.isNull()
    assert frame.width() >= 1200 and frame.height() >= 720


def test_the_empty_canvas_paints_its_explanation(qtbot, qt_theme_applied,
                                                 tmp_path):
    widget = screen_module.PipelineGraphScreen(
        project=str(tmp_path / "nothing"), threaded=False)
    qtbot.addWidget(widget)
    widget.resize(1200, 720)
    widget.show()

    assert not widget.grab().isNull()
    assert widget._canvas.node_rects() == {}


def test_copying_graphviz_puts_the_whole_graph_on_the_clipboard(screen, qapp):
    screen._on_copy_dot()

    text = qapp.clipboard().text()
    assert text.startswith("digraph spacr {")
    assert text.count("->") == len(screen.graph().edges)
    assert "copied as Graphviz DOT" in screen._verdict.text()


def test_copying_with_no_graph_says_so_rather_than_raising(qtbot):
    widget = screen_module.PipelineGraphScreen(threaded=False)
    qtbot.addWidget(widget)

    widget._on_copy_dot()

    assert "no graph to copy" in widget._verdict.text()


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------

def test_the_threaded_path_builds_the_same_graph_and_retires(qtbot, project):
    widget = screen_module.PipelineGraphScreen(threaded=True)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.graph_loaded, timeout=15000):
        widget.load_project(project)

    assert len(widget.graph()) == 4
    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=15000)
    assert widget.is_busy() is False
    widget.close()


def test_a_failing_build_reports_inline_and_never_modally(qtbot, monkeypatch):
    widget = screen_module.PipelineGraphScreen(threaded=False)
    qtbot.addWidget(widget)
    monkeypatch.setattr(
        screen_module, "build_graph",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("registry ate it")))

    widget.load_project("/nowhere/at/all")

    assert widget.last_error == "registry ate it"
    assert "registry ate it" in widget._verdict.text()


def test_reload_redraws_after_a_new_artifact_lands(screen, project):
    before = len(screen.graph())
    registry = artifacts.Registry(project=project)
    time.sleep(0.002)
    registry.register(
        module="umap", kind="embedding", role="embedding",
        path=_write(__import__("pathlib").Path(project) / "umap" / "e.csv"),
        settings={"n_neighbors": 15})

    screen._on_reload()

    assert len(screen.graph()) == before + 1
