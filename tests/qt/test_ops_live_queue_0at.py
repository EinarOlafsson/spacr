"""Item 0at: the OPS Live view on Mask Generation's figure queue (alpha).

Every tab shows one picture, graph or table at a time in the figure queue,
with the thumbnails on the left, on an opaque surface. Graphs are square.
Each tab has Sample, Settings and Run preview; Settings holds every setting
the tab's step alone reads; Apply to OPS module copies the tuned values.
Align & stitch chooses between the regions of the well; Objects & phenotype
covers the whole genotyped region. Clicking a gene in the clustering
embedding fills an information tile. The stitched mosaic is drawn again from
the tiles when zoomed. The preview has no font scale slider.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from spacr import ops_live  # noqa: E402

from tests.qt.test_ops_live_preview_971 import _fake_result  # noqa: E402

pytestmark = pytest.mark.qt

STEPS = ("align", "spots", "objects", "merge", "cluster", "voyager3")


class _Runner:
    """A stand-in for :func:`spacr.ops_live._run_step` that records its forms."""

    def __init__(self):
        self.calls, self.forms = [], []

    def __call__(self, form, step, cache, cancelled, steps):
        self.calls.append(step)
        self.forms.append(dict(form))
        result = dict(_fake_result(step), seconds=0.01, test_data=False)
        if step == "align":
            result["regions"] = [[1, 2], [3, 4], [5, 6]]
            result["region"] = int(form.get("ops_live_region") or 0)
        return result


@pytest.fixture
def live(qtbot):
    from spacr.qt.widgets.ops_live_preview import _OpsLivePanel

    made = []

    def build(form=None, propagate=None):
        runner = _Runner()
        panel = _OpsLivePanel(lambda: dict(form or {"genotype_source": "/somewhere"}),
                              run_step=runner)
        qtbot.addWidget(panel)
        panel._timer.setInterval(1)
        if propagate is not None:
            panel.set_propagate_callback(propagate)
        made.append(panel)
        return panel, runner
    yield build
    for panel in made:
        panel.shutdown()


def _run(qtbot, panel, runner, step):
    count = len(runner.calls)
    panel.show_step(step)
    panel.run_preview()
    qtbot.waitUntil(lambda: len(runner.calls) > count, timeout=10000)
    qtbot.waitUntil(lambda: not panel.preview_running() and panel._worker is None,
                    timeout=10000)


@pytest.mark.parametrize("step", STEPS)
def test_every_tab_shows_one_entry_at_a_time_in_the_figure_queue(qtbot, live, step):
    from spacr.qt.widgets.figure_queue import FigureQueue

    panel, runner = live()
    _run(qtbot, panel, runner, step)
    page = panel.page(step)
    queue = page.queue
    assert isinstance(queue, FigureQueue)
    assert queue._is_opaque()
    expected = len(page.views) + len(page.graphs) + 1
    assert queue.count() == expected
    assert queue._list.count() == expected, "every entry has a thumbnail on the left"
    shown = queue._stack.currentWidget()
    assert shown is queue._widget_at(queue._current_index())
    others = [queue._widget_at(i) for i in range(queue.count())
              if i != queue._current_index()]
    assert all(w is not shown for w in others)
    queue.show_index(queue.count() - 1)
    assert queue._stack.currentWidget().isAncestorOf(page.table)


def test_a_graph_is_kept_square(qtbot, live):
    from spacr.qt.widgets.ops_live_preview import _SquareBox

    panel, runner = live()
    _run(qtbot, panel, runner, "spots")
    page = panel.page("spots")
    box = page.graphs[0].parentWidget()
    assert isinstance(box, _SquareBox)
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QResizeEvent

    box.resize(600, 300)
    box.resizeEvent(QResizeEvent(QSize(600, 300), QSize(0, 0)))
    child = page.graphs[0].geometry()
    assert child.width() == child.height() == 300


def test_the_figure_queue_holds_widgets_and_clears_them(qtbot):
    from PySide6.QtWidgets import QLabel

    from spacr.qt.widgets.figure_queue import FigureQueue

    queue = FigureQueue()
    qtbot.addWidget(queue)
    first, second = QLabel("one"), QLabel("two")
    assert queue._add_widget(first, "One") == 0
    assert queue._add_widget(second, "Two") == 1
    assert queue.figure_titles() == ["One", "Two"]
    queue.show_index(0)
    assert queue._stack.currentWidget() is first
    queue.show_next()
    assert queue._stack.currentWidget() is second
    queue._set_figure_tools_visible(False)
    assert queue._fig_settings_btn.isHidden()
    queue.clear()
    assert queue.count() == 0 and queue._widget_at(0) is None


@pytest.mark.parametrize("step", STEPS)
def test_each_tab_has_sample_settings_and_run_preview_left_to_right(qapp, step):
    from spacr.qt.widgets.ops_live_preview import _OpsLivePanel

    panel = _OpsLivePanel(lambda: {"genotype_source": "/x"}, run_step=_Runner())
    try:
        page = panel.page(step)
        row = page.layout().itemAt(1).layout()
        order = [row.itemAt(i).widget() for i in range(row.count())]
        positions = [order.index(w) for w in (page.sample_spin, page.settings_btn,
                                              page.run_btn)]
        assert positions == sorted(positions)
        assert page.run_btn.text() == "Run preview"
    finally:
        panel.shutdown()


def test_settings_holds_every_setting_of_the_tab_and_only_those(qapp):
    from spacr.ops_settings import OPS_DEFAULTS
    from spacr.qt.widgets.ops_live_preview import _OpsLivePanel

    panel = _OpsLivePanel(lambda: {"genotype_source": "/x"}, run_step=_Runner())
    try:
        steps = ops_live._step_map()
        for step in STEPS:
            shown = set(panel.step_settings(step))
            own = {k for k in OPS_DEFAULTS if tuple(steps[k]) == (step,)}
            assert shown == own, step
        align = set(panel.step_settings("align"))
        assert {"ops_raster_overlap", "ops_stitch_edge_workers", "ops_compose_blend",
                "ops_registration_refine_px", "ops_placement_seed_mode",
                "ops_align_model"} <= align
        assert "ops_read_threshold" not in align and "genotype_source" not in align
        spots = set(panel.step_settings("spots"))
        assert {"ops_read_threshold", "ops_cell_rule", "ops_prefix_max_distance",
                "ops_decode_pixel_fallback", "ops_decoy_library_k"} <= spots
        dialog = panel.open_settings("align")
        assert set(dialog.editors) == align
        assert {"Stitching", "Field alignment"} <= set(dialog.groups)
        dialog.close()
    finally:
        panel.shutdown()


def test_apply_to_ops_copies_only_the_tuned_values(qtbot, live):
    copied = []
    panel, runner = live(propagate=copied.append)
    from spacr.qt.widgets.ops_live_preview import _editor_value

    dialog = panel.open_settings("align")
    editor = dialog.editors["ops_registration_refine_px"]
    editor.setValue(editor.value() + 2)
    qtbot.waitUntil(lambda: "align" in runner.calls, timeout=10000)
    panel.panel_setting_changed("spots", "ops_read_threshold", 300.0)
    assert panel.apply_to_ops("align") == 1
    assert copied == [{"ops_registration_refine_px": _editor_value(editor)}]
    assert "ops_read_threshold" in panel._overrides, "another tab keeps its values"
    dialog.close()


def test_the_sample_reaches_the_run_but_never_the_ops_form(qtbot, live):
    copied = []
    panel, runner = live(propagate=copied.append)
    page = panel.page("spots")
    panel.show_step("spots")
    page.sample_spin.setValue(600)
    qtbot.waitUntil(lambda: "spots" in runner.calls, timeout=10000)
    qtbot.waitUntil(lambda: not panel.preview_running(), timeout=10000)
    assert runner.forms[-1]["ops_live_sbs_crop"] == 600
    assert panel.apply_to_ops("spots") == 0 and copied == []


def test_align_lists_the_regions_and_a_choice_reruns_it(qtbot, live):
    panel, runner = live()
    _run(qtbot, panel, runner, "align")
    combo = panel.page("align").region
    assert combo.count() == 3 and combo.isEnabled()
    assert combo.itemText(0).startswith("Region 1: tiles 1, 2")
    panel.choose_region(2)
    qtbot.waitUntil(lambda: runner.calls.count("align") == 2, timeout=10000)
    qtbot.waitUntil(lambda: not panel.preview_running(), timeout=10000)
    assert runner.forms[-1]["ops_live_region"] == 2
    assert combo.currentIndex() == 2
    assert {"spots", "objects", "merge"} <= panel.stale_steps()


def test_regions_are_square_blocks_that_share_no_tile(monkeypatch):
    class Layout:
        def position(self, site):
            return (site % 6, site // 6)

    monkeypatch.setattr("spacr.ops_layout.round_well_layout", lambda n: Layout())
    index = {1: {site: {} for site in range(36)}}
    blocks = ops_live._sample_blocks(index, 1)
    assert len(blocks) >= 3 and all(len(b) == 4 for b in blocks)
    flat = [s for b in blocks for s in b]
    assert len(flat) == len(set(flat))
    assert ops_live._sample_block(index, 1) == blocks[0]
    small = {1: {321: {}, 322: {}, 331: {}, 332: {}}}
    assert ops_live._sample_blocks(small, 1) == [[321, 322, 331, 332]]


def test_objects_cover_every_field_over_the_genotyped_region():
    align = {"tiles": {1: (0.0, 0.0)}, "tile_shape": (400, 400),
             "fields": [
                 {"site": 10, "centre": (100.0, 100.0), "scale": 0.25,
                  "rotation": 0.0, "shape": (800, 800), "files": {"x": "y"}},
                 {"site": 11, "centre": (100.0, 300.0), "scale": 0.25,
                  "rotation": 0.0, "shape": (800, 800), "files": {"x": "y"}},
                 {"site": 12, "centre": (900.0, 900.0), "scale": 0.25,
                  "rotation": 0.0, "shape": (800, 800), "files": {"x": "y"}}]}
    windows = ops_live._region_windows(align, 1, (50, 150, 100))
    sites = sorted(info["site"] for info, _w in windows)
    assert sites == [10, 11], "every field over the crop, none beside it"
    assert all(height == 400 for _i, (_r, _c, height, _w) in windows)
    assert sum(width for _i, (_r, _c, _h, width) in windows) == 400, (
        "the two parts make up the whole genotyped region")
    point = ops_live._ph_to_well(windows[0][0], [[0, 0]])
    back = ops_live._well_to_field(windows[0][0], point)
    assert np.allclose(back, [[0, 0]])


def test_clicking_a_gene_fills_its_information_tile(qtbot, live):
    from spacr.qt.widgets.ops_live_preview import _PointGraph

    panel, runner = live()
    _run(qtbot, panel, runner, "cluster")
    page = panel.page("cluster")
    tile_view = page.point_graph
    assert isinstance(tile_view, _PointGraph)
    assert tile_view.graph.scatter_items
    item = tile_view.graph.scatter_items[0]
    points = item.points()
    item.sigClicked.emit(item, [points[3]], None)
    assert tile_view.shown_index == 3
    text = tile_view.tile.toPlainText()
    assert "G3" in text and "Cluster" in text and "Nearest genes" in text


def test_the_mosaic_is_drawn_again_from_the_tiles_when_zoomed(qtbot):
    from spacr.qt.widgets.live_preview import numpy_to_qpixmap
    from spacr.qt.widgets.ops_live_preview import _MosaicView

    asked = []

    def render(window, step):
        asked.append((window, step))
        rows = int(np.ceil((window[2] - window[0]) * 8 / step))
        cols = int(np.ceil((window[3] - window[1]) * 8 / step))
        return np.full((max(rows, 1), max(cols, 1), 3), 200, np.uint8)

    view = _MosaicView(render, 8)
    qtbot.addWidget(view)
    view.resize(400, 400)
    view.show()
    view.view.set_pixmap(numpy_to_qpixmap(np.zeros((800, 800, 3), np.uint8),
                                          normalise=False))
    view.refine()
    assert view.overlay is None, "at fit the coarse mosaic is enough"
    view.view._apply_zoom(12.0)
    view.refine()
    qtbot.waitUntil(lambda: view.overlay is not None, timeout=10000)
    assert asked and asked[-1][1] < 8
    assert view.overlay_step == asked[-1][1]
    assert view.overlay.scale() == pytest.approx(asked[-1][1] / 8)
    view.view.reset_zoom()
    view.refine()
    assert view.overlay is None
    view.wait()


def test_the_mosaic_window_reads_the_tiles_at_the_asked_step(monkeypatch):
    plane = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    monkeypatch.setattr("spacr.ops_engine._read_plane", lambda source, _u: plane)
    ops_live._PLANES.clear()
    result = {"tiles": {1: (0.0, 0.0), 2: (0.0, 64.0)}, "tile_shape": (64, 64),
              "tile_sources": {1: ("a", None), 2: ("b", None)}}
    full = ops_live._mosaic_window(result, (0, 0, 64, 128), 1)
    assert full.shape == (64, 128)
    assert full[10, 70] == plane[10, 6]
    half = ops_live._mosaic_window(result, (0, 0, 64, 128), 2)
    assert half.shape == (32, 64)


def test_the_font_scale_slider_is_gone(qapp):
    from spacr.qt.widgets.ops_live_preview import _OpsLivePanel

    panel = _OpsLivePanel(lambda: {"genotype_source": "/x"}, run_step=_Runner())
    try:
        assert getattr(panel, "preview_scaler", None) is None
        assert not panel.findChildren(type(panel), "PreviewScaleControl")
        from PySide6.QtWidgets import QWidget

        assert not [w for w in panel.findChildren(QWidget)
                    if w.objectName() in ("PreviewScaleControl", "PreviewScaleSlider")]
    finally:
        panel.shutdown()


def test_voyager3_is_its_own_tab_after_clustering(qapp):
    from spacr.ops_settings import OPS_DEFAULTS
    from spacr.qt.widgets.ops_live_preview import _OpsLivePanel

    panel = _OpsLivePanel(lambda: {"genotype_source": "/x"}, run_step=_Runner())
    try:
        titles = [panel._tabs.tabText(i) for i in range(panel._tabs.count())]
        assert titles[-2:] == ["Clustering", "Voyager3"]
        own = set(panel.step_settings("voyager3"))
        assert own == {k for k in OPS_DEFAULTS if k.startswith("ops_voyager3_")}
        assert not own & set(panel.step_settings("cluster"))
        page = panel.page("voyager3")
        assert page.sample_spin.value() == 3 and page.sample_key == "ops_live_voyager3_clusters"
    finally:
        panel.shutdown()


def test_voyager3_shows_its_clusters_and_carries_its_settings(qtbot, live):
    copied = []
    panel, runner = live(propagate=copied.append)
    _run(qtbot, panel, runner, "voyager3")
    page = panel.page("voyager3")
    assert page.queue.count() == len(page.graphs) + 1
    assert page.table.rowCount() == 2, "the shuffled control is not a row of its own"
    assert page.table.item(0, 2).text() == "ribosome"
    assert "named by" in page.sample.text()
    panel.panel_setting_changed("voyager3", "ops_voyager3_method", "enrichment")
    qtbot.waitUntil(lambda: runner.calls.count("voyager3") == 2, timeout=10000)
    qtbot.waitUntil(lambda: not panel.preview_running(), timeout=10000)
    assert runner.forms[-1]["ops_voyager3_method"] == "enrichment"
    assert panel.apply_to_ops("voyager3") == 1
    assert copied == [{"ops_voyager3_method": "enrichment"}]


def test_the_voyager3_step_names_the_largest_preview_clusters(tmp_path, monkeypatch):
    import json

    import pandas as pd

    seen = {}

    def fake_run(settings, report=None, out_dir=None, provider=None):
        seen.update(settings)
        report("Progress: 1/1, operation_type: Voyager3")
        os.makedirs(out_dir, exist_ok=True)
        table = pd.DataFrame({"cluster": ["1"], "kind": ["real"], "n_genes": [3],
                              "biological_role": ["x"], "confidence_rule": ["high"],
                              "support_fraction": ["1.0"], "supporting_genes": ["A"],
                              "candidates": [""]})
        paths = {"clusters": os.path.join(out_dir, "c.csv"),
                 "candidates": os.path.join(out_dir, "d.csv")}
        table.to_csv(paths["clusters"], index=False)
        pd.DataFrame({"cluster": []}).to_csv(paths["candidates"], index=False)
        provenance = os.path.join(out_dir, "p.json")
        with open(provenance, "w") as handle:
            json.dump({"method": "ai", "provider": "fake"}, handle)
        return {"outputs": paths, "provenance": provenance}

    import os

    monkeypatch.setattr("spacr.voyager3.run_voyager3", fake_run)
    clustered = {"genes": ["A", "B", "C", "D", "E", "F"], "labels": [1, 1, 1, 2, 2, 3]}
    ctx = ops_live._Context(settings={"ops_live_voyager3_clusters": 1}, test_root=None,
                            folder=tmp_path, well="A1", steps={})
    monkeypatch.setattr(ops_live, "_result", lambda c, step, cancelled=None: clustered)
    out = ops_live._run_voyager3(ctx)
    assert seen["ops_voyager3_cluster_ids"] == "1"
    table = pd.read_csv(seen["ops_voyager3_clusters"])
    assert list(table.columns) == ["gene", "cluster"] and len(table) == 6
    assert out["annotated"] == 1 and out["total_clusters"] == 2
    assert str(tmp_path) in str(seen["dst_root"])
