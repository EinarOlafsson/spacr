"""Item 1000: the OPS Live preview works like Mask Generation's Live view (alpha).

A tab never raises: a missing input is a notice saying what to run first, and
test data named by genotype_source is the test data. A tab says it runs while
it runs. Every tab has an image-quality slider, zoomable pictures, a Settings
panel with Close and Apply to OPS module, and in laptop mode a Refresh view button that is
the only thing that refreshes. Spots and Objects graphs are pyqtgraph plots.
Align & stitch shows the seam before and after stitching and the phenotype
field on the sequencing crop.
"""
from __future__ import annotations

import json
import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from spacr import ops_live  # noqa: E402

from tests.qt.test_ops_live_preview_971 import _fake_result  # noqa: E402

pytestmark = pytest.mark.qt


class _Runner:
    """A stand-in for :func:`spacr.ops_live._run_step` that records what ran."""

    def __init__(self, hold=None, fail=None):
        self.calls, self.forms = [], []
        self.hold, self.fail = hold, fail or {}

    def __call__(self, form, step, cache, cancelled, steps):
        self.calls.append(step)
        self.forms.append(dict(form))
        if self.hold is not None:
            while not self.hold.wait(0.01):
                if cancelled():
                    raise ops_live._Cancelled()
        if step in self.fail:
            raise self.fail[step]
        result = dict(_fake_result(step), seconds=0.01, test_data=False)
        if step in ("align", "spots", "objects"):
            result = _big_picture(step, result)
        return result


def _big_picture(step, result):
    """Give a step's picture a known size, so quality changes are measurable."""
    rng = np.random.default_rng(3)
    if step == "spots":
        result["dapi"] = rng.random((60, 100)).astype(np.float32)
        result["nuclei"] = np.zeros((60, 100), np.int32)
        result["nuclei"][10:20, 10:20] = 1
    return result


@pytest.fixture
def panel(qtbot, monkeypatch):
    """An OPS Live panel on a stand-in runner, outside laptop mode."""
    from spacr.qt.widgets import ops_live_preview as W

    monkeypatch.setattr(W, "_laptop_mode", lambda: False)
    made = []

    def build(runner, form=None, propagate=None):
        values = dict(form or {"genotype_source": "/somewhere"})
        built = W._OpsLivePanel(lambda: dict(values), run_step=runner)
        qtbot.addWidget(built)
        built._timer.setInterval(1)
        if propagate is not None:
            built.set_propagate_callback(propagate)
        made.append(built)
        return built
    yield build
    for built in made:
        built.shutdown()


def _idle(qtbot, panel):
    qtbot.waitUntil(lambda: not panel.preview_running()
                    and panel._worker is None, timeout=10000)


def _fake_test_data(root):
    """A folder shaped like the downloaded OPS test data."""
    (root / "settings").mkdir(parents=True)
    (root / "settings" / "ops_testdata_settings.json").write_text(json.dumps({}))
    (root / "reference").mkdir()
    (root / "reference" / "PROVENANCE_rev2_addendum.json").write_text("{}")
    sequencing = root / "20200202_6W-LaC024A" / "sequencing"
    sequencing.mkdir(parents=True)
    return sequencing


def test_a_genotype_source_inside_the_test_data_is_the_test_data(tmp_path):
    """B1: merge and clustering then read the test data's stored tables."""
    root = tmp_path / "ops_testdata_funk2022_A1"
    sequencing = _fake_test_data(root)
    assert ops_live._test_root_of(sequencing) == root.resolve()
    assert ops_live._test_root_of(tmp_path) is None
    settings, found = ops_live._effective_settings(
        {"genotype_source": str(sequencing), "ops_merge_threshold": 6.0})
    assert found == root.resolve()
    assert settings["genotype_source"] == str(sequencing)
    assert settings["ops_merge_threshold"] == 6.0, "the form runs as it is"


def test_a_missing_run_is_named_not_raised(tmp_path):
    """B1: without a finished run, the merge input says what is missing."""
    ctx = ops_live._Context(settings={"dst_root": str(tmp_path / "run")},
                            test_root=None, folder=tmp_path / "live", well="A1",
                            steps={})
    (tmp_path / "live").mkdir()
    with pytest.raises(ops_live._NotReady) as raised:
        ops_live._aggregation_inputs(ctx, str(tmp_path / "live" / "m.db"), "P1")
    assert raised.value.reason == "run"
    assert "measurements.db" in raised.value.detail["missing"]
    assert isinstance(raised.value, ValueError)


@pytest.mark.parametrize("step", ("merge", "cluster"))
def test_a_tab_whose_input_is_missing_says_what_to_run_first(qtbot, panel, step):
    """B1: the tab shows a sentence, never an exception's text."""
    missing = ops_live._NotReady("run", path="/data/run", missing="measurements.db")
    runner = _Runner(fail={step: missing})
    live = panel(runner)
    live.show_step(step)
    live.run_preview()
    qtbot.waitUntil(lambda: runner.calls == [step], timeout=10000)
    _idle(qtbot, live)
    page = live.page(step)
    text = page.notice.text()
    assert page.notice.isVisibleTo(page) and "dst_root" in text
    assert "/data/run" in text and "Run the OPS module" in text
    assert "ValueError" not in text and "Error" not in live.preview_status()
    if step == "cluster":
        assert text.startswith("Clustering reads the genes of Merge & aggregation")
    assert not page.is_running()


def test_every_missing_input_has_its_own_sentence():
    from spacr.qt.widgets.ops_live_preview import _not_ready_text

    reasons = {"run": {"path": "p", "missing": "m"}, "rows": {"path": "p", "table": "t"},
               "fields": {}, "phenotype": {}, "genes": {},
               "few_genes": {"n": 2, "floor": 10}}
    texts = {reason: _not_ready_text("merge", reason, detail)
             for reason, detail in reasons.items()}
    assert len(set(texts.values())) == len(texts)
    assert "{" not in "".join(texts.values())
    assert "2 genes" in texts["few_genes"] and "10 cells" in texts["few_genes"]


def test_a_running_tab_shows_it_in_the_tab(qtbot, panel):
    """B2: the spinner, the sentence and the tab title while the pass runs."""
    hold = threading.Event()
    runner = _Runner(hold)
    live = panel(runner)
    live.show_step("merge")
    live.run_preview()
    qtbot.waitUntil(lambda: runner.calls == ["merge"], timeout=10000)
    page = live.page("merge")
    assert page.is_running() and page.spinner.is_busy()
    assert "Merge & aggregation" in page.state.text()
    assert "(running)" in live._tabs.tabText(3)
    hold.set()
    _idle(qtbot, live)
    assert not page.is_running() and not page.spinner.is_busy()
    assert "(running)" not in live._tabs.tabText(3)


def test_opening_a_tab_says_it_starts_before_the_pass_does(qtbot, panel):
    """B2: no 'not run yet' while the tab's pass is already scheduled."""
    runner = _Runner()
    live = panel(runner)
    live.set_live(True)
    qtbot.waitUntil(lambda: runner.calls == ["align"], timeout=10000)
    _idle(qtbot, live)
    live._timer.setInterval(5000)
    live.show_step("objects")
    page = live.page("objects")
    assert page.is_running() and "Objects & phenotype" in page.state.text()
    assert "Not run yet" not in page.state.text()
    live.set_live(False)
    assert not page.is_running()


def test_the_quality_slider_sets_the_rendered_resolution(qtbot, panel):
    """F1: the picture is drawn at 10-100 % of its size, in Mask's zoom view."""
    from spacr.qt.widgets.live_preview import _ZoomView

    runner = _Runner()
    live = panel(runner)
    live.show_step("spots")
    live.run_preview()
    qtbot.waitUntil(lambda: runner.calls == ["spots"], timeout=10000)
    _idle(qtbot, live)
    page = live.page("spots")
    assert isinstance(page.views[0], _ZoomView)
    assert page.shown_size(0) == (100, 60)
    assert (page.quality.minimum(), page.quality.maximum()) == (10, 100)
    page.quality.setValue(50)
    assert page.quality_value.text() == "50 %"
    page._emit_quality()
    assert page.shown_size(0) == (50, 30)
    page.quality.setValue(10)
    page._emit_quality()
    assert page.shown_size(0) == (10, 6)
    assert runner.calls == ["spots"], "a quality change redraws, it does not rerun"


def test_a_picture_zooms_like_mask_generation(qtbot, panel):
    """F1: the wheel zooms the tab's picture."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    runner = _Runner()
    live = panel(runner)
    live.show_step("objects")
    live.run_preview()
    qtbot.waitUntil(lambda: runner.calls == ["objects"], timeout=10000)
    _idle(qtbot, live)
    view = live.page("objects").views[0]
    before = view.scale_factor()
    centre = QPointF(view.viewport().rect().center())
    view.wheelEvent(QWheelEvent(centre, centre, QPoint(0, 0), QPoint(0, 120),
                                Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False))
    assert view.scale_factor() > before


def test_the_settings_panel_reruns_its_tab_and_applies_to_ops(qtbot, panel):
    """0at: a change runs the tab with it; Apply to OPS module copies what was tuned."""
    copied = []
    runner = _Runner()
    live = panel(runner, form={"genotype_source": "/somewhere",
                               "ops_merge_threshold": 4.0}, propagate=copied.append)
    live.show_step("merge")
    dialog = live.open_settings("merge")
    assert dialog.close_btn is not None
    assert dialog.apply_btn.text() == "Apply to OPS module"
    dialog.apply_btn.click()
    assert copied == [] and "nothing" in dialog.note.text().lower()
    editor = dialog.editors["ops_merge_threshold"]
    editor.setValue(6.5)
    qtbot.waitUntil(lambda: runner.calls == ["merge"], timeout=10000)
    _idle(qtbot, live)
    assert runner.forms[-1]["ops_merge_threshold"] == 6.5
    assert copied == [], "nothing reaches the form before Apply to OPS module"
    dialog.apply_btn.click()
    assert copied == [{"ops_merge_threshold": 6.5}]
    dialog.close()


def test_a_form_edit_replaces_the_panel_value(qtbot, panel):
    runner = _Runner()
    live = panel(runner, form={"genotype_source": "/x", "ops_cluster_knn": 12})
    live.panel_setting_changed("cluster", "ops_cluster_knn", 6)
    assert live.settings_now()["ops_cluster_knn"] == 6
    live.setting_committed("ops_cluster_knn")
    assert live.settings_now()["ops_cluster_knn"] == 12


def test_laptop_mode_runs_once_on_live_then_waits_for_refresh(qtbot, panel, monkeypatch):
    """F1: Live on runs the tab on view once; later changes wait for Refresh view."""
    from spacr.qt.widgets import ops_live_preview as W

    runner = _Runner()
    live = panel(runner)
    monkeypatch.setattr(W, "_laptop_mode", lambda: True)
    live._sync_laptop()
    page = live.page("spots")
    assert not page.refresh_btn.isHidden() and page.refresh_btn.text() == "Refresh view"
    live.show_step("spots")
    live.set_live(True)
    qtbot.waitUntil(lambda: runner.calls == ["spots"], timeout=10000)
    _idle(qtbot, live)
    live.setting_committed("ops_read_threshold")
    live.panel_setting_changed("spots", "ops_read_threshold", 200.0)
    live.show_step("objects")
    live.show_step("spots")
    qtbot.wait(40)
    assert runner.calls == ["spots"], "after the first pass nothing runs by itself"
    assert "Refresh view" in page.state.text()
    page.refresh_btn.click()
    qtbot.waitUntil(lambda: runner.calls == ["spots", "spots"], timeout=10000)
    _idle(qtbot, live)
    assert runner.forms[-1]["ops_read_threshold"] == 200.0
    size = page.shown_size(0)
    page.quality.setValue(50)
    page._emit_quality()
    assert page.shown_size(0) == size and page.quality_pending()
    page.refresh_btn.click()
    assert page.shown_size(0) == (size[0] // 2, size[1] // 2)
    assert runner.calls == ["spots", "spots"], "Refresh of an up-to-date tab only redraws"
    dialog = live.open_settings("spots")
    assert not dialog.refresh_btn.isHidden()


@pytest.mark.parametrize("theme", ("light", "dark"))
def test_the_graphs_follow_the_theme(qtbot, panel, monkeypatch, theme):
    """The plots have no background of their own and use the theme's ink."""
    from PySide6.QtCore import Qt

    from spacr.qt import preferences
    from spacr.qt.theme import palette_for

    monkeypatch.setattr(preferences, "resolve_effective_theme", lambda: theme)
    colours = palette_for(theme)
    runner = _Runner()
    live = panel(runner)
    live.show_step("spots")
    live.run_preview()
    qtbot.waitUntil(lambda: runner.calls == ["spots"], timeout=10000)
    _idle(qtbot, live)
    graph = live.page("spots").graphs[0]
    assert graph.ink == {"fg": colours["fg"], "muted": colours["fg_muted"],
                         "border": colours["border"]}
    assert graph.plot.backgroundBrush().style() == Qt.NoBrush, "no plot background"


@pytest.mark.parametrize("step", ("spots", "objects", "merge", "cluster"))
def test_the_graphs_are_qt_graphs(qtbot, panel, step):
    """F3/F4/F5: the graphs are pyqtgraph plots that hold the data."""
    pg = pytest.importorskip("pyqtgraph")
    runner = _Runner()
    live = panel(runner)
    live.show_step(step)
    live.run_preview()
    qtbot.waitUntil(lambda: runner.calls == [step], timeout=10000)
    _idle(qtbot, live)
    page = live.page(step)
    shown = list(page.graphs)
    assert shown and all(isinstance(g.plot, pg.PlotWidget) for g in shown
                         if g.spec["kind"] != "message")
    kinds = {type(item).__name__ for g in shown for item in g.items()}
    assert kinds & {"BarGraphItem", "ScatterPlotItem"}
    for graph in shown:
        if graph.spec["kind"] in ("bars", "grouped"):
            low, high = graph.plot.getViewBox().viewRange()[1]
            assert low == pytest.approx(0.0) and high > 0, "bar axes start at 0"


def test_new_settings_redraw_the_graphs_with_the_image(qtbot, panel):
    """F3: a settings change runs the tab and redraws its picture and graphs."""
    runner = _Runner()
    live = panel(runner)
    live.show_step("spots")
    live.run_preview()
    qtbot.waitUntil(lambda: runner.calls == ["spots"], timeout=10000)
    _idle(qtbot, live)
    page = live.page("spots")
    first_plot = page.graphs[0].plot
    live.open_settings("spots").editors["ops_read_threshold"].setValue(250.0)
    qtbot.waitUntil(lambda: runner.calls == ["spots", "spots"], timeout=10000)
    _idle(qtbot, live)
    assert page.graphs[0].plot is not first_plot
    assert page.shown_size(0) == (100, 60)


def test_the_grid_search_inputs_live_in_the_spots_settings_panel(qtbot, panel):
    """F3: the grid inputs leave the tab for its Settings panel and come back."""
    from spacr.qt.widgets.ops_grid_search_panel import _attach_grid_search

    class Host:
        class _settings_model:
            @staticmethod
            def collect():
                return {"ops_grid_n_tiles": 4}

        @staticmethod
        def apply_settings_dict(values):
            return len(values)

    live = panel(_Runner())
    grid = _attach_grid_search(live, Host())
    page = live.page("spots")
    assert page.layout().indexOf(grid) >= 0
    assert grid.inputs.isHidden(), "the inputs are not inline on the tab"
    dialog = live.open_settings("spots")
    assert dialog.isAncestorOf(grid.inputs) and not grid.inputs.isHidden()
    assert grid.n_tiles.value() == 4
    dialog.close()
    assert grid.inputs.parent() is grid and grid.inputs.isHidden()


def test_every_tab_has_the_same_tools(qapp):
    from spacr.qt.widgets.ops_live_preview import _STEP_ORDER, _OpsLivePanel

    live = _OpsLivePanel(lambda: {"genotype_source": "/x"}, run_step=_Runner())
    try:
        for step in _STEP_ORDER:
            page = live.page(step)
            assert page.settings_btn.text() == "Settings…"
            assert page.refresh_btn.text() == "Refresh view"
            assert page.quality.accessibleName() == "Image quality"
            assert live.step_settings(step), step
    finally:
        live.shutdown()


def test_the_seam_is_drawn_on_the_raster_grid_and_once_stitched():
    """F2: the second tile is laid on the first at the grid and at the stitch."""
    rng = np.random.default_rng(0)
    world = rng.random((70, 120)).astype(np.float32)
    first = world[0:50, 0:60]
    second = world[3:53, 45:105]
    tiles = {1: (0.0, 0.0), 2: (3.0, 45.0)}
    assert ops_live._neighbour_pair(tiles, (50, 60)) == (1, 2, 1)
    window = (10, 40, 40, 60)
    after = ops_live._shifted(second, window, (3.0, 45.0))
    np.testing.assert_array_equal(after[:, 5:], first[10:40, 45:60])
    assert not after[:, :5].any(), "left of the second tile there is nothing"
    before = ops_live._shifted(second, window, (0.0, 50.0))
    assert not np.allclose(before[:, 10:], first[10:40, 50:60])
    assert ops_live._neighbour_pair({1: (0.0, 0.0)}, (50, 60)) is None


def test_the_placed_field_lands_on_the_sequencing_crop(monkeypatch):
    """F2: through its placement, the field's pixels are read back in place."""
    from spacr import ops_engine

    rng = np.random.default_rng(1)
    dna = rng.random((32, 32)).astype(np.float32)
    monkeypatch.setattr(ops_engine, "_ph_channels", lambda settings: ["DAPI"])
    monkeypatch.setattr(ops_live, "_ph_stack",
                        lambda ctx, info, channels: (dna[..., None], ("k",)))
    tiles = {5: (100.0, 200.0)}
    crop = (4, 6, 32)
    centre = (100 + 4 + 16, 200 + 6 + 16)
    field = {"site": 9, "files": {"DAPI": "x"}, "scale": 1.0, "rotation": 0.0,
             "centre": centre, "shape": (32, 32)}
    ctx = ops_live._Context(settings={}, test_root=None, folder=None, well="A1",
                            steps={})
    placed = ops_live._placement(ctx, field, tiles, 5, crop,
                                 np.zeros((32, 32), np.float32))
    assert placed["site"] == 9 and placed["scale"] == 1.0
    np.testing.assert_allclose(placed["phenotype"][:-1, :-1], dna[:-1, :-1],
                               atol=1e-5)
    assert ops_live._placement(ctx, None, tiles, 5, crop, dna) is None


def test_the_align_tab_draws_the_seam_and_the_placement(qapp):
    from spacr.qt.widgets.ops_live_preview import _DRAW, _theme_colours

    rng = np.random.default_rng(2)
    result = dict(_fake_result("align"))
    result["seam"] = {"sites": (1, 2), "axis": 1, "overlap": 8,
                      "nominal": (0.0, 40.0), "stitched": (1.0, 38.0),
                      "window": (0, 48, 30, 48),
                      "before": (rng.random((48, 18)), rng.random((48, 18))),
                      "after": (rng.random((48, 18)), rng.random((48, 18)))}
    result["placement"] = {"site": 7, "scale": 0.25, "rotation": 0.5,
                           "centre": (24.0, 40.0), "sbs": rng.random((32, 32)),
                           "phenotype": rng.random((32, 32))}
    drawn = _DRAW["align"](result, _theme_colours())
    titles = [title for title, _picture in drawn["images"]]
    assert len(titles) == 6
    assert titles[1].startswith("Seam of tiles 1 and 2 on the raster grid (+0, +40 px)")
    assert titles[2] == "The same seam after stitching (+1, +38 px)"
    assert titles[3].startswith("Phenotype field 7 (blue) on the sequencing crop")
    assert any(row[0] == "Seam 1|2" and row[1:3] == ["+1.0", "-2.0"]
               for row in drawn["rows"])
    assert drawn["graphs"][0]["kind"] == "grouped"
