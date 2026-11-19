"""Item 971: the OPS Live preview, one tab per pipeline step (alpha).

The OPS screen gets a Live switch behind the alpha gate; each of its five
tabs draws its step; a committed setting runs only its own tab and marks the
tabs that read from it stale; Cancel drops the pass in flight; nothing runs
while Live is off; and every OPS setting finds its tab from its registry
category.
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings  # noqa: E402

from spacr import ops_live  # noqa: E402

pytestmark = pytest.mark.qt

STEPS = ("align", "spots", "objects", "merge", "cluster")


def _labels(size=48, n=3):
    """A label mask with ``n`` square objects."""
    mask = np.zeros((size, size), np.int32)
    for i in range(n):
        mask[4 + 14 * i:12 + 14 * i, 6:16] = i + 1
    return mask


def _fake_result(step: str):
    """A tiny result of the shape each step's runner returns."""
    rng = np.random.default_rng(0)
    image = rng.random((48, 48)).astype(np.float32)
    align = {
        "tiles": {1: (0.0, 0.0), 2: (0.0, 40.0)}, "tile_shape": (48, 48),
        "origin": (0.0, 0.0), "mosaic": rng.random((6, 11)), "step": 8,
        "fields": [{"site": 7, "centre": (24.0, 40.0), "scale": 0.25,
                    "rotation": 0.0, "shape": (96, 96), "files": {}}],
        "sample_field": {"site": 7}, "crop_site": 1, "crop": (4, 4, 32),
        "cycle_shifts": {1: (0, 0), 2: (1, -2)}, "kept": [1, 2], "refused": [],
        "missing": [3], "last_cycle": 2, "before": (image, image),
        "after": (image, image),
    }
    if step == "align":
        return align
    if step == "spots":
        return {"site": 1, "crop": (4, 4, 32), "dapi": image, "nuclei": _labels(),
                "peaks": np.array([[10.0, 10.0], [20.0, 30.0]]),
                "strength": np.array([400.0, 500.0]), "reads": ["GTAC", "GTAA"],
                "read_quality": np.array([0.2, 0.05]),
                "cells": [{"object_id": 1, "guide": "GTACG", "barcode": "GTACG",
                           "reads": 3, "agreeing": 3, "fraction": 1.0,
                           "quality": 0.2, "decoy": False}],
                "n_nuclei": 3, "n_cycles": 2, "rule": "rule (a)",
                "threshold": 315.0, "library": 10}
    if step == "objects":
        table = pd.DataFrame({"site": 7, "label": [1, 2, 3],
                              "cell_area": [80.0, 90.0, 70.0],
                              "nucleus_dapi_mean_intensity": [1.0, 2.0, 3.0]})
        return {"site": 7, "window": (0, 0, 48),
                "stack": np.stack([image, image], axis=-1),
                "channels": ["DAPI", "AF750"], "nuclei": _labels(),
                "cells": _labels(), "layers": {"pathogen": _labels(n=1)},
                "table": table, "cell_channel": "AF750"}
    if step == "merge":
        cells = pd.DataFrame({"site": 7, "label": [1, 2], "y": [10.0, 20.0],
                              "x": [10.0, 30.0]})
        objects = pd.DataFrame({"object_id": [5, 6], "centroid_y": [11.0, 21.0],
                                "centroid_x": [10.0, 29.0]})
        merged = pd.DataFrame({"site": [7], "label": [1], "object_id": [5]})
        genes = pd.DataFrame({"gene": ["A", "B", "nontargeting"],
                              "n_cells": [3, 1, 4], "n_guides": [2, 1, 3],
                              "f1": [0.1, -0.2, 0.0]})
        return {"report": {"cells": 2, "merged": 1}, "used": {"kind": "test", "path": "ref"},
                "genes": genes, "merged": merged, "objects": objects,
                "cells": cells, "per_field": [{"site": 7, "cells": 2, "matched": 1}],
                "align": align}
    return {"genes": [f"G{i}" for i in range(8)],
            "components": rng.normal(size=(8, 3)), "n_components": 3,
            "n_features": 5, "labels": np.array([0, 0, 1, 1, 2, 2, 0, 1]),
            "kind": "knn", "k": 3, "resolution": 1.0, "seed": 42, "floor": 1,
            "note": "", "total_genes": 9}


def _colours():
    from spacr.qt.widgets.ops_live_preview import _theme_colours

    return _theme_colours()


@pytest.mark.parametrize("step", STEPS)
def test_each_tab_draws_its_step_on_a_tiny_fixture(qapp, step):
    """Every tab's picture has the same geometry and its table its header."""
    from spacr.qt.widgets.ops_live_preview import _DRAW

    picture, header, rows, sample = _DRAW[step](_fake_result(step), _colours(),
                                                (540, 150))
    assert picture.shape == (150, 540, 3) and picture.dtype == np.uint8
    assert picture.std() > 0, "the picture is not blank"
    assert header and all(len(row) <= len(header) for row in rows)
    assert isinstance(sample, str) and sample


def test_the_clustering_tab_says_why_when_the_extra_is_missing(qapp):
    from spacr.qt.widgets.ops_live_preview import _DRAW

    result = dict(_fake_result("cluster"), labels=None,
                  note="install the clustering extra")
    picture, header, rows, _sample = _DRAW["cluster"](result, _colours(), (540, 150))
    assert picture.shape == (150, 540, 3) and rows == []


class _Recorder:
    """A stand-in for :func:`spacr.ops_live._run_step` that records its calls."""

    def __init__(self, hold: threading.Event = None):
        self.calls = []
        self.hold = hold

    def __call__(self, form, step, cache, cancelled, steps):
        self.calls.append(step)
        hold = self.hold
        if hold is not None:
            while not hold.wait(0.01):
                if cancelled():
                    raise ops_live._Cancelled()
        if cancelled():
            raise ops_live._Cancelled()
        return dict(_fake_result(step), seconds=0.01, test_data=False)


@pytest.fixture
def make_panel(qtbot):
    from spacr.qt.widgets.ops_live_preview import _OpsLivePanel

    def build(recorder):
        panel = _OpsLivePanel(lambda: {"genotype_source": "/somewhere"},
                              run_step=recorder)
        qtbot.addWidget(panel)
        panel._timer.setInterval(1)
        return panel
    return build


def _idle(qtbot, panel):
    qtbot.waitUntil(lambda: not panel.preview_running()
                    and panel._worker is None, timeout=10000)


def test_nothing_runs_while_live_is_off(qtbot, make_panel):
    recorder = _Recorder()
    panel = make_panel(recorder)
    panel.apply_settings({})
    panel.setting_committed("ops_read_threshold")
    panel.setting_committed(None)
    qtbot.wait(50)
    assert recorder.calls == [] and panel.runs == []
    assert panel.stale_steps() == set(STEPS)


def test_switching_live_on_runs_the_tab_on_view(qtbot, make_panel):
    recorder = _Recorder()
    panel = make_panel(recorder)
    panel.set_live(True)
    qtbot.waitUntil(lambda: recorder.calls == ["align"], timeout=10000)
    _idle(qtbot, panel)
    assert not panel.page("align").picture.pixmap().isNull()
    assert "align" not in panel.stale_steps()


@pytest.mark.parametrize("step", STEPS)
def test_every_tab_renders_through_the_panel(qtbot, make_panel, step):
    recorder = _Recorder()
    panel = make_panel(recorder)
    panel.show_step(step)
    panel.run_preview()
    qtbot.waitUntil(lambda: step not in panel.stale_steps(), timeout=10000)
    _idle(qtbot, panel)
    page = panel.page(step)
    assert not page.picture.pixmap().isNull()
    assert page.table.columnCount() > 0 and page.sample.text()


def test_a_setting_runs_only_its_own_tab(qtbot, make_panel):
    """A merge setting brings Merge forward and runs it alone.

    Clustering reads from merge, so it is stale; the tabs before merge are
    not touched.
    """
    recorder = _Recorder()
    panel = make_panel(recorder)
    panel.set_live(True)
    qtbot.waitUntil(lambda: recorder.calls == ["align"], timeout=10000)
    _idle(qtbot, panel)
    recorder.calls.clear()
    assert panel._step_map()["ops_merge_threshold"] == ("merge",)
    panel.setting_committed("ops_merge_threshold")
    assert panel.current_step() == "merge"
    qtbot.waitUntil(lambda: recorder.calls == ["merge"], timeout=10000)
    _idle(qtbot, panel)
    qtbot.wait(30)
    assert recorder.calls == ["merge"]
    assert "cluster" in panel.stale_steps()
    assert "align" not in panel.stale_steps() and "merge" not in panel.stale_steps()


def test_an_input_setting_leaves_the_tab_on_view_in_front(qtbot, make_panel):
    recorder = _Recorder()
    panel = make_panel(recorder)
    panel.show_step("objects")
    panel.set_live(True)
    qtbot.waitUntil(lambda: recorder.calls == ["objects"], timeout=10000)
    _idle(qtbot, panel)
    recorder.calls.clear()
    panel.setting_committed("genotype_source")
    assert panel.current_step() == "objects"
    qtbot.waitUntil(lambda: recorder.calls == ["objects"], timeout=10000)
    _idle(qtbot, panel)
    assert panel.stale_steps() == set(STEPS) - {"objects"}


def test_cancel_drops_the_pass_in_flight(qtbot, make_panel):
    hold = threading.Event()
    recorder = _Recorder(hold)
    panel = make_panel(recorder)
    panel.run_preview()
    qtbot.waitUntil(lambda: recorder.calls == ["align"], timeout=10000)
    assert panel.preview_running()
    assert panel.cancel_preview() is True
    _idle(qtbot, panel)
    hold.set()
    qtbot.wait(30)
    page = panel.page("align")
    assert page.picture.pixmap().isNull(), "a cancelled pass draws nothing"
    assert "align" in panel.stale_steps()


def test_a_newer_request_supersedes_the_pass_in_flight(qtbot, make_panel):
    hold = threading.Event()
    recorder = _Recorder(hold)
    panel = make_panel(recorder)
    panel.run_preview()
    qtbot.waitUntil(lambda: recorder.calls == ["align"], timeout=10000)
    recorder.hold = None
    panel.show_step("spots")
    panel.run_preview()
    qtbot.waitUntil(lambda: recorder.calls == ["align", "spots"], timeout=10000)
    _idle(qtbot, panel)
    assert panel.page("align").picture.pixmap().isNull()
    assert not panel.page("spots").picture.pixmap().isNull()


def test_every_ops_setting_finds_its_tab_in_the_registry():
    from spacr.ops_settings import OPS_DEFAULTS

    steps = ops_live._step_map()
    assert set(steps) == set(OPS_DEFAULTS)
    assert all(set(found) <= set(STEPS) and found for found in steps.values())
    expected = {"ops_raster_overlap": ("align",), "ops_read_threshold": ("spots",),
                "ops_sbs_nucleus_model": ("spots",), "ops_ph_downsample": ("objects",),
                "ops_merge_threshold": ("merge",), "ops_cluster_knn": ("cluster",),
                "genotype_source": STEPS, "ops_gpu": STEPS}
    for key, found in expected.items():
        assert steps[key] == found, key


def test_a_subcategory_places_a_setting_by_its_name():
    """Item 969's subcategories reach the tabs without a key list."""
    assert ops_live._steps_of("ops_new_knob", "OPS phenotype: clustering") == ("cluster",)
    assert ops_live._steps_of("ops_new_knob", "OPS decoding: spots") == ("spots",)
    assert ops_live._steps_of("ops_new_knob", "OPS input") == STEPS
    assert ops_live._steps_of("ops_new_knob", "") == STEPS


def test_a_step_change_invalidates_what_reads_from_it():
    assert set(ops_live._downstream("align")) == {"spots", "objects", "merge", "cluster"}
    assert ops_live._downstream("merge") == ("cluster",)
    assert ops_live._downstream("cluster") == ()
    steps = ops_live._step_map()
    base = {"ops_merge_threshold": 4.0, "ops_cluster_knn": 12,
            "ops_raster_overlap": 213}
    moved = dict(base, ops_cluster_knn=6)
    assert ops_live._signature(base, "merge", steps) == ops_live._signature(moved, "merge", steps)
    assert ops_live._signature(base, "cluster", steps) != ops_live._signature(moved, "cluster", steps)
    moved = dict(base, ops_raster_overlap=200)
    assert all(ops_live._signature(base, s, steps) != ops_live._signature(moved, s, steps)
               for s in STEPS)


def test_the_test_data_is_the_base_and_form_edits_win(tmp_path, monkeypatch):
    import json

    root = tmp_path / "ops_testdata_funk2022_A1"
    (root / "settings").mkdir(parents=True)
    (root / "settings" / "ops_testdata_settings.json").write_text(json.dumps({
        "genotype_source": "plate/sequencing", "ops_ph_sites": "1255,1264",
        "ops_merge_threshold": 4.0, "n_workers": 2}))
    monkeypatch.setattr(ops_live, "_test_data_root", lambda: root)
    from spacr.ops_settings import OPS_DEFAULTS

    form = dict(OPS_DEFAULTS, ops_merge_threshold=6.0, ops_ph_sites="")
    settings, found = ops_live._effective_settings(form)
    assert found == root
    assert settings["genotype_source"] == str(root / "plate/sequencing")
    assert settings["ops_ph_sites"] == "1255,1264", "an empty form value keeps the test data's"
    assert settings["ops_merge_threshold"] == 6.0, "an edit on the form is what runs"
    assert settings["ops_design"].endswith("library/pool10_design.csv")


def test_the_ops_screen_has_the_live_switch_behind_the_alpha_gate(qtbot, tmp_path,
                                                                  monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.ops_live_preview import _OpsLivePanel

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    preferences.set_show_alpha(True)
    preferences._set_show_alpha_features(True)
    screen = AppScreen(app_key="ops")
    qtbot.addWidget(screen)
    try:
        switch = screen._preview_switch
        assert switch.objectName() == "OpsLiveSwitch"
        assert isinstance(screen._ops_live_preview, _OpsLivePanel)
        assert screen._preview_card_attr == "_ops_live_preview_card"
        assert not switch.isHidden()
        assert screen._ops_live_preview_card.isHidden(), "the card opens with Live"
        preferences._set_show_alpha_features(False)
        preferences._apply_alpha_widgets(screen)
        assert switch.isHidden()
        screen._on_preview_switch(False)
        assert switch.isHidden(), "moving the switch back does not show it"
        preferences._set_show_alpha_features(True)
        preferences._apply_alpha_widgets(screen)
        assert not switch.isHidden()
        panel = screen._ops_live_preview
        calls = []
        monkeypatch.setattr(panel, "setting_committed", calls.append)
        screen._observe_settings_commit("ops_read_threshold")
        assert calls == ["ops_read_threshold"]
        live = []
        monkeypatch.setattr(panel, "set_live", live.append)
        screen._on_preview_switch(True)
        assert live == [True] and not screen._ops_live_preview_card.isHidden()
        screen._on_preview_switch(False)
        assert live == [True, False]
    finally:
        screen._ops_live_preview.shutdown()
        screen.close()


def test_the_call_table_names_what_each_column_counts(qtbot, make_panel):
    """Reads (all) counts every read; Vote share is over library-matched reads.

    A nucleus with 8 reads, 2 of them for the guide and the other 6 matching
    no guide, has a vote share of 1.00; the headers say so and carry a
    tooltip each.
    """
    from spacr.qt.widgets.ops_live_preview import _DRAW

    result = _fake_result("spots")
    result["cells"] = [{"object_id": 74, "guide": "CGTAGTGGGAC",
                        "barcode": "CGTAGTGGGAA", "reads": 8, "agreeing": 2,
                        "fraction": 1.0, "quality": 0.084, "decoy": False,
                        "pixels": False}]
    _picture, header, rows, _sample = _DRAW["spots"](result, _colours(), (540, 150))
    labels = [h[0] for h in header]
    assert labels[3:6] == ["Reads (all)", "Reads for guide", "Vote share"]
    assert all(isinstance(h, tuple) and h[1] for h in header)
    assert rows[0][3:6] == ["8", "2", "1.00"]
    panel = make_panel(_Recorder())
    panel.page("spots").show_result({"picture": _picture, "header": header,
                                     "rows": rows, "sample": "s", "seconds": 0})
    table = panel.page("spots").table
    assert table.horizontalHeaderItem(5).text() == "Vote share"
    assert "library guide" in table.horizontalHeaderItem(5).toolTip()
