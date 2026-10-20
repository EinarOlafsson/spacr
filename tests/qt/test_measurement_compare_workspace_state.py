"""A saved comparison restores choices without computing or loading data."""
from __future__ import annotations

import json
import threading

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSignalBlocker

import spacr.qt.widgets.measurement_compare_dialog as mcd

pytestmark = pytest.mark.qt


@pytest.fixture
def objects():
    return pd.DataFrame({
        "prc": ["p1_r1_c1"] * 3 + ["p1_r1_c2"] * 3 + ["p1_r1_c3"] * 3,
        "plateID": ["p1"] * 9,
        "rowID": ["r1"] * 9,
        "columnID": ["c1"] * 3 + ["c2"] * 3 + ["c3"] * 3,
        "grna": ["g1", "x", "y", "g1", "x", "y", "q", "x", "y"],
        "cell_area": [float(v + 2) for v in range(9)],
        "cell_perimeter": [float(v + 12) for v in range(9)],
    })


@pytest.fixture
def panel(qtbot, objects):
    widget = mcd.MeasurementComparePanel(objects, {"picked": [0, 3, 6]})
    qtbot.addWidget(widget)
    return widget


def _choose(combo, value):
    index = combo.findData(value)
    assert index >= 0, value
    combo.setCurrentIndex(index)


def _fail(*args, **kwargs):
    pytest.fail("workspace restoration must not refresh, render, join or load")


def test_round_trip_defers_comparison_and_preserves_the_next_graph(
        panel, qtbot, objects, monkeypatch):
    choices = {
        "measurement": "cell_perimeter", "operator": "-",
        "second": "cell_area", "level": "cell", "kind": "bar",
        "spread": "sd", "scope": "wells", "contrast": "within_well",
    }
    blockers = [QSignalBlocker(getattr(panel, key)) for key in choices]
    for key, value in choices.items():
        _choose(getattr(panel, key), value)
    del blockers
    panel.set_selected_guides(["g1"])
    panel.set_selected_wells(["p1_r1_c2"])
    panel.controls.setText("control_a, control_b")
    panel.join_png_list.setChecked(False)
    panel.join_dependent.setChecked(True)
    _choose(panel.only, "picked")
    expected = panel.comparison()
    state = json.loads(json.dumps(panel.workspace_state()))

    fresh = mcd.MeasurementComparePanel(objects, {"picked": [0, 3, 6]})
    qtbot.addWidget(fresh)
    old_comparison, old_canvas = fresh.comparison(), fresh._canvas
    fresh._joined_once = True
    with monkeypatch.context() as guard:
        for name in ("refresh", "_draw", "_draw_and_report", "join_the_tables"):
            guard.setattr(fresh, name, _fail)
        for name in ("build", "with_statistics", "join_measurements", "save"):
            guard.setattr(mcd, name, _fail)
        guard.setattr(pd, "read_csv", _fail)
        assert fresh.apply_workspace_state(state) is True
        assert fresh.workspace_state() == state
    assert fresh.comparison() is old_comparison
    assert fresh._canvas is old_canvas
    assert fresh.second.isEnabled()
    assert not fresh.controls.isEnabled()
    assert not fresh.spread.isHidden()

    builds = []
    actual_build = mcd.build
    monkeypatch.setattr(mcd, "build", lambda *a, **k: (
        builds.append(1), actual_build(*a, **k))[1])
    actual = fresh.refresh()
    assert builds == [1]
    pd.testing.assert_frame_equal(actual.frame, expected.frame)
    assert actual.measurement == expected.measurement
    assert actual.level == "cell"
    assert fresh.only.currentData() == "picked"
    assert fresh._canvas.spec.kind == "bar"
    assert fresh._canvas.spec.spread == "sd"
    assert set(fresh._canvas.spec.frame["group"]) == {"picked"}
    assert set(fresh.scoped_objects()[0]["prc"]) == {"p1_r1_c2"}


@pytest.mark.parametrize("chosen", [None, [], ["p1_r1_c2"]])
@pytest.mark.parametrize("selected", [None, [], ["p1_r1_c2"]])
def test_null_and_empty_well_selections_keep_different_meanings(
        panel, chosen, selected):
    state = {"chosen_wells": chosen, "selected_guides": ["g1"],
             "selected_wells": selected, "scope": "wells"}
    assert panel.apply_workspace_state(state) is True
    saved = panel.workspace_state()
    assert saved["chosen_wells"] == chosen
    assert saved["selected_wells"] == selected
    assert panel.chosen_wells() == chosen
    expected = ["p1_r1_c1", "p1_r1_c2"] if selected is None else selected
    assert panel.selected_wells() == expected
    panel.refresh()
    assert set(panel.scoped_objects()[0]["prc"]) == set(expected)
    if chosen == [] or selected == []:
        assert panel.comparison().frame.empty


def test_explicit_empty_guides_survive_the_next_data_refresh(panel, objects):
    assert panel.apply_workspace_state({
        "scope": "guides", "selected_guides": [], "selected_wells": None,
    }) is True
    panel.set_data(objects.copy(), {"picked": [0, 3, 6]})
    assert panel.workspace_state()["selected_guides"] == []
    assert panel.selected_wells() == []
    assert panel.scoped_objects()[0].empty


def test_pending_class_filter_is_saved_and_resolved_after_refresh(
        panel, objects):
    assert panel.apply_workspace_state({"only": "later", "scope": "all"})
    assert panel.workspace_state()["only"] == "later"
    panel.set_data(objects.copy(), {"later": [0, 3, 6]})
    assert panel.only.currentData() == "later"
    assert set(panel._canvas.spec.frame["group"]) == {"later"}


def test_a_pending_class_filter_waits_through_a_refresh_without_that_class(
        panel, objects):
    assert panel.apply_workspace_state({"only": "later", "scope": "all"})
    panel.refresh()
    assert panel.only.currentData() == ""
    assert panel.workspace_state()["only"] == "later"
    panel.set_data(objects.copy(), {"later": [0, 3, 6]})
    assert panel.only.currentData() == "later"


def test_user_class_choice_supersedes_a_pending_saved_filter(panel):
    panel.apply_workspace_state({"only": "not_present"})
    _choose(panel.only, "picked")
    panel.refresh()
    assert panel.workspace_state()["only"] == "picked"


@pytest.mark.parametrize("state", [None, [], {}, {"unknown": "x"}, {
    "measurement": 3, "operator": "invalid",
    "selected_guides": "g1", "selected_wells": [3],
    "chosen_wells": {"bad": "shape"}, "controls": 42,
    "join_png_list": "false", "join_dependent": 0, "only": {},
}])
def test_invalid_or_unavailable_state_does_not_clear_existing_choices(
        panel, state, monkeypatch):
    before = panel.workspace_state()
    monkeypatch.setattr(panel, "refresh", _fail)
    assert panel.apply_workspace_state(state) is False
    assert panel.workspace_state() == before


def test_partial_restore_updates_control_enablement_without_a_graph(panel):
    assert panel.apply_workspace_state({
        "operator": "+", "contrast": "against_controls", "kind": "box",
    }) is True
    assert panel.second.isEnabled()
    assert panel.controls.isEnabled()
    assert panel.spread.isHidden()
    assert panel.apply_workspace_state({"operator": "", "contrast": ""})
    assert not panel.second.isEnabled()
    assert not panel.controls.isEnabled()


def test_unavailable_joined_measurements_survive_until_columns_arrive(
        panel, objects):
    state = {"measurement": "nucleus_area", "second": "nucleus_perimeter",
             "operator": "/", "scope": "all"}
    assert panel.apply_workspace_state(state)
    panel.refresh()
    assert panel.workspace_state()["measurement"] == "nucleus_area"
    assert panel.workspace_state()["second"] == "nucleus_perimeter"
    wide = objects.assign(nucleus_area=24.0, nucleus_perimeter=6.0)
    panel.set_data(wide, {"picked": [0, 3, 6]})
    assert panel.measurement.currentData() == "nucleus_area"
    assert panel.second.currentData() == "nucleus_perimeter"
    assert panel.comparison().frame["value"].tolist() == [4.0] * 6
    panel.set_data(wide.copy(), {"picked": [0, 3, 6]})
    assert panel.second.currentData() == "nucleus_perimeter"


def test_user_measurement_choices_supersede_pending_joined_columns(panel):
    panel.apply_workspace_state({"measurement": "nucleus_area",
                                 "second": "nucleus_perimeter"})
    _choose(panel.measurement, "cell_perimeter")
    _choose(panel.second, "cell_perimeter")
    state = panel.workspace_state()
    assert state["measurement"] == "cell_perimeter"
    assert state["second"] == "cell_perimeter"


@pytest.mark.parametrize("name,pending", [
    ("measurement", "nucleus_area"),
    ("second", "nucleus_area"),
    ("only", "later"),
])
def test_activating_the_current_fallback_replaces_a_pending_choice(
        panel, objects, name, pending, monkeypatch):
    combo = getattr(panel, name)
    chosen = str(combo.currentData() or "")
    assert panel.apply_workspace_state({name: pending})
    assert panel.workspace_state()[name] == pending
    with monkeypatch.context() as guard:
        guard.setattr(panel, "refresh", _fail)
        guard.setattr(panel, "_draw", _fail)
        combo.activated.emit(combo.currentIndex())
        assert panel.workspace_state()[name] == chosen
    panel.set_data(objects.assign(nucleus_area=42.0), {"later": [0, 3, 6]})
    assert combo.currentData() == chosen
    assert panel.workspace_state()[name] == chosen


def test_restore_abandons_an_old_join_without_rendering_its_late_result(
        panel, objects, qtbot, monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls = []

    def old_join(*args, **kwargs):
        calls.append(kwargs)
        started.set()
        assert release.wait(5)
        return objects.assign(nucleus_area=123.0), ""

    monkeypatch.setattr(mcd, "join_measurements", old_join)
    panel._databases = ("/unused/no-database-opened.db",)
    panel.join_the_tables()
    try:
        qtbot.waitUntil(started.is_set, timeout=5000)
        old_comparison, old_canvas = panel.comparison(), panel._canvas
        monkeypatch.setattr(panel, "refresh", _fail)
        monkeypatch.setattr(panel, "_draw", _fail)
        assert panel.apply_workspace_state({"kind": "bar",
                                             "join_png_list": False})
        assert not panel._joining
        release.set()
        qtbot.waitUntil(lambda: not panel._jobs._jobs, timeout=5000)
        assert len(calls) == 1
        assert panel.comparison() is old_comparison
        assert panel._canvas is old_canvas
        assert "nucleus_area" not in panel._objects.columns
        assert panel.workspace_state()["join_png_list"] is False
    finally:
        release.set()
        panel._jobs.shutdown()
