"""The Power screen's arrayed-assay planner is an alpha feature.

``PowerArrayedPlanner`` is registered in ``spacr.settings.ALPHA_FEATURES``,
so it is hidden until Preferences -> "Show alpha features" is on, and a
plan asked for while it is hidden still reads the values on its form.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")


def _pilot_csv(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for plate in range(3):
        for well in range(6):
            for field in range(4):
                offset = rng.normal(0, 0.4)
                for value in 5 + offset + rng.normal(0, 1.5, 15):
                    rows.append({"area": value, "plateID": f"p{plate}",
                                 "prc": f"p{plate}_r1_c{well}",
                                 "fieldID": field})
    path = tmp_path / "pilot.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_the_planner_is_hidden_until_alpha_features_are_shown(
        qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens.power import PowerScreen
    from spacr.settings import ALPHA_FEATURES

    assert ALPHA_FEATURES[585] == {"widgets": ("PowerArrayedPlanner",)}
    for shown in (False, True):
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: s)
        screen = PowerScreen(threaded=False)
        qtbot.addWidget(screen)
        assert screen._arrayed_planner.isHidden() is (not shown)
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: not s)
        preferences._apply_alpha_widgets(screen)
        assert screen._arrayed_planner.isHidden() is shown


def test_a_plan_made_while_hidden_uses_the_form(
        qtbot, qt_theme_applied, monkeypatch, tmp_path):
    from spacr.qt import preferences
    from spacr.qt.screens.power import PowerScreen

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: False)
    screen = PowerScreen(threaded=False)
    qtbot.addWidget(screen)
    assert screen._arrayed_planner.isHidden()
    screen._pilot_path.setText(str(_pilot_csv(tmp_path)))
    screen._refresh_pilot_columns()
    assert screen._pilot_columns["value"].findText("area") >= 0
    screen._pilot_columns["value"].setEditText("area")
    screen._pilot_columns["replicate"].setEditText("plateID")
    screen._plan_effect.setValue(1.0)
    screen._plan_power.setValue(0.9)
    designs = screen._plan_from_pilot()
    assert designs is not None and not designs.empty
    assert (designs["power"] >= 0.9).all()
    assert screen._plan_table.rowCount() == min(10, len(designs))
    assert "Cheapest design" in screen._plan_summary.text()
    screen._pilot_columns["value"].setEditText("nope")
    assert screen._plan_from_pilot() is None
    assert "Could not read the pilot" in screen._plan_summary.text()


@pytest.fixture
def planner(qtbot, qt_theme_applied, monkeypatch, tmp_path):
    from spacr.qt import preferences
    from spacr.qt.screens.power import PowerScreen

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    screen = PowerScreen(threaded=False)
    qtbot.addWidget(screen)
    screen._pilot_path.setText(str(_pilot_csv(tmp_path)))
    screen._pilot_columns["value"].setEditText("area")
    screen._plan_effect.setValue(1.0)
    screen._plan_power.setValue(0.9)
    return screen


def test_save_preserves_the_computed_design_after_form_edits(
        planner, monkeypatch, tmp_path):
    from spacr.qt.screens.power import QFileDialog
    from spacr.sp_stats import _plan_arrayed_design

    designs = planner._plan_from_pilot()
    assert not designs.empty
    pilot_path = planner._pilot_path.text()
    planner._plan_effect.setValue(17.0)
    planner._plan_power.setValue(0.5)
    planner._plan_paired.setChecked(True)
    planner._pilot_path.setText("another-pilot.csv")
    planner._pilot_columns["value"].setEditText("other_measurement")
    target = tmp_path / "saved-plan.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *args: (str(target), ""))
    assert planner._save_plan.isEnabled()
    planner._save_plan.click()
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["schema"] == "spacr-arrayed-plan-v1"
    assert saved["pilot"]["path"] == pilot_path
    assert saved["pilot"]["columns"]["value"] == "area"
    assert saved["design_inputs"]["effect"] == 1.0
    assert saved["design_inputs"]["power"] == 0.9
    assert saved["design_inputs"]["paired"] is False
    assert saved["variance_components"]["replicate"] is None
    assert saved["variance_components"]["estimated"]["replicate"] is False
    assert "NaN" not in target.read_text(encoding="utf-8")
    assert saved["designs"] == designs.to_dict(orient="records")
    assert saved["recommendation_index"] == 0
    assert saved["simulation"]["design_index"] == 0
    assert saved["simulation"]["n_sim"] == 500
    assert saved["simulation"]["seed"] == 0
    assert 0 <= saved["simulation"]["power"] <= 1
    replayed = _plan_arrayed_design(saved["variance_components"],
                                   **saved["design_inputs"])
    pd.testing.assert_frame_equal(replayed, designs)


def test_cancelling_save_keeps_the_result_and_writes_nothing(
        planner, monkeypatch, tmp_path):
    from spacr.qt.screens.power import QFileDialog

    planner._plan_from_pilot()
    before = set(tmp_path.iterdir())
    summary = planner._plan_summary.text()
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *args: ("", ""))
    assert planner._save_arrayed_plan() is False
    assert set(tmp_path.iterdir()) == before
    assert planner._plan_summary.text() == summary
    assert planner._save_plan.isEnabled()


@pytest.mark.parametrize("state", ["initial", "unreadable", "unreachable"])
def test_no_result_cannot_open_a_save_dialog(planner, monkeypatch, state):
    from spacr.qt.screens.power import QFileDialog

    def unexpected_dialog(*args):
        pytest.fail("A planner without a result must not ask for a destination")

    if state != "initial":
        assert not planner._plan_from_pilot().empty
        if state == "unreadable":
            planner._pilot_path.setText("missing-pilot.csv")
            assert planner._plan_from_pilot() is None
        else:
            planner._plan_effect.setValue(0.0)
            assert planner._plan_from_pilot().empty
    monkeypatch.setattr(QFileDialog, "getSaveFileName", unexpected_dialog)
    assert not planner._save_plan.isEnabled()
    assert planner._save_arrayed_plan() is False


def test_replanning_with_an_active_sort_preserves_whole_design_rows(
        planner, qtbot):
    from PySide6.QtCore import Qt

    planner._plan_from_pilot()
    qtbot.wait(1)
    planner._plan_table.sortItems(2, Qt.DescendingOrder)
    designs = planner._plan_from_pilot()
    qtbot.wait(1)
    expected = {
        (str(row.replicates), str(row.wells), str(row.fields),
         f"{row.power:.3f}", f"{row.cost:.4g}")
        for row in designs.head(10).itertuples()}
    actual = {
        tuple(planner._plan_table.item(row, col).text() for col in range(5))
        for row in range(planner._plan_table.rowCount())}
    assert actual == expected


def test_failed_save_preserves_the_destination_and_allows_retry(
        planner, monkeypatch, tmp_path):
    from spacr.qt.screens.power import QFileDialog

    planner._plan_from_pilot()
    target = tmp_path / "existing-directory"
    target.mkdir()
    retained = target / "retained.txt"
    retained.write_text("untouched", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *args: (str(target), ""))
    assert planner._save_arrayed_plan() is False
    assert retained.read_text(encoding="utf-8") == "untouched"
    assert "Export failed" in planner._plan_summary.text()
    assert planner._save_plan.isEnabled()
