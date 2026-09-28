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
        for control in (*screen._plan_costs.values(),
                        *screen._plan_limits.values()):
            assert screen._arrayed_planner.isAncestorOf(control)
            assert control.isVisibleTo(screen._arrayed_planner)
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


def test_cost_and_limit_controls_keep_bounded_defaults(planner):
    """The form starts with bounded defaults explained on its row labels.

    :param planner: the alpha-enabled Power screen with a readable pilot.
    """
    form = planner._arrayed_planner.layout()
    for key, default in (("replicate", 20.0), ("well", 1.0), ("field", 0.1)):
        control = planner._plan_costs[key]
        assert control.value() == default
        assert control.minimum() == 0.0
        assert control.maximum() == 1e6
        assert control.decimals() == 3
        label = form.labelForField(control)
        assert label is not None, key
        assert label.property("settingHelpLabel"), key
        assert control.toolTip() == "", key
        hint = label.toolTip()
        assert ("Relative cost per condition: one replicate, one well "
                "within a replicate, or one field within a well.") in hint
        assert "Use common units; zero ignores this cost." in hint
        assert f"Default {default:g}." in hint
    for key, minimum, maximum, default in (
            ("replicates", 2, 24, 12), ("wells", 1, 24, 12),
            ("fields", 1, 50, 25)):
        control = planner._plan_limits[key]
        assert (control.minimum(), control.maximum(), control.value()) == (
            minimum, maximum, default)
        label = form.labelForField(control)
        assert label is not None, key
        assert label.property("settingHelpLabel"), key
        assert control.toolTip() == "", key
        hint = label.toolTip()
        assert ("Largest count considered: biological replicates per "
                "condition, wells per condition per replicate, or fields "
                "per well.") in hint
        assert f"Default {default}." in hint


def test_search_limits_change_reachable_designs_and_empty_summary(planner):
    """Small ceilings can rule out designs that a larger search finds.

    :param planner: the alpha-enabled Power screen with a readable pilot.
    """
    for key, value in zip(("replicates", "wells", "fields"), (2, 1, 1)):
        planner._plan_limits[key].setValue(value)
    assert planner._plan_from_pilot().empty
    assert "2 replicates, 1 wells per condition and 1 fields per well" in (
        planner._plan_summary.text())
    assert planner._arrayed_plan is None
    assert not planner._save_plan.isEnabled()
    previous_count = 0
    for limits in ((4, 3, 5), (8, 4, 8)):
        for key, value in zip(("replicates", "wells", "fields"), limits):
            planner._plan_limits[key].setValue(value)
        designs = planner._plan_from_pilot()
        assert len(designs) > previous_count
        previous_count = len(designs)
        for key, maximum in zip(("replicates", "wells", "fields"), limits):
            assert designs[key].between(2 if key == "replicates" else 1,
                                        maximum).all()
            assert planner._arrayed_plan["design_inputs"]["max_" + key] == maximum
        assert (designs["power"] >= planner._plan_power.value()).all()


def test_cost_weights_change_ranking_without_changing_reachable_designs(planner):
    """Each cost changes the ranking of the same feasible search grid.

    :param planner: the alpha-enabled Power screen with a readable pilot.
    """
    for key, value in zip(("replicates", "wells", "fields"), (12, 4, 8)):
        planner._plan_limits[key].setValue(value)
    candidates = None
    recommendations = set()
    for costs in ((1000.0, 0.0, 0.0), (0.0, 1000.0, 0.0),
                  (0.0, 0.0, 1000.0)):
        for key, value in zip(("replicate", "well", "field"), costs):
            planner._plan_costs[key].setValue(value)
        designs = planner._plan_from_pilot()
        assert not designs.empty
        expected_cost = designs["replicates"] * (
            costs[0] + designs["wells"] * (
                costs[1] + designs["fields"] * costs[2]))
        pd.testing.assert_series_equal(designs["cost"], expected_cost,
                                       check_names=False)
        assert designs["cost"].is_monotonic_increasing
        comparable = (designs.drop(columns="cost")
                      .sort_values(["replicates", "wells", "fields"])
                      .reset_index(drop=True))
        if candidates is None:
            candidates = comparable
        else:
            pd.testing.assert_frame_equal(comparable, candidates)
        recommendations.add(tuple(designs.iloc[0][
            ["replicates", "wells", "fields"]]))
        assert planner._arrayed_plan["design_inputs"]["costs"] == costs
    assert len(recommendations) > 1


def test_save_preserves_the_computed_design_after_form_edits(
        planner, monkeypatch, tmp_path):
    from spacr.qt.screens.power import QFileDialog
    from spacr.sp_stats import _plan_arrayed_design

    for key, value in zip(("replicate", "well", "field"), (120.5, 3.25, 0.05)):
        planner._plan_costs[key].setValue(value)
    for key, value in zip(("replicates", "wells", "fields"), (8, 3, 6)):
        planner._plan_limits[key].setValue(value)
    designs = planner._plan_from_pilot()
    assert not designs.empty
    pilot_path = planner._pilot_path.text()
    summary = planner._plan_summary.text()
    planner._plan_effect.setValue(17.0)
    planner._plan_power.setValue(0.5)
    planner._plan_paired.setChecked(True)
    planner._pilot_path.setText("another-pilot.csv")
    planner._pilot_columns["value"].setEditText("other_measurement")
    for control in planner._plan_costs.values():
        control.setValue(0.0)
    for key, value in zip(("replicates", "wells", "fields"), (9, 5, 7)):
        planner._plan_limits[key].setValue(value)
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
    assert saved["design_inputs"]["costs"] == [120.5, 3.25, 0.05]
    assert saved["design_inputs"]["max_replicates"] == 8
    assert saved["design_inputs"]["max_wells"] == 3
    assert saved["design_inputs"]["max_fields"] == 6
    assert saved["variance_components"]["replicate"] is None
    assert saved["variance_components"]["estimated"]["replicate"] is False
    assert "NaN" not in target.read_text(encoding="utf-8")
    assert saved["designs"] == designs.to_dict(orient="records")
    assert saved["recommendation_index"] == 0
    assert saved["simulation"]["design_index"] == 0
    assert saved["simulation"]["n_sim"] == 500
    assert saved["simulation"]["seed"] == 0
    assert 0 <= saved["simulation"]["power"] <= 1
    assert saved["summary"] == summary
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
