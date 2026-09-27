"""The Power screen's arrayed-assay planner is an alpha feature.

``PowerArrayedPlanner`` is registered in ``spacr.settings.ALPHA_FEATURES``,
so it is hidden until Preferences -> "Show alpha features" is on, and a
plan asked for while it is hidden still reads the values on its form.
"""
from __future__ import annotations

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
