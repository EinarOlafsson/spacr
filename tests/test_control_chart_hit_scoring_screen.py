"""The hit-scoring option of the Control Charts screen (item 570).

The option scores every well against the negative control the form already
names, lists the ranked hits, and exports the report. It runs on its own job
runner, so a campaign too short for a control chart can still be scored.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

pytestmark = pytest.mark.qt

from spacr.qt.screens.control_chart import (             # noqa: E402
    HIT_ALPHA_WIDGETS, ControlChartScreen)

PLANTED = {("P1", 4, 7), ("P2", 7, 9)}


def _plates() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for plate in ("P1", "P2"):
        for r in range(1, 9):
            for c in range(1, 13):
                value = 50.0 + rng.normal(0, 1.0)
                kind = "neg" if c == 1 else ("pos" if c == 12 else "sample")
                if kind == "pos":
                    value += 20.0
                if (plate, r, c) in PLANTED:
                    value += 15.0
                rows.append({"plateID": plate, "rowID": f"r{r}",
                             "columnID": f"c{c}", "well_type": kind,
                             "gene": f"g{c}", "signal": value})
    return pd.DataFrame(rows)


@pytest.fixture
def screen(qtbot):
    made = ControlChartScreen(threaded=False)
    qtbot.addWidget(made)
    return made


def _turn_on(screen):
    screen.set_frame(_plates())
    screen._value.setCurrentText("signal")
    screen._control_column.setCurrentText("well_type")
    screen._on_control_column("well_type")
    screen._negative.setCurrentText("neg")
    screen._positive.setCurrentText("pos")
    screen._hit_score.setChecked(True)


def test_every_alpha_widget_of_the_option_exists_by_name(screen):
    """The alpha gate finds these by object name; a renamed widget would
    silently escape it."""
    from PySide6.QtCore import QObject

    for name in HIT_ALPHA_WIDGETS:
        assert screen.findChildren(QObject, name), name


def test_the_option_is_registered_alpha_and_follows_the_switch(
        screen, tmp_path, monkeypatch):
    """Item 569's gate: hidden while Show alpha features is off, shown once
    it is on, and every name the screen exports is in the registry."""
    from PySide6.QtCore import QObject, QSettings

    from spacr.qt import preferences
    from spacr.settings import ALPHA_FEATURES, _alpha_names

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    assert set(HIT_ALPHA_WIDGETS) == set(ALPHA_FEATURES[570]["widgets"])
    assert set(HIT_ALPHA_WIDGETS) <= _alpha_names("widgets")
    preferences._apply_alpha_widgets(screen)
    for name in HIT_ALPHA_WIDGETS:
        assert screen.findChild(QObject, name).isHidden(), name
    preferences._set_show_alpha_features(True)
    preferences._apply_alpha_widgets(screen)
    for name in HIT_ALPHA_WIDGETS:
        assert not screen.findChild(QObject, name).isHidden(), name


def test_off_by_default_and_scores_nothing(screen):
    screen.set_frame(_plates())
    assert not screen._hit_score.isChecked()
    assert screen.hit_result is None
    assert screen.hit_table.rowCount() == 0


def test_scoring_lists_the_planted_hits_first(screen):
    _turn_on(screen)
    result = screen.hit_result
    assert result is not None
    hits = result.hits()
    called = {(p, int(r), int(c)) for p, r, c in zip(
        hits["plateID"],
        result.wells.set_index("prc").loc[hits["prc"], "row_index"],
        result.wells.set_index("prc").loc[hits["prc"], "column_index"])}
    assert called == PLANTED
    assert screen.hit_table.rowCount() == len(hits)
    assert screen.hit_table.item(0, 0).text() == "1"
    assert "Z'" in screen.hit_summary.text()


def test_changing_the_statistic_rescores(screen):
    _turn_on(screen)
    screen._hit_rank.setCurrentIndex(screen._hit_rank.findData("b_score"))
    assert screen.hit_result.options["rank_by"] == "b_score"
    screen._hit_threshold.setValue(40.0)
    assert screen.hit_result.options["thresholds"]["b_score"] == 40.0
    assert screen.hit_table.rowCount() == 0


def test_without_a_negative_control_it_says_so(screen):
    screen.set_frame(_plates())
    screen._hit_score.setChecked(True)
    assert screen.hit_result is None
    assert "negative control" in screen.hit_summary.text()


def test_a_campaign_too_short_to_chart_is_still_scored(screen):
    """Two plates are far below the control chart's baseline minimum; the
    chart is refused and the hit scores are not."""
    _turn_on(screen)
    assert screen.result is None
    assert screen.hit_result is not None


def test_export_writes_the_report(screen, tmp_path):
    _turn_on(screen)
    screen._hit_treatment.setCurrentText("gene")
    assert screen._hit_treatment.currentText() == "gene"
    written = screen.export_hits(str(tmp_path / "hits"))
    assert os.path.exists(written["hit_table"])
    assert os.path.exists(written["hit_treatments"])
    assert any(key.startswith("hit_heatmap_") for key in written)


def test_export_before_scoring_writes_nothing(screen, tmp_path):
    assert screen.export_hits(str(tmp_path / "hits")) is None
    assert not (tmp_path / "hits").exists()
