"""The anomaly option of Control Charts, behind the alpha gate.

Hidden with Preferences -> Show alpha features off and shown with it on; a
scoring set up while the option is hidden still reaches the export,
because hiding is a display decision only.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

pytestmark = pytest.mark.qt

from spacr.qt.screens.control_chart import ControlChartScreen  # noqa: E402

from tests.test_anomaly_detection_vs_controls import _HITS, _screen  # noqa: E402,E501

_ALPHA = ("ControlChartAnomaly", "ControlChartAnomalySection")


@pytest.fixture
def screen(qtbot):
    made = ControlChartScreen(threaded=False)
    qtbot.addWidget(made)
    return made


@pytest.fixture
def gate(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_registered_alpha_and_follows_the_switch(screen, gate):
    from PySide6.QtCore import QObject

    from spacr.settings import ALPHA_FEATURES, _alpha_names

    assert set(_ALPHA) == set(ALPHA_FEATURES[563]["widgets"])
    assert set(_ALPHA) <= _alpha_names("widgets")
    gate._apply_alpha_widgets(screen)
    for name in _ALPHA:
        assert screen.findChild(QObject, name).isHidden(), name
    gate._set_show_alpha_features(True)
    gate._apply_alpha_widgets(screen)
    for name in _ALPHA:
        assert not screen.findChild(QObject, name).isHidden(), name


def test_a_scoring_set_up_while_hidden_still_reaches_the_export(
        screen, gate, tmp_path):
    from PySide6.QtCore import QObject

    gate._apply_alpha_widgets(screen)
    assert screen.findChild(QObject, "ControlChartAnomaly").isHidden()
    screen.set_frame(_screen())
    screen._value.setCurrentText("feature_0")
    screen._control_column.setCurrentText("condition")
    screen._on_control_column("condition")
    screen._negative.setCurrentText("neg")
    screen._anomaly_method.setCurrentIndex(1)
    screen._anomaly_hits.setText(", ".join(_HITS))
    screen._anomaly_score.setChecked(True)
    result = screen._anomaly
    assert result is not None, screen.anomaly_summary.text()
    assert result.method == "knn"
    assert result.auroc >= 0.95
    assert screen.anomaly_table.rowCount() == 160
    assert screen.anomaly_figure.axes
    written = screen._export_anomalies(str(tmp_path / "anomalies"))
    assert os.path.exists(written["anomaly_wells"])


def test_without_a_negative_control_it_says_so(screen):
    screen.set_frame(_screen())
    screen._negative.setCurrentText("")
    screen._anomaly_score.setChecked(True)
    assert screen._anomaly is None
    assert "negative control" in screen.anomaly_summary.text()
