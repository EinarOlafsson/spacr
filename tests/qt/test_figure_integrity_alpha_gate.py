"""The figure integrity check is an alpha Preferences toggle.

Pinned here:

* with Show alpha features off the toggle is not built;
* with it on, the toggle is on the Figures tab, off by default, and Save
  stores it;
* a value saved while the toggle is hidden still reaches the figure writer.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402
from PySide6.QtWidgets import QDialogButtonBox, QWidget          # noqa: E402

from spacr.settings import ALPHA_FEATURES                         # noqa: E402

WIDGETS = ALPHA_FEATURES[572]["widgets"]


@pytest.fixture
def prefs(tmp_path, monkeypatch, qt_theme_applied):
    from spacr.qt import preferences

    path = tmp_path / "integrity.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    monkeypatch.delenv("SPACR_FIGURE_INTEGRITY", raising=False)
    return preferences


def _dialog(qtbot, prefs):
    dlg = prefs.PreferencesDialog()
    qtbot.addWidget(dlg)
    return dlg


def test_registered_as_alpha():
    assert WIDGETS == ("FigureIntegrityCheck",)


def test_hidden_the_toggle_is_not_built(qtbot, prefs):
    prefs._set_show_alpha_features(False)
    dlg = _dialog(qtbot, prefs)
    for name in WIDGETS:
        assert dlg.findChild(QWidget, name) is None, name
    assert prefs._is_alpha_visible("widgets", WIDGETS[0]) is False


def test_shown_the_toggle_is_off_and_saves(qtbot, prefs):
    prefs._set_show_alpha_features(True)
    dlg = _dialog(qtbot, prefs)
    toggle = dlg.findChild(QWidget, "FigureIntegrityCheck")
    assert toggle is not None
    assert toggle.isChecked() is False
    toggle.setChecked(True)
    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert prefs._get_figure_integrity() is True


def test_a_value_saved_while_hidden_reaches_the_writer(prefs, tmp_path,
                                                       monkeypatch):
    from matplotlib.figure import Figure

    from spacr import plot

    prefs._set_show_alpha_features(False)
    prefs._set_figure_integrity(True)
    monkeypatch.setattr(plot, "figure_output_preferences",
                        lambda: ("png", 100))
    fig = Figure(figsize=(2, 2), dpi=100)
    fig.add_subplot().imshow(np.arange(64 * 64).reshape(64, 64) % 97,
                             cmap="gray")
    written = plot.save_figure(fig, tmp_path / "hidden.png")
    assert os.path.isfile(plot._provenance_sidecar_path(written))

    prefs._set_figure_integrity(False)
    again = plot.save_figure(fig, tmp_path / "off.png")
    assert not os.path.exists(plot._provenance_sidecar_path(again))
