"""A recording never shows the alpha species pages (item 634), even with alpha modules switched on for a lesson."""
from pathlib import Path
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path[:0] = [str(Path(__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[1])]

import pytest

pytest.importorskip('PySide6')
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import capture_host_mode  # noqa: E402


def test_capture_profile_keeps_species_and_features_off():
    profile = capture_host_mode.CAPTURE_PROFILE
    assert profile['show_alpha_species'] is False and profile['show_alpha_features'] is False


def test_species_pages_stay_hidden_when_alpha_modules_are_on(tmp_path, monkeypatch):
    QApplication.instance() or QApplication([])
    from spacr.qt import app as gui, preferences as prefs
    store = QSettings(str(tmp_path / 'qt.ini'), QSettings.IniFormat)
    monkeypatch.setattr(prefs, '_settings', lambda: store)
    prefs._set_show_alpha_species(capture_host_mode.CAPTURE_PROFILE['show_alpha_species'])
    prefs._set_show_alpha_features(capture_host_mode.CAPTURE_PROFILE['show_alpha_features'])
    prefs.set_show_alpha(True)                                  # a lesson of an alpha module (SPACR_TUTORIAL_SHOW_ALPHA=1)
    assert gui.app_is_visible('pipeline_graph') is True          # the alpha module it shows
    for species in ('plasmodium', 'candida'):
        assert gui.app_is_visible(species) is False
