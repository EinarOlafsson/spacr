"""Item 1007 lane B: the Convert tab alpha OME-Zarr 0.6 upgrade runs upgrade_ngff."""
import json
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from spacr.qt import preferences
from spacr.qt.screens import convert as screen_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_ome_zarr_upgrade_1007 import _store  # noqa: E402


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Keep feature visibility out of the user preference file."""
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: settings)
    preferences._set_show_alpha_features(True)
    return preferences


def test_the_upgrade_row_runs_the_converter_and_shows_its_report(qtbot, prefs, tmp_path):
    """Run the Convert row and display verification or a destination refusal."""
    screen = screen_module.ConvertScreen(threaded=False)
    qtbot.addWidget(screen)
    assert not screen._btn_ngff_upgrade.isEnabled() and screen._ngff_link.isChecked()
    src = _store(tmp_path)
    screen._ngff_src.setText(str(src))
    screen._ngff_dst.setText(str(tmp_path / "out.zarr"))
    assert screen._btn_ngff_upgrade.isEnabled()
    screen._ngff_link.setChecked(False)
    assert screen.run_ngff_upgrade()
    report = json.loads((tmp_path / "out.zarr" / "ngff_upgrade.json").read_text())
    assert report["verdict"] == "verified" and report["copied"] == report["chunks"]
    assert "verified" in screen.summary_text() and report["chunk_list_md5"] in screen.summary_text()
    screen._ngff_dst.setText(str(src / "inside.zarr"))
    assert not screen.run_ngff_upgrade()
    assert "outside the source" in screen.status_text()


def test_the_upgrade_row_is_alpha(qtbot, prefs):
    """Hide the upgrade row when alpha features are disabled."""
    from spacr.settings import ALPHA_FEATURES

    assert "ConvertNgffUpgrade" in ALPHA_FEATURES[1007]["widgets"]
    preferences._set_show_alpha_features(False)
    screen = screen_module.ConvertScreen(threaded=False)
    qtbot.addWidget(screen)
    assert screen._ngff_panel.isHidden()
