"""Map Barcodes' spatial transcriptomics panel, behind the alpha gate.

Hidden with Preferences -> Show alpha features off and shown with it on; a
panel filled in while it is hidden still runs, because hiding is a display
decision only.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyarrow")

pytestmark = pytest.mark.qt

from spacr.qt.screens import map_barcodes                 # noqa: E402

from tests.test_spatial_transcriptomics import (          # noqa: E402
    _masks, xenium_bundle)


@pytest.fixture
def gate(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


@pytest.fixture
def screen(qtbot, gate):
    from spacr.qt.screens.app_screen import AppScreen

    made = AppScreen(app_key="map_barcodes")
    qtbot.addWidget(made)
    assert map_barcodes._install_spatial_transcriptomics(made) is not None
    return made


def test_registered_alpha_and_follows_the_switch(screen, gate):
    from PySide6.QtCore import QObject

    from spacr.settings import ALPHA_FEATURES

    names = map_barcodes._SPATIAL_ALPHA_WIDGETS
    assert set(names) == set(ALPHA_FEATURES[534]["widgets"])
    toggle = screen.findChild(QObject, "MapBarcodesSpatialToggle")
    assert toggle.isHidden()
    gate._set_show_alpha_features(True)
    gate._apply_alpha_widgets(screen)
    assert not toggle.isHidden()
    toggle.setChecked(True)
    assert not screen.findChild(QObject, "MapBarcodesSpatialCard").isHidden()
    gate._set_show_alpha_features(False)
    gate._apply_alpha_widgets(screen)
    for name in names:
        assert screen.findChild(QObject, name).isHidden(), name


def test_a_panel_filled_while_hidden_still_runs(screen, gate, tmp_path):
    from spacr.tabular import read_table

    panel = screen._spatial_panel
    assert screen._spatial_toggle.isHidden()
    folder = xenium_bundle(tmp_path / "x")
    cell, _nucleus, pathogen = _masks()
    np.save(tmp_path / "plate1_r1_c1_f1.npy", cell)
    np.save(tmp_path / "pathogen.npy", pathogen)
    panel.folder.setText(folder)
    panel.masks["cell"].setText(str(tmp_path / "plate1_r1_c1_f1.npy"))
    panel.masks["pathogen"].setText(str(tmp_path / "pathogen.npy"))
    db = str(tmp_path / "measurements.db")
    panel.db.setText(db)
    assert panel.load()
    assert panel.gene.count() == 2
    assert panel.figure.axes
    summary = panel.run()
    assert summary is not None and summary["prcf"] == "plate1_r1_c1_f1"
    wide = read_table(db, table="cell_expression", report=None)
    assert len(wide) == 20
    assert "20" in panel.status.text()


def test_missing_inputs_are_named(screen):
    panel = screen._spatial_panel
    assert panel.load() is False
    assert panel.status.text()
    assert panel.run() is None
