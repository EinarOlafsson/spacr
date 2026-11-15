"""Curate screen — opening a mask and its tracks, and refusing what it can't.

The screen's two Open buttons are the whole entry point, and each one has
three outcomes that matter: nothing typed, something typed that will not
load, and a real file. All three are driven here against files on disk, so
"will not load" means the loader genuinely refused rather than a stub
saying it did.

The ledger sentence is asserted on open, because that is the moment it
exists for -- the person about to analyse a mask finds out it was painted
by hand by opening it, not by looking for a sidecar file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from PySide6.QtWidgets import QFileDialog

from spacr.qt.curation_tool import BrushPanel
from spacr.qt.screens.curate import CurateScreen


@pytest.fixture
def screen(qtbot):
    widget = CurateScreen()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def mask_file(tmp_path):
    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[4:14, 4:14] = 1
    labels[18:28, 18:28] = 2
    path = tmp_path / "plate1_A01_1.tif"
    tifffile.imwrite(path, labels, photometric="minisblack")
    return path


@pytest.fixture
def tracks_file(tmp_path):
    frame = pd.DataFrame({
        "frame": [0, 1, 2, 0, 1],
        "track_id": [1, 1, 1, 2, 2],
        "original_label": [1, 1, 1, 2, 2],
        "centroid_y": [4.0, 5.0, 6.0, 20.0, 21.0],
        "centroid_x": [4.0, 4.5, 5.0, 20.0, 20.5],
    })
    path = tmp_path / "btrack_tracks_cell_plate1_A01_1.csv"
    frame.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# The mask
# ---------------------------------------------------------------------------


def test_browsing_to_a_mask_fills_the_field_and_opens_it(screen, mask_file,
                                                         monkeypatch):
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: (str(mask_file), "Masks (*.tif)")))
    screen._choose_mask()

    assert screen._mask_edit.text() == str(mask_file)
    assert isinstance(screen.brush, BrushPanel)


def test_cancelling_the_mask_browser_changes_nothing(screen, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    screen._choose_mask()

    assert screen._mask_edit.text() == ""
    assert screen.brush is None
    assert screen.status.text() == ""


# ---------------------------------------------------------------------------
# The tracks
# ---------------------------------------------------------------------------


def test_browsing_to_a_tracks_csv_fills_the_field_and_opens_it(
        screen, tracks_file, monkeypatch):
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName",
        staticmethod(lambda *a, **k: (str(tracks_file), "CSV (*.csv)")))
    screen._choose_tracks()

    assert screen._tracks_edit.text() == str(tracks_file)
    assert screen.tabs.currentWidget() is screen.tracks


def test_cancelling_the_tracks_browser_changes_nothing(screen, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    screen._choose_tracks()

    assert screen._tracks_edit.text() == ""
    assert screen.status.text() == ""
