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

from spacr.curation import CurationLog
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

def test_opening_nothing_asks_for_a_file_rather_than_failing(screen):
    assert screen.open_mask() is None
    assert screen.status.text() == "Choose a mask file first."
    assert screen.brush is None


def test_a_path_that_is_not_a_file_is_the_same_as_nothing(screen, tmp_path):
    screen._mask_edit.setText(str(tmp_path / "never_written.tif"))
    assert screen.open_mask() is None
    assert screen.status.text() == "Choose a mask file first."


def test_a_file_the_loader_refuses_names_the_file(screen, tmp_path):
    """A .tif that is not a TIFF -- the commonest way this goes wrong."""
    broken = tmp_path / "not_really.tif"
    broken.write_text("this is a note, not a mask")
    screen._mask_edit.setText(str(broken))

    assert screen.open_mask() is None
    assert str(broken) in screen.status.text()
    assert "Could not load" in screen.status.text()
    assert screen.brush is None


def test_opening_a_mask_puts_a_brush_on_it_and_announces_the_path(
        screen, mask_file, qtbot):
    screen._mask_edit.setText(str(mask_file))

    with qtbot.waitSignal(screen.mask_opened, timeout=2000) as caught:
        panel = screen.open_mask()

    assert isinstance(panel, BrushPanel)
    assert screen.brush is panel
    assert caught.args == [str(mask_file)]
    assert screen._mask_path == str(mask_file)
    assert "as the pipeline produced it" in screen.status.text()


def test_opening_a_curated_mask_says_it_was_edited_by_hand(screen,
                                                            mask_file):
    log = CurationLog(str(mask_file))
    log.append("paint", 1, n_changed=12)
    log.write_beside(mask_file)

    screen._mask_edit.setText(str(mask_file))
    screen.open_mask()

    assert "curated by hand" in screen.status.text()
    assert "curation.json" in screen.status.text()


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


def test_a_second_mask_replaces_the_first_brush(screen, mask_file, tmp_path):
    """Two brushes on one canvas would paint two layers from one stroke."""
    screen._mask_edit.setText(str(mask_file))
    first = screen.open_mask()
    first.start_painting()

    second_file = tmp_path / "plate1_A02_1.tif"
    tifffile.imwrite(second_file, np.zeros((32, 32), np.uint16),
                     photometric="minisblack")
    screen._mask_edit.setText(str(second_file))
    second = screen.open_mask()

    assert second is not first
    assert screen.brush is second
    assert screen._brush_hint.isVisible() is False


# ---------------------------------------------------------------------------
# The tracks
# ---------------------------------------------------------------------------

def test_opening_no_tracks_asks_for_a_csv(screen):
    assert screen.open_tracks() is False
    assert screen.status.text() == "Choose a tracks CSV first."


def test_a_tracks_path_that_is_not_a_file_is_the_same_as_nothing(screen,
                                                                 tmp_path):
    screen._tracks_edit.setText(str(tmp_path / "gone.csv"))
    assert screen.open_tracks() is False
    assert screen.status.text() == "Choose a tracks CSV first."


def test_a_csv_that_is_not_a_track_table_does_not_open(screen, tmp_path):
    """No ``track_id`` column: the panel says why and the tab does not move."""
    wrong = tmp_path / "measurements.csv"
    pd.DataFrame({"area": [1, 2], "intensity": [3, 4]}).to_csv(
        wrong, index=False)
    screen._tracks_edit.setText(str(wrong))

    assert screen.open_tracks() is False
    assert screen.tabs.currentWidget() is not screen.tracks
    assert "track_id" in screen.tracks.status.text()


def test_opening_tracks_brings_the_track_panel_forward(screen, tracks_file):
    screen._tracks_edit.setText(str(tracks_file))

    assert screen.open_tracks() is True
    assert screen.tabs.currentWidget() is screen.tracks
    assert "as the pipeline produced it" in screen.status.text()


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
