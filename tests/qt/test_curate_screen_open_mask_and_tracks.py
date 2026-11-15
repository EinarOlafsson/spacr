"""Curate screen: what the two Open buttons do with nothing, junk and real files.

``open_mask`` and ``open_tracks`` are the screen's entry points. Each one is
walked over a table of inputs written to disk -- nothing typed, a path that is
not a file, a file the loader genuinely refuses, a real file and a hand-curated
one -- and the exact status sentence, return value and widget state are read
back. The ledger sentence is asserted on open because that is when it exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr.curation import CurationLog
from spacr.qt.curation_tool import BrushPanel
from spacr.qt.screens.curate import CurateScreen
from spacr.qt.theme import active_palette

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    widget = CurateScreen()
    qtbot.addWidget(widget)
    return widget


def _mask(path, labels=True):
    data = np.zeros((32, 32), dtype=np.uint16)
    if labels:
        data[4:14, 4:14] = 1
        data[18:28, 18:28] = 2
    tifffile.imwrite(path, data, photometric="minisblack")
    return path


def test_open_mask_refuses_what_it_cannot_load_and_brushes_what_it_can(
        screen, tmp_path, qtbot, subtests):
    """Nothing, a missing path and a non-TIFF refuse; a mask gets one brush."""
    broken = tmp_path / "not_really.tif"
    broken.write_text("this is a note, not a mask", encoding="utf-8")
    refusals = [
        ("nothing typed", "", "Choose a mask file first."),
        ("a path that is not a file", str(tmp_path / "never_written.tif"),
         "Choose a mask file first."),
        ("a .tif that is not a TIFF", str(broken),
         f"Could not load {broken} as a label mask."),
    ]
    for case_id, typed, said in refusals:
        with subtests.test(msg=case_id):
            screen._mask_edit.setText(typed)
            assert screen.open_mask() is None
            assert screen.status.text() == said
            assert screen.brush is None
            assert screen._mask_path == ""

    raw = _mask(tmp_path / "plate1_A01_1.tif")
    with subtests.test(msg="a pipeline mask: brush, signal, untouched note"):
        screen._mask_edit.setText(f"  {raw}  ")
        with qtbot.waitSignal(screen.mask_opened, timeout=2000) as caught:
            first = screen.open_mask()
        assert isinstance(first, BrushPanel)
        assert screen.brush is first
        assert caught.args == [str(raw)]
        assert screen._mask_path == str(raw)
        assert screen.status.text() == (
            "plate1_A01_1.tif is as the pipeline produced it.")
        assert screen.status.styleSheet() == ""
        assert screen._brush_hint.isVisible() is False

    with subtests.test(msg="a second mask replaces the first brush"):
        first.start_painting()
        second_file = _mask(tmp_path / "plate1_A02_1.tif", labels=False)
        log = CurationLog(str(second_file))
        log.append("paint", 1, n_changed=12)
        log.write_beside(second_file)
        screen._mask_edit.setText(str(second_file))
        second = screen.open_mask()
        assert isinstance(second, BrushPanel)
        assert second is not first
        assert screen.brush is second
        assert screen._brush_layout.indexOf(second) == 0
        assert screen._brush_layout.indexOf(first) == -1
        assert screen._mask_path == str(second_file)
        assert screen.status.text() == (
            "plate1_A02_1.tif has been curated by hand — see "
            "plate1_A02_1.tif.curation.json for what changed.")
        assert screen.status.styleSheet() == (
            f"color: {active_palette()['warning']};")


def test_open_tracks_needs_a_track_table_and_brings_its_panel_forward(
        screen, tmp_path, subtests):
    """No path or no file asks for a CSV; a non-track CSV stays put."""
    wrong = tmp_path / "measurements.csv"
    pd.DataFrame({"area": [1, 2], "intensity": [3, 4]}).to_csv(
        wrong, index=False)
    for case_id, typed in [("nothing typed", ""),
                           ("a path that is not a file",
                            str(tmp_path / "gone.csv"))]:
        with subtests.test(msg=case_id):
            screen._tracks_edit.setText(typed)
            assert screen.open_tracks() is False
            assert screen.status.text() == "Choose a tracks CSV first."
            assert screen.tabs.currentWidget() is not screen.tracks

    with subtests.test(msg="a csv with no track_id column does not open"):
        screen.status.setText("")
        screen._tracks_edit.setText(str(wrong))
        assert screen.open_tracks() is False
        assert screen.tabs.currentWidget() is not screen.tracks
        assert screen.status.text() == ""
        assert screen.tracks.status.text() == (
            "a track table needs ['frame', 'track_id']; this one has "
            "['area', 'intensity']... The canonical columns are "
            "['frame', 'track_id', 'original_label', 'x', 'y'].")

    tracks = tmp_path / "btrack_tracks_cell_plate1_A01_1.csv"
    pd.DataFrame({
        "frame": [0, 1, 2, 0, 1],
        "track_id": [1, 1, 1, 2, 2],
        "original_label": [1, 1, 1, 2, 2],
        "centroid_y": [4.0, 5.0, 6.0, 20.0, 21.0],
        "centroid_x": [4.0, 4.5, 5.0, 20.0, 20.5],
    }).to_csv(tracks, index=False)
    with subtests.test(msg="a real tracks table opens on its tab"):
        screen._tracks_edit.setText(str(tracks))
        assert screen.open_tracks() is True
        assert screen.tabs.currentWidget() is screen.tracks
        assert screen.status.text() == (
            "btrack_tracks_cell_plate1_A01_1.csv is as the pipeline "
            "produced it.")
