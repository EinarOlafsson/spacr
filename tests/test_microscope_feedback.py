"""A folder watch can send the events it finds back to the microscope.

The request was to send the positions of hits or rare events found by live
analysis to the microscope for re-imaging, with a simulation mode for testing
without a microscope. It is done, in simulation, when detected events produce
correctly transformed stage positions.

The simulated microscope answers the same calls a pycro-manager ``Core``
does and acquires from a folder of field images laid out on its stage, so a
re-imaged view centred on an event shows the object at its centre only when
the pixel-to-stage transform is right.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest

tifffile = pytest.importorskip("tifffile")

from spacr import core                                            # noqa: E402
from tests.test_watch_folder_and_analyse import (                 # noqa: E402,F401
    FAST, MASK, MEASURE, _channels, _name, real_pipeline)


def _positions(folder, rows):
    path = folder / "positions.csv"
    pd.DataFrame(rows, columns=["field", "x", "y", "z"][:len(rows[0])]).to_csv(
        path, index=False)
    return str(path)


def _ledger(src):
    with open(os.path.join(src, "spacr_watch", "watch_ledger.json")) as handle:
        return json.load(handle)["fields"]


class MeasuredField:
    """Stand-in pipeline: writes a merged stack and a cell table per field."""

    def __init__(self, objects, shape=(64, 64)):
        self.objects, self.shape = objects, shape

    def __call__(self, field_dir, settings):
        key = os.path.basename(field_dir)
        os.makedirs(os.path.join(field_dir, "merged"))
        np.save(os.path.join(field_dir, "merged", key + ".npy"),
                np.zeros(self.shape + (3,), np.float32))
        os.makedirs(os.path.join(field_dir, "measurements"))
        frame = pd.DataFrame(
            [{"object_label": label, "cell_area": area,
              "cell_channel_0_centroid_weighted-0": row,
              "cell_channel_0_centroid_weighted-1": column}
             for label, area, row, column in self.objects])
        with sqlite3.connect(os.path.join(field_dir, "measurements",
                                          "measurements.db")) as connection:
            frame.to_sql("cell", connection, index=False)


def _feedback(folder, positions, **extra):
    settings = {"src": str(folder), "channels": [0, 1], "metadata_type":
                "cellvoyager", "custom_regex": None, "watch_folder": True,
                "watch_pipeline": "mask_measure", "microscope_feedback": True,
                "microscope_driver": "simulated",
                "microscope_positions": positions}
    settings.update(FAST)
    settings.update(extra)
    return settings


def test_a_pixel_becomes_the_stage_position_that_centres_it():
    matrix = core._microscope_matrix(
        {"microscope_stage_transform": "[0.5, 0.0, 0.0, -0.5]"})
    x, y = core._microscope_stage_position((45.5, 43.5), (64, 64),
                                           (1000.0, 2000.0), matrix)
    assert (x, y) == (1006.0, 1993.0)
    assert core._microscope_stage_position((31.5, 31.5), (64, 64),
                                           (1000.0, 2000.0), matrix) == (
        1000.0, 2000.0)
    turned = core._microscope_matrix(
        {"microscope_stage_transform": [0.0, -2.0, 2.0, 0.0]})
    assert core._microscope_stage_position((41.5, 31.5), (64, 64), (0.0, 0.0),
                                           turned) == (-20.0, 0.0)
    for bad in ("[1, 2]", [0, 0, 0, 0], "not numbers"):
        with pytest.raises(ValueError, match="microscope_stage_transform"):
            core._microscope_matrix({"microscope_stage_transform": bad})


def test_the_simulated_microscope_shows_the_field_under_the_stage(tmp_path):
    field = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    tifffile.imwrite(tmp_path / _name("A01", 1), field)
    tifffile.imwrite(tmp_path / _name("A01", 2), field + 1)
    matrix = core._microscope_matrix(
        {"microscope_stage_transform": [0.5, 0.0, 0.0, -0.5]})
    scope = core._SimulatedMicroscope(
        str(tmp_path), {"plate1_A01_0001_001": (1000.0, 2000.0)}, matrix,
        {"metadata_type": "cellvoyager", "custom_regex": None})
    scope.set_xy_position(1000.0, 2000.0)
    scope.snap_image()
    view = np.asarray(scope.get_image()).reshape(scope.get_image_height(),
                                                 scope.get_image_width())
    np.testing.assert_array_equal(view, field)
    scope.set_xy_position(1006.0, 1993.0)
    scope.wait_for_device(scope.get_xy_stage_device())
    scope.snap_image()
    view = scope.get_image().reshape(64, 64)
    np.testing.assert_array_equal(view[:50, :52], field[14:, 12:])
    assert not view[50:].any() and not view[:, 52:].any()
    assert [call[0] for call in scope.log] == [
        "set_xy_position", "snap_image", "set_xy_position", "wait_for_device",
        "snap_image"]


def test_events_are_placed_queued_and_imaged_during_a_watch(tmp_path):
    for well in ("A01", "A02"):
        for channel in (1, 2):
            tifffile.imwrite(tmp_path / _name(well, channel),
                             np.full((64, 64), channel, np.uint16))
    positions = _positions(tmp_path, [("plate1_A01_0001_001", 1000, 2000, 7.5)])
    pipeline = MeasuredField([(1, 576, 17.5, 17.5), (2, 672, 45.5, 43.5),
                              (3, 900, 10.0, 50.0)])
    result = core._watch_folder_and_analyse(
        _feedback(tmp_path, positions, microscope_event_query="cell_area > 600",
                  microscope_stage_transform=[0.5, 0.0, 0.0, -0.5],
                  microscope_max_events=1, microscope_timepoints=2,
                  microscope_interval_seconds=0.05), pipeline)
    assert result["done"] == ["plate1_A01_0001_001", "plate1_A02_0001_001"]
    fields = _ledger(tmp_path)
    [event] = fields["plate1_A01_0001_001"]["events"]
    assert event["object"] == "2" and event["pixel"] == [45.5, 43.5]
    assert event["stage"] == [1006.0, 1993.0, 7.5]
    assert event["status"] == "acquired"
    assert event["files"] == ["plate1_A01_0001_001_e000_t000.tif",
                              "plate1_A01_0001_001_e000_t001.tif"]
    reimaged = tmp_path / "spacr_watch" / "reimaged"
    assert sorted(os.listdir(reimaged)) == event["files"]
    [missing] = fields["plate1_A02_0001_001"]["events"]
    assert missing["status"] == "no_position" and missing["stage"] is None


def test_events_still_queued_are_imaged_when_the_next_watch_starts(tmp_path):
    for channel in (1, 2):
        tifffile.imwrite(tmp_path / _name("A01", channel),
                         np.full((64, 64), 3, np.uint16))
    positions = _positions(tmp_path, [("plate1_A01_0001_001", 0, 0)])
    settings = _feedback(tmp_path, positions)
    original = core._microscope_drain
    core._microscope_drain = lambda context: 0
    try:
        core._watch_folder_and_analyse(settings,
                                       MeasuredField([(1, 50, 31.5, 31.5)]))
    finally:
        core._microscope_drain = original
    [event] = _ledger(tmp_path)["plate1_A01_0001_001"]["events"]
    assert event["status"] == "queued" and event["stage"] == [0.0, 0.0]
    core._watch_folder_and_analyse(settings, MeasuredField([]))
    [event] = _ledger(tmp_path)["plate1_A01_0001_001"]["events"]
    assert event["status"] == "acquired"


def test_what_feedback_needs_is_asked_for_by_name(tmp_path, monkeypatch):
    positions = _positions(tmp_path, [("f", 0, 0)])
    with pytest.raises(ValueError, match="mask_measure"):
        core._watch_folder_and_analyse(
            _feedback(tmp_path, positions, watch_pipeline="mask"), None)
    with pytest.raises(ValueError, match="microscope_positions"):
        core._watch_folder_and_analyse(_feedback(tmp_path, ""), None)
    with pytest.raises(ValueError, match="microscope_driver"):
        core._watch_folder_and_analyse(
            _feedback(tmp_path, positions, microscope_driver="zeiss"), None)
    monkeypatch.setitem(sys.modules, "pycromanager", None)
    with pytest.raises(ImportError, match="pip install pycromanager"):
        core._watch_folder_and_analyse(
            _feedback(tmp_path, positions, microscope_driver="pycromanager"),
            None)


def test_measured_events_are_imaged_centred_on_their_objects(
        tmp_path, real_pipeline):
    source = tmp_path / "acquired"
    source.mkdir()
    fields = {}
    for shift, well in enumerate(("A01", "A02")):
        images = _channels(shift)
        fields[well] = images[0]
        for channel, image in enumerate(images, start=1):
            tifffile.imwrite(source / _name(well, channel), image)
    names = sorted(os.listdir(source))
    settings_file = tmp_path / "measure_settings.json"
    settings_file.write_text(json.dumps(MEASURE))
    positions = _positions(tmp_path, [("plate1_A01_0001_001", 1000, 2000),
                                      ("plate1_A02_0001_001", 5000, 2000)])
    watched = tmp_path / "watched"
    watched.mkdir()

    def acquire():
        for name in names:
            time.sleep(0.2)
            partial = watched / (name + ".part")
            shutil.copyfile(source / name, partial)
            os.replace(partial, watched / name)

    writer = threading.Thread(target=acquire)
    writer.start()
    try:
        result = core.preprocess_generate_masks(dict(
            MASK, src=str(watched), watch_folder=True,
            watch_pipeline="mask_measure",
            watch_measure_settings=str(settings_file),
            watch_settle_seconds=0.3, watch_poll_seconds=0.05,
            watch_idle_minutes=4.0 / 60.0, microscope_feedback=True,
            microscope_driver="simulated", microscope_positions=positions,
            microscope_stage_transform=[0.5, 0.0, 0.0, -0.5],
            microscope_event_query="cell_area > 600"))
    finally:
        writer.join()

    assert len(result["done"]) == 2 and not result["failed"]
    ledger = _ledger(watched)
    for well, centre in (("A01", (1000.0, 2000.0)), ("A02", (5000.0, 2000.0))):
        [event] = ledger[f"plate1_{well}_0001_001"]["events"]
        assert event["pixel"] == pytest.approx([45.5, 43.5])
        assert event["stage"] == pytest.approx([centre[0] + 6.0,
                                                centre[1] - 7.0])
        view = tifffile.imread(watched / "spacr_watch" / "reimaged"
                               / event["files"][0])
        np.testing.assert_array_equal(view[:50, :52], fields[well][14:, 12:])
        assert view[31:33, 31:33].min() == fields[well][45, 43] > 40
