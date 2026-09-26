"""Make Masks can watch a folder and analyse each field as it arrives.

The request was "live watch a folder and analyze incoming images", and the
item is done when copying a plate's files into the watched folder one by one
gives the same results as a batch run, with each field taken within a set
delay of arriving.

The engine tests hand the watch a recording stand-in for the pipeline, so they
pin what is taken and when: a growing file waits, a field waits for all of its
channels, a half-written TIFF waits, every field runs once, the record lets a
restart skip finished fields, a failure is recorded and retried on the next
start, and Stop leaves a record the next start continues from.

The acceptance test runs the real preprocessing, the real merge and the real
Measure, with only the Cellpose forward pass replaced by a labeller of the
painted squares, once as a batch run and once through the watch.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from spacr import core                                            # noqa: E402
from spacr.cancellation import (CancellationToken, PipelineCancelled,  # noqa: E402
                                installed_token)

FAST = {"watch_settle_seconds": 0.3, "watch_poll_seconds": 0.05,
        "watch_idle_minutes": 1.5 / 60.0}


def _image(value=1, size=16):
    return np.full((size, size), value, np.uint16)


def _write(folder, name, value=1):
    tifffile.imwrite(os.path.join(folder, name), _image(value))


def _name(well, channel):
    return f"plate1_{well}_T0001F001L01A01Z01C0{channel}.tif"


class Recorder:
    """Stand-in pipeline: records each call and what the field folder held."""

    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)

    def __call__(self, field_dir, settings):
        names = sorted(os.listdir(field_dir))
        self.calls.append((os.path.basename(field_dir), names, time.time()))
        if os.path.basename(field_dir) in self.fail:
            raise RuntimeError("segmentation failed")
        merged = os.path.join(field_dir, "merged")
        os.makedirs(merged)
        np.save(os.path.join(merged, os.path.basename(field_dir) + ".npy"),
                np.zeros((2, 2), np.uint16))


def _settings(folder, **extra):
    settings = {"src": str(folder), "channels": [0, 1], "metadata_type":
                "cellvoyager", "custom_regex": None, "watch_folder": True}
    settings.update(FAST)
    settings.update(extra)
    return settings


def _ledger(src):
    with open(os.path.join(src, "spacr_watch", "watch_ledger.json")) as handle:
        return json.load(handle)["fields"]


def test_the_mask_entry_point_hands_a_watch_to_the_watcher(tmp_path,
                                                           monkeypatch):
    seen = []
    monkeypatch.setattr(core, "_watch_folder_and_analyse",
                        lambda settings: seen.append(settings) or "watched")
    assert core.preprocess_generate_masks(_settings(tmp_path)) == "watched"
    assert seen and seen[0]["src"] == str(tmp_path)


def test_fields_already_there_run_once_each_and_nothing_else_runs(tmp_path):
    for well in ("A01", "A02"):
        for channel in (1, 2):
            _write(tmp_path, _name(well, channel))
    (tmp_path / "notes.txt").write_text("not an image")
    recorder = Recorder()
    result = core._watch_folder_and_analyse(_settings(tmp_path), recorder)
    keys = [call[0] for call in recorder.calls]
    assert keys == ["plate1_A01_0001_001", "plate1_A02_0001_001"]
    assert recorder.calls[0][1] == [_name("A01", 1), _name("A01", 2)]
    assert result["done"] == keys and result["failed"] == []
    merged = sorted(os.listdir(tmp_path / "spacr_watch" / "merged"))
    assert merged == [key + ".npy" for key in keys]
    fields = _ledger(tmp_path)
    assert {key: fields[key]["status"] for key in fields} == {
        key: "done" for key in keys}
    assert set(fields[keys[0]]["files"]) == {_name("A01", 1), _name("A01", 2)}


def test_a_field_waits_for_every_channel_and_a_growing_file_waits(tmp_path):
    recorder = Recorder()
    _write(tmp_path, _name("A01", 1))
    growing = tmp_path / _name("A01", 2)
    written = {}

    def acquire():
        time.sleep(0.4)
        with open(growing, "wb") as handle:
            for _step in range(6):
                handle.write(b"\0" * 200)
                handle.flush()
                time.sleep(0.12)
        _write(tmp_path, _name("A01", 2), value=7)
        written["at"] = time.time()

    writer = threading.Thread(target=acquire)
    writer.start()
    try:
        result = core._watch_folder_and_analyse(_settings(tmp_path), recorder)
    finally:
        writer.join()
    assert [call[0] for call in recorder.calls] == ["plate1_A01_0001_001"]
    started = recorder.calls[0][2]
    assert started >= written["at"] + FAST["watch_settle_seconds"] - 0.05
    assert result["done"] == ["plate1_A01_0001_001"]
    entry = _ledger(tmp_path)["plate1_A01_0001_001"]
    assert entry["waited"] <= (FAST["watch_settle_seconds"]
                               + FAST["watch_poll_seconds"] + 1.0)


def test_a_half_written_tif_is_not_taken(tmp_path):
    name = _name("A01", 1)
    _write(tmp_path, name)
    whole = (tmp_path / name).read_bytes()
    (tmp_path / name).write_bytes(whole[: len(whole) // 3])
    assert core._watch_unreadable(str(tmp_path / name)) is not None
    recorder = Recorder()
    result = core._watch_folder_and_analyse(
        _settings(tmp_path, channels=[0]), recorder)
    assert recorder.calls == []
    assert result["incomplete"] == ["plate1_A01_0001_001"]
    (tmp_path / name).write_bytes(whole)
    assert core._watch_unreadable(str(tmp_path / name)) is None
    result = core._watch_folder_and_analyse(
        _settings(tmp_path, channels=[0]), recorder)
    assert [call[0] for call in recorder.calls] == ["plate1_A01_0001_001"]


def test_a_restart_skips_fields_already_analysed(tmp_path):
    for channel in (1, 2):
        _write(tmp_path, _name("A01", channel))
    first = Recorder()
    core._watch_folder_and_analyse(_settings(tmp_path), first)
    for channel in (1, 2):
        _write(tmp_path, _name("B01", channel))
    second = Recorder()
    result = core._watch_folder_and_analyse(_settings(tmp_path), second)
    assert [call[0] for call in first.calls] == ["plate1_A01_0001_001"]
    assert [call[0] for call in second.calls] == ["plate1_B01_0001_001"]
    assert result["done"] == ["plate1_A01_0001_001", "plate1_B01_0001_001"]


def test_a_failed_field_is_recorded_shown_and_retried_on_the_next_start(
        tmp_path, capsys):
    for well in ("A01", "A02"):
        for channel in (1, 2):
            _write(tmp_path, _name(well, channel))
    failing = Recorder(fail={"plate1_A01_0001_001"})
    result = core._watch_folder_and_analyse(_settings(tmp_path), failing)
    out = capsys.readouterr().out
    assert result["failed"] == ["plate1_A01_0001_001"]
    assert result["done"] == ["plate1_A02_0001_001"]
    assert [call[0] for call in failing.calls].count("plate1_A01_0001_001") == 1
    assert "watch_folder: ERROR plate1_A01_0001_001 failed: RuntimeError" in out
    assert "watch_folder: 1 analysed, 0 waiting, 1 failed" in out
    entry = _ledger(tmp_path)["plate1_A01_0001_001"]
    assert entry["status"] == "failed" and "segmentation failed" in entry["error"]
    retry = Recorder()
    result = core._watch_folder_and_analyse(_settings(tmp_path), retry)
    assert [call[0] for call in retry.calls] == ["plate1_A01_0001_001"]
    assert result["failed"] == []


def test_stop_ends_the_watch_cleanly_and_the_next_start_continues(tmp_path):
    for well in ("A01", "A02"):
        for channel in (1, 2):
            _write(tmp_path, _name(well, channel))
    token = CancellationToken()
    calls = []

    def stop_during_the_second(field_dir, settings):
        calls.append(os.path.basename(field_dir))
        if len(calls) == 2:
            token.cancel("stopped by the user")
            from spacr.cancellation import checkpoint
            checkpoint()
        Recorder()(field_dir, settings)

    with installed_token(token):
        with pytest.raises(PipelineCancelled):
            core._watch_folder_and_analyse(
                _settings(tmp_path, watch_idle_minutes=0),
                stop_during_the_second)
    fields = _ledger(tmp_path)
    assert fields["plate1_A01_0001_001"]["status"] == "done"
    assert fields["plate1_A02_0001_001"]["status"] == "interrupted"
    resumed = Recorder()
    core._watch_folder_and_analyse(_settings(tmp_path), resumed)
    assert [call[0] for call in resumed.calls] == ["plate1_A02_0001_001"]
    assert _ledger(tmp_path)["plate1_A02_0001_001"]["status"] == "done"


def test_stop_while_waiting_ends_the_watch(tmp_path):
    token = CancellationToken()
    threading.Timer(0.3, token.cancel).start()
    started = time.time()
    with installed_token(token):
        with pytest.raises(PipelineCancelled):
            core._watch_folder_and_analyse(
                _settings(tmp_path, watch_idle_minutes=0,
                          watch_poll_seconds=30), Recorder())
    assert time.time() - started < 5


def test_a_field_is_appended_to_the_combined_database_once(tmp_path):
    field_db = tmp_path / "field.db"
    with sqlite3.connect(field_db) as connection:
        connection.execute("CREATE TABLE cell (prcfo TEXT, area REAL)")
        connection.execute("INSERT INTO cell VALUES ('p1_r1_c1_f1_o1', 3.0)")
    later_db = tmp_path / "later.db"
    with sqlite3.connect(later_db) as connection:
        connection.execute(
            "CREATE TABLE cell (prcfo TEXT, area REAL, extra REAL)")
        connection.execute(
            "INSERT INTO cell VALUES ('p1_r1_c2_f1_o1', 4.0, 1.5)")
    combined = tmp_path / "out" / "measurements.db"
    assert core._watch_merge_database(str(field_db), str(combined), "a")
    assert not core._watch_merge_database(str(field_db), str(combined), "a")
    assert core._watch_merge_database(str(later_db), str(combined), "b")
    with sqlite3.connect(combined) as connection:
        rows = connection.execute(
            "SELECT prcfo, area, extra FROM cell ORDER BY prcfo").fetchall()
    assert rows == [("p1_r1_c1_f1_o1", 3.0, None),
                    ("p1_r1_c2_f1_o1", 4.0, 1.5)]


@pytest.mark.parametrize("extra, message", [
    ({"timelapse": True}, "timelapse"),
    ({"z_stack": "True"}, "z_stack"),
    ({"watch_pipeline": "classify"}, "watch_pipeline"),
    ({"watch_settle_seconds": "soon"}, "watch_settle_seconds"),
    ({"src": ["a", "b"]}, "one folder"),
])
def test_what_a_watch_cannot_do_is_refused_by_name(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        core._watch_folder_and_analyse(_settings(tmp_path, **extra),
                                       Recorder())


def test_a_name_without_a_channel_is_a_field_by_itself():
    cache = {}
    settings = {"metadata_type": "cellvoyager", "custom_regex": None}
    assert core._watch_field_of("scan 01.czi", settings, cache) == (
        "scan_01", None)
    assert core._watch_field_of(_name("B03", 2), settings, cache) == (
        "plate1_B03_0001_001", "02")


SIZE = 64
CELLS = ((slice(6, 30), slice(6, 30)), (slice(34, 58), slice(30, 58)))
NUCLEI = ((slice(14, 22), slice(14, 22)), (slice(42, 50), slice(40, 48)))


def _channels(shift):
    """Nucleus and cell images of one field; ``shift`` varies the brightness."""
    nucleus = np.full((SIZE, SIZE), 40, np.uint16)
    cell = np.full((SIZE, SIZE), 40, np.uint16)
    for box in CELLS:
        cell[box] = 700 + 60 * shift
    for box in NUCLEI:
        nucleus[box] = 900 + 40 * shift
    return [nucleus, cell]


def _label_the_painted_squares(src, settings, object_type):
    """Stand-in for the Cellpose pass: label the painted cells or nuclei."""
    from scipy import ndimage

    from spacr.io import _listdir_visible

    out = os.path.join(src, f"{object_type}_mask_stack")
    os.makedirs(out, exist_ok=True)
    position = settings[f"cellpose_{object_type}_channel"]
    for name in sorted(_listdir_visible(src)):
        if not name.endswith(".npz"):
            continue
        with np.load(os.path.join(src, name), allow_pickle=True) as archive:
            data, files = archive["data"], archive["filenames"]
        for index, filename in enumerate(files):
            labels, _count = ndimage.label(data[index, :, :, position] > 0.5)
            np.save(os.path.join(out, str(filename)), labels.astype(np.uint16))


MASK = {"metadata_type": "cellvoyager", "custom_regex": None,
        "channels": [0, 1], "nucleus_channel": 0, "cell_channel": 1,
        "pathogen_channel": None, "organelle_channel": None,
        "preprocess": True, "masks": True, "plot": False, "verbose": False,
        "test_mode": False, "timelapse": False, "n_jobs": 1,
        "adjust_cells": False, "consolidate": False, "save": True,
        "batch_size": 1, "randomize": False, "normalize": True}
MEASURE = {"timelapse": False, "channels": [0, 1], "cell_min_size": 0,
           "nucleus_min_size": 0, "pathogen_min_size": 0, "save_png": False,
           "save_arrays": False, "plot": False, "save_measurements": True,
           "n_jobs": 1, "verbose": False, "radial_dist": False,
           "homogeneity": False, "calculate_correlation": False,
           "experiment": "watch"}


def _rows(db, table):
    """A table's rows without the columns that name where files live."""
    with sqlite3.connect(db) as connection:
        columns = [row[1] for row in connection.execute(
            f'PRAGMA table_info("{table}")')]
        kept = [column for column in columns
                if "path" not in column.lower() and column != "file_name"]
        listed = ", ".join(f'"{column}"' for column in kept)
        rows = connection.execute(f'SELECT {listed} FROM "{table}"').fetchall()
    return kept, sorted(rows, key=repr)


@pytest.fixture
def real_pipeline(monkeypatch):
    import spacr.measure as measure
    import spacr.object as spacr_object

    monkeypatch.setattr(spacr_object, "generate_cellpose_masks_sam",
                        _label_the_painted_squares)
    monkeypatch.setattr(
        measure, "_load_zernike_moments",
        lambda: (_ for _ in ()).throw(ImportError("not needed here")))
    monkeypatch.setattr(measure, "_ZERNIKE_AVAILABLE", None)


def test_files_copied_in_one_by_one_give_the_batch_runs_results(
        tmp_path, real_pipeline):
    from spacr.core import preprocess_generate_masks
    from spacr.measure import measure_crop

    source = tmp_path / "acquired"
    source.mkdir()
    wells = ("A01", "A02", "B01")
    for shift, well in enumerate(wells):
        for channel, image in enumerate(_channels(shift), start=1):
            tifffile.imwrite(source / _name(well, channel), image)
    names = sorted(os.listdir(source))

    batch = tmp_path / "batch"
    shutil.copytree(source, batch)
    preprocess_generate_masks(dict(MASK, src=str(batch)))
    measure_crop(dict(MEASURE, src=str(batch / "merged")))

    settings_file = tmp_path / "measure_settings.json"
    settings_file.write_text(json.dumps(MEASURE))
    watched = tmp_path / "watched"
    watched.mkdir()
    arrivals = {}

    def acquire():
        for name in names:
            time.sleep(0.2)
            partial = watched / (name + ".part")
            shutil.copyfile(source / name, partial)
            os.replace(partial, watched / name)
            arrivals[name] = time.time()

    writer = threading.Thread(target=acquire)
    writer.start()
    try:
        result = core.preprocess_generate_masks(dict(
            MASK, src=str(watched), watch_folder=True,
            watch_pipeline="mask_measure",
            watch_measure_settings=str(settings_file),
            watch_settle_seconds=0.3, watch_poll_seconds=0.05,
            watch_idle_minutes=4.0 / 60.0))
    finally:
        writer.join()

    assert len(result["done"]) == len(wells) and not result["failed"]
    combined = watched / "spacr_watch"
    batch_merged = sorted(p for p in os.listdir(batch / "merged")
                          if p.endswith(".npy"))
    watch_merged = sorted(p for p in os.listdir(combined / "merged")
                          if p.endswith(".npy"))
    assert watch_merged == batch_merged and len(batch_merged) == len(wells)
    for name in batch_merged:
        np.testing.assert_array_equal(np.load(combined / "merged" / name),
                                      np.load(batch / "merged" / name))
    for table in ("cell", "nucleus"):
        batch_rows = _rows(batch / "measurements" / "measurements.db", table)
        watch_rows = _rows(combined / "measurements" / "measurements.db",
                           table)
        assert watch_rows == batch_rows and len(batch_rows[1]) == 2 * len(wells)
    fields = _ledger(watched)
    for entry in fields.values():
        last = max(arrivals[name] for name in entry["files"])
        assert entry["started"] - last <= (0.3 + 0.05 + 1.0) + sum(
            other["seconds"] for other in fields.values()
            if other["started"] < entry["started"])
