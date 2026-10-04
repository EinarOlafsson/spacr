"""The refusals of measured lineage colouring, one guard at a time."""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import _lineage_measurements as lm
from tests.test_lineage_measurements_f537 import project  # noqa: F401


def _write(path, frame):
    frame.to_csv(path, index=False)
    return path


def test_a_failed_json_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / "out.json"

    def refuse(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(lm.os, "replace", refuse)
    with pytest.raises(OSError):
        lm._atomic_json(target, {"a": 1})
    assert list(tmp_path.iterdir()) == []


def test_an_oversized_tracks_table_is_refused(tmp_path, monkeypatch):
    path = _write(tmp_path / "t.csv", pd.DataFrame({"frame": [0], "track_id": [1]}))
    real_open = open

    class _Huge:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size):
            return b"x" * size

    monkeypatch.setattr("builtins.open", lambda p, mode="r", *a, **k: (
        _Huge() if str(p) == str(path) and "b" in mode else real_open(p, mode, *a, **k)))
    with pytest.raises(ValueError, match="exceeds 128 MiB"):
        lm._tracks_snapshot(path)


@pytest.mark.parametrize("frame, message", [
    (pd.DataFrame({"frame": [0]}), "no track_id"),
    (pd.DataFrame({"frame": [0.5], "track_id": [1]}), "frame must contain"),
    (pd.DataFrame({"frame": [0, 0], "track_id": [1, 1]}), "duplicate"),
])
def test_a_malformed_tracks_table_is_refused(tmp_path, frame, message):
    with pytest.raises(ValueError, match=message):
        lm._tracks_snapshot(_write(tmp_path / "t.csv", frame))


def test_source_names_must_be_basenames_of_one_field():
    with pytest.raises(ValueError, match="basenames"):
        lm._source_fields(["a/plate1_A01_1_100.npy"])
    with pytest.raises(ValueError, match="one imaging field"):
        lm._source_fields(["plate1_A01_1_100.npy", "plate1_A01_2_103.npy"])


def test_preparing_sources_checks_object_frames_and_labels(project):  # noqa: F811
    path, _db, _tracks, names, labels = project
    with pytest.raises(ValueError, match="cell, nucleus or pathogen"):
        lm._prepare_lineage_sources(path, "organelle", names, labels)
    with pytest.raises(ValueError, match="unmapped frame"):
        lm._prepare_lineage_sources(path, "cell", names[:-1], labels[:-1])


def test_an_oversized_mapping_is_refused(project, monkeypatch):  # noqa: F811
    path = project[0]
    manifest = str(path) + lm._SUFFIX
    real_open = lm.Path.open

    class _Huge:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size):
            return b" " * size

    monkeypatch.setattr(lm.Path, "open", lambda self, *a, **k: (
        _Huge() if str(self) == manifest else real_open(self, *a, **k)))
    with pytest.raises(ValueError, match="exceeds 32 MiB"):
        lm._load_lineage_sources(path)


def _rewrite(path, change):
    manifest = lm.Path(str(path) + lm._SUFFIX)
    data = json.loads(manifest.read_text())
    change(data)
    manifest.write_text(json.dumps(data))


def test_frames_that_are_not_a_list_are_refused(project):  # noqa: F811
    path = project[0]
    _rewrite(path, lambda d: d.__setitem__("frames", {"0": 1}))
    with pytest.raises(ValueError, match="frames must be a list"):
        lm._load_lineage_sources(path)


def test_a_mapping_with_a_duplicate_object_is_refused(project):  # noqa: F811
    path = project[0]

    def duplicate(data):
        frame = next(f for f in data["frames"] if f["labels"])
        frame["labels"].append(dict(frame["labels"][0]))

    _rewrite(path, duplicate)
    with pytest.raises(ValueError, match="duplicate object identities"):
        lm._load_lineage_sources(path)


def test_a_feature_named_like_an_identity_is_refused(project):  # noqa: F811
    path, db = project[0], project[1]
    with pytest.raises(ValueError, match="numeric measurement feature"):
        lm._measured_lineage_inputs(db, path, "object_label")


def test_measured_labels_must_be_positive_integers(project):  # noqa: F811
    path, db = project[0], project[1]
    with sqlite3.connect(db) as con:
        con.execute("UPDATE cell SET object_label = 0 WHERE rowid = 1")
    with pytest.raises(ValueError, match="positive integers"):
        lm._measured_lineage_inputs(db, path, "cell_intensity")


def test_measured_rows_that_repeat_an_object_are_refused(project):  # noqa: F811
    path, db = project[0], project[1]
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO cell SELECT * FROM cell WHERE rowid = 1")
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        lm._measured_lineage_inputs(db, path, "cell_intensity")


def test_a_distance_given_as_text_must_be_a_number():
    with pytest.raises(ValueError, match="finite number"):
        lm._measured_lineage_options(
            {"timelapse_lineage_max_distance": "far"}, None)
    assert lm._measured_lineage_options(
        {"timelapse_lineage_max_distance": 0}, None) == (None, 30.0)
    assert lm._measured_lineage_options(
        {"frame_interval_s": 60}, 60.0) == (60.0, 30.0)


def test_one_role_given_as_text_is_read_as_a_list(project, capsys):  # noqa: F811
    db = project[1]
    results = lm._run_measured_lineage_step(
        db, {"timelapse_objects": "nucleus", "save": False})
    assert results == []
    assert "no tracks tables found" in capsys.readouterr().out
    with pytest.raises(ValueError, match="cell, nucleus or pathogen"):
        lm._run_measured_lineage_step(db, {"timelapse_objects": ["organelle"]})


def test_segment_statistics_colour_without_reading_measurements(project):  # noqa: F811
    db = project[1]
    result, = lm._run_measured_lineage_step(
        db, {"timelapse_objects": ["cell"], "save": False,
             "timelapse_lineage_color_by": "generation_time"})
    assert "segments" in result


def test_a_manifest_for_another_object_is_reported(project, capsys):  # noqa: F811
    path, db = project[0], project[1]
    other = path.parent / path.name.replace("_cell_", "_nucleus_")
    other.write_bytes(path.read_bytes())
    lm.Path(str(other) + lm._SUFFIX).write_bytes(
        lm.Path(str(path) + lm._SUFFIX).read_bytes())
    lm._run_measured_lineage_step(
        db, {"timelapse_objects": ["nucleus"], "save": False,
             "timelapse_lineage_color_by": "generation_time"})
    assert "object type differs" in capsys.readouterr().out


def test_frames_that_are_not_records_are_malformed(project):  # noqa: F811
    path = project[0]
    _rewrite(path, lambda d: d.__setitem__("frames", [1, 2]))
    with pytest.raises(ValueError, match="malformed lineage source mapping"):
        lm._load_lineage_sources(path)
