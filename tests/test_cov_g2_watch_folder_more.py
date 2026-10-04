"""More watch-folder and microscope edges: unstable files, failed steps, feedback."""
from __future__ import annotations

import os
import sqlite3
import sys
import types

import numpy as np
import pytest

from spacr import core
from tests.test_cov_g2_microscope_edges import _microscope


def _map(tmp_path):
    from spacr.convert import MAP_FILENAME

    path = tmp_path / MAP_FILENAME
    path.write_text("target,source\n")
    return path


def test_a_map_replaced_while_opened_is_refused(tmp_path, monkeypatch):
    path = _map(tmp_path)
    real = os.stat

    def moved(target, *a, **k):
        info = real(target, *a, **k)
        if str(target) == str(path) and not k.get("follow_symlinks", True):
            return os.stat_result((info.st_mode, info.st_ino + 1) + tuple(info)[2:])
        return info

    monkeypatch.setattr(core.os, "stat", moved)
    with pytest.raises(ValueError, match="not a stable regular file"):
        core._watch_map_bytes(str(tmp_path))


def test_a_map_that_grows_while_read_is_refused(tmp_path, monkeypatch):
    _map(tmp_path)
    real = os.fstat
    calls = []

    def growing(fd):
        info = real(fd)
        calls.append(1)
        if len(calls) == 2:
            values = list(info)
            values[6] += 1
            return os.stat_result(values)
        return info

    monkeypatch.setattr(core.os, "fstat", growing)
    with pytest.raises(ValueError, match="changed while being read"):
        core._watch_map_bytes(str(tmp_path))


def test_a_field_whose_steps_leave_nothing_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "preprocess_generate_masks", lambda run: None)
    with pytest.raises(RuntimeError, match="no merged stack"):
        core._watch_analyse_field(str(tmp_path), {})
    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / "f.npy", np.zeros((4, 4, 2)))
    monkeypatch.setattr(core, "_overlay_candidates", lambda folder: ["f.npy"])
    import spacr.measure as measure

    monkeypatch.setattr(measure, "measure_crop", lambda settings: None)
    monkeypatch.setattr(core, "measure_crop", lambda settings: None, raising=False)
    with pytest.raises(RuntimeError, match="no measurements.db"):
        core._watch_analyse_field(str(tmp_path),
                                  {"watch_pipeline": "mask_measure"})


def test_a_field_that_cannot_be_appended_is_rolled_back(tmp_path):
    field = tmp_path / "field.db"
    with sqlite3.connect(field) as con:
        con.execute("CREATE TABLE cells (a INTEGER)")
        con.execute("INSERT INTO cells VALUES (1)")
    combined = tmp_path / "out" / "measurements.db"
    combined.parent.mkdir()
    with sqlite3.connect(combined) as con:
        con.execute("CREATE TABLE base (a INTEGER)")
        con.execute("CREATE VIEW cells AS SELECT a FROM base")
    with pytest.raises(sqlite3.OperationalError):
        core._watch_merge_database(str(field), str(combined), "k1")
    with sqlite3.connect(combined) as con:
        assert not con.execute("SELECT * FROM spacr_watch_fields").fetchall()


def test_an_artifact_that_grows_or_changes_while_hashed_is_refused(
        tmp_path, monkeypatch):
    path = tmp_path / "a.csv"
    path.write_bytes(b"abc")
    real_read = os.read

    def longer(fd, n):
        return real_read(fd, n) + b"!"

    monkeypatch.setattr(core.os, "read", longer)
    with pytest.raises(ValueError, match="grew while hashing"):
        core._watch_artifact_sha256(str(path))
    monkeypatch.setattr(core.os, "read", real_read)
    real_identity = core._watch_file_identity
    calls = []

    def drifting(target):
        calls.append(1)
        identity = real_identity(target)
        return identity if len(calls) == 1 else identity[:4] + [0]

    monkeypatch.setattr(core, "_watch_file_identity", drifting)
    with pytest.raises(ValueError, match="changed while hashing"):
        core._watch_artifact_sha256(str(path))


def test_snapshots_of_mismatched_unreadable_or_short_sources_are_none(
        tmp_path, monkeypatch):
    source = tmp_path / "a.tif"
    source.write_bytes(b"abcd")
    expected = core._watch_file_identity(str(source))
    fake = expected[:2] + [expected[2] + 1] + expected[3:]
    monkeypatch.setattr(core, "_watch_file_identity", lambda p: fake)
    assert core._watch_copy_snapshot(str(source), str(tmp_path / "b"), fake) is None
    monkeypatch.undo()

    def refuse(fd, n):
        raise OSError("io error")

    monkeypatch.setattr(core.os, "read", refuse)
    assert core._watch_copy_snapshot(str(source), str(tmp_path / "c"),
                                     expected) is None
    monkeypatch.undo()
    monkeypatch.setattr(core.os.path, "getsize", lambda p: 0)
    assert core._watch_copy_snapshot(str(source), str(tmp_path / "d"),
                                     expected) is None


def test_deferring_skips_files_never_seen(tmp_path):
    field_dir = tmp_path / "stage"
    field_dir.mkdir()
    context = {"seen": {}, "ledger": {"fields": {"k": {}}},
               "ledger_path": str(tmp_path / "ledger.json")}
    core._watch_defer_snapshot("k", [("gone.tif", "1")], str(field_dir), context)
    assert context["ledger"]["fields"]["k"]["status"] == "waiting"


def test_a_vanished_unseen_file_is_not_a_change(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "_watch_images", lambda src: ["ghost.tif"])
    context = {"seen": {}, "src": str(tmp_path)}
    assert not core._watch_observe(context, 0.0)


def test_a_stage_far_from_every_field_sees_nothing(tmp_path):
    scope = _microscope(tmp_path, {"plate1_A01_0001_001": (0.0, 0.0)})
    scope.set_xy_position(1000.0, 1000.0)
    scope.snap_image()
    assert scope.get_image().max() == 0


def test_pycromanager_opens_its_core(monkeypatch, tmp_path):
    module = types.ModuleType("pycromanager")
    module.Core = lambda: "core"
    monkeypatch.setitem(sys.modules, "pycromanager", module)
    assert core._microscope_open({"microscope_driver": "pycromanager"},
                                 str(tmp_path)) == "core"


def test_failed_acquisitions_and_feedback_are_recorded(tmp_path, monkeypatch,
                                                       capsys):
    from spacr.cancellation import PipelineCancelled

    event = {"id": "e1", "status": "queued", "stage": (1.0, 2.0)}
    context = {"ledger": {"fields": {"k": {"events": [event], "finished": 1}}},
               "ledger_path": str(tmp_path / "ledger.json")}

    def broken(event, context):
        raise RuntimeError("stage stuck")

    monkeypatch.setattr(core, "_microscope_acquire", broken)
    assert core._microscope_drain(context) == 0
    assert event["status"] == "failed"
    event["status"] = "queued"

    def cancelled(event, context):
        raise PipelineCancelled()

    monkeypatch.setattr(core, "_microscope_acquire", cancelled)
    with pytest.raises(PipelineCancelled):
        core._microscope_drain(context)
    monkeypatch.setattr(core, "_microscope_queue", lambda key, field_dir, c: 1)
    with pytest.raises(PipelineCancelled):
        core._microscope_feedback("k", str(tmp_path), context)

    def queue_fails(key, field_dir, context):
        raise ValueError("no measurements")

    monkeypatch.setattr(core, "_microscope_queue", queue_fails)
    core._microscope_feedback("k", str(tmp_path), context)
    assert "no measurements" in context["ledger"]["fields"]["k"]["feedback_error"]
    assert "microscope feedback for k failed" in capsys.readouterr().out
