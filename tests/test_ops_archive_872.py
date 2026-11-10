"""Per-well copies to archive storage with verification and done markers (item 872, B9).

A synthetic well is segmented and measured with ``ops_archive_root`` set:
its files must land in the archive under the plate's name with a
``WELL_DONE`` marker that lists each file's size and MD5, and the plate
gets ``PLATE_DONE``. A rerun must resume from the marker without measuring
anything again; a truncated archive file must be found and copied again; a
copy that fails its check must leave every local file in place and write no
marker.
"""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("tifffile")

from spacr import ops_engine, ops_run_infra                          # noqa: E402
from tests.test_ops_phenotype_run_v2 import (                        # noqa: E402,F401
    PLATE, SETTINGS, _placed, _write_tiles, models)

PHASES = ("ph_segment", "ph_measure")


@pytest.fixture
def plate(tmp_path, models):
    """Tiles and placement of a two-field well."""
    _write_tiles(tmp_path / "tiles")
    (tmp_path / "run").mkdir()
    _placed(str(tmp_path / "run" / "measurements.db"))
    return tmp_path


def _run(root, **extra):
    """``run_ops`` with an archive under ``root``."""
    settings = {**SETTINGS, "dst_root": str(root / "run"),
                "phenotype_source": str(root / "tiles"),
                "ops_archive_root": str(root / "archive"), **extra}
    return ops_engine.run_ops(settings, phases=PHASES)


def _marker(root):
    """The well's WELL_DONE record."""
    with open(root / "archive" / PLATE / "A1" / "WELL_DONE",
              encoding="utf-8") as handle:
        return json.load(handle)


def test_a_finished_well_is_copied_verified_and_marked(plate):
    out = _run(plate)
    archive = out["wells"]["A1"]["archive"]
    record = _marker(plate)
    names = sorted(entry["path"] for entry in record["files"])
    assert "ops_phenotype_cells_A1.parquet" in names
    assert os.path.join("A1", "phenotype_masks", "site-0.npz") in names
    assert os.path.join("A1", "ops_report.json") in names
    assert archive["files"] == len(names) and archive["recopied"] == []
    assert record["phases"] == ["ph_measure", "ph_segment"]
    for entry in record["files"]:
        copy = plate / "archive" / PLATE / entry["path"]
        assert ops_run_infra.file_md5(str(copy)) == entry["md5"]
        assert ops_run_infra.file_md5(str(plate / "run" / entry["path"])) == \
            entry["md5"]
    assert out["archive_plate_done"].endswith(os.path.join(PLATE, "PLATE_DONE"))
    assert archive["deleted_local"] == 0


def test_a_rerun_resumes_from_the_marker_without_measuring_again(
        plate, monkeypatch):
    first = _run(plate)
    monkeypatch.setattr(ops_engine, "_ph_fields", lambda *a, **k: pytest.fail(
        "the archived well was measured again"))
    again = _run(plate)
    report = again["wells"]["A1"]
    assert report["archive"]["resumed"] is True
    assert report["ph_fields"]["ph_measure"]["cells"] == \
        first["wells"]["A1"]["ph_fields"]["ph_measure"]["cells"]


def test_a_marker_of_fewer_phases_does_not_skip_the_well(plate):
    _run(plate)
    settings = {**SETTINGS, "dst_root": str(plate / "run"),
                "phenotype_source": str(plate / "tiles"),
                "ops_archive_root": str(plate / "archive")}
    out = ops_engine.run_ops(settings, phases=(*PHASES, "ph_masks"))
    assert "resumed" not in out["wells"]["A1"]["archive"]
    assert _marker(plate)["phases"] == ["ph_masks", "ph_measure", "ph_segment"]


def test_a_truncated_copy_is_found_and_copied_again(plate, monkeypatch):
    _run(plate)
    damaged = plate / "archive" / PLATE / "A1" / "phenotype_masks" / "site-1.npz"
    data = damaged.read_bytes()
    damaged.write_bytes(data[:len(data) // 2])
    monkeypatch.setattr(ops_engine, "_ph_fields", lambda *a, **k: pytest.fail(
        "a damaged copy with its local files was measured again"))
    out = _run(plate)
    archive = out["wells"]["A1"]["archive"]
    assert archive["damaged"] == [os.path.join("A1", "phenotype_masks",
                                               "site-1.npz")]
    assert damaged.read_bytes() == data
    assert ops_run_infra.archived_well(str(plate / "archive"), PLATE, "A1",
                                       phases=PHASES)["ok"]


def test_a_damaged_copy_whose_local_files_are_gone_runs_the_well(plate):
    _run(plate, ops_archive_delete_local=True)
    assert not (plate / "run" / "A1").exists()
    assert not (plate / "run" / "ops_phenotype_cells_A1.parquet").exists()
    (plate / "archive" / PLATE / "A1" / "ops_report.json").write_text("{}")
    out = _run(plate)
    assert "resumed" not in out["wells"]["A1"]["archive"]
    assert out["wells"]["A1"]["ph_fields"]["ph_segment"]["segmented"] == 2


def test_nothing_local_is_deleted_before_the_copy_verifies(plate, monkeypatch):
    def corrupt(source, target, verify, expected):
        """Write a truncated copy, as a failing network write would."""
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(source, "rb") as handle:
            data = handle.read()
        with open(target, "wb") as handle:
            handle.write(data[:-1])
        return ops_run_infra._matches(target, verify, expected)

    monkeypatch.setattr(ops_run_infra, "_copy_verified", corrupt)
    with pytest.raises(OSError, match="does not match the local file"):
        _run(plate, ops_archive_delete_local=True)
    assert (plate / "run" / "ops_phenotype_cells_A1.parquet").is_file()
    assert (plate / "run" / "A1" / "phenotype_masks" / "site-0.npz").is_file()
    assert not (plate / "archive" / PLATE / "A1" / "WELL_DONE").exists()


def test_a_copy_that_fails_once_is_copied_again(tmp_path):
    run = tmp_path / "run"
    (run / "A1").mkdir(parents=True)
    (run / "A1" / "x.bin").write_bytes(b"abc" * 100)
    (run / "ops_thing_A1.parquet").write_bytes(b"p")
    (run / "ops_thing_A10.parquet").write_bytes(b"q")
    (run / "ops_thing_A1.parquet.part").write_bytes(b"r")
    tries = []

    def flaky(source, target, verify, expected):
        """Fail the first copy of each file."""
        tries.append(source)
        if tries.count(source) == 1:
            return False
        return ops_run_infra._copy_verified(source, target, verify, expected)

    out = ops_run_infra.archive_well(str(run), str(tmp_path / "arc"), "P", "A1",
                                     phases=("ph_segment",), verify="size",
                                     copy=flaky, delete_local=True)
    assert sorted(out["recopied"]) == [os.path.join("A1", "x.bin"),
                                       "ops_thing_A1.parquet"]
    assert out["deleted_local"] == 2 and not (run / "A1").exists()
    assert (run / "ops_thing_A10.parquet").exists()
    assert ops_run_infra.archived_well(str(tmp_path / "arc"), "P", "A1",
                                       phases=("ph_segment",))["ok"]
    assert ops_run_infra.archived_well(str(tmp_path / "arc"), "P", "A2",
                                       phases=()) is None
    assert ops_run_infra.mark_plate_done(str(tmp_path / "arc"), "P",
                                         ["A1", "A2"]) is None
    with pytest.raises(ValueError, match="md5 or size"):
        ops_run_infra.archive_well(str(run), str(tmp_path / "arc"), "P", "A1",
                                   phases=(), verify="crc")
    with pytest.raises(ValueError, match="md5 or size"):
        ops_engine._ph_archive_verify({"ops_archive_verify": "crc"})
