"""Convert's barcode linkage refuses every input it cannot trust."""
from __future__ import annotations

import os
import types
from pathlib import Path

import pandas as pd
import pytest

from spacr import convert as cv
from spacr.errors import ConfigurationError
from tests.test_convert_barcode_f583 import acquisition  # noqa: F401


def test_a_sidecar_that_is_not_a_regular_file_is_refused(tmp_path):
    fifo = tmp_path / "barcode.txt"
    os.mkfifo(fifo)
    with pytest.raises(ConfigurationError, match="stable regular file"):
        cv._conversion_barcode_sidecar_bytes(str(fifo))


def test_a_sidecar_that_grows_while_read_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "barcode.txt"
    path.write_text("00123")
    real = os.fstat
    calls = []

    def growing(fd):
        result = real(fd)
        calls.append(1)
        if len(calls) == 2:
            return os.stat_result((result.st_mode, result.st_ino, result.st_dev,
                                   result.st_nlink, result.st_uid, result.st_gid,
                                   result.st_size + 1, 0, 0, 0))
        return result

    monkeypatch.setattr(cv.os, "fstat", growing)
    with pytest.raises(ConfigurationError, match="changed while being read"):
        cv._conversion_barcode_sidecar_bytes(str(path))


def _plan(tmp_path, *, layout="well", source=None, outputs=("plate1",),
          relative=("A01", "f.tif")):
    root = tmp_path / "raw"
    image = Path(source) if source else root.joinpath(*relative)
    sources = [types.SimpleNamespace(path=str(image), meta={"layout": layout})]
    mappings = [types.SimpleNamespace(source_plate="raw", plate=plate,
                                      source=str(image)) for plate in outputs]
    return types.SimpleNamespace(sources=sources, mappings=mappings), root


@pytest.mark.parametrize("kwargs, message", [
    ({"source": "/elsewhere/f.tif"}, "inside the source tree"),
    ({"layout": "plate_well", "relative": ("f.tif",)}, "identify this source plate"),
    ({"layout": None}, "unambiguous scanned source layout"),
    ({"outputs": ("plate1", "plate2")}, "multiple output plate identities"),
])
def test_discovery_refuses_what_it_cannot_place(tmp_path, kwargs, message):
    plan, root = _plan(tmp_path, **kwargs)
    root.mkdir()
    with pytest.raises(ConfigurationError, match=message):
        cv._discover_conversion_barcodes(plan, root, {})


def test_inputs_inside_the_destination_are_refused(acquisition, tmp_path):  # noqa: F811
    inside = Path(acquisition["dst"]) / "own.csv"
    inside.parent.mkdir(parents=True)
    inside.write_text(Path(acquisition["profiling_metadata"]).read_text())
    with pytest.raises(ConfigurationError, match="outside the destination"):
        cv.convert_folder(acquisition, profiling_metadata=str(inside))


def test_a_database_inside_the_bundle_is_refused(acquisition):  # noqa: F811
    bundle = Path(acquisition["dst"]) / "plate_barcode_linkage" / "m.db"
    with pytest.raises(ConfigurationError, match="plate barcode bundle"):
        cv.convert_folder(acquisition, db_path=str(bundle))


def test_a_source_that_changes_during_preflight_is_refused(acquisition,  # noqa: F811
                                                          monkeypatch):
    real = cv._barcode_input_bytes
    reads = []

    def drifting(path):
        data = real(path)
        if path == str(Path(acquisition["plate_barcode_source"]).resolve()):
            reads.append(1)
            if len(reads) == 2:
                return data + b"\n"
        return data

    monkeypatch.setattr(cv, "_barcode_input_bytes", drifting)
    with pytest.raises(ConfigurationError, match="changed during preflight"):
        cv.convert_folder(acquisition)


def test_an_unreadable_records_csv_is_refused(acquisition, monkeypatch):  # noqa: F811
    def broken(*a, **k):
        raise ValueError("bad bytes")

    monkeypatch.setattr(cv.pd, "read_csv", broken)
    with pytest.raises(ConfigurationError, match="Cannot read plate barcode CSV"):
        cv.convert_folder(acquisition)


def test_a_plan_with_no_wells_has_nothing_to_link(tmp_path):
    records = tmp_path / "r.csv"
    records.write_text("barcode,well\n1,A01\n")
    plan = types.SimpleNamespace(sources=[], mappings=[])
    with pytest.raises(ConfigurationError, match="No planned image wells"):
        cv._prepare_conversion_barcodes(
            {"plate_barcode_source": str(records)}, plan, tmp_path / "raw",
            tmp_path / "out")


def test_an_input_changed_while_linking_is_refused(acquisition, monkeypatch):  # noqa: F811
    import spacr.plate_qc as plate_qc

    real = plate_qc._link_barcode_wells

    def link_then_edit(*args, **kwargs):
        result = real(*args, **kwargs)
        own = Path(acquisition["profiling_metadata"])
        own.write_text(own.read_text() + "plate1,A02,DMSO,2\n")
        return result

    monkeypatch.setattr(plate_qc, "_link_barcode_wells", link_then_edit)
    with pytest.raises(ConfigurationError, match="input changed during preflight"):
        cv.convert_folder(acquisition)


def test_cleanup_of_a_failed_bundle_tolerates_stuck_files(acquisition,  # noqa: F811
                                                          monkeypatch):
    def failed(*args, **kwargs):
        raise OSError("publication failed")

    real_unlink = Path.unlink

    def stuck_unlink(self, missing_ok=False):
        if self.parent.name == "plate_barcode_linkage":
            raise OSError("busy")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(cv.os, "link", failed)
    monkeypatch.setattr(Path, "unlink", stuck_unlink)
    with pytest.raises(OSError, match="publication failed"):
        cv.convert_folder(acquisition)
    assert (Path(acquisition["dst"]) / "plate_barcode_linkage").exists()
    assert pd


def test_a_database_beside_the_bundle_is_accepted(acquisition):  # noqa: F811
    db = Path(acquisition["dst"]) / "tables" / "m.db"
    result = cv.convert_folder(acquisition, db_path=str(db))
    assert (Path(result.dst) / "plate_barcode_linkage" / "complete.json").exists()
