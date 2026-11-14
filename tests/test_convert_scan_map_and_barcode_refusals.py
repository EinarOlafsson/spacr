"""Table-driven tests of ``spacr.convert``: scanning, the map table and barcode linkage refusals.

* ``scan`` + ``plan`` number channels per plate (``C01`` names one stain
  inside a plate and is free to name another on the next) and take a CZI's
  scene axis apart only when the file has one;
* ``populate_db_from_map`` loads maps with and without ``prcf`` and replaces
  the previous generation of rows instead of appending to it;
* ``convert_folder`` with barcode linkage refuses every input it cannot trust
  (inputs inside the destination, a database inside the bundle, inputs that
  drift during preflight, an unreadable records CSV), leaves stuck files in
  place when publication fails, and accepts a database beside the bundle.

Everything is synthetic and offline; CZI files are served by a ``czifile``
stand-in.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import convert as cv
from spacr.errors import ConfigurationError


def _write(path, value=1, shape=(6, 6)):
    """Write a synthetic TIFF whose pixels say which file it is."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, np.full(shape, value, np.uint16))
    return path


# ---------------------------------------------------------------------------
# scan: per-plate channel numbering and CZI scenes
# ---------------------------------------------------------------------------

# (case id, {plate: (condition, channels)}, channel_map, plate_map, targets)
_NUMBERING_CASES = [
    ("one plate numbers from one", {"run1": ("wt", (1, 2, 3))},
     {("run1", "C1"): 1, ("run1", "C2"): 2, ("run1", "C3"): 3},
     {"run1": "plate1"},
     ["plate1_A01_T0001F001L01A01Z01C01.tif",
      "plate1_A01_T0001F001L01A01Z01C02.tif",
      "plate1_A01_T0001F001L01A01Z01C03.tif"]),
    ("numbering restarts on every plate",
     {"runA": ("wt", (1, 2)), "runB": ("ko", (2,))},
     {("runA", "C1"): 1, ("runA", "C2"): 2, ("runB", "C2"): 1},
     {"runA": "plate1", "runB": "plate2"},
     ["plate1_A01_T0001F001L01A01Z01C01.tif",
      "plate1_A01_T0001F001L01A01Z01C02.tif",
      "plate2_A01_T0001F001L01A01Z01C01.tif"]),
]


def test_plan_of_a_scan_numbers_channels_per_plate(tmp_path, subtests):
    """Channel ids restart at 1 on each plate; plates get their own tokens and targets."""
    for index, (case_id, plates, channel_map, plate_map, targets) in enumerate(_NUMBERING_CASES):
        with subtests.test(msg=case_id):
            root = tmp_path / f"src{index}"
            for plate, (condition, channels) in plates.items():
                for channel in channels:
                    _write(str(root / plate / condition / f"fov01_C{channel}.tif"),
                           value=10 + channel)
            conversion_plan = cv.plan(cv.scan(str(root)))
            assert conversion_plan.channel_map == channel_map
            assert conversion_plan.plate_map == plate_map
            assert sorted(m.target for m in conversion_plan.mappings) == targets


def _fake_czifile(array, axes):
    """A ``czifile`` module stand-in serving ``array`` under ``axes``."""

    class FakeCziFile:
        def __init__(self, path):
            self.shape, self.axes, self.dtype = array.shape, axes, array.dtype

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def asarray(self):
            return array

    module = types.ModuleType("czifile")
    module.CziFile = FakeCziFile
    return module


def _czi_without_scenes():
    array = np.zeros((2, 3, 6, 6), np.uint16)  # C Z Y X
    for c in range(2):
        for z in range(3):
            array[c, z] = 10 * c + z
    return array


def _czi_with_scenes():
    array = np.zeros((2, 2, 6, 6), np.uint16)  # S C Y X
    for s in range(2):
        for c in range(2):
            array[s, c] = 100 * s + c
    return array


# (case id, array, axes, sources, (t, z, n_channels) of the first source, written, {target: pixel})
_CZI_CASES = [
    ("no scene axis keeps every plane", _czi_without_scenes(), "CZYX", 1, (1, 3, 2), 6,
     {"plate1_A01_T0001F001L01A01Z01C01.tif": 0,
      "plate1_A01_T0001F001L01A01Z03C01.tif": 2,
      "plate1_A01_T0001F001L01A01Z03C02.tif": 12}),
    ("scene axis splits into one source per scene", _czi_with_scenes(), "SCYX", 2,
     (1, 1, 2), 4,
     {"plate1_A01_T0001F001L01A01Z01C01.tif": 0,
      "plate1_A01_T0001F001L01A01Z01C02.tif": 1,
      "plate1_A01_T0001F002L01A01Z01C01.tif": 100,
      "plate1_A01_T0001F002L01A01Z01C02.tif": 101}),
]


def test_scan_of_a_czi_splits_scenes_only_when_the_file_has_them(tmp_path, monkeypatch, subtests):
    """A CZI without ``S`` is one source with all its planes; with ``S`` one source per scene."""
    for index, (case_id, array, axes, n_sources, tzc, n_written, pixels) in enumerate(_CZI_CASES):
        with subtests.test(msg=case_id):
            monkeypatch.setitem(sys.modules, "czifile", _fake_czifile(array, axes))
            root = tmp_path / f"src{index}"
            root.mkdir()
            (root / "scan.czi").write_bytes(b"czi")
            sources = cv.scan(str(root))
            assert len(sources) == n_sources
            assert (sources[0].t, sources[0].z, sources[0].n_channels) == tzc
            out = tmp_path / f"out{index}"
            result = cv.convert(cv.plan(sources), str(out))
            assert result.n_written == n_written
            for target, value in pixels.items():
                plane = tifffile.imread(str(out / target))
                assert plane.shape == (6, 6)
                assert int(plane.min()) == int(plane.max()) == value


# ---------------------------------------------------------------------------
# populate_db_from_map
# ---------------------------------------------------------------------------

def _row(channel=1, source="/data/run1/wt/fov01_C1.tif", prcf=None):
    row = {"target": f"plate1_A01_T0001F001L01A01Z01C{channel:02d}.tif",
           "source": source, "plate": "plate1", "well": "A01",
           "field": 1, "channel": channel, "z": 1, "t": 1}
    if prcf is not None:
        row["prcf"] = prcf
    return row


def _db_state(db_path):
    """(index names, rows ordered by channel) of the conversion table."""
    connection = sqlite3.connect(str(db_path))
    try:
        indexes = sorted(name for (name,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index'") if name)
        frame = pd.read_sql(
            f"SELECT * FROM {cv.CONVERSION_TABLE} ORDER BY channel", connection)
    finally:
        connection.close()
    return indexes, frame


_T = cv.CONVERSION_TABLE
# (case id, generations of rows written one after another, final index names)
_MAP_CASES = [
    ("map without prcf loads and indexes only target", [[_row()]],
     [f"idx_{_T}_target"]),
    ("map with prcf also indexes prcf", [[_row(prcf="plate1_A_01_1")]],
     [f"idx_{_T}_prcf", f"idx_{_T}_target"]),
    ("repopulating replaces the previous generation",
     [[_row()], [_row(1, "/data/a.tif"), _row(2, "/data/b.tif")]],
     [f"idx_{_T}_target"]),
]


def test_populate_db_from_map_loads_exactly_the_latest_map(tmp_path, subtests):
    """Each call returns its row count and leaves only that map's rows, indexed by the columns present."""
    for index, (case_id, generations, want_indexes) in enumerate(_MAP_CASES):
        with subtests.test(msg=case_id):
            path = tmp_path / f"map{index}.csv"
            db = tmp_path / f"m{index}.db"
            for rows in generations:
                pd.DataFrame(rows).to_csv(path, index=False)
                assert cv.populate_db_from_map(str(db), str(path)) == len(rows)
            indexes, frame = _db_state(db)
            assert [i for i in indexes if i.startswith(f"idx_{_T}_")] == want_indexes
            last = generations[-1]
            assert frame["target"].tolist() == [r["target"] for r in last]
            assert frame["source"].tolist() == [r["source"] for r in last]
            assert frame["channel"].tolist() == [r["channel"] for r in last]
            assert ("prcf" in frame.columns) == ("prcf" in last[0])


# ---------------------------------------------------------------------------
# convert_folder with barcode linkage
# ---------------------------------------------------------------------------

def _acquisition(root):
    """Two TIFF wells, a zero-prefixed barcode records CSV and an own plate map."""
    raw = root / "raw"
    for well in ("A01", "A02"):
        (raw / well).mkdir(parents=True)
        tifffile.imwrite(raw / well / "field01_C1.tif",
                         np.arange(64, dtype=np.uint16).reshape(8, 8))
    records = root / "records.csv"
    records.write_text("barcode,well,strain,compound,concentration\n"
                       "00123,A01,RH,DMSO,0\n00123,A03,ME49,drug,5\n", encoding="utf-8")
    own = root / "own.csv"
    own.write_text("plateID,well,compound,concentration\nplate1,A01,DMSO,1\n",
                   encoding="utf-8")
    return {"src": str(raw), "dst": str(root / "converted"),
            "plate_barcode_source": str(records),
            "plate_barcodes": {"plate1": "00123"},
            "profiling_metadata": str(own), "preview_rows": 0}


def _inside_destination(settings, mp):
    inside = Path(settings["dst"]) / "own.csv"
    inside.parent.mkdir(parents=True)
    inside.write_text(Path(settings["profiling_metadata"]).read_text(encoding="utf-8"),
                      encoding="utf-8")
    return {"profiling_metadata": str(inside)}, (
        f"Plate barcode inputs must be outside the destination: {inside.resolve()}")


def _db_in_bundle(settings, mp):
    return {"db_path": str(Path(settings["dst"]) / "plate_barcode_linkage" / "m.db")}, (
        "db_path must not write into the plate barcode bundle.")


def _source_drifts(settings, mp):
    real, reads = cv._barcode_input_bytes, []
    source = str(Path(settings["plate_barcode_source"]).resolve())

    def drifting(path):
        data = real(path)
        if path == source:
            reads.append(1)
            if len(reads) == 2:
                return data + b"\n"
        return data

    mp.setattr(cv, "_barcode_input_bytes", drifting)
    return {}, "Plate barcode source changed during preflight."


def _records_unreadable(settings, mp):
    def broken(*args, **kwargs):
        raise ValueError("bad bytes")

    mp.setattr(cv.pd, "read_csv", broken)
    return {}, "Cannot read plate barcode CSV: bad bytes"


def _own_map_edited_while_linking(settings, mp):
    import spacr.plate_qc as plate_qc

    real = plate_qc._link_barcode_wells
    own = Path(settings["profiling_metadata"])

    def link_then_edit(*args, **kwargs):
        result = real(*args, **kwargs)
        own.write_text(own.read_text(encoding="utf-8") + "plate1,A02,DMSO,2\n",
                       encoding="utf-8")
        return result

    mp.setattr(plate_qc, "_link_barcode_wells", link_then_edit)
    return {}, f"Plate barcode input changed during preflight: {own.resolve()}"


_REFUSALS = [
    ("an input inside the destination", _inside_destination),
    ("a database inside the bundle", _db_in_bundle),
    ("the source changes during preflight", _source_drifts),
    ("an unreadable records CSV", _records_unreadable),
    ("an input changed while linking", _own_map_edited_while_linking),
]


def test_convert_folder_refuses_untrusted_barcode_inputs_before_any_bundle(tmp_path, subtests):
    """Every untrusted barcode input raises ConfigurationError with its exact message and no bundle."""
    for index, (case_id, arrange) in enumerate(_REFUSALS):
        with subtests.test(msg=case_id), pytest.MonkeyPatch.context() as mp:
            settings = _acquisition(tmp_path / f"case{index}")
            extra, message = arrange(settings, mp)
            with pytest.raises(ConfigurationError) as raised:
                cv.convert_folder(settings, **extra)
            assert str(raised.value) == message
            assert not (Path(settings["dst"]) / "plate_barcode_linkage").exists()


def test_convert_folder_publishes_or_leaves_stuck_files_of_the_barcode_bundle(tmp_path, monkeypatch):
    """A database beside the bundle is accepted; a failed publication keeps files it cannot unlink."""
    settings = _acquisition(tmp_path / "beside")
    db = Path(settings["dst"]) / "tables" / "m.db"
    result = cv.convert_folder(settings, db_path=str(db))
    bundle = Path(result.dst) / "plate_barcode_linkage"
    assert sorted(p.name for p in bundle.iterdir()) == [
        "complete.json", "plate_barcode_mismatches.csv", "plate_map_lims.csv"]
    receipt = json.loads((bundle / "complete.json").read_text(encoding="utf-8"))
    assert (receipt["complete"], receipt["linked_wells"]) == (True, 1)
    assert db.exists()

    def failed(*args, **kwargs):
        raise OSError("publication failed")

    real_unlink = Path.unlink

    def stuck_unlink(self, missing_ok=False):
        if self.parent.name == "plate_barcode_linkage":
            raise OSError("busy")
        return real_unlink(self, missing_ok=missing_ok)

    settings = _acquisition(tmp_path / "stuck")
    monkeypatch.setattr(cv.os, "link", failed)
    monkeypatch.setattr(Path, "unlink", stuck_unlink)
    with pytest.raises(OSError, match="^publication failed$"):
        cv.convert_folder(settings)
    bundle = Path(settings["dst"]) / "plate_barcode_linkage"
    assert sorted(p.name for p in bundle.iterdir()) == [
        ".complete.json.pending", "plate_barcode_mismatches.csv", "plate_map_lims.csv"]
