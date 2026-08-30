"""Conversion branches that only a second plate, a one-scene file, a
half-written field or an older map file ever take.

Each test here pins a decision :mod:`spacr.convert` makes silently on the
happy path and that only shows up when the input is slightly unusual:

* channel ids are handed out **per plate**, so ``C01`` names the same stain
  in every well of one plate and is free to name a different one on the next;
* a vendor file that records no scene axis must not be indexed as though it
  had one — the whole stack has to survive the read;
* when a write dies halfway through a field, the planes already on disk are
  reported as written, not swept into the failed list along with the rest;
* a map file written before ``prcf`` existed still loads into the database,
  because the index is an optimisation and not a schema requirement.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import convert as cv


def _write(path, value=1, shape=(6, 6), **kwargs):
    """Write a synthetic TIFF whose pixels say which file it is."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, np.full(shape, value, np.uint16), **kwargs)
    return path


def _indexes(db_path):
    """Return the index names sqlite actually holds for the given database."""
    connection = sqlite3.connect(str(db_path))
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index'").fetchall()
    finally:
        connection.close()
    return sorted(name for (name,) in rows if name is not None)


# ---------------------------------------------------------------------------
# Channel numbering is per plate
# ---------------------------------------------------------------------------

def test_channel_ids_restart_on_every_plate_so_c01_means_one_stain(tmp_path):
    """A two-plate run must number channels inside each plate separately.

    ``C01`` is what every downstream measurement joins on. If the channel
    ids were pooled across plates, the second plate's only stain would come
    out as ``C03`` on one run and ``C01`` on another depending on which
    plates happened to be in the folder, and two experiments analysed a week
    apart would silently disagree about which image is the nucleus channel.
    """
    root = tmp_path / "src"
    for channel in (1, 2):
        _write(str(root / "runA" / "wt" / f"fov01_C{channel}.tif"),
               value=10 + channel)
    _write(str(root / "runB" / "ko" / "fov01_C2.tif"), value=99)

    conversion_plan = cv.plan(cv.scan(str(root)))

    assert conversion_plan.channel_map == {
        ("runA", "C1"): 1, ("runA", "C2"): 2, ("runB", "C2"): 1,
    }, "channel ids were not restarted per plate"
    # The plate token, not the channel id, is what separates the two runs.
    assert conversion_plan.plate_map == {"runA": "plate1", "runB": "plate2"}
    targets = sorted(m.target for m in conversion_plan.mappings)
    assert targets == [
        "plate1_A01_T0001F001L01A01Z01C01.tif",
        "plate1_A01_T0001F001L01A01Z01C02.tif",
        "plate2_A01_T0001F001L01A01Z01C01.tif",
    ]


def test_one_plate_still_numbers_its_channels_from_one(tmp_path):
    """The per-plate loop must not need a second plate to hand out ids.

    This is the ordinary single-plate run: it shares the code path with the
    test above, and pinning it here is what proves the two-plate result came
    from restarting the numbering rather than from a different code path.
    """
    root = tmp_path / "src"
    for channel in (1, 2, 3):
        _write(str(root / "run1" / "wt" / f"fov01_C{channel}.tif"),
               value=channel)

    conversion_plan = cv.plan(cv.scan(str(root)))

    assert conversion_plan.channel_map == {
        ("run1", "C1"): 1, ("run1", "C2"): 2, ("run1", "C3"): 3,
    }
    assert len(conversion_plan.mappings) == 3


# ---------------------------------------------------------------------------
# A CZI that records no scene axis
# ---------------------------------------------------------------------------

def _install_fake_czifile(monkeypatch, array, axes):
    """Register a ``czifile`` stub that serves ``array`` under ``axes``."""

    class FakeCziFile:
        def __init__(self, path):
            self.shape = array.shape
            self.axes = axes
            self.dtype = array.dtype

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def asarray(self):
            return array

    module = types.ModuleType("czifile")
    module.CziFile = FakeCziFile
    monkeypatch.setitem(sys.modules, "czifile", module)
    return module


def test_a_czi_without_a_scene_axis_keeps_every_plane_it_has(
        tmp_path, monkeypatch):
    """A single-scene CZI must not be indexed along a scene axis it lacks.

    ``czifile`` omits ``S`` entirely for a file acquired without tiling.
    Taking ``series`` out of the first axis anyway would silently drop
    channels — a two-channel three-slice acquisition would arrive as one
    plane and nothing in the map would say a thing was lost.
    """
    array = np.zeros((2, 3, 6, 6), np.uint16)          # C Z Y X, no S
    for c in range(2):
        for z in range(3):
            array[c, z] = 10 * c + z
    _install_fake_czifile(monkeypatch, array, "CZYX")

    root = tmp_path / "src"
    root.mkdir(parents=True)
    (root / "scan.czi").write_bytes(b"czi")

    sources = cv.scan(str(root))
    assert len(sources) == 1, "a file with no S axis is one series"
    assert (sources[0].t, sources[0].z, sources[0].n_channels) == (1, 3, 2)

    out = tmp_path / "out"
    result = cv.convert(cv.plan(sources), str(out))

    assert result.n_written == 2 * 3
    # Z03/C02 is z index 2 of channel index 1 -> 10 * 1 + 2.
    value = tifffile.imread(
        str(out / "plate1_A01_T0001F001L01A01Z03C02.tif"))
    assert int(value[0, 0]) == 12
    assert int(tifffile.imread(
        str(out / "plate1_A01_T0001F001L01A01Z01C01.tif"))[0, 0]) == 0


def test_a_czi_with_a_scene_axis_is_split_into_one_source_per_scene(
        tmp_path, monkeypatch):
    """The scene axis, when present, still has to be taken apart.

    The counterpart to the test above: if the ``S`` handling were dropped
    altogether, six scenes would land on one field id and five sixths of the
    plate would be overwritten by the last scene read.
    """
    array = np.zeros((2, 2, 6, 6), np.uint16)          # S C Y X
    for s in range(2):
        for c in range(2):
            array[s, c] = 100 * s + c
    _install_fake_czifile(monkeypatch, array, "SCYX")

    root = tmp_path / "src"
    root.mkdir(parents=True)
    (root / "scan.czi").write_bytes(b"czi")

    sources = cv.scan(str(root))
    assert len(sources) == 2, "two scenes must scan as two sources"

    out = tmp_path / "out"
    result = cv.convert(cv.plan(sources), str(out))

    assert result.n_written == 2 * 2
    value = tifffile.imread(
        str(out / "plate1_A01_T0001F002L01A01Z01C02.tif"))
    assert int(value[0, 0]) == 100 * 1 + 1


# ---------------------------------------------------------------------------
# A write that dies halfway through a field
# ---------------------------------------------------------------------------

def test_a_plane_already_written_is_not_also_reported_as_failed(
        tmp_path, monkeypatch):
    """When a source dies mid-field, the planes already on disk stay written.

    The failure sweep walks the whole group for the source that raised. If
    it did not exclude the targets already in ``written``, the ledger would
    claim a file failed while a complete, readable TIFF of it sits in the
    output folder — and a user reconciling the map against the folder would
    find one more image than the run says it produced.
    """
    root = tmp_path / "src"
    stack = np.stack([np.full((6, 6), 10 + z, np.uint16) for z in range(4)])
    os.makedirs(str(root / "run1" / "wt"))
    tifffile.imwrite(str(root / "run1" / "wt" / "fov01.tif"), stack,
                     metadata={"axes": "ZYX"})

    conversion_plan = cv.plan(cv.scan(str(root)))
    assert len(conversion_plan.mappings) == 4, "a 4-plane stack is 4 targets"

    real_write = cv._atomic_write

    def _die_on_the_second_plane(path, array):
        if path.endswith("Z02C01.tif"):
            raise RuntimeError("the disk went away mid-field")
        real_write(path, array)

    monkeypatch.setattr(cv, "_atomic_write", _die_on_the_second_plane)

    out = tmp_path / "out"
    result = cv.convert(conversion_plan, str(out))

    written = sorted(m.target for m in result.written)
    failed = sorted(m.target for m in result.failed)
    assert written == ["plate1_A01_T0001F001L01A01Z01C01.tif"]
    assert failed == [
        "plate1_A01_T0001F001L01A01Z02C01.tif",
        "plate1_A01_T0001F001L01A01Z03C01.tif",
        "plate1_A01_T0001F001L01A01Z04C01.tif",
    ]
    # The one written plane is a real file, and is not in the failed list.
    assert set(written).isdisjoint(failed)
    assert int(tifffile.imread(str(out / written[0]))[0, 0]) == 10
    assert [os.path.basename(p) for p, _ in result.skipped] == ["fov01.tif"]
    assert "disk went away" in result.skipped[0][1]
    # The half-written field is never marked complete on the checkpoint.
    assert result.resumed_fields == []
    rows = pd.read_csv(result.map_path)
    assert sorted(rows.loc[rows["status"] == "failed", "target"]) == failed
    assert sorted(rows.loc[rows["status"] == "converted", "target"]) == written

    # And the counterpart that proves the empty list above came from the
    # half-written field and not from resume being dead: with the disk back,
    # a resumed run finishes the field, and only then does a third run skip
    # it as already done.
    monkeypatch.setattr(cv, "_atomic_write", real_write)
    repaired = cv.convert(conversion_plan, str(out), resume=True)
    assert repaired.resumed_fields == [], "nothing was complete to resume yet"
    assert sorted(m.target for m in repaired.written) == failed
    again = cv.convert(conversion_plan, str(out), resume=True)
    assert again.resumed_fields == ["plate1/A01/f0001"]
    assert again.n_written == 0
    assert sorted(m.target for m in again.existing) == sorted(
        m.target for m in conversion_plan.mappings)


# ---------------------------------------------------------------------------
# An older map file, missing the prcf column
# ---------------------------------------------------------------------------

def _minimal_map(path, *, with_prcf):
    """Write the smallest CSV ``read_map`` accepts, with or without prcf."""
    row = {
        "target": "plate1_A01_T0001F001L01A01Z01C01.tif",
        "source": "/data/run1/wt/fov01_C1.tif",
        "plate": "plate1", "well": "A01",
        "field": 1, "channel": 1, "z": 1, "t": 1,
    }
    if with_prcf:
        row["prcf"] = "plate1_A_01_1"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def test_a_map_written_before_prcf_existed_still_loads_into_the_database(
        tmp_path):
    """A map without ``prcf`` must populate, not crash on its own index.

    ``prcf`` is not one of the columns :func:`read_map` requires, so a map
    from an older spaCR — or one a user trimmed by hand — legitimately
    arrives without it. Indexing it unconditionally would raise
    ``no such column`` and lose the whole table, when the only thing missing
    is a lookup speed-up.
    """
    old = _minimal_map(tmp_path / "old_map.csv", with_prcf=False)
    old_db = tmp_path / "old.db"
    assert cv.populate_db_from_map(str(old_db), str(old)) == 1

    frame = pd.read_sql(f"SELECT * FROM {cv.CONVERSION_TABLE}",
                        sqlite3.connect(str(old_db)))
    assert list(frame["target"]) == ["plate1_A01_T0001F001L01A01Z01C01.tif"]
    assert "prcf" not in frame.columns
    old_indexes = _indexes(old_db)
    assert f"idx_{cv.CONVERSION_TABLE}_target" in old_indexes
    assert f"idx_{cv.CONVERSION_TABLE}_prcf" not in old_indexes

    # Same call on a map that DOES carry prcf: the index appears, which is
    # what proves the absence above came from the column, not from the
    # indexing being dead code.
    new = _minimal_map(tmp_path / "new_map.csv", with_prcf=True)
    new_db = tmp_path / "new.db"
    assert cv.populate_db_from_map(str(new_db), str(new)) == 1
    new_indexes = _indexes(new_db)
    assert f"idx_{cv.CONVERSION_TABLE}_prcf" in new_indexes
    assert f"idx_{cv.CONVERSION_TABLE}_target" in new_indexes


def test_repopulating_from_a_map_replaces_the_previous_generation_of_rows(
        tmp_path):
    """Re-running a conversion must not leave two generations in the table.

    The conversion table is joined against measurements by ``target``. Two
    generations of rows would double every joined measurement, which reads
    as twice the cells rather than as a bug.
    """
    path = _minimal_map(tmp_path / "map.csv", with_prcf=False)
    db = tmp_path / "measurements.db"
    assert cv.populate_db_from_map(str(db), str(path)) == 1

    second = pd.DataFrame([
        {"target": "plate1_A01_T0001F001L01A01Z01C01.tif",
         "source": "/data/a.tif", "plate": "plate1", "well": "A01",
         "field": 1, "channel": 1, "z": 1, "t": 1},
        {"target": "plate1_A01_T0001F001L01A01Z01C02.tif",
         "source": "/data/b.tif", "plate": "plate1", "well": "A01",
         "field": 1, "channel": 2, "z": 1, "t": 1},
    ])
    second.to_csv(path, index=False)
    assert cv.populate_db_from_map(str(db), str(path)) == 2

    frame = pd.read_sql(f"SELECT * FROM {cv.CONVERSION_TABLE}",
                        sqlite3.connect(str(db)))
    assert len(frame) == 2, "the old generation of rows survived"
    assert sorted(frame["channel"]) == [1, 2]
