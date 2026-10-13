"""The decode on fields that give it less than a whole field: what is skipped, and what is said.

PART 9 hole 1: a field with fewer than three usable cycles is not decoded,
and a field whose reference cycle cannot be read has nothing to align onto.
Both are reported with their reason instead of ending the well. A field in
which nothing changes from cycle to cycle has no reads, which is an answer
and not a failure. PART 9 hole 2: a well whose spots do not lie on its
nuclei is reported loudly, because its barcodes would be wrong. And a well
decoded in worker processes counts its fields out as it goes.

Every field is written to disk as the acquisition names it: cycle 1 as a
DAPI-CY3-A594-CY5-CY7 stack, later cycles one file per base channel.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from scipy import ndimage

from spacr import ops_engine
from spacr.ops_sbs import BASES
from spacr.ops_store import write_table

SIZE = 160
CENTRES = [(50, 50), (50, 110), (110, 80)]
CODES = ["GTAC", "CATG", "AAGG"]
#: The equivalent area of a radius-6 nucleus.
AREA = math.pi * 36


def _rings():
    """Three reads on the rim of each planted nucleus, 3 px outside its boundary.

    :returns: ``[(y, x, barcode)]``.
    """
    return [(int(round(cy + 9 * math.sin(angle))), int(round(cx + 9 * math.cos(angle))), code)
            for (cy, cx), code in zip(CENTRES, CODES) for angle in (0.3, 2.4, 4.4)]


def _field(root, site, cycles, reads=(), *, noise=True, skip=()):
    """Write one field's cycles.

    :param root: the acquisition folder.
    :param site: the site number.
    :param cycles: how many cycles the acquisition has.
    :param reads: ``[(y, x, barcode)]``; each lights its base's channel.
    :param noise: add independent noise to every plane. Without it and
        without reads, every plane of every cycle is the same image.
    :param skip: cycles whose files this site does not have.
    """
    rng = np.random.default_rng(0)
    # The cell background every plane shares: it is what the channel and
    # cycle registrations lock onto.
    texture = ndimage.gaussian_filter(rng.random((SIZE, SIZE)), 6)
    texture = (texture - texture.min()) / np.ptp(texture) * 600 + 200
    for cycle in range(1, cycles + 1):
        planes = []
        for channel in range(4):
            plane = texture + (rng.normal(0, 10, texture.shape) if noise else 0.0)
            for y, x, code in reads:
                if code[cycle - 1] == BASES[channel]:
                    plane[y - 1:y + 2, x - 1:x + 2] += 6000
            planes.append(np.clip(plane, 0, 65535).astype(np.uint16))
        if cycle in skip:
            continue
        folder = root / f"c{cycle}"
        folder.mkdir(parents=True, exist_ok=True)
        if cycle == 1:
            dapi = np.full((SIZE, SIZE), 500, np.uint16)
            tifffile.imwrite(folder / f"10X_c1_A1_DAPI-CY3-A594-CY5-CY7_Site-{site}.tif",
                             np.stack([dapi] + planes))
            continue
        for name, plane in zip(ops_engine._BASE_CHANNELS, planes):
            tifffile.imwrite(folder / f"10X_c{cycle}_A1_{name}_Site-{site}.tif", plane)


def _task(root, site):
    """One field's decode task, built the way the decode phase builds it.

    :param root: the acquisition folder.
    :param site: the site.
    :returns: the task, with the three planted nuclei as its objects.
    """
    index = ops_engine._index_tiles(str(root))["A1"]
    cycles = sorted(index)
    return {
        "site": site, "cycles": cycles, "reference": 1, "gpu": False,
        "planes": {cycle: [ops_engine._plane_sources(index[cycle].get(site, {})).get(name)
                           for name in ops_engine._BASE_CHANNELS]
                   for cycle in cycles},
        "centroids": np.array(CENTRES, float), "areas": np.full(3, AREA),
        "ids": np.array([1, 2, 3]), "owned": np.ones(3, bool),
    }


def _well(tmp_path, sites, tile):
    """An output folder whose ``ops_geometry`` places ``sites``.

    :param tmp_path: the test's folder.
    :param sites: ``site -> (y, x)``.
    :param tile: the tile edge.
    :returns: ``(settings, db)`` for a decode of well A1 read from ``raw``.
    """
    out = tmp_path / "out"
    out.mkdir()
    db = str(out / "measurements.db")
    write_table(db, "ops_geometry", pd.DataFrame([{
        "plate": "plate", "well": "A1", "cycle": 1, "site": site,
        "y": float(y), "x": float(x), "origin_y": 0.0, "origin_x": 0.0,
        "tile_height": tile, "tile_width": tile} for site, (y, x) in sites.items()]))
    settings = {"genotype_source": str(tmp_path / "raw"), "dst_root": str(out),
                "plate": "plate", "ops_gpu": False, "n_workers": 1}
    return settings, db


def _objects(db, centres, area):
    """Store well A1's ``ops_objects``, replacing any before.

    :param db: the measurements database.
    :param centres: ``[(y, x)]`` in the well frame.
    :param area: every object's area.
    """
    write_table(db, "ops_objects", pd.DataFrame({
        "plate": "plate", "well": "A1", "object_id": np.arange(1, len(centres) + 1),
        "centroid_y": [float(y) for y, _ in centres],
        "centroid_x": [float(x) for _, x in centres],
        "area": np.full(len(centres), float(area))}))


def test_a_field_whose_reference_cycle_cannot_be_read_is_skipped_with_its_reason(
        tmp_path, monkeypatch):
    """Every cycle is placed against the reference, so without it the field has no frame.

    The field is skipped before any registration runs, the reference is
    listed as missing and the truncated file as unreadable, and the three
    cycles that did read are not blamed.
    """
    monkeypatch.setattr(ops_engine, "_LIBRARY", frozenset())
    _field(tmp_path, 3, 4, _rings())
    stack = tmp_path / "c1" / "10X_c1_A1_DAPI-CY3-A594-CY5-CY7_Site-3.tif"
    data = stack.read_bytes()
    stack.write_bytes(data[: len(data) // 20])

    result = ops_engine._decode_field(_task(tmp_path, 3))

    assert result["skipped"] == "the reference cycle could not be read"
    assert result["missing"] == [1]
    assert "kept" not in result
    assert result["unreadable"]
    assert {path for path, _reason in result["unreadable"]} == {str(stack)}
    assert all(reason.startswith("ValueError: failed to read")
               for _path, reason in result["unreadable"])


def test_a_field_left_with_fewer_than_three_cycles_is_not_decoded(tmp_path, monkeypatch):
    """Two cycles give two-letter reads, which no library can tell apart.

    Site 7 has no files for cycles 2 and 3; site 8 beside it has all four.
    Site 7 is aligned, found to keep cycles 1 and 4, and skipped with that
    count; its absent files are missing, not unreadable. Site 8 decodes.
    """
    monkeypatch.setattr(ops_engine, "_LIBRARY", frozenset(CODES))
    _field(tmp_path, 7, 4, _rings(), skip=(2, 3))
    _field(tmp_path, 8, 4, _rings())

    lost = ops_engine._decode_field(_task(tmp_path, 7))
    whole = ops_engine._decode_field(_task(tmp_path, 8))

    assert lost["skipped"] == "2 usable cycles"
    assert lost["kept"] == [1, 4] and lost["missing"] == [2, 3]
    assert lost["unreadable"] == []
    assert "skipped" not in whole and whole["n_cycles"] == 4
    assert sorted(set(whole["calls"])) == sorted(CODES)


def test_a_field_where_nothing_changes_between_cycles_decodes_to_no_reads(
        tmp_path, monkeypatch):
    """Reads are found by their change across cycles; a field without change has none.

    Every plane of every cycle is the same image, so the registrations all
    accept and the field is decoded, not skipped. It yields no spots and no
    calls, and says so in its counts, instead of failing on an empty list.
    """
    monkeypatch.setattr(ops_engine, "_LIBRARY", frozenset(CODES))
    _field(tmp_path, 5, 4, reads=(), noise=False)

    result = ops_engine._decode_field(_task(tmp_path, 5))

    assert "skipped" not in result and result["kept"] == [1, 2, 3, 4]
    assert result["spots"] == 0
    assert result["attributed"] == 0 and result["ambiguous"] == 0
    assert result["exact"] == 0
    assert result["calls"] == []
    assert result["ids"].shape == (0,) and result["quality"].shape == (0,)


def test_reads_that_lie_away_from_every_nucleus_are_reported_loudly(tmp_path, capsys):
    """PART 9 hole 2's gate, driven both ways on one field.

    With the well's objects placed away from the reads, no spot lies within
    the 10 px footprint: the report flags the containment below the gate and
    the engine prints the warning, and no object is given a barcode. With the
    objects moved onto the nuclei the reads surround, the same spots are
    contained, the warning is not printed, and all three objects are assigned.
    """
    _field(tmp_path / "raw", 0, 4, _rings())
    settings, db = _well(tmp_path, {0: (0, 0)}, SIZE)

    _objects(db, [(12, 12), (148, 148)], 36.0)
    far = ops_engine.run_ops(settings, phases=("decode",))["wells"]["A1"]["decode"]
    loud = capsys.readouterr().out
    _objects(db, CENTRES, AREA)
    near = ops_engine.run_ops(settings, phases=("decode",))["wells"]["A1"]["decode"]
    quiet = capsys.readouterr().out

    assert far["spots"] == near["spots"] >= len(_rings())
    assert far["containment"] == 0.0 and far["containment_below_gate"] is True
    assert "ONLY 0.0% OF SPOTS LIE WITHIN 10 PX OF A NUCLEUS" in loud
    assert far["objects_assigned"] == 0 and far["ops_barcodes_rows"] == 0
    assert near["containment"] >= ops_engine._MIN_CONTAINMENT
    assert near["containment_below_gate"] is False
    assert "OF SPOTS LIE WITHIN" not in quiet
    assert near["objects_assigned"] == 3 and near["ops_barcodes_rows"] == 3


def test_a_well_decoded_in_worker_processes_counts_its_fields_out(tmp_path, capsys):
    """Every 25th field is reported as the pool returns it.

    Twenty-five fields are placed but only the nuclear tile of one was ever
    acquired, so every field is skipped for its unreadable reference. The
    well still finishes: its report lists all 25 reasons, and its
    ``ops_barcodes`` holds no rows rather than being left unwritten.
    """
    raw = tmp_path / "raw" / "c1"
    raw.mkdir(parents=True)
    tifffile.imwrite(raw / "10X_c1_A1_DAPI_Site-0.tif", np.zeros((64, 64), np.uint16))
    sites = {5 * row + column: (56.0 * row, 56.0 * column)
             for row in range(5) for column in range(5)}
    settings, db = _well(tmp_path, sites, 64)
    _objects(db, [(y + 32, x + 32) for y, x in sites.values()], 80.0)

    decode = ops_engine.run_ops({**settings, "n_workers": 2},
                                phases=("decode",))["wells"]["A1"]["decode"]

    assert decode["workers"] == 2
    assert decode["fields"] == 25 and decode["fields_decoded"] == 0
    assert decode["skipped"] == {str(site): "the reference cycle could not be read"
                                 for site in sites}
    assert decode["ops_barcodes_rows"] == 0 and decode["objects_assigned"] == 0
    assert "OPS: A1 decode: 25 of 25 fields" in capsys.readouterr().out


def test_a_zero_byte_cycle_file_is_warned_about_and_does_not_end_the_well(
        tmp_path, capsys):
    """372, 2026-09-26: one empty sequencing file failed all of well B3.

    ``c4/10X_c4_B3_CY3_Site-59.tif`` is 0 bytes on the NAS, an lftp download
    that never finished. tifffile says "not a TIFF file" with an error that
    is not a ValueError, and it ended the well at decode field 59. Here the
    same kind of file costs cycle 3 of the one field: the well completes,
    every nucleus is still assigned, the file and its reason are in the
    report, and the run prints a WARNING line that it did not print before.
    """
    _field(tmp_path / "raw", 0, 4, _rings())
    settings, db = _well(tmp_path, {0: (0, 0)}, SIZE)
    _objects(db, CENTRES, AREA)

    whole = ops_engine.run_ops(settings, phases=("decode",))["wells"]["A1"]["decode"]
    before = capsys.readouterr().out
    empty = tmp_path / "raw" / "c3" / "10X_c3_A1_A594_Site-0.tif"
    empty.write_bytes(b"")
    lost = ops_engine.run_ops(settings, phases=("decode",))["wells"]["A1"]["decode"]
    after = capsys.readouterr().out

    assert whole["unreadable"] == [] and "WARNING" not in before
    assert whole["objects_assigned"] == 3
    assert [path for path, _reason in lost["unreadable"]] == [str(empty)]
    assert lost["unreadable"][0][1].startswith("TiffFileError: not a TIFF file")
    assert lost["cycles_missing"] == {"0": [3]}
    assert lost["skipped"] == {}
    assert lost["objects_assigned"] == 3
    assert "OPS: A1 decode: WARNING 1 source file(s) could not be read" in after
    assert "10X_c3_A1_A594_Site-0.tif (TiffFileError: not a TIFF file" in after
