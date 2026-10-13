"""What the OPS engine refuses, and what it passes over without refusing.

`spacr.ops_engine.run_ops` is what the OPS button runs (372, THE SWITCH). A
refusal is all an operator sees of a run that could not start, so each one
here is driven from tiles and tables planted in ``tmp_path`` and checked for
the sentence that names the fix -- and, where it matters, for the absence of
anything written on the way to it.

The other half is what must NOT be refused: a file beside the tiles that is
not a tile, a read that fails when nobody asked for the reason, a second well
or a second plate in the same database. None of those may cost the well that
is being run.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pd = pytest.importorskip("pandas")

from spacr import ops_engine
from spacr.ops_store import read_table, row_count, write_table


def _tile(path, image):
    """Write one image, making its folder first.

    :param path: where, named as the acquisition names its tiles.
    :param image: the pixels.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(image, np.uint16))


def _truncate(path, keep=0.3):
    """Cut a file short, as an interrupted copy leaves it.

    :param path: the file.
    :param keep: the fraction of its bytes to keep.
    """
    data = path.read_bytes()
    path.write_bytes(data[: int(len(data) * keep)])


def _geometry(plate, well, sites, tile=64):
    """``ops_geometry`` rows with the columns the stitch phase writes.

    :param plate: the plate name.
    :param well: the well name.
    :param sites: ``site -> (y, x)``.
    :param tile: the tile edge in pixels.
    :returns: the frame.
    """
    return pd.DataFrame([{
        "plate": plate, "well": well, "cycle": 1, "site": site,
        "y": float(y), "x": float(x), "origin_y": 0.0, "origin_x": 0.0,
        "tile_height": tile, "tile_width": tile,
    } for site, (y, x) in sites.items()])


def _settings(tmp_path):
    """OPS settings over an empty acquisition folder and a separate output.

    :param tmp_path: the test's folder.
    :returns: the settings.
    """
    (tmp_path / "raw").mkdir(exist_ok=True)
    return {"genotype_source": str(tmp_path / "raw"),
            "dst_root": str(tmp_path / "out"), "plate": "plate",
            "ops_gpu": False, "n_workers": 1}


def test_a_file_not_named_as_a_tile_is_not_indexed(tmp_path):
    """An overview, a metadata sidecar and a notes file sit beside real tiles.

    Only a name that carries magnification, cycle, well, channel and site is
    a tile. Anything else is passed over rather than guessed at, and does not
    add a well, a cycle or a site to the index.
    """
    tile = tmp_path / "c1" / "10X_c1_A1_DAPI_Site-0.tif"
    _tile(tile, np.zeros((8, 8)))
    _tile(tmp_path / "c1" / "A1_overview.tif", np.zeros((8, 8)))
    (tmp_path / "c1" / "10X_c1_A1_DAPI_Site-0.txt").write_text("exposure 200 ms")
    (tmp_path / "notes.csv").write_text("well,comment\nB2,empty\n")

    assert ops_engine._index_tiles(str(tmp_path)) == {
        "A1": {1: {0: {"DAPI": str(tile)}}}}


def test_every_failed_read_is_none_when_no_list_asks_why(tmp_path):
    """The reason list is optional; leaving it out must not turn a loss into a crash.

    A missing path, a truncated file (ValueError, 372 PART 14-L) and a plane
    the stack does not hold each give None with no list to write to. A
    channel no file offers is not a failure at all, so it records no reason
    even when a list is given. A good plane of the same stack still reads.
    """
    good = tmp_path / "10X_c2_A1_CY3-CY5_Site-0.tif"
    _tile(good, np.full((2, 16, 16), 9))
    short = tmp_path / "10X_c2_A1_A594_Site-0.tif"
    _tile(short, np.ones((256, 256)))
    _truncate(short)
    folder = tmp_path / "10X_c2_A1_CY7_Site-0.tif"
    folder.mkdir()

    assert ops_engine._read_plane((str(folder), None)) is None
    assert ops_engine._read_plane((str(short), None)) is None
    assert ops_engine._read_plane((str(good), 2)) is None
    reasons = []
    assert ops_engine._read_plane(None, reasons) is None
    assert reasons == []
    assert np.all(ops_engine._read_plane((str(good), 1)) == 9)


def test_a_file_with_more_axes_than_a_plane_gives_its_first_plane(tmp_path):
    """A single-channel file holding a z-stack, and a channel stack of z-stacks.

    The engine works on 2-D planes. A file that carries extra leading axes is
    reduced to its first plane on each of them rather than handed on with a
    shape every later step would misread.
    """
    z_stack = np.stack([np.full((16, 16), 100 + z) for z in range(3)]).astype(np.uint16)
    single = tmp_path / "10X_c2_A1_CY3_Site-0.tif"
    # Grey pages, said explicitly: tifffile would otherwise store a leading
    # axis of three as RGB components, a default it has deprecated.
    tifffile.imwrite(single, z_stack, photometric="minisblack")
    stack = tmp_path / "10X_c1_A1_DAPI-CY3_Site-0.tif"
    tifffile.imwrite(stack, np.stack([z_stack, z_stack + 50]), photometric="minisblack")

    plane = ops_engine._read_plane((str(single), None))
    assert plane.dtype == np.float32 and plane.shape == (16, 16)
    assert np.all(plane == 100)
    plane = ops_engine._read_plane((str(stack), 1))
    assert plane.shape == (16, 16) and np.all(plane == 150)


def test_a_well_with_no_nuclear_plane_in_any_cycle_is_refused(tmp_path):
    """Every phase is placed on the nuclear stain, so a well without one cannot start.

    The refusal names the well and the stain, and nothing is written for it.
    """
    settings = _settings(tmp_path)
    raw = tmp_path / "raw"
    _tile(raw / "c1" / "10X_c1_A1_DAPI_Site-0.tif", np.zeros((8, 8)))
    for cycle in (1, 2):
        _tile(raw / f"c{cycle}" / f"10X_c{cycle}_B2_CY3_Site-0.tif", np.zeros((8, 8)))

    with pytest.raises(ValueError, match="well B2 has no DAPI plane in any cycle"):
        ops_engine.run_ops(settings, wells=["b2"])
    assert not os.path.exists(os.path.join(settings["dst_root"], "B2"))


def test_a_well_whose_nuclear_tiles_are_all_truncated_is_refused_at_the_stitch(tmp_path):
    """No readable nuclear tile means no well frame, and nothing to store.

    One truncated tile costs that tile (PART 14-L); when every tile is
    truncated the stitch says so rather than solving a layout of nothing, and
    no ``ops_geometry`` table is left behind.
    """
    settings = _settings(tmp_path)
    for site in range(3):
        path = (tmp_path / "raw" / "c1"
                / f"10X_c1_A1_DAPI-CY3-A594-CY5-CY7_Site-{site}.tif")
        _tile(path, np.ones((5, 64, 64)))
        _truncate(path, keep=0.05)

    with pytest.raises(ValueError, match="no nuclear tile of well A1 could be read"):
        ops_engine.run_ops(settings, phases=("stitch",))
    db = os.path.join(settings["dst_root"], "measurements.db")
    assert row_count(db, "ops_geometry") is None


def test_decoding_before_the_objects_exist_says_which_half_is_missing(tmp_path):
    """The readiness gate's reason is the refusal: no database, then no table.

    No sequencing channel is read until ``ops_objects`` is on disk and
    counted. A decode asked for first is refused with the gate's own
    sentence, which differs between "nothing has run" and "the stitch ran but
    the objects phase did not".
    """
    settings = _settings(tmp_path)
    _tile(tmp_path / "raw" / "c1" / "10X_c1_A1_DAPI_Site-0.tif", np.zeros((64, 64)))

    with pytest.raises(ValueError, match="there is no database"):
        ops_engine.run_ops(settings, phases=("decode",))
    db = os.path.join(settings["dst_root"], "measurements.db")
    write_table(db, "ops_geometry", _geometry("plate", "A1", {0: (0, 0)}))
    with pytest.raises(ValueError, match="has no ops_objects table"):
        ops_engine.run_ops(settings, phases=("decode",))


def test_decoding_a_well_the_objects_phase_never_reached_is_refused(tmp_path):
    """The objects table exists for another well, so the gate passes and the well still has none.

    Decoding A1 against A2's objects would attribute every read to nothing.
    The refusal names the well and the phase to run, and no barcodes are
    written.
    """
    settings = _settings(tmp_path)
    _tile(tmp_path / "raw" / "c1" / "10X_c1_A1_DAPI_Site-0.tif", np.zeros((64, 64)))
    db = os.path.join(settings["dst_root"], "measurements.db")
    os.makedirs(settings["dst_root"])
    write_table(db, "ops_geometry", _geometry("plate", "A1", {0: (0, 0)}))
    write_table(db, "ops_objects", pd.DataFrame({
        "plate": ["plate"], "well": ["A2"], "object_id": [1],
        "centroid_y": [30.0], "centroid_x": [30.0], "area": [80]}))

    with pytest.raises(ValueError,
                       match="well A1 has no ops_objects rows; run the objects phase first"):
        ops_engine.run_ops(settings, wells=["A1"], phases=("decode",))
    assert row_count(db, "ops_barcodes") is None


def test_rewriting_one_well_keeps_every_other_wells_rows(tmp_path):
    """A well run again replaces its own rows and nobody else's.

    The store's writer replaces a whole table (its parquet cache has to
    describe the same rows), so the engine reads the other wells back and
    writes them with this one. Another well of the same plate and the same
    well name on another plate both survive.
    """
    db = str(tmp_path / "measurements.db")
    ops_engine._replace_well_rows(
        db, "ops_geometry", _geometry("plate", "A1", {0: (0, 0), 1: (0, 50)}),
        "plate", "A1")
    ops_engine._replace_well_rows(
        db, "ops_geometry", _geometry("plate", "A2", {0: (5, 5)}), "plate", "A2")
    ops_engine._replace_well_rows(
        db, "ops_geometry", _geometry("other", "A1", {0: (7, 7)}), "other", "A1")

    stored = ops_engine._replace_well_rows(
        db, "ops_geometry", _geometry("plate", "A1", {0: (1, 2)}), "plate", "A1")

    def placed(plate, well):
        """``[(site, y, x)]`` of one well, as stored."""
        rows = ops_engine._well_rows(db, "ops_geometry", plate, well)
        return [(int(r.site), r.y, r.x) for r in rows.itertuples()]

    assert stored == 3
    assert placed("plate", "A1") == [(0, 1.0, 2.0)]
    assert placed("plate", "A2") == [(0, 5.0, 5.0)]
    assert placed("other", "A1") == [(0, 7.0, 7.0)]


def test_a_geometry_table_that_names_no_well_is_never_read_as_this_one(tmp_path):
    """A table written without ``plate`` and ``well`` columns belongs to no well.

    Such a table cannot say whose placements it holds, so the objects phase
    treats the well as unstitched and says to run the stitch. Stitching the
    well then replaces the table: rows that name no well cannot be kept
    beside rows that do.
    """
    settings = _settings(tmp_path)
    _tile(tmp_path / "raw" / "c1" / "10X_c1_A1_DAPI_Site-0.tif", np.zeros((64, 64)))
    os.makedirs(settings["dst_root"])
    db = os.path.join(settings["dst_root"], "measurements.db")
    write_table(db, "ops_geometry", pd.DataFrame({
        "site": [0, 1], "y": [0.0, 0.0], "x": [0.0, 50.0],
        "tile_height": [64, 64], "tile_width": [64, 64]}))

    assert ops_engine._well_rows(db, "ops_geometry", "plate", "A1") is None
    with pytest.raises(ValueError, match="run the stitch phase first"):
        ops_engine.run_ops(settings, phases=("objects",))

    stored = ops_engine._replace_well_rows(
        db, "ops_geometry", _geometry("plate", "A1", {0: (0, 0)}), "plate", "A1")
    assert stored == 1
    table = read_table(db, "ops_geometry")
    assert list(table["well"]) == ["A1"] and list(table["site"]) == [0]


@pytest.mark.parametrize("column", ["prefix", "barcode", "sequence"])
def test_a_guide_library_csv_is_read_from_the_column_that_names_it(tmp_path, column):
    """Any of the three names a guide table uses for its barcodes, as a str or a Path.

    A row with no barcode (a non-targeting control listed by gene only) is
    not a guide and is left out rather than matched as an empty read.
    """
    path = tmp_path / "library.csv"
    path.write_text(f"gene,{column}\nTSG101,GTACGT\nNTC,\nVPS4A,CATGCA\n",
                    encoding="utf-8")

    assert ops_engine._load_library(path) == ["GTACGT", "CATGCA"]
    assert ops_engine._load_library(str(path)) == ["GTACGT", "CATGCA"]


def test_a_library_with_both_a_prefix_and_a_sequence_is_read_by_its_prefix(tmp_path):
    """The prefix is the part of the guide the sequencing cycles read.

    A full protospacer is longer than the reads and would never match one,
    so when a table carries both, the prefix is the column taken.
    """
    path = tmp_path / "library.csv"
    path.write_text("sequence,prefix\nGTACGTAACCGGTTAACCGG,GTACGT\n", encoding="utf-8")

    assert ops_engine._load_library(path) == ["GTACGT"]


def test_a_library_csv_with_no_barcode_column_is_refused_before_any_work(tmp_path):
    """The refusal names the three accepted columns, and no database is started."""
    settings = _settings(tmp_path)
    _tile(tmp_path / "raw" / "c1" / "10X_c1_A1_DAPI_Site-0.tif", np.zeros((64, 64)))
    path = tmp_path / "guides.csv"
    path.write_text("gene,guide\nTSG101,GTACGT\n", encoding="utf-8")

    with pytest.raises(ValueError, match="has no prefix, barcode or sequence column"):
        ops_engine.run_ops(settings, library=str(path))
    assert not os.path.exists(os.path.join(settings["dst_root"], "measurements.db"))
