"""Item 951: the OPS stitch, objects and decode phases on chosen sites.

``ops_sites`` limits the sequencing phases to a list of sites. The layout
still comes from the whole acquisition's file names; only the listed tiles
are read. On the synthetic five-field well of
``test_the_ops_engine_runs_a_well_end_to_end`` (a plus: centre site 2, arms
0, 1, 3 and 4) the subset runs are checked against the whole-well run.
"""
from __future__ import annotations

import json
import os
import shutil
from tests.ops_store_sql import ops_connect, read_sql

import numpy as np
import pytest

pytest.importorskip("tifffile")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")
pd = pytest.importorskip("pandas")

from spacr import ops_engine
from tests.test_the_ops_engine_runs_a_well_end_to_end import (
    MARGIN, OVERLAP, _plate, _ThresholdModel)


@pytest.fixture(scope="module")
def well(tmp_path_factory):
    """The synthetic well, run whole once.

    :returns: ``(raw, root, settings, library, whole, tops)``.
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ops_engine, "_RASTER_OVERLAP", OVERLAP)
    monkeypatch.setattr(ops_engine, "_WINDOW", 256)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 48)
    monkeypatch.setattr(ops_engine, "_cellpose_model",
                        lambda settings, gpu: _ThresholdModel())
    root = tmp_path_factory.mktemp("ops_sites")
    raw = root / "raw"
    raw.mkdir()
    _nuclei, codes, _centre, tops = _plate(raw)
    settings = {"genotype_source": str(raw), "dst_root": str(root / "whole"),
                "plate": "synthetic", "ops_gpu": False, "n_workers": 1}
    library = sorted(set(codes)) + ["TTTT", "GGGG"]
    whole = ops_engine.run_ops(settings, library=library)
    yield raw, root, settings, library, whole, tops
    monkeypatch.undo()


def _rows(db, table):
    """One table, ordered."""
    with ops_connect(db) as conn:
        return read_sql(f"SELECT * FROM {table}", conn)


def _owner(geometry, tile, y, x):
    """The site whose tile centre is nearest to a point."""
    centres = geometry[["y", "x"]].to_numpy(float) + tile / 2
    return int(geometry["site"].to_numpy()[
        np.argmin(np.hypot(centres[:, 0] - y, centres[:, 1] - x))])


def _calls(db):
    """``(centroid_y, centroid_x, barcode, mapped_guide)`` of every call."""
    with ops_connect(db) as conn:
        return conn.execute(
            "SELECT o.centroid_y, o.centroid_x, b.barcode, b.mapped_guide "
            "FROM ops_barcodes b JOIN ops_objects o ON b.object_id = o.object_id "
            "AND b.plate = o.plate AND b.well = o.well").fetchall()


@pytest.mark.parametrize("sites, groups", [((0, 4), 2), ((2, 3), 1)])
def test_listed_sites_are_placed_read_and_decoded_alone(well, tmp_path, sites,
                                                        groups):
    """Two arms land on the layout's grid; the centre and an arm register.

    The synthetic tiles were cut on the layout's exact grid, so both kinds of
    placement land where the tiles were cut. Every call of the subset run on
    a nucleus the whole run gives to a listed tile is the whole run's call.
    A nucleus in the overlap with a tile that is not listed belongs to the
    listed tile here: the centre field lost a cycle-3 file, so such a call
    can differ.
    """
    _raw, _root, settings, library, whole, tops = well
    run = ops_engine.run_ops({**settings, "dst_root": str(tmp_path),
                              "ops_sites": ",".join(map(str, sites))},
                             library=library)
    report = run["wells"]["A1"]
    stitch = report["stitch"]
    assert report["ops_sites"] == list(sites)
    assert stitch["placement"] == "ops_sites" and stitch["groups"] == groups
    assert stitch["layout_sites"] == 5 and stitch["placed"] == len(sites)
    expected = "layout_grid" if groups == len(sites) else "registered"
    assert set(stitch["placement_sources"].values()) == {expected}
    geometry = _rows(run["db"], "ops_geometry")
    assert sorted(geometry["site"]) == list(sites)
    for row in geometry.itertuples():
        top, left = tops[row.site]
        assert abs(row.y - (top - MARGIN)) <= 1 and abs(row.x - (left - MARGIN)) <= 1
    assert report["decode"]["fields"] == len(sites)
    assert report["decode"]["ops_sites"] == list(sites)
    saved = json.loads((tmp_path / "A1" / "ops_report.json").read_text())
    assert saved["ops_sites"] == list(sites)

    reference = _calls(whole["db"])
    placed = _rows(whole["db"], "ops_geometry")
    tile = int(placed["tile_height"].iloc[0])
    found = [call for call in _calls(run["db"])
             if _owner(placed, tile, call[0], call[1]) in sites]
    assert found
    for y, x, barcode, guide in found:
        away = [np.hypot(y - a, x - b) for a, b, _, _ in reference]
        match = reference[int(np.argmin(away))]
        assert min(away) < 2.5 and (barcode, guide) == match[2:]


def test_decoding_chosen_fields_of_a_whole_stitch_keeps_the_well_territories(
        well, tmp_path):
    """Decode only, on the whole run's stitch and objects: the same rows.

    Each nucleus still belongs to its nearest tile among every placed site,
    so the listed fields give exactly the whole run's calls for the nuclei
    they own.
    """
    _raw, root, settings, library, whole, _tops = well
    copy = tmp_path / "decode_only"
    shutil.copytree(root / "whole", copy)
    run = ops_engine.run_ops({**settings, "dst_root": str(copy),
                              "ops_sites": [0, 2]},
                             phases=("decode",), library=library)
    assert run["wells"]["A1"]["decode"]["fields"] == 2
    geometry = _rows(whole["db"], "ops_geometry")
    tile = int(geometry["tile_height"].iloc[0])
    objects = _rows(whole["db"], "ops_objects").set_index("object_id")
    reference = _rows(whole["db"], "ops_barcodes")
    owned = reference[[
        _owner(geometry, tile, objects.loc[o, "centroid_y"],
               objects.loc[o, "centroid_x"]) in (0, 2)
        for o in reference["object_id"]]]
    found = _rows(run["db"], "ops_barcodes")
    pd.testing.assert_frame_equal(found.reset_index(drop=True),
                                  owned.reset_index(drop=True))


def test_a_folder_too_small_to_fit_a_layout_names_the_ways_out(well, tmp_path):
    """Sites 0-2 alone fit no round well; the refusal names both fixes."""
    raw = well[0]
    partial = tmp_path / "partial"
    for path in raw.rglob("*.tif"):
        if int(path.stem.rsplit("-", 1)[1]) <= 2:
            target = partial / path.relative_to(raw)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, target)
    settings = {**well[2], "genotype_source": str(partial),
                "dst_root": str(tmp_path / "out"), "ops_sites": "0-2"}
    with pytest.raises(ValueError, match="ops_placement_seed") as refusal:
        ops_engine.run_ops(settings, phases=("stitch",))
    assert "genotype_source" in str(refusal.value)


@pytest.mark.parametrize("value, match", [
    ("0,b", "must list site numbers"),
    ([-1], "0 or more"),
    ("7,9", r"sites \[7, 9\]"),
])
def test_bad_sites_are_refused_before_any_pixel(well, tmp_path, value, match):
    """A typo, a negative site and a site the well lacks."""
    settings = {**well[2], "dst_root": str(tmp_path), "ops_sites": value}
    with pytest.raises(ValueError, match=match):
        ops_engine.run_ops(settings)
    assert not os.path.exists(tmp_path / "measurements.db")


@pytest.mark.parametrize("value", [None, ""])
def test_empty_runs_every_site(value):
    """None and an empty string both mean every site."""
    assert ops_engine._ops_sites({"ops_sites": value}) is None
    assert ops_engine._ops_sites({}) is None
    assert ops_engine._ops_sites({"ops_sites": "46, 148,160-162"}) == {
        46, 148, 160, 161, 162}


def test_the_fdr_report_names_the_sites(tmp_path):
    """The decode report, its FDR report and decode_fdr_<well>.json."""
    path = tmp_path / "decode_fdr_A1.json"
    path.write_text(json.dumps({"pct": 1.0}), encoding="utf-8")
    decode = {"fdr": {"pct": 1.0, "paths": {"fdr": str(path)}}}
    ops_engine._name_sites_in_fdr(decode, [46, 148])
    assert decode["ops_sites"] == decode["fdr"]["ops_sites"] == [46, 148]
    assert json.loads(path.read_text())["ops_sites"] == [46, 148]


def test_a_library_without_a_barcode_column_names_its_columns(tmp_path):
    """The refusal lists the columns found and the accepted names."""
    path = tmp_path / "guides.csv"
    path.write_text("gene,sgRNA\nTSG101,GTACGT\n", encoding="utf-8")
    with pytest.raises(ValueError) as refusal:
        ops_engine._load_library(str(path))
    text = str(refusal.value)
    assert "'gene', 'sgRNA'" in text and "prefix, barcode or sequence" in text
    empty = tmp_path / "empty.csv"
    empty.write_text("gene\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        ops_engine._load_library(str(empty))
