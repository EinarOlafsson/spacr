"""Cross-field ownership of OPS phenotype cells (item 874, alpha).

The planted fields are the reference script's self-test: identity rotation,
scale 1, fields offset in x and y, and disks cut by a field's edge. Each test
names the physical cells and their owners it plants and checks that exactly
those come back.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from spacr import ops_engine
from spacr.ops_store import read_table, write_table

PLATE = "P1"


def _disk(shape, centre, radius, label, out):
    """Paint a disk of ``label`` into ``out``."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    out[(yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius ** 2] = label


def _transform(offset):
    """Scale 1, no rotation, translated by ``offset``."""
    return (1.0, ops_engine._ph_rotation(0.0), np.asarray(offset, float))


def _areas(image):
    """``label -> pixels`` of a label image."""
    labels, counts = np.unique(image[image > 0], return_counts=True)
    return pd.Series(counts, index=labels)


def _fields(images, offsets):
    """The ``fields`` argument of the ownership pass."""
    return {site: {"transform": _transform(offsets[site]), "shape": (100, 100),
                   "areas": _areas(images[site]), "cells": images[site]}
            for site in images}


def _own(images, offsets, **kwargs):
    """Ownership of planted fields, indexed by ``(site, label)``."""
    kwargs.setdefault("metric", "overlap_fraction")
    kwargs.setdefault("min_overlap", 0.5)
    cells, edges, pairs = ops_engine._ph_cross_field_fields(
        _fields(images, offsets), **kwargs)
    return cells.set_index(["site", "label"]), edges, pairs


def test_a_cut_copy_is_owned_by_the_whole_one():
    a = np.zeros((100, 100), np.uint32)
    b = np.zeros((100, 100), np.uint32)
    _disk(a.shape, (50, 92), 12, 1, a)
    _disk(b.shape, (50, 22), 12, 1, b)
    _disk(a.shape, (20, 20), 8, 2, a)
    cells, edges, pairs = _own({0: a, 1: b}, {0: (0, 0), 1: (0, 70)})
    assert pairs == 1 and len(edges) == 1
    assert cells.loc[(0, 1), "n_copies"] == 2 and cells.loc[(1, 1), "n_copies"] == 2
    assert not cells.loc[(0, 1), "is_owner"] and cells.loc[(1, 1), "is_owner"]
    assert cells.loc[(0, 1), "owner_field"] == 1
    assert 0 < cells.loc[(0, 1), "owned_area_share"] < 1
    assert cells.loc[(0, 2), "n_copies"] == 1 and cells.loc[(0, 2), "is_owner"]
    assert cells.loc[(0, 2), "owned_area_share"] == 1.0


def test_three_copies_are_one_cell_owned_by_the_whole_copy():
    offsets = {0: (0, 0), 1: (0, 60), 2: (60, 60)}
    images = {}
    for site, (oy, ox) in offsets.items():
        image = np.zeros((100, 100), np.uint32)
        _disk(image.shape, (80 - oy, 80 - ox), 10, 7, image)
        images[site] = image
    images[0][:, 85:] = 0
    images[1][85:, :] = 0
    cells, _edges, _pairs = _own(images, offsets)
    assert (cells["n_copies"] == 3).all() and cells["is_owner"].sum() == 1
    assert cells.index[cells["is_owner"]].tolist() == [(2, 7)]


def test_a_tie_goes_to_the_smaller_site():
    a = np.zeros((100, 100), np.uint32)
    b = np.zeros((100, 100), np.uint32)
    _disk(a.shape, (50, 85), 8, 3, a)
    _disk(b.shape, (50, 15), 8, 4, b)
    cells, _edges, _pairs = _own({5: a, 9: b}, {5: (0, 0), 9: (0, 70)})
    assert (cells["n_copies"] == 2).all()
    assert cells.index[cells["is_owner"]].tolist() == [(5, 3)]


def test_fields_that_do_not_overlap_have_no_pairs():
    a = np.zeros((100, 100), np.uint32)
    b = np.zeros((100, 100), np.uint32)
    _disk(a.shape, (50, 50), 8, 1, a)
    _disk(b.shape, (50, 50), 8, 1, b)
    cells, edges, pairs = _own({0: a, 1: b}, {0: (0, 0), 1: (0, 500)})
    assert pairs == 0 and len(edges) == 0
    assert cells["is_owner"].all() and (cells["n_copies"] == 1).all()


def test_a_sliver_below_the_threshold_is_not_a_copy():
    a = np.zeros((100, 100), np.uint32)
    b = np.zeros((100, 100), np.uint32)
    _disk(a.shape, (50, 85), 10, 1, a)
    b[40:60, 12:18] = 2
    cells, edges, _pairs = _own({0: a, 1: b}, {0: (0, 0), 1: (0, 70)},
                                metric="iou", min_overlap=0.5)
    assert (cells["n_copies"] == 1).all() and len(edges) == 0
    cells, edges, _pairs = _own({0: a, 1: b}, {0: (0, 0), 1: (0, 70)},
                                metric="overlap_fraction", min_overlap=0.5)
    assert len(edges) == 1 and cells.loc[(0, 1), "is_owner"]


def test_an_unknown_metric_is_refused():
    with pytest.raises(ValueError, match="ops_cross_field_metric"):
        ops_engine._ph_pair_edges(np.array([1]), np.array([1]), np.array([5]),
                                  {1: 5}, {1: 5}, metric="dice")
    with pytest.raises(ValueError, match="ops_cross_field_policy"):
        ops_engine._ph_cross_field_policy({"ops_cross_field_policy": "x"})


def _run_folder(tmp_path):
    """Two overlapping fields saved as a run: masks, mask pass, placements."""
    a = np.zeros((100, 100), np.uint32)
    b = np.zeros((100, 100), np.uint32)
    _disk(a.shape, (50, 92), 12, 1, a)
    _disk(b.shape, (50, 22), 12, 1, b)
    _disk(a.shape, (20, 20), 8, 2, a)
    _disk(b.shape, (20, 60), 8, 2, b)
    masks = tmp_path / "A1" / "phenotype_masks"
    masks.mkdir(parents=True)
    rows, passed = [], []
    for site, (image, ox) in enumerate(((a, 0.0), (b, 70.0))):
        np.savez_compressed(masks / f"site-{site}.npz", cells=image,
                            nuclei=image)
        rows.append({"plate": PLATE, "well": "A1", "site": site,
                     "centre_y": 50.0, "centre_x": ox + 50.0, "is_anchor": 1,
                     "scale": 1.0, "rotation_deg": 0.0, "offset_y": 0.0,
                     "offset_x": ox, "field_height": 100,
                     "field_width": 100, "transform_source": "anchor",
                     "refined": 0})
        areas = _areas(image)
        for label, count in areas.items():
            passed.append({"plate": PLATE, "well": "A1", "site": site,
                           "label": int(label), "cell_px": int(count),
                           "touches_field_edge": bool(site == 0 and label == 1)})
    db = str(tmp_path / "measurements.db")
    write_table(db, "ops_phenotype", pd.DataFrame(rows))
    pd.DataFrame(passed).to_parquet(
        ops_engine._ph_maskpass_path(str(tmp_path), "A1"))
    return db


def test_the_phase_writes_the_table_and_leaves_ops_phenotype_alone(tmp_path):
    db = _run_folder(tmp_path)
    before = read_table(db, "ops_phenotype")
    settings = {"dst_root": str(tmp_path), "plate": PLATE}
    out = ops_engine.run_ops(settings, phases=("ph_ownership",))
    assert out["wells"]["A1"]["ph_ownership"] == {
        "skipped": "ops_cross_field_ownership is off"}
    out = ops_engine.run_ops({**settings, "ops_cross_field_ownership": True},
                             phases=("ph_ownership",))
    report = out["wells"]["A1"]["ph_ownership"]
    assert report["fields"] == 2 and report["edges"] == 1
    assert report["physical_cells"] == 3 and report["non_owner_copies"] == 1
    table = pd.read_parquet(ops_engine._ph_cross_field_path(str(tmp_path),
                                                            "A1"))
    assert list(table.columns) == [
        "plate", "well", "site", "label", "area", "touches_field_edge",
        "component", "n_copies", "owner_field", "owner_label", "is_owner",
        "owned_area_share"]
    cut = table[(table.site == 0) & (table.label == 1)].iloc[0]
    assert cut.touches_field_edge and not cut.is_owner and cut.owner_field == 1
    edges = pd.read_parquet(ops_engine._ph_cross_field_edges_path(
        str(tmp_path), "A1"))
    assert edges[["site_a", "site_b", "label_a", "label_b"]].values.tolist() \
        == [[0, 1, 1, 1]]
    pd.testing.assert_frame_equal(read_table(db, "ops_phenotype"), before)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        ops_engine.run_ops({**settings, "ops_cross_field_ownership": True,
                            "ops_cross_field_min_overlap": 0},
                           phases=("ph_ownership",))


def _genotyped():
    """Genotyped cells: both copies of the shared cell and one lone cell."""
    return pd.DataFrame({"well": "A1", "site": [0, 1, 0, 1],
                         "label": [1, 1, 2, 9],
                         "object_id": [10, 11, 12, 13]})


@pytest.mark.parametrize("policy, kept, key", [
    ("one_per_cell", [(1, 1), (0, 2), (1, 9)], "dropped_extra_copies"),
    ("owner_only", [(1, 1), (0, 2), (1, 9)], "dropped_non_owner_copies"),
    ("flag_only", [(0, 1), (1, 1), (0, 2), (1, 9)], None)])
def test_policies_choose_the_rows_the_aggregate_pools(tmp_path, policy, kept,
                                                      key):
    db = _run_folder(tmp_path)
    ops_engine.run_ops({"dst_root": str(tmp_path), "plate": PLATE,
                        "ops_cross_field_ownership": True},
                       phases=("ph_ownership",))
    cells, report = ops_engine._ph_cross_field_filter(
        _genotyped(), str(tmp_path), ["A1"], {"ops_cross_field_policy": policy})
    assert sorted(zip(cells.site, cells.label)) == sorted(kept)
    if key:
        assert report[key] == 1
    del db


def test_one_per_cell_keeps_the_larger_copy_even_when_the_owner_was_not_linked(
        tmp_path):
    _run_folder(tmp_path)
    ops_engine.run_ops({"dst_root": str(tmp_path), "plate": PLATE,
                        "ops_cross_field_ownership": True},
                       phases=("ph_ownership",))
    only_cut = pd.DataFrame({"well": "A1", "site": [0], "label": [1],
                             "object_id": [10]})
    kept, report = ops_engine._ph_cross_field_filter(
        only_cut, str(tmp_path), ["A1"], {})
    assert len(kept) == 1 and report["dropped_extra_copies"] == 0
    dropped, report = ops_engine._ph_cross_field_filter(
        only_cut, str(tmp_path), ["A1"], {"ops_cross_field_policy":
                                          "owner_only"})
    assert dropped.empty and report["dropped_non_owner_copies"] == 1
    with pytest.raises(ValueError, match="ph_ownership"):
        ops_engine._ph_cross_field_filter(only_cut, str(tmp_path), ["B2"], {})
