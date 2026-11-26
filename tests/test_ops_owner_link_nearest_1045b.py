"""Item 1045b: where the 'inherit_nn' nearest-unused step links, and why it rarely does.

The merge pairs nuclei by mutual nearest neighbours within
``ops_merge_threshold`` (4 px), and the 1045 inheritance gives the owner copy
any link a copy of its 1026 component holds. What is left for the
nearest-unused step is an owner whose nearest SBS nucleus prefers a
neighbouring cell that pairs elsewhere: the nucleus stays unused and the owner
unlinked although they lie within 3 px. These tests plant that geometry in a
real merge, and plant the paper emulation's other case, two copies of one
cell more than 2 px apart (two physical cells under the paper's 2 px rule,
one 1026 component in spaCR), where the link comes by inheritance instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from spacr import ops_engine
from spacr.ops_merge import field_transform
from spacr.ops_store import read_table, write_table

PLATE = "P1"
SIZE = 200
ANCHORS = [(40.0 + 30.0 * i, 40.0 + 30.0 * j) for i in range(5) for j in range(5)
           if (i, j) != (2, 2)]


def _one_field(tmp_path, cells, nuclei):
    """One field at the well origin; ``cells`` and ``nuclei`` are well ``(y, x)``."""
    db = str(tmp_path / "measurements.db")
    transform = field_transform(1.0, 0.0, (100.0, 100.0), (SIZE, SIZE))
    write_table(db, "ops_phenotype", pd.DataFrame([{
        "plate": PLATE, "well": "A1", "site": 0, "centre_y": 100.0,
        "centre_x": 100.0, "is_anchor": 1, "scale": 1.0, "rotation_deg": 0.0,
        "offset_y": float(transform[2][0]), "offset_x": float(transform[2][1]),
        "field_height": SIZE, "field_width": SIZE,
        "transform_source": "anchor", "refined": 0}]))
    write_table(db, "ops_objects", pd.DataFrame({
        "plate": PLATE, "well": "A1", "object_id": np.arange(1, len(nuclei) + 1),
        "centroid_y": [p[0] for p in nuclei], "centroid_x": [p[1] for p in nuclei]}))
    frame = pd.DataFrame({"plate": PLATE, "well": "A1", "site": 0,
                          "label": np.arange(1, len(cells) + 1),
                          "centroid_row": [p[0] for p in cells],
                          "centroid_col": [p[1] for p in cells]})
    frame.to_parquet(ops_engine._ph_cells_path(str(tmp_path), "A1"))
    frame[["plate", "well", "site", "label"]].assign(
        cell_px=80, touches_field_edge=False).to_parquet(
        ops_engine._ph_maskpass_path(str(tmp_path), "A1"))
    return db


def _merged(db, tmp_path, mode):
    settings = ops_engine.ops_defaults({"ops_align_model": "similarity",
                                        "ops_owner_link_inherit": mode})
    ops_engine._ph_cross_field_ownership(db, PLATE, "A1", settings,
                                         str(tmp_path))
    report = ops_engine._ph_merge(db, PLATE, "A1", settings, str(tmp_path),
                                  False)
    return report, read_table(db, "ops_merge")


def test_an_owner_whose_nearest_nucleus_prefers_a_neighbour_takes_it(tmp_path):
    neighbour, owner = (100.0, 101.4), (100.0, 105.5)
    taken, left = (100.0, 100.0), (100.0, 103.0)
    db = _one_field(tmp_path, ANCHORS + [neighbour, owner],
                    ANCHORS + [taken, left])
    owner_label, left_id = len(ANCHORS) + 2, len(ANCHORS) + 2
    links = {}
    for mode in ("inherit", "inherit_nn"):
        report, merged = _merged(db, tmp_path, mode)
        links[mode] = dict(zip(merged.label.astype(int),
                               merged.object_id.astype(int)))
    assert owner_label not in links["inherit"]
    assert left_id not in links["inherit"].values()
    assert links["inherit_nn"][owner_label] == left_id
    assert report["owner_link"]["nearest_unused_links"] == 1
    assert {k: v for k, v in links["inherit_nn"].items()
            if k != owner_label} == links["inherit"]
    row = merged[merged.label == owner_label].iloc[0]
    assert 2.0 < row.distance <= 3.0


def test_the_step_never_takes_a_nucleus_another_cell_holds(tmp_path):
    neighbour, owner = (100.0, 101.4), (100.0, 102.9)
    taken = (100.0, 100.0)
    db = _one_field(tmp_path, ANCHORS + [neighbour, owner], ANCHORS + [taken])
    report, merged = _merged(db, tmp_path, "inherit_nn")
    assert report["owner_link"]["nearest_unused_links"] == 0
    assert len(ANCHORS) + 2 not in set(merged.label.astype(int))
    assert not merged.object_id.duplicated().any()


def test_copies_more_than_2_px_apart_get_their_link_by_inheritance():
    owners = pd.DataFrame({"well": "A1", "site": [0, 1], "label": [7, 3],
                           "component": [5, 5], "is_owner": [True, False],
                           "owner_field": [0.0, 0.0],
                           "owner_label": [7.0, 7.0]})
    twin_link = pd.DataFrame({"site": [1], "label": [3], "object_id": [42],
                              "distance": [0.5]})
    counts = {}
    moved = ops_engine._ph_inherit_owner_links(twin_link, owners, counts)
    assert moved[["site", "label", "object_id"]].values.tolist() == [[0, 7, 42]]
    assert counts["links_moved_to_owner"] == 1
    positions = pd.DataFrame({"site": [0, 1], "label": [7, 3],
                              "well_y": [0.0, 0.0], "well_x": [0.0, 2.5]})
    nearest = {}
    out = ops_engine._ph_nearest_unused_links(
        moved, owners, positions, np.array([[0.0, 2.4]]), [42],
        counts=nearest)
    pd.testing.assert_frame_equal(out, moved)
    assert nearest == {"nearest_unused_links": 0, "owners_without_link": 0}
