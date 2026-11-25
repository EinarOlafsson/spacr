"""Item 1026: the seam owner is the field nearest ONE centroid per cell.

The ownership docstring says a cell imaged in overlapping fields belongs to
"the field whose centre is nearest the cell centroid". Before settings
schema 970 each copy's own centroid was measured to its own field centre,
so a few pixels of registration error between the copies could name a
field that is not the nearest one. ``ops_cross_field_owner_centroid``
``'linked'`` (the default) measures the mean of the whole copies to every
field centre; ``'per_copy'`` reproduces the old owners exactly, and files
stamped before 970 replay it.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from spacr import ops_engine
from spacr import ops_merge as FO
from spacr import settings as S
from spacr.ops_settings import OPS_CATEGORIES, OPS_DEFAULTS, OPS_TOOLTIPS

SIDE = 100


def _transform(offset):
    return (1.0, np.eye(2), np.asarray(offset, float))


def _centres(offsets):
    return {s: FO._field_centre(_transform(o), (SIDE, SIDE))
            for s, o in offsets.items()}


def _straddling():
    """Two copies of one cell near the cutoff between fields 1 and 2.

    Field centres are (50, 50) and (50, 110); the cutoff is x = 80. The
    copies sit 8 px apart (registration error): copy 1 at (50, 79.5) is
    29.5 px from its own centre, copy 2 at (42, 81.5) 29.6 px from its own,
    so the per-copy rule picks field 1. Their mean (46, 80.5) is 0.5 px
    past the cutoff, nearer field 2.
    """
    offsets = {1: (0.0, 0.0), 2: (0.0, 60.0)}
    cells = pd.DataFrame({"site": [1, 2], "label": [1, 1],
                          "area": [80.0, 80.0], "whole": [True, True],
                          "well_y": [50.0, 42.0], "well_x": [79.5, 81.5]})
    edges = pd.DataFrame({"site_a": [1], "label_a": [1], "site_b": [2],
                          "label_b": [1], "distance": [8.2]})
    return offsets, cells, edges


def test_linked_follows_the_docstring_where_per_copy_does_not():
    offsets, cells, edges = _straddling()
    centres = _centres(offsets)
    linked = FO._assign_field_owners(cells, edges, centres=centres)
    old = FO._assign_field_owners(cells, edges, centres=centres,
                                  centroid="per_copy")
    mean = cells[["well_y", "well_x"]].mean().to_numpy()
    nearest = min(centres, key=lambda s: (np.hypot(*(mean - centres[s])), s))
    assert nearest == 2
    assert linked.loc[linked.is_owner, "site"].tolist() == [nearest]
    assert linked["nearest_field"].tolist() == [2, 2]
    assert old.loc[old.is_owner, "site"].tolist() == [1]
    assert "nearest_field" not in old
    assert np.allclose(linked["centre_distance_px"],
                       [np.hypot(*(mean - centres[1])),
                        np.hypot(*(mean - centres[2]))])


def test_the_phase_uses_the_linked_centroid_by_default():
    offsets, cells, _edges = _straddling()
    fields = {}
    for site, (oy, ox) in offsets.items():
        mine = cells[cells.site == site]
        labels = mine["label"].to_numpy(np.int64)
        fields[site] = {
            "transform": _transform((oy, ox)), "shape": (SIDE, SIDE),
            "areas": pd.Series(mine["area"].to_numpy(float), index=labels),
            "whole": pd.Series(mine["whole"].to_numpy(bool), index=labels),
            "centroids": {int(l): (float(y - oy), float(x - ox)) for l, y, x
                          in zip(labels, mine["well_y"], mine["well_x"])},
            "cells": None}
    new, _, _ = ops_engine._ph_cross_field_fields(fields, metric="centroid")
    old, _, _ = ops_engine._ph_cross_field_fields(fields, metric="centroid",
                                                  centroid="per_copy")
    assert new.loc[new.is_owner, "site"].tolist() == [2]
    assert old.loc[old.is_owner, "site"].tolist() == [1]
    options = ops_engine._ph_cross_field_options(ops_engine.ops_defaults({}))
    assert options["centroid"] == "linked"
    with pytest.raises(ValueError, match="ops_cross_field_owner_centroid"):
        ops_engine._ph_cross_field_options(
            {"ops_cross_field_owner_centroid": "median"})


def test_a_copy_cut_by_its_field_edge_still_never_owns():
    """The nearest field's copy is cut: the nearest whole field owns, as before."""
    offsets = {1: (0.0, 0.0), 2: (0.0, 60.0)}
    centres = _centres(offsets)
    cells = pd.DataFrame({"site": [1, 2, 1, 2], "label": [1, 1, 2, 2],
                          "area": 80.0, "whole": [False, True, False, False],
                          "well_y": [50.0, 50.0, 10.0, 10.0],
                          "well_x": [70.0, 71.0, 85.0, 86.0]})
    edges = pd.DataFrame({"site_a": [1, 1], "label_a": [1, 2],
                          "site_b": [2, 2], "label_b": [1, 2],
                          "distance": [1.0, 1.0]})
    for centroid in FO._FIELD_OWNER_CENTROIDS:
        owned = FO._assign_field_owners(cells, edges, centres=centres,
                                        centroid=centroid)
        assert owned.loc[owned.is_owner, ["site", "label"]].values.tolist() \
            == [[2, 1]]
        assert not owned.loc[~owned.whole.astype(bool), "is_owner"].any()
        cut_everywhere = owned[owned.label == 2]
        assert cut_everywhere.owner_field.isna().all()
    linked = FO._assign_field_owners(cells, edges, centres=centres)
    first = linked[linked.label == 1]
    assert (first.nearest_field == 1).all() and (first.owner_field == 2).all()


def _old_assign_field_owners(cells, edges, *, centres):
    """The ``'nearest_centre'`` owner of settings schema 965, kept verbatim."""
    cells = cells.reset_index(drop=True).copy()
    index = pd.Series(np.arange(len(cells)), index=pd.MultiIndex.from_arrays(
        [cells["site"].astype(np.int64), cells["label"].astype(np.int64)]))
    first = index.reindex(pd.MultiIndex.from_arrays(
        [edges["site_a"].astype(np.int64),
         edges["label_a"].astype(np.int64)])).to_numpy(float)
    second = index.reindex(pd.MultiIndex.from_arrays(
        [edges["site_b"].astype(np.int64),
         edges["label_b"].astype(np.int64)])).to_numpy(float)
    known = ~(np.isnan(first) | np.isnan(second))
    first, second = first[known].astype(np.int64), second[known].astype(
        np.int64)
    key = edges["distance"].to_numpy(float)[known]
    count = len(cells)
    component, refused = FO._copy_components(
        count, first, second, cells["site"].to_numpy(np.int64),
        np.argsort(key, kind="stable"))
    cells["component"] = component
    cells["n_copies"] = cells.groupby("component")["label"].transform("size")
    rank = cells[["component", "site", "label"]].copy()
    centre = np.array([centres[int(s)] for s in cells["site"]],
                      float).reshape(-1, 2)
    distance = np.hypot(cells["well_y"].to_numpy(float) - centre[:, 0],
                        cells["well_x"].to_numpy(float) - centre[:, 1])
    cells["centre_distance_px"] = distance
    rank["key"] = distance
    eligible = np.asarray(cells["whole"], bool)
    order = rank[eligible].sort_values(["component", "key", "site", "label"],
                                       kind="mergesort")
    owners = order.drop_duplicates("component")[["component", "site",
                                                 "label"]]
    owners.columns = ["component", "owner_field", "owner_label"]
    cells = cells.merge(owners, on="component", how="left")
    cells["is_owner"] = ((cells["site"] == cells["owner_field"])
                         & (cells["label"] == cells["owner_label"]))
    owner_area = cells.loc[cells["is_owner"]].set_index("component")["area"]
    cells["owned_area_share"] = cells["area"] / cells["component"].map(
        owner_area)
    cells.attrs["refused_edges"] = int(refused)
    return cells


def _neighbourhood(seed):
    """A 3 x 3 block of overlapping fields with jittered copies."""
    rng = np.random.default_rng(seed)
    offsets = {3 * i + j + 1: (60.0 * i, 60.0 * j)
               for i in range(3) for j in range(3)}
    rows = []
    for y, x in rng.uniform(0, 160, (400, 2)):
        for site, (oy, ox) in offsets.items():
            r, c = y - oy, x - ox
            if 0 <= r < SIDE and 0 <= c < SIDE:
                dy, dx = rng.normal(0, 2.0, 2)
                rows.append({"site": site, "well_y": y + dy, "well_x": x + dx,
                             "whole": min(r, SIDE - 1 - r, c,
                                          SIDE - 1 - c) > 5,
                             "area": float(rng.integers(50, 120))})
    cells = pd.DataFrame(rows)
    cells["label"] = cells.groupby("site").cumcount() + 1
    return offsets, cells


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_per_copy_is_byte_identical_to_the_old_owner(seed):
    offsets, cells = _neighbourhood(seed)
    centres = _centres(offsets)
    neighbours = FO.overlapping_fields(
        {s: FO.field_footprint(_transform(o), (SIDE, SIDE))
         for s, o in offsets.items()})
    edges = FO._match_copies_by_centroid(cells, neighbours, 6.0)
    new = FO._assign_field_owners(cells, edges, centres=centres,
                                  centroid="per_copy")
    old = _old_assign_field_owners(cells, edges, centres=centres)
    pd.testing.assert_frame_equal(new, old, check_exact=True)

    def parquet(frame):
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        return buffer.getvalue()

    assert parquet(new) == parquet(old)
    linked = FO._assign_field_owners(cells, edges, centres=centres)
    assert int(linked.is_owner.sum()) == int(old.is_owner.sum())
    assert (linked.owner_field.isna() == old.owner_field.isna()).all()


def test_settings_files_before_970_replay_the_per_copy_rule():
    from spacr.cli import _in_its_own_era

    assert S.SETTINGS_SCHEMA_VERSION >= 970
    assert S.DEFAULT_CHANGES[970] == {
        "ops_cross_field_owner_centroid": "per_copy"}
    assert OPS_DEFAULTS["ops_cross_field_owner_centroid"] == "linked"
    assert ops_engine.ops_defaults({})[
        "ops_cross_field_owner_centroid"] == "linked"
    known = set(OPS_DEFAULTS)
    for version in (0, 963, 965, 969):
        old = _in_its_own_era({S.SETTINGS_VERSION_KEY: version}, known)
        assert old["ops_cross_field_owner_centroid"] == "per_copy"
    new = _in_its_own_era({S.SETTINGS_VERSION_KEY: 970}, known)
    assert "ops_cross_field_owner_centroid" not in new
    chosen = _in_its_own_era({S.SETTINGS_VERSION_KEY: 965,
                              "ops_cross_field_owner_centroid": "linked"},
                             known)
    assert chosen["ops_cross_field_owner_centroid"] == "linked"


def test_the_setting_is_documented_and_categorised():
    text = OPS_TOOLTIPS["ops_cross_field_owner_centroid"]
    assert len(text) <= 600
    assert text.endswith("Default 'linked'. Alpha.")
    owners = OPS_CATEGORIES["Cell ownership"]
    assert "ops_cross_field_owner_centroid" in owners and len(owners) <= 20
