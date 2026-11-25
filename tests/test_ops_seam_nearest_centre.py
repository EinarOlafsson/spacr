"""Seam ownership: one owner per physical cell, the nearest field centre.

A cell in the overlap of two OPS phenotype fields is segmented in both. The
maintainer rule (2026-11-24): the cell belongs to the field whose centre is
nearest the cell centroid in stitched coordinates, among the fields that
hold it whole; ties go to the lower field. :mod:`spacr.field_ownership` is
the one implementation; these tests plant overlaps, ties, edge-cut cells
and a 3 x 3 neighbourhood and check that exactly the planted owners come
back, from centroids only, with no label image read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from spacr import field_ownership as FO
from spacr import ops_engine
from spacr import settings as S
from spacr.ops_merge import deduplicate_matches, field_footprint, overlapping_fields
from spacr.ops_store import write_table

PLATE = "P1"
SIDE = 100


def _transform(offset):
    return (1.0, np.eye(2), np.asarray(offset, float))


def _plant(offsets, cells, radius=5.0, jitter=0.0, seed=0):
    """Copies of physical cells in every field that sees their centroid.

    :param offsets: ``{site: (y, x)}`` field origins in the well.
    :param cells: ``(N, 2)`` well centroids of the physical cells.
    :returns: ``(copies, truth)``: one row per copy (``site``, ``label``,
        ``row``, ``col`` in field pixels, ``whole``, ``cell``) and the
        planted owner of each physical cell (None when no field holds it
        whole), by the rule.
    """
    rng = np.random.default_rng(seed)
    rows, truth = [], {}
    for cell, (y, x) in enumerate(np.asarray(cells, float)):
        best = None
        for site in sorted(offsets):
            oy, ox = offsets[site]
            r, c = y - oy, x - ox
            if not (0 <= r < SIDE and 0 <= c < SIDE):
                continue
            whole = min(r, SIDE - 1 - r, c, SIDE - 1 - c) > radius
            label = 1 + sum(1 for row in rows if row["site"] == site)
            dy, dx = rng.uniform(-jitter, jitter, 2) if jitter else (0.0, 0.0)
            rows.append({"site": site, "label": label, "row": r + dy,
                         "col": c + dx, "whole": whole, "cell": cell})
            if whole:
                distance = np.hypot(y - (oy + SIDE / 2), x - (ox + SIDE / 2))
                if best is None or distance < best[0] - 1e-9:
                    best = (distance, site)
        truth[cell] = None if best is None else best[1]
    return pd.DataFrame(rows), truth


def _fields(offsets, copies):
    fields = {}
    for site in offsets:
        mine = copies[copies.site == site]
        labels = mine["label"].to_numpy(np.int64)
        fields[site] = {
            "transform": _transform(offsets[site]), "shape": (SIDE, SIDE),
            "areas": pd.Series(np.full(len(mine), 80.0), index=labels),
            "whole": pd.Series(mine["whole"].to_numpy(bool), index=labels),
            "centroids": {int(l): (float(r), float(c)) for l, r, c in
                          zip(labels, mine["row"], mine["col"])},
            "cells": None}
    return fields


def _own(offsets, copies, **kwargs):
    kwargs.setdefault("metric", "centroid")
    kwargs.setdefault("radius", 8.0)

    def refuse(site):
        raise AssertionError("the centroid metric read a label image")

    owned, edges, pairs = ops_engine._ph_cross_field_fields(
        _fields(offsets, copies), load=refuse, **kwargs)
    return owned.merge(copies[["site", "label", "cell"]], on=["site", "label"])


def _owners(owned):
    """``{physical cell: owner site}`` and the copies owned per cell."""
    owners = owned[owned.is_owner].groupby("cell")["site"].apply(list)
    return {int(c): s for c, s in owners.items()}


def test_a_whole_cell_in_two_fields_has_one_owner_the_nearer_centre():
    offsets = {1: (0, 0), 2: (0, 60)}
    copies, truth = _plant(offsets, [(50, 75), (50, 85)])
    owners = _owners(_own(offsets, copies))
    assert owners == {0: [1], 1: [2]}
    assert truth == {0: 1, 1: 2}


def test_a_tie_goes_to_the_lower_field():
    offsets = {1: (0, 0), 2: (0, 60)}
    copies, _ = _plant(offsets, [(50, 80)])
    assert _owners(_own(offsets, copies)) == {0: [1]}
    swapped = {7: (0, 0), 3: (0, 60)}
    copies, _ = _plant(swapped, [(50, 80)])
    assert _owners(_own(swapped, copies)) == {0: [3]}


def test_a_cell_cut_by_the_nearer_field_is_owned_by_the_field_that_holds_it_whole():
    offsets = {1: (0, 0), 2: (0, 60)}
    copies, truth = _plant(offsets, [(50, 63)], radius=5.0)
    nearer = copies[(copies.site == 2)]
    assert not bool(nearer.whole.iloc[0])
    owned = _own(offsets, copies)
    assert _owners(owned) == {0: [1]} and truth == {0: 1}


def test_a_cell_cut_by_every_field_that_holds_it_has_no_owner():
    offsets = {1: (0, 0)}
    copies, truth = _plant(offsets, [(50, 97)], radius=5.0)
    owned = _own(offsets, copies)
    assert truth == {0: None} and not owned.is_owner.any()
    assert owned.owner_field.isna().all()
    assert FO.owner_counts(owned) == (1, 0, 1)


@pytest.mark.parametrize("jitter", [0.0, 3.0])
def test_a_three_by_three_neighbourhood_gives_every_whole_cell_one_owner(jitter):
    offsets = {3 * i + j + 1: (60 * i, 60 * j) for i in range(3) for j in range(3)}
    rng = np.random.default_rng(4)
    span = 60 * 2 + SIDE
    grid = np.array([(y, x) for y in range(4, span - 4, 26)
                     for x in range(4, span - 4, 26)], float)
    grid += rng.uniform(-2, 2, grid.shape)
    copies, truth = _plant(offsets, grid, jitter=jitter, seed=1)
    owned = _own(offsets, copies)
    owners = _owners(owned)
    for cell, site in truth.items():
        if site is None:
            assert cell not in owners
        else:
            assert len(owners[cell]) == 1
            if not jitter:
                assert owners[cell] == [site]
    seen = copies.groupby("cell").size()
    assert (seen > 1).sum() > 25
    edge_only = copies[copies.whole].groupby("cell").size()
    assert (edge_only > 1).sum() > 12
    assert int(owned.is_owner.sum()) == sum(v is not None for v in truth.values())
    if not jitter:
        physical, _owned, unowned = FO.owner_counts(owned)
        assert physical == len(truth)
        assert unowned == sum(v is None for v in truth.values()) > 0


def test_the_old_rule_survives_only_as_a_named_deprecated_option():
    assert FO.FIELD_OWNER_RULES[0] == "nearest_centre"
    assert FO.DEPRECATED_FIELD_OWNER_RULES == ("largest",)
    with pytest.raises(ValueError, match="field owner rule"):
        FO.field_owner_rule("deepest")
    offsets = {1: (0, 0), 2: (0, 60)}
    copies, _ = _plant(offsets, [(50, 80)])
    cells = copies.assign(area=[50.0, 90.0], well_y=0.0, well_x=0.0)
    owned = FO.assign_field_owners(
        cells, pd.DataFrame({"site_a": [1], "label_a": [1], "site_b": [2],
                             "label_b": [1]}), rule="largest")
    assert owned.loc[owned.is_owner, "site"].tolist() == [2]


def _run_folder(tmp_path):
    """Two overlapping fields saved as a run without label images."""
    offsets = {0: (0.0, 0.0), 1: (0.0, 60.0)}
    copies, truth = _plant(offsets, [(50, 75), (50, 85), (50, 63), (20, 20)])
    placed, passed, measured = [], [], []
    for site, (oy, ox) in offsets.items():
        placed.append({"plate": PLATE, "well": "A1", "site": site,
                       "centre_y": oy + 50.0, "centre_x": ox + 50.0,
                       "is_anchor": 1, "scale": 1.0, "rotation_deg": 0.0,
                       "offset_y": oy, "offset_x": ox, "field_height": SIDE,
                       "field_width": SIDE, "transform_source": "anchor",
                       "refined": 0})
    for row in copies.itertuples():
        passed.append({"plate": PLATE, "well": "A1", "site": row.site,
                       "label": row.label, "cell_px": 80,
                       "touches_field_edge": not row.whole})
        measured.append({"plate": PLATE, "well": "A1", "site": row.site,
                         "label": row.label, "centroid_row": row.row,
                         "centroid_col": row.col})
    db = str(tmp_path / "measurements.db")
    write_table(db, "ops_phenotype", pd.DataFrame(placed))
    pd.DataFrame(passed).to_parquet(
        ops_engine._ph_maskpass_path(str(tmp_path), "A1"))
    pd.DataFrame(measured).to_parquet(
        ops_engine._ph_cells_path(str(tmp_path), "A1"))
    return db, copies, truth


def test_the_default_phase_needs_no_label_images_and_owns_each_cell_once(tmp_path):
    _db, copies, truth = _run_folder(tmp_path)
    assert not (tmp_path / "A1" / "phenotype_masks").exists()
    settings = {"dst_root": str(tmp_path), "plate": PLATE}
    report = ops_engine.run_ops(settings, phases=("ph_ownership",))[
        "wells"]["A1"]["ph_ownership"]
    assert report["metric"] == "centroid" and report["rule"] == "nearest_centre"
    assert report["physical_cells"] == 4 and report["owned_cells"] == 4
    cells = copies[["site", "label"]]
    full = ops_engine.ops_defaults(settings)
    owned = ops_engine._ph_owned_cells(str(tmp_path), "A1", cells, full)
    got = {}
    for (cell, flag, site) in zip(copies.cell, owned, copies.site):
        if flag:
            got.setdefault(int(cell), []).append(int(site))
    assert got == {c: [s] for c, s in truth.items()}
    off = ops_engine._ph_owned_cells(
        str(tmp_path), "A1", cells,
        {**full, "ops_cross_field_ownership": False})
    assert int(off.sum()) == int(copies.whole.sum()) == 6


def test_the_merge_gives_the_sequencing_link_to_the_owner_not_the_closer_copy(tmp_path):
    _db, copies, _ = _run_folder(tmp_path)
    settings = ops_engine.ops_defaults({"dst_root": str(tmp_path),
                                        "plate": PLATE})
    ops_engine.run_ops(settings, phases=("ph_ownership",))
    owners = pd.read_parquet(ops_engine._ph_cross_field_path(str(tmp_path), "A1"))
    shared = owners[owners.n_copies == 2]
    owner = shared[shared.is_owner].iloc[0]
    other = shared[~shared.is_owner].iloc[0]
    links = pd.DataFrame({"site": [int(owner.site), int(other.site)],
                          "label": [int(owner.label), int(other.label)],
                          "object_id": [5, 5], "distance": [2.0, 0.5]})
    ranked = ops_engine._ph_owner_first_links(links, str(tmp_path), "A1",
                                              settings)
    kept, dropped = deduplicate_matches(ranked, by=("_owner_rank", "distance"))
    assert dropped == 1 and kept.site.tolist() == [int(owner.site)]
    closer, _ = deduplicate_matches(links)
    assert closer.site.tolist() == [int(other.site)]
    untouched = ops_engine._ph_owner_first_links(
        links, str(tmp_path), "A1", {**settings, "ops_cross_field_ownership": False})
    assert untouched is links


def test_the_merge_runs_ownership_first_when_it_was_not_named(tmp_path):
    _run_folder(tmp_path)
    settings = ops_engine.ops_defaults({})
    assert ops_engine._ph_ownership_with_merge(settings, ("merge",),
                                               str(tmp_path), "A1")
    assert not ops_engine._ph_ownership_with_merge(
        settings, ("merge", "ph_ownership"), str(tmp_path), "A1")
    assert not ops_engine._ph_ownership_with_merge(
        {**settings, "ops_cross_field_ownership": False}, ("merge",),
        str(tmp_path), "A1")
    assert not ops_engine._ph_ownership_with_merge(settings, ("merge",),
                                                   str(tmp_path), "B2")


def test_a_well_without_a_mask_pass_keeps_its_rows(tmp_path):
    cells = pd.DataFrame({"well": "C3", "site": [0, 1], "label": [1, 1]})
    kept, report = ops_engine._ph_cross_field_filter(cells, str(tmp_path),
                                                     ["C3"], {})
    assert len(kept) == 2 and report["wells_without_mask_pass"] == ["C3"]


def test_ph_rebuild_names_the_phases_that_rebuild_tables_from_stored_masks():
    assert ops_engine._run_phases({}, ("ph_rebuild",)) == (
        "ph_ownership", "merge", "aggregate", "ph_tables")
    assert ops_engine._run_phases({"ops_phases": "ph_rebuild,ph_crops"}, None) == (
        "ph_ownership", "merge", "aggregate", "ph_tables", "ph_crops")
    assert ops_engine._run_phases({}, ("merge", "ph_rebuild")) == (
        "merge", "ph_ownership", "aggregate", "ph_tables")
    segmenting = {"ph_segment", "ph_measure", "ph_masks"}
    assert not segmenting & set(ops_engine._PH_REBUILD_PHASES)


def test_schema_963_keeps_the_old_rule_for_old_settings_files():
    assert S.SETTINGS_SCHEMA_VERSION >= 963
    assert S.DEFAULT_CHANGES[963] == {
        "ops_cross_field_ownership": False,
        "ops_cross_field_metric": "overlap_fraction",
        "ops_cross_field_policy": "one_per_cell",
        "ops_cross_field_rule": "largest"}
    known = set(S.DEFAULT_CHANGES[963])
    old = {}
    S._fold_older_defaults(old, 962, known=known)
    assert old == S.DEFAULT_CHANGES[963]
    new = {}
    S._fold_older_defaults(new, 963, known=known)
    assert new == {}
    defaults = ops_engine.ops_defaults({})
    assert defaults["ops_cross_field_ownership"] is True
    assert defaults["ops_cross_field_metric"] == "centroid"
    assert defaults["ops_cross_field_rule"] == "nearest_centre"
    assert defaults["ops_cross_field_policy"] == "owner_only"


def test_overlapping_fields_of_the_neighbourhood_are_its_eight_neighbours():
    offsets = {3 * i + j + 1: (60 * i, 60 * j) for i in range(3) for j in range(3)}
    prints = {s: field_footprint(_transform(o), (SIDE, SIDE)) for s, o in offsets.items()}
    assert sorted(overlapping_fields(prints)[5]) == [1, 2, 3, 4, 6, 7, 8, 9]
