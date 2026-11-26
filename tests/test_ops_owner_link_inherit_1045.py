"""Item 1045: the owner copy of a seam cell inherits the physical cell's SBS link.

Two overlapping phenotype fields image the same cells. The cross-field
ownership pass (1026, the 'linked' centroid) names one owner copy per
physical cell, its ``component``; the merge pairs copies to SBS nuclei by
the closer match, so a twin can hold the link the owner should carry. With
``ops_owner_link_inherit`` 'inherit' the owner takes that link and the twin
keeps none; with 'inherit_nn' (the default) an owner whose physical cell has
no link at all also takes the nearest SBS nucleus within 3 px that no cell
holds. Copies linked to two nuclei are a reported conflict; 'off', the
value for files stamped before schema 982, leaves the links where the merge
put them.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from spacr import ops_engine
from spacr import settings as S
from spacr.ops_merge import field_transform
from spacr.ops_store import read_table, write_table

PLATE = "P1"
SIZE = 200
XS = (50.0, 160.0, 190.0, 300.0, 170.0)


def _run(tmp_path):
    """Fields 0 and 1 overlap in well x 150-200; field 1's copies match closer.

    Physical cells at well x 50 (field 0 only), 160 (both; owner field 0,
    whose centre is nearer), 190 (both; owner field 1), 300 (field 1 only)
    and 170 (both, no SBS nucleus). SBS nuclei 1-4 sit at 50, 160, 190, 300.
    """
    db = str(tmp_path / "measurements.db")
    rows = []
    for site, cx in ((0, 100.0), (1, 250.0)):
        transform = field_transform(1.0, 0.0, (100.0, cx), (SIZE, SIZE))
        rows.append({"plate": PLATE, "well": "A1", "site": site,
                     "centre_y": 100.0, "centre_x": cx, "is_anchor": 1,
                     "scale": 1.0, "rotation_deg": 0.0,
                     "offset_y": float(transform[2][0]),
                     "offset_x": float(transform[2][1]),
                     "field_height": SIZE, "field_width": SIZE,
                     "transform_source": "anchor", "refined": 0})
    write_table(db, "ops_phenotype", pd.DataFrame(rows))
    write_table(db, "ops_objects", pd.DataFrame({
        "plate": PLATE, "well": "A1", "object_id": [1, 2, 3, 4],
        "centroid_y": 100.0, "centroid_x": [50.0, 160.0, 190.0, 300.0]}))
    cells = []
    for site, origin, jitter in ((0, 0.0, 1.0), (1, 150.0, 0.5)):
        label = 0
        for x in XS:
            local = x - origin
            if not 0 <= local <= SIZE:
                continue
            label += 1
            cells.append({"plate": PLATE, "well": "A1", "site": site,
                          "label": label,
                          "centroid_row": 100.0 + (0.0 if x == 170.0 else jitter),
                          "centroid_col": local, "well_x": x})
    frame = pd.DataFrame(cells)
    frame.drop(columns="well_x").to_parquet(
        ops_engine._ph_cells_path(str(tmp_path), "A1"))
    frame[["plate", "well", "site", "label"]].assign(
        cell_px=80, touches_field_edge=False).to_parquet(
        ops_engine._ph_maskpass_path(str(tmp_path), "A1"))
    return db, frame


def _settings(**extra):
    return ops_engine.ops_defaults({"ops_align_model": "similarity", **extra})


def _merge(db, tmp_path, settings):
    report = ops_engine._ph_merge(db, PLATE, "A1", settings, str(tmp_path),
                                  False)
    return report, read_table(db, "ops_merge")


def _own(db, tmp_path, settings):
    return ops_engine._ph_cross_field_ownership(db, PLATE, "A1", settings,
                                                str(tmp_path))


def _link_sites(links, frame):
    """``{well x of the physical cell: [sites holding its link]}``."""
    got = links.merge(frame[["site", "label", "well_x"]], on=["site", "label"])
    return {x: sorted(g.site.astype(int)) for x, g in got.groupby("well_x")}


def _owner_link_share(links, tmp_path, settings, frame):
    """Owner copies (the tables' rows) linked, of all owner copies."""
    owned = ops_engine._ph_owned_cells(str(tmp_path), "A1",
                                       frame[["site", "label"]], settings)
    kept = frame[owned][["site", "label"]]
    linked = kept.merge(links[["site", "label"]], on=["site", "label"])
    return len(linked), len(kept)


def _sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def test_the_owner_inherits_the_twin_link_and_the_twin_keeps_no_row(tmp_path):
    db, frame = _run(tmp_path)
    on = _settings()
    off = _settings(ops_owner_link_inherit="off")
    no_table = {**on, "ops_cross_field_ownership": False}
    _report, merged = _merge(db, tmp_path, no_table)
    assert _link_sites(merged, frame)[160.0] == [1]
    _own(db, tmp_path, on)
    before = ops_engine._ph_merge_links(db, PLATE, ["A1"], str(tmp_path), off)
    after = ops_engine._ph_merge_links(db, PLATE, ["A1"], str(tmp_path), on)
    assert _link_sites(before, frame) == {50.0: [0], 160.0: [1], 190.0: [1],
                                          300.0: [1]}
    assert _link_sites(after, frame) == {50.0: [0], 160.0: [0], 190.0: [1],
                                         300.0: [1]}
    assert len(after) == len(before) == 4
    assert sorted(after.object_id) == [1, 2, 3, 4]
    assert _owner_link_share(before, tmp_path, on, frame) == (3, 5)
    assert _owner_link_share(after, tmp_path, on, frame) == (4, 5)
    report, merged = _merge(db, tmp_path, on)
    assert _link_sites(merged, frame) == _link_sites(after, frame)
    assert report["owner_link"]["conflicting_cells"] == 0
    pd.testing.assert_frame_equal(
        ops_engine._ph_merge_links(db, PLATE, ["A1"], str(tmp_path), on),
        ops_engine._ph_merge_links(db, PLATE, ["A1"]))
    genotype_rows = merged.merge(frame[["site", "label", "well_x"]],
                                 on=["site", "label"])
    assert int((genotype_rows.well_x == 160.0).sum()) == 1


def test_copies_linked_to_two_nuclei_are_a_reported_conflict_not_a_pick(tmp_path):
    db, frame = _run(tmp_path)
    on = _settings()
    _own(db, tmp_path, on)
    owners = ops_engine._ph_cross_field_owners(str(tmp_path), "A1")
    copies = owners.merge(frame[["site", "label", "well_x"]],
                          on=["site", "label"])
    owner = copies[(copies.well_x == 160.0) & copies.is_owner].iloc[0]
    twin = copies[(copies.well_x == 160.0) & ~copies.is_owner].iloc[0]
    links = pd.DataFrame({"well": "A1",
                          "site": [int(owner.site), int(twin.site)],
                          "label": [int(owner.label), int(twin.label)],
                          "object_id": [2, 7], "distance": [1.0, 0.5]})
    counts = {}
    kept = ops_engine._ph_inherit_owner_links(links, owners, counts)
    pd.testing.assert_frame_equal(kept, links)
    assert counts["conflicting_cells"] == 1 and counts["conflicting_links"] == 2
    assert counts["links_moved_to_owner"] == 0
    assert sorted(counts["conflicts"].object_id) == [2, 7]
    lone = links.iloc[[1]].reset_index(drop=True)
    moved = ops_engine._ph_inherit_owner_links(lone, owners, counts)
    assert moved[["site", "label", "object_id"]].values.tolist() == [
        [int(owner.site), int(owner.label), 7]]
    assert counts["conflicting_cells"] == 0
    write_table(db, "ops_objects", pd.DataFrame({
        "plate": PLATE, "well": "A1", "object_id": [1, 2, 3, 4, 7],
        "centroid_y": [100.0] * 4 + [101.0],
        "centroid_x": [50.0, 160.0, 190.0, 300.0, 160.0]}), if_exists="replace")
    report, merged = _merge(db, tmp_path, on)
    table = pd.read_parquet(ops_engine._ph_owner_link_conflicts_path(
        str(tmp_path), "A1"))
    assert report["owner_link"]["conflicting_cells"] == len(
        table.component.unique())
    for component, group in table.groupby("component"):
        assert group.object_id.nunique() > 1


def test_off_replays_the_merge_byte_for_byte(tmp_path):
    db, frame = _run(tmp_path)
    off = _settings(ops_owner_link_inherit="off")
    _own(db, tmp_path, off)
    _report, merged = _merge(db, tmp_path, off)
    links = ops_engine._ph_merge_links(db, PLATE, ["A1"])
    assert ops_engine._ph_inherited_well_links(links, str(tmp_path), off) is links
    assert not os.path.isfile(ops_engine._ph_owner_link_conflicts_path(
        str(tmp_path), "A1"))
    assert "owner_link" not in _report
    flag_only = _settings(ops_cross_field_policy="flag_only")
    assert not ops_engine._ph_owner_link_inherit(flag_only)
    assert ops_engine._ph_inherited_well_links(
        links, str(tmp_path), flag_only) is links
    pd.testing.assert_frame_equal(
        ops_engine._ph_merge_links(db, PLATE, ["A1"], str(tmp_path), off),
        links)


def test_the_982_row_replays_the_old_links_for_older_files():
    assert S.SETTINGS_SCHEMA_VERSION >= 982
    assert S.DEFAULT_CHANGES[982] == {"ops_owner_link_inherit": "off"}
    assert ops_engine.ops_defaults({})["ops_owner_link_inherit"] == "inherit_nn"
    keys = set(ops_engine.ops_defaults({}))
    old = {"plate": "P1"}
    applied = S._fold_older_defaults(old, 981, known=keys)
    assert applied["ops_owner_link_inherit"] == ("off", 982)
    assert old["ops_owner_link_inherit"] == "off"
    assert ops_engine._ph_owner_link_mode(ops_engine.ops_defaults(old)) == "off"
    explicit = {"ops_owner_link_inherit": "inherit"}
    S._fold_older_defaults(explicit, 981, known=keys)
    assert explicit["ops_owner_link_inherit"] == "inherit"
    with pytest.raises(ValueError, match="ops_owner_link_inherit"):
        ops_engine._ph_owner_link_mode({"ops_owner_link_inherit": "nearest"})
    current = {"plate": "P1"}
    assert "ops_owner_link_inherit" not in S._fold_older_defaults(
        current, 982, known=keys)


def test_inheriting_links_leaves_the_physical_cells_unchanged(tmp_path):
    db, frame = _run(tmp_path)
    on = _settings()
    off = _settings(ops_owner_link_inherit="off")
    report = _own(db, tmp_path, on)
    path = ops_engine._ph_cross_field_path(str(tmp_path), "A1")
    sha = _sha(path)
    owned = {}
    for name, settings in (("off", off), ("inherit", _settings(
            ops_owner_link_inherit="inherit")), ("on", on)):
        _merge(db, tmp_path, settings)
        assert _sha(path) == sha
        owned[name] = ops_engine._ph_owned_cells(
            str(tmp_path), "A1", frame[["site", "label"]], settings)
    np.testing.assert_array_equal(owned["off"], owned["on"])
    np.testing.assert_array_equal(owned["off"], owned["inherit"])
    table = pd.read_parquet(path)
    assert table.component.nunique() == report["physical_cells"] == 5


def _owners_table(rows):
    """A cross-field table: ``(site, label, component, is_owner)`` rows."""
    return pd.DataFrame(rows, columns=["site", "label", "component",
                                       "is_owner"]).assign(well="A1")


def _no_links():
    return pd.DataFrame({"site": pd.Series([], dtype=np.int64),
                         "label": pd.Series([], dtype=np.int64),
                         "object_id": pd.Series([], dtype=np.int64),
                         "distance": pd.Series([], dtype=float)})


def test_an_unlinked_owner_takes_the_nearest_unused_nucleus():
    owners = _owners_table([(0, 1, 0, True), (1, 1, 0, False),
                            (0, 2, 1, True)])
    positions = pd.DataFrame({"site": [0, 1, 0], "label": [1, 1, 2],
                              "well_y": [0.0, 0.0, 50.0],
                              "well_x": [0.0, 0.0, 50.0]})
    target = np.array([[0.0, 2.0], [0.0, 1.0], [50.0, 52.0]])
    counts = {}
    out = ops_engine._ph_nearest_unused_links(
        _no_links(), owners, positions, target, [10, 11, 12], counts=counts)
    got = {(int(r.site), int(r.label)): (int(r.object_id), r.distance)
           for r in out.itertuples()}
    assert got == {(0, 1): (11, 1.0), (0, 2): (12, 2.0)}
    assert counts == {"nearest_unused_links": 2, "owners_without_link": 2}
    twin_linked = pd.DataFrame({"site": [1], "label": [1], "object_id": [11],
                                "distance": [0.5]})
    out = ops_engine._ph_nearest_unused_links(
        twin_linked, owners, positions, target, [10, 11, 12])
    assert out[["site", "label", "object_id"]].values.tolist() == [
        [1, 1, 11], [0, 2, 12]]


def test_a_nucleus_another_cell_holds_is_skipped():
    owners = _owners_table([(0, 1, 0, True), (0, 2, 1, True)])
    positions = pd.DataFrame({"site": [0, 0], "label": [1, 2],
                              "well_y": [0.0, 40.0], "well_x": [0.0, 40.0]})
    target = np.array([[0.0, 1.0], [0.0, 2.5], [40.0, 40.0]])
    held = pd.DataFrame({"site": [0], "label": [2], "object_id": [20],
                         "distance": [0.0]})
    out = ops_engine._ph_nearest_unused_links(
        held, owners, positions, target, [21, 22, 20])
    assert out[["site", "label", "object_id"]].values.tolist() == [
        [0, 2, 20], [0, 1, 21]]
    taken = pd.DataFrame({"site": [9], "label": [9], "object_id": [21],
                          "distance": [0.0]})
    out = ops_engine._ph_nearest_unused_links(
        pd.concat([held, taken], ignore_index=True), owners, positions,
        target, [21, 22, 20])
    assert out.set_index(["site", "label"]).loc[(0, 1), "object_id"] == 22


def test_three_px_exactly_links_and_just_beyond_does_not():
    owners = _owners_table([(0, 1, 0, True), (0, 2, 1, True)])
    positions = pd.DataFrame({"site": [0, 0], "label": [1, 2],
                              "well_y": [0.0, 100.0], "well_x": [0.0, 100.0]})
    target = np.array([[3.0, 0.0], [100.0, 103.001]])
    out = ops_engine._ph_nearest_unused_links(
        _no_links(), owners, positions, target, [1, 2])
    assert out[["site", "label", "object_id"]].values.tolist() == [[0, 1, 1]]
    assert out.distance.tolist() == [3.0]
    assert ops_engine._PH_OWNER_LINK_NN_PX == 3.0


def test_two_owner_copies_never_take_the_same_nucleus():
    rng = np.random.default_rng(0)
    n = 60
    owners = _owners_table([(0, i + 1, i, True) for i in range(n)])
    cells = rng.uniform(0, 20, (n, 2))
    positions = pd.DataFrame({"site": 0, "label": np.arange(1, n + 1),
                              "well_y": cells[:, 0], "well_x": cells[:, 1]})
    target = rng.uniform(0, 20, (25, 2))
    out = ops_engine._ph_nearest_unused_links(
        _no_links(), owners, positions, target, np.arange(100, 125))
    assert not out.object_id.duplicated().any()
    assert not out.duplicated(["site", "label"]).any()
    ops_engine._ph_one_link_per_cell(out)
    assert (out.distance <= 3.0).all() and len(out) > 0
    pair = _owners_table([(0, 1, 0, True), (0, 2, 1, True)])
    same = pd.DataFrame({"site": [0, 0], "label": [1, 2],
                         "well_y": [0.0, 0.0], "well_x": [-1.0, 1.0]})
    out = ops_engine._ph_nearest_unused_links(
        _no_links(), pair, same, np.array([[0.0, 0.0]]), [5])
    assert out[["site", "label", "object_id"]].values.tolist() == [[0, 1, 5]]


def test_the_merge_gives_an_unlinked_owner_its_nearest_unused_nucleus(tmp_path):
    db, frame = _run(tmp_path)
    write_table(db, "ops_objects", pd.DataFrame({
        "plate": PLATE, "well": "A1", "object_id": [1, 2, 3, 4, 5],
        "centroid_y": 100.0,
        "centroid_x": [50.0, 160.0, 190.0, 300.0, 172.5]}), if_exists="replace")
    base = dict(ops_merge_threshold=1.5)
    _own(db, tmp_path, _settings(**base))
    sites = {}
    for mode in ("off", "inherit", "inherit_nn"):
        report, merged = _merge(db, tmp_path, _settings(
            ops_owner_link_inherit=mode, **base))
        sites[mode] = _link_sites(merged, frame)
        assert not merged.object_id.duplicated().any()
        if mode == "inherit_nn":
            assert report["owner_link"]["nearest_unused_links"] == 1
            row = merged.merge(frame[["site", "label", "well_x"]],
                               on=["site", "label"])
            row = row[row.well_x == 170.0].iloc[0]
            assert int(row.object_id) == 5 and 1.5 < row.distance <= 3.0
            assert (merged.plate == PLATE).all() and (merged.well == "A1").all()
    assert 170.0 not in sites["off"] and 170.0 not in sites["inherit"]
    assert sites["inherit_nn"][170.0] == [0]
