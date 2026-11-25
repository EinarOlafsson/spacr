"""OPS and Measure correctness audit before plate A (L5).

Checked:

* ph_tables and the crops refuse a well whose cross-field ownership table
  is missing while ``ops_cross_field_ownership`` is on, and report how many
  copies another field owns;
* the keys of ``_ph_table_frames`` cannot join the features by position
  after a link table with a repeated cell;
* one rule decides the outside ring of every object, so the outside ring
  block of Measure and the ``outside`` profiling family write the same
  ``nucleus_..._outside_`` values, and ``'background'`` replays the old
  profiling values;
* ``haralick_correlation`` is NaN for a flat object;
* ph_tables replaces only the wells it writes: other wells stay registered,
  A1 then A2 into one store gives the parts one run of both gives, and a
  rerun of a well is byte-identical.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

from spacr import measure, ops_engine, settings as spacr_settings, store  # noqa: E402
from tests.test_ops_phtables_store import (                            # noqa: E402
    PLATE, SETTINGS, _build, _field, _hashes, _owners, _tables,
    shipped_dtype_map)                                                 # noqa: F401


def test_ph_tables_refuses_a_well_without_its_cross_field_table(tmp_path):
    run, _cells = _build(tmp_path)
    os.remove(ops_engine._ph_cross_field_path(str(run), "A1"))
    with pytest.raises(ValueError, match="no cross-field ownership table"):
        _tables(run)
    report = _tables(run, ops_cross_field_ownership=False)
    assert report["rows"]["cell"] == len(_cells) - 2
    flag_only = _tables(run, ops_cross_field_policy="flag_only")
    assert flag_only["rows"]["cell"] == len(_cells) - 2


def test_the_crops_and_the_owned_cells_refuse_it_too(tmp_path):
    run, cells = _build(tmp_path)
    os.remove(ops_engine._ph_cross_field_path(str(run), "A1"))
    with pytest.raises(ValueError, match="ph_ownership"):
        ops_engine._ph_owned_cells(str(run), "A1", cells, SETTINGS)
    assert ops_engine._ph_wells_without_cross_field_table(
        str(run), ["A1"], SETTINGS) == ["A1"]
    assert ops_engine._ph_wells_without_cross_field_table(
        str(run), ["A1"], {**SETTINGS, "ops_cross_field_ownership": False}) == []


def test_the_report_counts_the_copies_another_field_owns(tmp_path):
    run, cells = _build(tmp_path)
    owners = pd.read_parquet(ops_engine._ph_cross_field_path(str(run), "A1"))
    owners.loc[[5, 6, 7], "is_owner"] = False
    owners.to_parquet(ops_engine._ph_cross_field_path(str(run), "A1"),
                      index=False)
    report = _tables(run)
    well = report["wells"]["A1"]
    assert well["copies_owned_elsewhere"] == 3
    assert well["cut_at_field_edge"] == 2
    assert report["copies_owned_elsewhere"] == 3
    assert well["rows"] == len(cells) - 5


def test_a_repeated_link_cannot_shift_the_keys(tmp_path):
    rng = np.random.default_rng(1)
    cells = _field(3, rng)
    links = cells[["site", "label"]].copy()
    links["object_id"] = np.arange(len(links))
    links = pd.concat([links, links.iloc[:1]], ignore_index=True)
    owned = np.ones(len(cells), bool)
    with pytest.raises(ValueError, match="more than once"):
        ops_engine._ph_table_frames(PLATE, "A1", cells, links, None, owned,
                                    "rule")
    keys, features = ops_engine._ph_table_frames(
        PLATE, "A1", cells, links.iloc[:-1], None, owned, "rule")
    assert len(keys) == len(features["cell"]) == len(cells)
    assert list(keys["object_label"]) == list(cells["label"])


def _touching():
    labels = np.zeros((40, 40), np.int32)
    labels[5:15, 5:15] = 1
    labels[5:15, 17:27] = 2
    labels[25:35, 8:18] = 3
    rng = np.random.default_rng(0)
    image = rng.gamma(3, 100, (40, 40, 2))
    image[5:15, 17:27] += 2000
    return labels, image


def test_one_rule_decides_every_outside_ring():
    assert measure._outside_excludes_foreground("cell")
    assert measure._outside_excludes_foreground("cytoplasm")
    assert not measure._outside_excludes_foreground("nucleus")
    assert not measure._outside_excludes_foreground("pathogen")
    assert measure._outside_excludes_foreground("nucleus", "background")
    assert measure._outside_excludes_foreground(None)
    with pytest.raises(ValueError, match="profile_outside_rule"):
        measure._outside_excludes_foreground("nucleus", "ring")


def test_the_profiling_family_writes_what_the_outside_block_writes():
    labels, image = _touching()
    block = measure._outside_intensity(
        labels, image[..., 0],
        exclude_foreground=measure._outside_excludes_foreground("nucleus"))
    family = measure._measure_profile_features(labels, image, ["outside"],
                                              role="nucleus")
    assert np.allclose(family["channel_0_outside_mean"],
                       [row[1] for row in block])
    old = measure._measure_profile_features(labels, image, ["outside"],
                                           role="nucleus",
                                           outside_rule="background")
    assert not np.allclose(old["channel_0_outside_mean"],
                           family["channel_0_outside_mean"])
    assert np.allclose(old["channel_0_outside_mean"],
                       measure.measure_outside(labels, image)[
                           "channel_0_outside_mean"])
    cell = measure._measure_profile_features(labels, image, ["outside"],
                                            role="cell")
    assert np.allclose(cell["channel_0_outside_mean"],
                       old["channel_0_outside_mean"])


def test_older_settings_files_replay_the_old_ring():
    assert spacr_settings.SETTINGS_SCHEMA_VERSION >= 965
    assert spacr_settings.DEFAULT_CHANGES[965] == {
        "profile_outside_rule": "background",
        "ops_ph_profile_outside_rule": "background"}
    from spacr.ops_settings import OPS_TO_MEASURE_KEYS, ops_defaults

    assert OPS_TO_MEASURE_KEYS["ops_ph_profile_outside_rule"] == \
        "profile_outside_rule"
    assert ops_defaults({})["ops_ph_profile_outside_rule"] == "object"


def test_the_haralick_correlation_of_a_flat_object_is_nan():
    labels = np.zeros((30, 30), np.int32)
    labels[2:12, 2:12] = 1
    labels[15:28, 15:28] = 2
    image = np.random.default_rng(2).random((30, 30, 1))
    image[2:12, 2:12, 0] = 7.0
    table = measure.measure_texture(labels, image, pftas=False)
    flat, varying = table["channel_0_haralick_correlation"]
    assert np.isnan(flat)
    assert np.isfinite(varying) and varying != 1.0
    assert np.isfinite(table["channel_0_haralick_asm"].iloc[0])


def _two_wells(root):
    run, cells = _build(root)
    rng = np.random.default_rng(5)
    folder = run / "A2" / "phenotype_cells"
    folder.mkdir(parents=True)
    frames = []
    for site in (1, 2):
        frame = _field(site, rng)
        frame.to_parquet(folder / f"site-{site}.parquet", index=False)
        frames.append(frame)
    other = pd.concat(frames, ignore_index=True)
    other.insert(0, "well", "A2")
    other.insert(0, "plate", PLATE)
    other.to_parquet(ops_engine._ph_cells_path(str(run), "A2"), index=False)
    edge = other[["site", "label"]].copy()
    edge["touches_field_edge"] = False
    edge.to_parquet(ops_engine._ph_maskpass_path(str(run), "A2"), index=False)
    _owners(run, "A2", other)
    return run, cells, other


def _run(run, wells, **extra):
    return ops_engine._ph_tables(str(run / "measurements.db"), PLATE, wells,
                                 {**SETTINGS, "dst_root": str(run), **extra},
                                 str(run))


def _rows(folder):
    with store.open(folder) as opened:
        cell = opened.query(table="cell")
        wells = sorted(set(cell["well"].astype(str)))
    return len(cell), wells


def test_one_store_holds_a_plate_written_well_by_well(tmp_path):
    together, cells, other = _two_wells(tmp_path / "together")
    _run(together, ["A1", "A2"])
    apart, _c, _o = _two_wells(tmp_path / "apart")
    _run(apart, ["A1"])
    _run(apart, ["A2"])
    assert _hashes(str(apart / "measurements")) == _hashes(
        str(together / "measurements"))
    assert _rows(str(apart / "measurements")) == (
        len(cells) - 2 + len(other), ["A1", "A2"])
    before = _hashes(str(apart / "measurements"))
    _run(apart, ["A1"])
    assert _hashes(str(apart / "measurements")) == before
    assert _rows(str(apart / "measurements"))[1] == ["A1", "A2"]


def test_a_well_dropped_from_the_layout_leaves_no_stale_wide_table(tmp_path):
    run, _cells, _other = _two_wells(tmp_path)
    _run(run, ["A1", "A2"], ops_tables_layout="wide")
    _run(run, ["A1"])
    with store.open(str(run / "measurements")) as opened:
        wide = opened.query(table="cell_wide")
    assert sorted(set(wide["well"].astype(str))) == ["A2"]
    _run(run, ["A2"])
    with store.open(str(run / "measurements")) as opened:
        assert "cell_wide" not in opened.tables()
    assert not (run / "measurements" / "tables" / "cell_wide").exists()


def test_a_chain_of_matches_never_joins_two_cells_of_one_field():
    from spacr import ops_merge

    cells = pd.DataFrame({"site": [1, 2, 1], "label": [1, 1, 2],
                          "area": [100.0, 90.0, 80.0]})
    edges = pd.DataFrame({"site_a": [1, 2], "label_a": [1, 1],
                          "site_b": [2, 1], "label_b": [1, 2],
                          "distance": [2.0, 5.0]})
    owned = ops_merge._assign_field_owners(cells, edges, rule="largest")
    assert list(owned["component"]) == [0, 0, 1]
    assert owned.attrs["refused_edges"] == 1
    assert not owned.duplicated(["component", "site"]).any()
    nearer = edges.assign(distance=[5.0, 2.0])
    owned = ops_merge._assign_field_owners(cells, nearer, rule="largest")
    assert list(owned["component"]) == [0, 1, 1]
    component, refused = ops_merge._copy_components(
        4, [0, 2], [1, 3], [1, 2, 1, 3])
    assert list(component) == [0, 0, 1, 1] and refused == 0


def test_a_cell_missing_from_the_ownership_table_is_counted_by_one_rule(
        tmp_path):
    run, cells = _build(tmp_path)
    path = ops_engine._ph_cross_field_path(str(run), "A1")
    owners = pd.read_parquet(path)
    owners[owners["site"] != 40].to_parquet(path, index=False)
    unplaced = int((cells["site"] == 40).sum())
    counts = {}
    owned = ops_engine._ph_owned_cells(str(run), "A1", cells, SETTINGS,
                                       counts=counts)
    assert not owned[(cells["site"] == 40).to_numpy()].any()
    assert counts["cells_without_ownership_row"] == unplaced
    kept, report = ops_engine._ph_cross_field_filter(
        cells.assign(well="A1"), str(run), ["A1"],
        {**SETTINGS, "ops_cross_field_policy": "owner_only"})
    assert report["cells_without_ownership_row"] == unplaced
    assert not (kept["site"] == 40).any()
    assert len(kept) == len(cells) - unplaced


def test_a_sequencing_nucleus_called_twice_is_named(tmp_path):
    table = pd.DataFrame({"plate": PLATE, "well": ["A1", "A1", "A1"],
                          "object_id": [4, 4, 9],
                          "mapped_guide": ["ACGT", "TTGA", "GGCA"]})
    path = tmp_path / "calls.csv"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="A1:4"):
        ops_engine._ph_genotype_table(str(path), PLATE, ["A1"])
    with pytest.raises(ValueError, match="ops_barcodes"):
        ops_engine._ph_one_call_per_nucleus(table, "ops_barcodes")
    ops_engine._ph_one_call_per_nucleus(table.iloc[1:], "ops_barcodes")


def test_ph_crops_refuses_a_well_without_its_cross_field_table(tmp_path):
    """A3-BUG1: ph_crops raises as ph_tables does, even with no tiles to cut."""
    run, cells = _build(tmp_path)
    os.remove(ops_engine._ph_cross_field_path(str(run), "A1"))
    settings = {**SETTINGS, "dst_root": str(run)}
    for index in ({}, {"A1": {1: {1: "tile.tif"}}}):
        with pytest.raises(ValueError,
                           match=f"no cross-field ownership table.*{len(cells)} "
                                 "cell copies"):
            ops_engine._ph_crops(str(run / "measurements.db"), PLATE, ["A1"],
                                 index, settings, str(run))
    with pytest.raises(ValueError, match=f"{len(cells)} cell copies"):
        _tables(run)
    report = ops_engine._ph_crops(
        str(run / "measurements.db"), PLATE, ["A1"], {},
        {**settings, "ops_cross_field_ownership": False}, str(run))
    assert report["wells"]["A1"] == {"skipped": "no cell table or no tiles"}


def test_a_cell_linked_twice_is_refused_where_links_are_read(tmp_path):
    """R3b: one cell with two object_ids stops ph_tables, the crops and the guard."""
    from spacr.ops_store import read_table, write_table

    run, _cells = _build(tmp_path)
    db = str(run / "measurements.db")
    links = read_table(db, "ops_merge")
    twice = pd.concat([links, links.iloc[[0]].assign(object_id=10 ** 6)],
                      ignore_index=True)
    write_table(db, "ops_merge", twice)
    cell = ":".join(str(twice.iloc[0][k]) for k in ("plate", "well", "site", "label"))
    with pytest.raises(ValueError, match="links 1 cells more than once"):
        ops_engine._ph_one_link_per_cell(twice)
    with pytest.raises(ValueError, match=cell):
        _tables(run)
    with pytest.raises(ValueError, match="more than once"):
        ops_engine._ph_crops(db, PLATE, ["A1"], {},
                             {**SETTINGS, "dst_root": str(run)}, str(run))
    assert ops_engine._ph_one_link_per_cell(links) is links
    assert ops_engine._ph_one_link_per_cell(None) is None


def test_link_genotypes_refuses_one_cell_repeated_across_links(tmp_path):
    """R3b2: links whose every row is label 1 of one field name one cell many times."""
    run, _cells = _build(tmp_path)
    db = str(run / "measurements.db")
    links = pd.DataFrame({"plate": PLATE, "well": "A1", "site": 3, "label": 1,
                          "object_id": np.arange(1, 6, dtype=np.int64)})
    with pytest.raises(ValueError, match="more than once"):
        ops_engine._ph_link_genotypes(db, PLATE, ["A1"], links, SETTINGS)
    with pytest.raises(ValueError, match="more than once"):
        ops_engine._ph_table_frames(PLATE, "A1", _cells, links, None,
                                    np.ones(len(_cells), bool), "x")
