"""The OPS aggregate phase pools by feature block.

The block-wise pooling must give the tables the whole-frame pooling gave,
value for value. The whole-frame reduction is kept here verbatim, as it was
before spaCR 1.5.1.8, as the reference.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest

from spacr import ops_engine
from spacr.gene_measurement_compare import (aggregate_genes, aggregate_guides,
                                            bootstrap_gene_pvalues,
                                            mitotic_calls, robust_z_to_controls)
from spacr.ops_store import write_table

PLATE = "P1"
GUIDES = ["AAAA", "CCCC", "GGGG", "TTTT", "ACAC", "AGAG", "CTCT", "GAGA"]
GENES = ["nontargeting", "nontargeting_b", "nontargeting_c", "GENE1",
         "GENE1", "GENE2", "GENE2", "GENE3"]


def _old_aggregate(db, plate, wells, settings, destination):
    """The whole-frame _ph_aggregate before block-wise pooling, verbatim
    except that it returns its two tables instead of writing them. Its gene
    test is the cell bootstrap, which test_gene_pvalue_calibrated_null.py
    checks against the earlier function."""
    design_path = settings.get("ops_design")
    control_prefix = str(settings.get("ops_control_prefix") or "nontargeting")
    merged_rows = ops_engine._rows_for_wells(db, "ops_merge", plate, wells)
    barcodes = ops_engine._rows_for_wells(db, "ops_barcodes", plate, wells)
    called = barcodes.loc[barcodes["mapped_guide"].fillna("").astype(str) != "",
                          ["well", "object_id", "mapped_guide"]]
    length = int(called["mapped_guide"].str.len().mode().iloc[0]) if len(called) else 11
    design, ambiguous = ops_engine._ph_design(str(design_path), length)
    frames = []
    for well in wells:
        path = ops_engine._ph_cells_path(destination, well)
        if os.path.isfile(path):
            frames.append(pd.read_parquet(path))
    cells = pd.concat(frames, ignore_index=True)
    features = ops_engine._ph_feature_columns(cells.columns)
    cells = cells.merge(merged_rows[["well", "site", "label", "object_id"]],
                        on=["well", "site", "label"], how="inner")
    cells = cells.merge(called, on=["well", "object_id"], how="left")
    with_guide = cells["mapped_guide"].notna()
    cells = cells[with_guide].merge(
        design.rename(columns={"prefix": "mapped_guide"}),
        on="mapped_guide", how="inner")
    control = cells["gene"].str.lower().str.startswith(control_prefix.lower())
    channels = [c.lower() for c in ops_engine._ph_channels(settings)]
    dna_column = f"nucleus_{channels[0]}_mean_intensity" if channels else ""
    if dna_column not in cells or "nucleus_area" not in cells:
        dna_column = next((c for c in features if c.startswith("nucleus_")
                           and c.endswith("mean_intensity")), "")
    mitotic_z = float(settings.get("ops_mitotic_dna_z") or 3.0)
    if dna_column and "nucleus_area" in cells:
        cells["mitotic"] = mitotic_calls(cells, dna=dna_column,
                                         area="nucleus_area", dna_z=mitotic_z)
    else:
        cells["mitotic"] = False
    normalised = robust_z_to_controls(cells, features, control=control)
    scored = pd.concat([cells[["well", "sgRNA", "gene", "mitotic"]],
                        normalised], axis=1)
    guides = aggregate_guides(scored, features, guide="sgRNA", gene="gene",
                              mitotic="mitotic")
    genes = aggregate_genes(guides, features)
    boot = int(settings.get("ops_bootstrap") or 0)
    if boot > 0 and control.any():
        tested = [c for c in (
            "nucleus_area", dna_column,
            f"nucleus_{channels[0]}_integrated_intensity" if channels else "",
            *(f"cell_{c}_mean_intensity" for c in channels[1:]))
            if c in features]
        targeting = guides[~guides["gene"].str.lower().str.startswith(
            control_prefix.lower())]
        pvalues = bootstrap_gene_pvalues(
            normalised.loc[control.to_numpy()], targeting, tested,
            n_boot=boot, null="cell_bootstrap")
        genes = genes.merge(pvalues, on="gene", how="left")
    return guides, genes


def _plate(tmp_path, *, wells=("A1", "A2", "B1"), n=400, seed=0,
           drop_column_in=None):
    """Three merged, guide-called wells with a well shift, two genes with
    effects, a nearly constant distance column and a few cells without a
    guide or a merge row."""
    rng = np.random.default_rng(seed)
    db = str(tmp_path / "measurements.db")
    merges, calls = [], []
    for w, well in enumerate(wells):
        ids = np.arange(n) + 1
        guide_of = rng.choice(GUIDES, size=n)
        effect = np.where(np.isin(guide_of, ["TTTT", "ACAC"]), 3.0,
                          np.where(guide_of == "AGAG", -2.0, 0.0))
        merged = ids[ids % 23 != 0]
        merges.append(pd.DataFrame({
            "plate": PLATE, "well": well, "site": ids % 5,
            "label": ids, "object_id": ids + 1000 * w, "distance": 0.5,
            "refined": 1}).set_index("label").loc[merged].reset_index())
        mapped = np.where(ids % 17 == 0, "", guide_of)
        calls.append(pd.DataFrame({
            "plate": PLATE, "well": well, "object_id": ids + 1000 * w,
            "barcode": "x", "mapped_guide": mapped}))
        distance = np.zeros(n)
        distance[rng.random(n) < 0.03] = rng.integers(1, 40, n)[:1]
        cells = pd.DataFrame({
            "plate": PLATE, "well": well, "site": ids % 5, "label": ids,
            "centroid_row": rng.random(n), "centroid_col": rng.random(n),
            "nucleus_area": rng.normal(100 + 5 * w, 5, n) + 5 * effect,
            "nucleus_dapi_mean_intensity": rng.normal(500, 20, n),
            "nucleus_dapi_integrated_intensity": rng.normal(5e4, 3e3, n),
            "cell_gfp_mean_intensity": rng.normal(50, 3, n) + effect,
            "cell_area": rng.normal(400, 30, n).astype(np.float32),
            "cell_euler_number": np.ones(n),
            "cell_gfp_distance_to_nucleus": distance,
            "cytoplasm_gfp_mean_intensity": rng.normal(20, 2, n),
        })
        cells.loc[:5, "nucleus_dapi_mean_intensity"] = 900.0
        cells.loc[:5, "nucleus_area"] = 70.0
        cells.loc[7, "cell_area"] = np.nan
        if drop_column_in == well:
            cells = cells.drop(columns=["cytoplasm_gfp_mean_intensity"])
        cells.to_parquet(tmp_path / f"ops_phenotype_cells_{well}.parquet")
    write_table(db, "ops_merge", pd.concat(merges, ignore_index=True))
    barcodes = pd.concat(calls, ignore_index=True)
    write_table(db, "ops_barcodes", barcodes)
    pd.DataFrame({"sgRNA": [g + "TTTT" for g in GUIDES],
                  "gene_symbol": GENES}).to_csv(tmp_path / "design.csv",
                                                index=False)
    return db, list(wells), barcodes


def _settings(tmp_path, **extra):
    base = {"ops_design": str(tmp_path / "design.csv"),
            "ops_ph_channels": "DAPI,GFP", "ops_bootstrap": 200,
            "ops_pvalue_null": "cell_bootstrap", "ops_feature_qc": False}
    base.update(extra)
    return base


def _tables(report):
    return (pd.read_parquet(report["outputs"]["guides_parquet"]),
            pd.read_parquet(report["outputs"]["genes_parquet"]))


@pytest.mark.parametrize("block", [1, 3, 48])
@pytest.mark.parametrize("drop_column_in", [None, "A2"])
def test_block_wise_pooling_gives_the_whole_frame_tables_exactly(
        tmp_path, monkeypatch, block, drop_column_in):
    db, wells, _ = _plate(tmp_path, drop_column_in=drop_column_in)
    settings = _settings(tmp_path)
    old_guides, old_genes = _old_aggregate(db, PLATE, wells, settings,
                                           str(tmp_path))
    monkeypatch.setattr(ops_engine, "_PH_POOL_BLOCK", block)
    report = ops_engine._ph_aggregate(db, PLATE, wells, settings,
                                      str(tmp_path))
    guides, genes = _tables(report)
    pd.testing.assert_frame_equal(guides, old_guides, check_exact=True)
    pd.testing.assert_frame_equal(genes, old_genes, check_exact=True)
    assert "nucleus_area_pvalue" in genes
    assert report["features_dropped"] == []


def test_feature_qc_leaves_out_only_columns_the_old_tables_had_as_nan(
        tmp_path):
    db, wells, _ = _plate(tmp_path)
    old_guides, old_genes = _old_aggregate(
        db, PLATE, wells, _settings(tmp_path), str(tmp_path))
    report = ops_engine._ph_aggregate(
        db, PLATE, wells, _settings(tmp_path, ops_feature_qc=True),
        str(tmp_path))
    guides, genes = _tables(report)
    dropped = {row["feature"]: row for row in report["features_dropped"]}
    assert set(dropped) == {"cell_euler_number",
                            "cell_gfp_distance_to_nucleus"}
    assert old_guides[list(dropped)].isna().all().all()
    assert old_genes[list(dropped)].isna().all().all()
    pd.testing.assert_frame_equal(
        guides, old_guides.drop(columns=list(dropped)), check_exact=True)
    pd.testing.assert_frame_equal(
        genes, old_genes.drop(columns=list(dropped)), check_exact=True)
    reason = dropped["cell_gfp_distance_to_nucleus"]["reason"]
    assert "median absolute deviation" in reason
    assert "centre of mass" in reason
    assert "topology" in dropped["cell_euler_number"]["reason"]
    assert dropped["cell_euler_number"]["control_share_at_median"] == 1.0
    assert dropped["cell_euler_number"]["wells_with_zero_control_mad"] == 3
    table = pd.read_csv(report["outputs"]["feature_qc_csv"])
    assert sorted(table["feature"]) == sorted(dropped)


def test_a_genotype_table_replaces_the_stored_calls(tmp_path):
    db, wells, barcodes = _plate(tmp_path)
    settings = _settings(tmp_path)
    stored = _tables(ops_engine._ph_aggregate(db, PLATE, wells, settings,
                                              str(tmp_path)))
    path = tmp_path / "genotype.parquet"
    barcodes.to_parquet(path)
    report = ops_engine._ph_aggregate(
        db, PLATE, wells, {**settings, "ops_genotype_table": str(path)},
        str(tmp_path))
    assert report["genotype_table"] == str(path)
    for left, right in zip(_tables(report), stored):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    # Another decode: every GENE3 call is lost, so GENE3 disappears.
    other = barcodes.assign(mapped_guide=barcodes["mapped_guide"].replace(
        "GAGA", ""))
    other.drop(columns=["plate"]).to_csv(tmp_path / "other.csv", index=False)
    guides, genes = _tables(ops_engine._ph_aggregate(
        db, PLATE, wells,
        {**settings, "ops_genotype_table": str(tmp_path / "other.csv")},
        str(tmp_path)))
    assert "GENE3" not in set(genes["gene"])
    assert "GENE3" in set(stored[1]["gene"])


def test_a_genotype_table_must_name_each_nucleus_once(tmp_path):
    db, wells, barcodes = _plate(tmp_path)
    twice = pd.concat([barcodes, barcodes.iloc[:3]])
    twice.to_parquet(tmp_path / "twice.parquet")
    with pytest.raises(ValueError, match="more than once"):
        ops_engine._ph_aggregate(
            db, PLATE, wells, _settings(
                tmp_path, ops_genotype_table=str(tmp_path / "twice.parquet")),
            str(tmp_path))
    barcodes.drop(columns=["object_id"]).to_parquet(tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="lacks"):
        ops_engine._ph_aggregate(
            db, PLATE, wells, _settings(
                tmp_path, ops_genotype_table=str(tmp_path / "bad.parquet")),
            str(tmp_path))
    with pytest.raises(ValueError, match="ops_genotype_table"):
        ops_engine._ph_aggregate(
            db, PLATE, wells, _settings(
                tmp_path, ops_genotype_table=str(tmp_path / "none.csv")),
            str(tmp_path))


def test_a_well_subset_pools_like_a_run_of_those_wells(tmp_path):
    db, wells, _ = _plate(tmp_path)
    settings = _settings(tmp_path)
    old_guides, old_genes = _old_aggregate(db, PLATE, ["A1", "B1"], settings,
                                           str(tmp_path))
    report = ops_engine._ph_aggregate(
        db, PLATE, wells, {**settings, "ops_pool_wells": "a1, B1"},
        str(tmp_path))
    assert report["wells"] == ["A1", "B1"]
    assert report["outputs"]["genes_csv"].endswith(
        f"ops_genes_{PLATE}_wells-A1-B1.csv")
    guides, genes = _tables(report)
    pd.testing.assert_frame_equal(guides, old_guides, check_exact=True)
    pd.testing.assert_frame_equal(genes, old_genes, check_exact=True)
    with pytest.raises(ValueError, match="ops_pool_wells"):
        ops_engine._ph_aggregate(db, PLATE, wells,
                                 {**settings, "ops_pool_wells": "C9"},
                                 str(tmp_path))


def test_interphase_pooling_normalises_on_every_cell_and_pools_interphase(
        tmp_path):
    db, wells, _ = _plate(tmp_path)
    settings = _settings(tmp_path, ops_bootstrap=0)
    report = ops_engine._ph_aggregate(
        db, PLATE, wells, {**settings, "ops_pool_interphase": True},
        str(tmp_path))
    assert report["pooled"] == "interphase"
    assert report["outputs"]["guides_csv"].endswith(
        f"ops_guides_{PLATE}_interphase.csv")
    assert report["cells_pooled"] < report["cells_in_design"]
    guides, genes = _tables(report)
    assert "mitotic_fraction" not in guides
    # The reference: the whole-frame z over every cell, then interphase only.
    merged = ops_engine._rows_for_wells(db, "ops_merge", PLATE, wells)
    calls = ops_engine._rows_for_wells(db, "ops_barcodes", PLATE, wells)
    calls = calls[calls["mapped_guide"] != ""]
    design, _ = ops_engine._ph_design(str(tmp_path / "design.csv"), 4)
    cells = pd.concat([pd.read_parquet(tmp_path / f"ops_phenotype_cells_{w}"
                                       ".parquet") for w in wells],
                      ignore_index=True)
    features = ops_engine._ph_feature_columns(cells.columns)
    cells = cells.merge(merged[["well", "site", "label", "object_id"]]).merge(
        calls[["well", "object_id", "mapped_guide"]]).merge(
        design.rename(columns={"prefix": "mapped_guide"}))
    control = cells["gene"].str.startswith("nontargeting")
    mitotic = mitotic_calls(cells, dna="nucleus_dapi_mean_intensity",
                            area="nucleus_area")
    assert mitotic.any()
    z = robust_z_to_controls(cells, features, control=control)
    scored = pd.concat([cells[["sgRNA", "gene"]], z], axis=1)[~mitotic]
    expected = aggregate_guides(scored, features, guide="sgRNA", gene="gene")
    pd.testing.assert_frame_equal(guides, expected, check_exact=True)


def test_named_test_features_and_the_calibrated_null_reach_the_report(
        tmp_path):
    db, wells, _ = _plate(tmp_path)
    report = ops_engine._ph_aggregate(
        db, PLATE, wells,
        _settings(tmp_path, ops_pvalue_null="calibrated",
                  ops_test_features="cell_gfp_mean_intensity,nucleus_area"),
        str(tmp_path))
    assert report["bootstrap_features"] == ["cell_gfp_mean_intensity",
                                            "nucleus_area"]
    assert report["pvalue_null"] == "calibrated"
    assert set(report["null_parameters"]) == {"cell_gfp_mean_intensity",
                                              "nucleus_area"}
    genes = _tables(report)[1].set_index("gene")
    assert genes.loc["GENE1", "nucleus_area_pvalue"] < 0.05
    with pytest.raises(ValueError, match="ops_test_features"):
        ops_engine._ph_aggregate(
            db, PLATE, wells, _settings(tmp_path, ops_test_features="nope"),
            str(tmp_path))
    with pytest.raises(ValueError, match="ops_pvalue_null"):
        ops_engine._ph_aggregate(
            db, PLATE, wells, _settings(tmp_path, ops_pvalue_null="exact"),
            str(tmp_path))
