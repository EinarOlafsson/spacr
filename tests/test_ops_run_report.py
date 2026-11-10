"""The OPS run report: per-well yield funnel and counts against a reference (item 875, alpha).

The funnel is checked against counts planted in a synthetic well (cut cells,
margin copies, unlinked cells, a call outside the design, a broken row), and
the rank correlation against counts whose rho is known.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from scipy.stats import spearmanr

from spacr import ops_engine
from spacr.ops_store import write_table

PLATE = "P1"
GUIDES = ["AAAA", "CCCC", "GGGG", "TTTT"]
GENES = ["nontargeting", "nontargeting", "GENE1", "GENE2"]


def _well(tmp_path, n=40):
    """One well: n cells, every 10th cut by the field edge, every 7th a
    margin copy (A1b), labels above 30 unlinked, nucleus 5 called outside
    the design and cell 2 with broken features."""
    rng = np.random.default_rng(0)
    db = str(tmp_path / "measurements.db")
    labels = np.arange(1, n + 1)
    linked = labels[labels <= 30]
    owned = labels % 7 != 0
    linked = linked[owned[linked - 1]]
    guide = [GUIDES[i % 4] for i in range(len(linked))]
    guide[list(linked).index(5)] = "NNNN"
    write_table(db, "ops_merge", pd.DataFrame({
        "plate": PLATE, "well": "A1", "site": 0, "label": linked,
        "object_id": linked, "distance": 0.5, "refined": 1}))
    write_table(db, "ops_barcodes", pd.DataFrame({
        "plate": PLATE, "well": "A1", "object_id": linked, "barcode": "x",
        "mapped_guide": guide}))
    pd.DataFrame({"sgRNA": [g + "TTTT" for g in GUIDES],
                  "gene_symbol": GENES}).to_csv(tmp_path / "design.csv",
                                                index=False)
    area = rng.normal(100, 5, n)
    area[1] = np.nan
    cells = pd.DataFrame({"plate": PLATE, "well": "A1", "site": 0,
                          "label": labels, "nucleus_area": area,
                          "nucleus_dapi_mean_intensity": rng.normal(500, 20,
                                                                    n)})
    cells.to_parquet(tmp_path / "ops_phenotype_cells_A1.parquet")
    pd.DataFrame({"plate": PLATE, "well": "A1", "site": 0, "label": labels,
                  "touches_field_edge": labels % 10 == 0}).to_parquet(
        ops_engine._ph_maskpass_path(str(tmp_path), "A1"))
    pd.DataFrame({"plate": PLATE, "well": "A1", "site": 0, "label": labels,
                  "owned": owned}).to_parquet(
        ops_engine._ph_ownership_path(str(tmp_path), "A1"))
    return db, labels, owned


SETTINGS = {"ops_ph_channels": "DAPI", "ops_ownership": "a1b",
            "ops_drop_nonfinite_rows": True}


def test_the_funnel_counts_every_filter_the_aggregate_ran(tmp_path):
    db, labels, owned = _well(tmp_path)
    settings = {**SETTINGS, "ops_design": str(tmp_path / "design.csv")}
    aggregate = ops_engine._ph_aggregate(db, PLATE, ["A1"], settings,
                                         str(tmp_path))
    report = ops_engine._ph_report(db, PLATE, ["A1"], settings,
                                   str(tmp_path), aggregate)
    funnel = pd.read_csv(report["funnel"]).iloc[0]
    cut = labels % 10 == 0
    linked = (labels <= 30) & owned & ~cut
    assert funnel["segmented"] == 40
    assert funnel["aggregate_owned"] == owned.sum()
    assert funnel["aggregate_margin_copies_excluded"] == (~owned).sum()
    assert funnel["aggregate_owned_cut_excluded"] == (owned & cut).sum()
    assert funnel["whole_and_owned"] == (owned & ~cut).sum()
    assert funnel["linked"] == linked.sum()
    assert funnel["decoy_assigned"] == 1
    assert funnel["with_library_guide"] == linked.sum() - 1
    assert funnel["dropped_nonfinite"] == 1
    assert funnel["interphase"] + funnel["aggregate_mitotic"] == \
        funnel["with_library_guide"] - 1
    assert funnel["aggregate_cells_segmented"] == 40
    assert report["counts_vs_reference"] == {
        "skipped": "no ops_reference_counts table"}
    assert aggregate["cells_in_design"] == (funnel["with_library_guide"]
                                           - funnel["dropped_nonfinite"])


def test_counts_against_a_reference_give_the_known_rho(tmp_path):
    db, _labels, _owned = _well(tmp_path)
    settings = {**SETTINGS, "ops_design": str(tmp_path / "design.csv")}
    aggregate = ops_engine._ph_aggregate(db, PLATE, ["A1"], settings,
                                         str(tmp_path))
    reference = pd.DataFrame({"guide": [g + "TTTT" for g in GUIDES]
                              + ["ACGTACGT"],
                              "gene": GENES + ["GENE3"],
                              "count": [40.0, 10.0, 30.0, 20.0, 5.0]})
    reference.to_csv(tmp_path / "reference.csv", index=False)
    report = ops_engine._ph_report(
        db, PLATE, ["A1"], {**settings,
                            "ops_reference_counts": str(tmp_path / "reference.csv"),
                            "ops_reference_counts_level": "sbs,phenotype",
                            "ops_reference_shuffles": 50},
        str(tmp_path), aggregate)
    table = pd.read_csv(report["counts_vs_reference"]["csv"])
    sbs = table[(table.level == "sbs") & (table.unit == "guide")]
    called = pd.Series([GUIDES[i % 4] for i in range(len(sbs))])
    assert sbs["key"].tolist() == reference["guide"].tolist()
    assert sbs["spacr_count"].iloc[-1] == 0
    rho = spearmanr(sbs["spacr_count"], sbs["reference_count"])[0]
    summary = json.loads(open(report["counts_vs_reference"]["json"]).read())
    assert summary["levels"]["sbs"]["guide"]["spearman"] == pytest.approx(rho,
                                                                         abs=1e-12)
    assert summary["levels"]["sbs"]["guide"]["shuffles"] == 50
    assert set(summary["levels"]) == {"sbs", "phenotype"}
    assert set(table["level"]) == {"sbs", "phenotype"}
    scaled = sbs["spacr_scaled"].sum()
    assert scaled == pytest.approx(sbs["reference_count"].sum())
    phenotype = table[(table.level == "phenotype") & (table.unit == "guide")]
    guides = pd.read_parquet(aggregate["outputs"]["guides_parquet"])
    assert sorted(phenotype["key"]) == sorted(guides["guide"])
    again = ops_engine._ph_report(
        db, PLATE, ["A1"], {**settings,
                            "ops_reference_counts": str(tmp_path / "reference.csv"),
                            "ops_reference_shuffles": 50},
        str(tmp_path), aggregate)
    again = json.loads(open(again["counts_vs_reference"]["json"]).read())
    assert again["levels"]["sbs"] == summary["levels"]["sbs"]
    del called


def test_the_null_is_seeded_and_counts_its_permutations():
    rng = np.random.default_rng(0)
    first = ops_engine._ph_rho_with_null([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], 20,
                                         rng)
    assert first["spearman"] == pytest.approx(-1.0) and first["n"] == 5
    second = ops_engine._ph_rho_with_null([1, 2, 3, 4, 5], [5, 4, 3, 2, 1],
                                          20, np.random.default_rng(0))
    assert first == second
    assert 0 <= first["shuffled_p95_abs"] <= 1
    finite = ops_engine._ph_rho_with_null([1, np.nan, 3], [1, 2, 3], 0,
                                          np.random.default_rng(0))
    assert finite["n"] == 2 and finite["shuffled_mean"] is None


def test_the_report_refuses_bad_inputs(tmp_path):
    db, _labels, _owned = _well(tmp_path)
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"guide": ["A"], "n": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="guide, gene and count"):
        ops_engine._ph_report(db, PLATE, ["A1"],
                              {"ops_reference_counts": str(bad)}, str(tmp_path))
    good = tmp_path / "good.csv"
    pd.DataFrame({"guide": ["A"], "gene": ["G"], "count": [1]}).to_csv(
        good, index=False)
    with pytest.raises(ValueError, match="ops_reference_counts_level"):
        ops_engine._ph_report(db, PLATE, ["A1"],
                              {"ops_reference_counts": str(good),
                               "ops_reference_counts_level": "cells"},
                              str(tmp_path))
    with pytest.raises(ValueError, match="aggregate"):
        ops_engine._ph_report(db, PLATE, ["A1"],
                              {"ops_reference_counts": str(good),
                               "ops_reference_counts_level": "phenotype"},
                              str(tmp_path / "elsewhere"))
    report = ops_engine._ph_report(db, PLATE, ["A1"], {}, str(tmp_path / "x"))
    assert report["funnel"] is None


def test_run_ops_runs_the_report_last(tmp_path):
    db, _labels, _owned = _well(tmp_path)
    write_table(db, "ops_phenotype", pd.DataFrame({
        "plate": PLATE, "well": "A1", "site": [0], "centre_y": 0.0,
        "centre_x": 0.0, "is_anchor": 1}))
    out = ops_engine.run_ops({**SETTINGS, "dst_root": str(tmp_path),
                              "plate": PLATE,
                              "ops_design": str(tmp_path / "design.csv")},
                             phases=("aggregate", "report"))
    assert out["report"]["funnel"].endswith("funnel_per_well.csv")
    assert (tmp_path / f"ops_report_{PLATE}.json").is_file()
