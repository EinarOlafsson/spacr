"""Item 883: a gene profile as the median of all of its cells, and NT rows.

Brieflow (Di Bernardo et al., Nat Commun 2026) reduces a gene as the median
over every cell of the gene. With the option off, spaCR's tables stay the
median of guide medians, value for value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import ops_engine
from spacr.gene_measurement_compare import (aggregate_genes, aggregate_guides,
                                            nt_row_names)
from tests.test_ops_aggregate_blockwise import PLATE, _plate, _settings, _tables


def _cells():
    """Two guides of one gene with very different cell counts."""
    return pd.DataFrame({
        "guide": ["g1"] * 3 + ["g2"] * 6 + ["n1"] * 2,
        "gene": ["A"] * 9 + ["nontargeting"] * 2,
        "x": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 5.0, 6.0],
    })


def test_the_cell_level_takes_the_median_over_every_cell_of_the_gene():
    cells = _cells()
    guides = aggregate_guides(cells, ["x"])
    by_guide = aggregate_genes(guides, ["x"])
    by_cell = aggregate_genes(guides, ["x"], level="cells", cells=cells)
    a = by_guide.set_index("gene").loc["A"]
    b = by_cell.set_index("gene").loc["A"]
    assert a["x"] == pytest.approx(np.median([1.0, 12.5]))
    assert b["x"] == pytest.approx(np.median([0, 1, 2, 10, 11, 12, 13, 14, 15]))
    assert (b["n_guides"], b["n_cells"]) == (2, 9)
    assert list(by_cell.columns) == list(by_guide.columns)


def test_min_cells_leaves_out_genes_with_that_many_cells_or_fewer():
    cells = _cells()
    guides = aggregate_guides(cells, ["x"])
    kept = aggregate_genes(guides, ["x"], level="cells", cells=cells,
                           min_cells=2)
    assert list(kept["gene"]) == ["A"]


def test_the_level_is_checked_and_the_cell_level_needs_cells():
    guides = aggregate_guides(_cells(), ["x"])
    with pytest.raises(ValueError, match="level must be one of"):
        aggregate_genes(guides, ["x"], level="wells")
    with pytest.raises(ValueError, match="needs the cell table"):
        aggregate_genes(guides, ["x"], level="cells")


def test_barcode_rows_name_every_control_guide_on_its_own():
    names = nt_row_names(["g1", "n1", "n2"], ["A", "nontargeting",
                                              "NonTargeting"], rows="barcode")
    assert list(names) == ["A", "nontargeting_n1", "nontargeting_n2"]


def test_pseudo_genes_are_seeded_groups_of_four_and_a_short_group_is_dropped():
    guides = [f"nt{i:02d}" for i in range(10)] + ["g1"]
    genes = ["nontargeting"] * 10 + ["A"]
    names = nt_row_names(guides, genes, rows="pseudo")
    again = nt_row_names(guides, genes, rows="pseudo")
    assert list(names) == list(again)
    counts = pd.Series(names[:10]).value_counts()
    assert counts["nontargeting_drop"] == 2
    assert sorted(counts.drop("nontargeting_drop")) == [4, 4]
    assert names[-1] == "A"
    rank = np.argsort(np.random.default_rng(0).permutation(10))
    assert list(names[:10]) == [
        f"NTpseudo_{k // 4:02d}" if k < 8 else "nontargeting_drop"
        for k in rank]
    kept = nt_row_names(guides, genes, rows="pseudo", drop_incomplete=False)
    assert "nontargeting_drop" not in set(kept)
    other = nt_row_names(guides[:4], genes[:4], rows="pseudo",
                         nt_guides=guides[:10], drop_incomplete=False)
    assert list(other) == list(kept[:4])
    with pytest.raises(ValueError, match="rows must be one of"):
        nt_row_names(guides, genes, rows="wells")


def test_options_off_leave_the_aggregate_tables_unchanged(tmp_path):
    db, wells, _ = _plate(tmp_path)
    base = _tables(ops_engine._ph_aggregate(db, PLATE, wells,
                                            _settings(tmp_path), str(tmp_path)))
    same = _tables(ops_engine._ph_aggregate(
        db, PLATE, wells, _settings(tmp_path, ops_aggregate_level="guides",
                                    ops_nt_rows=None), str(tmp_path)))
    for a, b in zip(base, same):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_the_aggregate_phase_takes_genes_over_cells_and_barcode_rows(tmp_path):
    db, wells, _ = _plate(tmp_path)
    report = ops_engine._ph_aggregate(
        db, PLATE, wells, _settings(tmp_path, ops_aggregate_level="cells",
                                    ops_nt_rows="barcode"), str(tmp_path))
    guides, genes = _tables(report)
    assert report["aggregate_level"] == "cells"
    assert report["nt_rows"] == "barcode"
    control = genes[genes["gene"].str.startswith("nontargeting")]
    assert len(control) == 3
    assert set(control["gene"]) == {f"nontargeting_{g}TTTT"
                                    for g in ("AAAA", "CCCC", "GGGG")}
    gene1 = genes.set_index("gene").loc["GENE1"]
    assert gene1["n_cells"] == guides.loc[guides["gene"] == "GENE1",
                                          "n_cells"].sum()
    assert gene1["nucleus_area_pvalue"] == gene1["nucleus_area_pvalue"]


def test_pseudo_rows_in_the_aggregate_phase_stay_controls(tmp_path):
    db, wells, _ = _plate(tmp_path)
    report = ops_engine._ph_aggregate(
        db, PLATE, wells, _settings(tmp_path, ops_nt_rows="pseudo"),
        str(tmp_path))
    _guides, genes = _tables(report)
    assert not genes["gene"].str.startswith("nontargeting_pseudo").any()
    assert "nontargeting_drop" not in set(genes["gene"])
    assert genes["gene"].str.startswith("GENE").all()


def test_the_aggregate_phase_refuses_unknown_options(tmp_path):
    db, wells, _ = _plate(tmp_path)
    with pytest.raises(ValueError, match="ops_aggregate_level"):
        ops_engine._ph_aggregate(db, PLATE, wells,
                                 _settings(tmp_path, ops_nt_rows="wells"),
                                 str(tmp_path))


def test_tvn_profiles_take_the_gene_over_its_cells():
    from spacr.ops_tvn import tvn_profiles

    cells = _cells().rename(columns={"guide": "sgRNA"})
    embedding = cells[["x"]].to_numpy(np.float32)
    _g, by_guide = tvn_profiles(cells, embedding)
    _g, by_cell = tvn_profiles(cells, embedding, level="cells")
    assert by_guide.set_index("gene").loc["A", "PC_0"] == pytest.approx(6.75)
    assert by_cell.set_index("gene").loc["A", "PC_0"] == pytest.approx(11.0)


def profile_plate(tmp_path, *, wells=("A1", "A2", "B1"), n=1800, seed=0):
    """A merged, guide-called plate large enough for the profile options:
    40 genes in four groups that share a phenotype, three guides each, and
    twelve control guides; cells are G1, S/G2 (1.6 times the DNA) or
    mitotic (bright, compact nuclei)."""
    from spacr.ops_store import write_table

    rng = np.random.default_rng(seed)
    genes = [f"GENE{g:02d}" for g in range(40)]
    guides = [f"{g}_{k}" for g in genes for k in range(3)]
    controls = [f"NT{k:02d}" for k in range(12)]
    seq = {name: "".join(rng.choice(list("ACGT"), 12)) for name in guides + controls}
    design = pd.DataFrame({"sgRNA": [seq[x] + "TTTT" for x in guides + controls],
                           "gene_symbol": [x.rsplit("_", 1)[0] for x in guides]
                           + ["nontargeting"] * len(controls)})
    design.to_csv(tmp_path / "design.csv", index=False)
    effect = {g: np.eye(4)[i % 4] * (2.0 + (i % 5) * 0.3) for i, g in enumerate(genes)}
    db = str(tmp_path / "measurements.db")
    merges, calls = [], []
    for w, well in enumerate(wells):
        ids = np.arange(n) + 1
        names = rng.choice(guides + controls, size=n, p=None)
        prefix = np.array([seq[x][:11] for x in names])
        shift = np.array([effect[x.rsplit("_", 1)[0]] if x in effect or x.rsplit("_", 1)[0] in effect
                          else np.zeros(4) for x in names])
        mitotic = rng.random(n) < 0.06
        sg2 = (rng.random(n) < 0.3) & ~mitotic
        merges.append(pd.DataFrame({"plate": PLATE, "well": well, "site": ids % 6,
                                    "label": ids, "object_id": ids + 10000 * w,
                                    "distance": 0.5, "refined": 1}))
        calls.append(pd.DataFrame({"plate": PLATE, "well": well,
                                   "object_id": ids + 10000 * w, "barcode": "x",
                                   "mapped_guide": prefix}))
        area = np.where(mitotic, 60.0, 100.0) + rng.normal(0, 4, n) + 3 * shift[:, 0]
        dna_mean = np.where(mitotic, 900.0, 500.0) + rng.normal(0, 15, n)
        integrated = 5e4 * np.where(sg2, 1.6, 1.0) * np.exp(rng.normal(0, 0.03, n))
        cells = pd.DataFrame({
            "plate": PLATE, "well": well, "site": ids % 6, "label": ids,
            "centroid_row": rng.random(n), "centroid_col": rng.random(n),
            "nucleus_area": area, "nucleus_dapi_mean_intensity": dna_mean,
            "nucleus_dapi_integrated_intensity": integrated,
            "cell_gfp_mean_intensity": rng.normal(50, 3, n) + 3 * shift[:, 1],
            "cell_area": rng.normal(400, 30, n) + 30 * shift[:, 2],
            "cytoplasm_gfp_mean_intensity": rng.normal(20, 2, n) + 2 * shift[:, 3],
            "cell_gfp_integrated_intensity": rng.normal(2e4, 1e3, n),
            "cytoplasm_area": rng.normal(300, 20, n),
        })
        cells.to_parquet(tmp_path / f"ops_phenotype_cells_{well}.parquet")
    write_table(db, "ops_merge", pd.concat(merges, ignore_index=True))
    write_table(db, "ops_barcodes", pd.concat(calls, ignore_index=True))
    return db, list(wells)


def profile_settings(tmp_path, **extra):
    """TVN aggregate settings for :func:`profile_plate`."""
    base = {"ops_design": str(tmp_path / "design.csv"), "ops_ph_channels": "DAPI,GFP",
            "ops_bootstrap": 0, "ops_feature_qc": False,
            "ops_profile_normalisation": "tvn", "ops_tvn_whiten_epsilon": 0}
    base.update(extra)
    return base
