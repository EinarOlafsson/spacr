"""The OPS phenotype path: field transforms, merge and per-gene aggregation.

Every case is planted truth: a transform chosen here, points moved by it, and
the answer checked against the choice. Nothing reads an image or a model.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spacr import ops_engine
from spacr.gene_measurement_compare import (aggregate_genes, aggregate_guides,
                                            bootstrap_gene_pvalues,
                                            mitotic_calls,
                                            robust_z_to_controls)
from spacr.ops_merge import (deduplicate_matches, field_to_well,
                             field_transform, merge_field,
                             refine_field_transform)
from spacr.ops_sample import pair_nuclei_with_cells
from spacr.ops_store import read_table, write_table

FIELD = (2960, 2960)
PLATE = "P1"


def _old_phenotype_rows(centres):
    """``ops_phenotype`` rows as a database written before the transforms."""
    return pd.DataFrame([{
        "plate": PLATE, "well": "A1", "site": site, "centre_y": cy,
        "centre_x": cx, "sbs_site": 0, "is_anchor": int(site == 0),
        "inliers": 0, "alignment_residual_px": np.nan,
        "raster_residual_px": np.nan} for site, (cy, cx) in centres.items()])


def test_a_field_transform_puts_the_field_centre_on_the_stored_centre():
    transform = field_transform(0.25, 0.4, (5000.0, 7000.0), FIELD)
    centre = field_to_well([[FIELD[0] / 2, FIELD[1] / 2]], transform)[0]
    assert np.allclose(centre, [5000.0, 7000.0])
    corner = field_to_well([[0.0, 0.0]], transform)[0]
    assert np.hypot(*(centre - corner)) == pytest.approx(
        0.25 * np.hypot(*np.array(FIELD) / 2))


def test_old_rows_are_migrated_from_their_report_once(tmp_path):
    db = str(tmp_path / "measurements.db")
    write_table(db, "ops_phenotype",
                _old_phenotype_rows({0: (1000.0, 1000.0), 1: (1000.0, 2267.0)}))
    (tmp_path / "A1").mkdir()
    (tmp_path / "A1" / "ops_report.json").write_text(json.dumps({
        "phenotype": {"expected_scale": 0.25, "anchors": [
            {"site": 0, "scale": 0.2498, "degrees": 0.02},
            {"site": 5, "scale": 0.2500, "degrees": -0.02},
            {"site": 9, "scale": 0.2502, "degrees": 0.00}]}}))
    migrated = ops_engine._ph_transforms(db, PLATE, "A1", str(tmp_path),
                                         field_shape=FIELD)
    assert set(ops_engine._PH_TRANSFORM_COLUMNS) <= set(migrated.columns)
    assert migrated["scale"].tolist() == [pytest.approx(0.25)] * 2
    assert migrated["transform_source"].tolist() == ["anchor", "raster"]
    stored = read_table(db, "ops_phenotype")
    assert stored["offset_y"].notna().all()
    again = ops_engine._ph_transforms(db, PLATE, "A1", str(tmp_path))
    row = again.iloc[1]
    transform = field_transform(row.scale, row.rotation_deg,
                                (row.centre_y, row.centre_x), FIELD)
    assert np.allclose(transform[2], [row.offset_y, row.offset_x])


def test_old_rows_without_a_report_or_size_are_refused(tmp_path):
    db = str(tmp_path / "measurements.db")
    write_table(db, "ops_phenotype", _old_phenotype_rows({0: (10.0, 10.0)}))
    with pytest.raises(ValueError, match="no scale"):
        ops_engine._ph_transforms(db, PLATE, "A1", str(tmp_path),
                                  field_shape=FIELD)


def _planted(seed=0, n=400):
    rng = np.random.default_rng(seed)
    points = rng.uniform(50, FIELD[0] - 50, size=(n, 2))
    truth = field_transform(0.25, 0.35, (8000.0, 9000.0), FIELD)
    targets = field_to_well(points, truth) + rng.normal(0, 0.3, size=(n, 2))
    extra = rng.uniform(7600, 9400, size=(150, 2))
    return points, np.vstack([targets, extra]), truth


def test_refinement_recovers_a_planted_transform_from_a_shifted_seed():
    points, targets, truth = _planted()
    seed = field_transform(0.25, 0.0, (8005.0, 8996.0), FIELD)
    found, inliers, residual, refined = refine_field_transform(
        points, targets, seed)
    assert refined and inliers > 350
    assert found[0] == pytest.approx(0.25, rel=2e-3)
    assert np.allclose(field_to_well(points, found),
                       field_to_well(points, truth), atol=1.0)
    assert residual < 1.0


def test_refinement_keeps_the_seed_when_nothing_agrees():
    points, _targets, _truth = _planted()
    seed = field_transform(0.25, 0.0, (8000.0, 9000.0), FIELD)
    far = np.random.default_rng(3).uniform(0, 30000, size=(300, 2))
    found, _inliers, _residual, refined = refine_field_transform(
        points, far, seed)
    assert not refined
    assert np.allclose(found[2], seed[2])


def test_merge_pairs_every_planted_nucleus_once():
    points, targets, truth = _planted()
    pairs = merge_field(points, targets, truth, threshold=4.0)
    assert len(pairs) >= 395
    assert (pairs[:, 0] == pairs[:, 1]).mean() > 0.99
    assert pairs[:, 2].max() <= 4.0


def test_an_overlap_keeps_the_closer_observation():
    frame = pd.DataFrame({"site": [1, 2, 2], "label": [5, 9, 10],
                          "object_id": [77, 77, 78],
                          "distance": [2.5, 0.5, 1.0]})
    kept, dropped = deduplicate_matches(frame)
    assert dropped == 1
    assert kept.set_index("object_id").loc[77, "site"] == 2


def test_two_segmentations_are_paired_one_nucleus_per_cell():
    cells = np.zeros((40, 40), int)
    cells[0:20, 0:20] = 3
    cells[0:20, 20:40] = 7
    cells[20:40, 0:20] = 9
    nuclei = np.zeros_like(cells)
    nuclei[5:10, 5:10] = 1
    nuclei[8:12, 18:24] = 2
    nuclei[2:4, 30:32] = 4
    nuclei[30:34, 30:34] = 5
    paired_n, paired_c, counts = pair_nuclei_with_cells(nuclei, cells)
    assert counts["paired"] == 2
    assert counts["cells_without_nucleus"] == 1
    assert counts["nuclei_without_cell"] == 1
    assert counts["extra_nuclei_in_cell"] == 1
    for label in np.unique(paired_n[paired_n > 0]):
        assert np.all(paired_c[paired_n == label] == label)


def _guide_cells(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for well, offset in (("A1", 0.0), ("A2", 50.0)):
        for guide, gene, shift in (("NT1", "nontargeting", 0.0),
                                   ("NT2", "nontargeting_b", 0.0),
                                   ("G1a", "GENE1", 2.0),
                                   ("G1b", "GENE1", 2.0),
                                   ("G2a", "GENE2", 0.0)):
            n = 400 if gene.startswith("nontargeting") else 60
            values = rng.normal(100 + offset + 10 * shift, 10, size=n)
            for value in values:
                rows.append({"well": well, "guide": guide, "gene": gene,
                             "nucleus_area": value})
    return pd.DataFrame(rows)


def test_aggregation_reports_a_planted_gene_effect_per_well():
    cells = _guide_cells()
    control = cells["gene"].str.startswith("nontargeting")
    z = robust_z_to_controls(cells, ["nucleus_area"], control=control)
    scored = pd.concat([cells[["well", "guide", "gene"]], z], axis=1)
    guides = aggregate_guides(scored, ["nucleus_area"])
    genes = aggregate_genes(guides, ["nucleus_area"]).set_index("gene")
    assert genes.loc["GENE1", "nucleus_area"] == pytest.approx(2.0, abs=0.35)
    assert abs(genes.loc["GENE2", "nucleus_area"]) < 0.5
    assert genes.loc["GENE1", "n_guides"] == 2
    pvalues = bootstrap_gene_pvalues(
        z[control.to_numpy()], guides[~guides["gene"].str.startswith(
            "nontargeting")], ["nucleus_area"], n_boot=500).set_index("gene")
    assert pvalues.loc["GENE1", "nucleus_area_pvalue"] < 0.01
    assert pvalues.loc["GENE2", "nucleus_area_pvalue"] > 0.01
    assert "nucleus_area_fdr" in pvalues


def test_mitotic_calls_need_bright_and_compact_chromatin():
    rng = np.random.default_rng(1)
    cells = pd.DataFrame({"well": "A1",
                          "dna": rng.normal(1000, 50, 500),
                          "area": rng.normal(1500, 100, 500)})
    cells.loc[:9, "dna"] = 2000
    cells.loc[:4, "area"] = 900
    cells.loc[5:9, "area"] = 2500
    calls = mitotic_calls(cells, dna="dna", area="area")
    assert calls[:5].all() and not calls[5:10].any()
    assert calls.sum() <= 7


def test_merge_and_aggregate_run_without_a_sequencing_folder(tmp_path):
    points, targets, truth = _planted(n=200)
    targets = targets[:200]
    scale, rotation, translation = truth
    write_table(str(tmp_path / "measurements.db"), "ops_phenotype",
                ops_engine._ph_transform_columns(
                    _old_phenotype_rows({4: (8000.0, 9000.0)}), scale,
                    0.35, FIELD))
    db = str(tmp_path / "measurements.db")
    write_table(db, "ops_objects", pd.DataFrame({
        "plate": PLATE, "well": "A1", "object_id": np.arange(200) + 1,
        "centroid_y": targets[:, 0], "centroid_x": targets[:, 1]}))
    guides = ["AAAAAAAAAAA", "CCCCCCCCCCC", "GGGGGGGGGGG"]
    write_table(db, "ops_barcodes", pd.DataFrame({
        "plate": PLATE, "well": "A1", "object_id": np.arange(200) + 1,
        "barcode": "x", "mapped_guide": [guides[i % 3] for i in range(200)]}))
    pd.DataFrame({"sgRNA": [g + "TTTTTTTTT" for g in guides],
                  "gene_symbol": ["nontargeting", "nontargeting_2", "GENE1"]}
                 ).to_csv(tmp_path / "design.csv", index=False)
    rng = np.random.default_rng(2)
    pd.DataFrame({"plate": PLATE, "well": "A1", "site": 4,
                  "label": np.arange(200) + 1,
                  "centroid_row": points[:, 0], "centroid_col": points[:, 1],
                  "nucleus_area": rng.normal(1500, 100, 200),
                  "nucleus_dapi_mean_intensity": rng.normal(1000, 50, 200)}
                 ).to_parquet(tmp_path / "ops_phenotype_cells_A1.parquet")
    out = ops_engine.run_ops(
        {"dst_root": str(tmp_path), "plate": PLATE, "ops_gpu": False,
         "ops_design": str(tmp_path / "design.csv"),
         "ops_ph_channels": "DAPI"},
        wells=["A1"], phases=("merge", "aggregate"))
    merge = out["wells"]["A1"]["merge"]
    assert merge["match_rate"] > 0.97
    merged = read_table(db, "ops_merge")
    assert (merged["label"] == merged["object_id"]).mean() > 0.99
    aggregate = out["aggregate"]
    assert aggregate["guides"] == 3 and aggregate["genes"] == 3
    genes = pd.read_csv(tmp_path / f"ops_genes_{PLATE}.csv")
    assert "mitotic_index" in genes
    assert (tmp_path / "A1" / "ops_report.json").is_file()


def test_an_unknown_phase_is_still_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown phases"):
        ops_engine.run_ops({"dst_root": str(tmp_path)}, phases=("ph_paint",))
