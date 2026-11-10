"""Item 890: cell-cycle phase calls and cell-cycle-resolved gene profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import (cell_cycle_phases, g1_mode,
                                            phase_columns, phase_profiles)


def _cells(seed=0, n=6000):
    """Two wells with G1 peaks a factor of two apart; S/G2 cells carry
    1.6 times G1 DNA."""
    rng = np.random.default_rng(seed)
    well = np.where(np.arange(n) < n // 2, "A1", "A2")
    base = np.where(well == "A1", 1e5, 2e5)
    sg2 = rng.random(n) < 0.3
    dna = base * np.where(sg2, 1.6, 1.0) * np.exp(rng.normal(0, 0.03, n))
    gene = rng.choice(["nontargeting", "GENE1", "GENE2"], n)
    mitotic = rng.random(n) < 0.05
    return pd.DataFrame({"well": well, "dna": dna, "gene": gene,
                         "mitotic": mitotic, "sg2": sg2,
                         "guide": [f"{g}_{k}" for g, k in
                                   zip(gene, rng.integers(0, 3, n))]})


def test_the_g1_mode_is_the_peak_of_the_smoothed_histogram():
    rng = np.random.default_rng(1)
    values = np.r_[rng.normal(16.0, 0.05, 5000), rng.normal(16.7, 0.05, 2000)]
    assert g1_mode(values) == pytest.approx(16.0, abs=0.03)


def test_phases_split_interphase_cells_by_dna_above_each_wells_peak():
    cells = _cells()
    control = cells["gene"] == "nontargeting"
    called = cell_cycle_phases(cells, dna="dna", control=control)
    interphase = ~cells["mitotic"]
    assert (called.loc[cells["mitotic"], "phase"] == "M").all()
    agree = (called.loc[interphase, "phase"] == "SG2") == cells.loc[interphase, "sg2"]
    assert agree.mean() > 0.99
    for well in ("A1", "A2"):
        rel = called.loc[(cells.well == well) & interphase & ~cells.sg2, "dna_log2_rel"]
        assert abs(rel.median()) < 0.03
    assert called["dna_log2_rel"].dtype == np.float32
    later = cell_cycle_phases(cells, dna="dna", control=control, split_log2=0.9)
    assert (later.loc[interphase, "phase"] == "G1").all()


def test_a_well_without_anchor_cells_gets_no_phase_distance():
    cells = _cells(seed=2)
    control = (cells["gene"] == "nontargeting") & (cells["well"] == "A1")
    called = cell_cycle_phases(cells, dna="dna", control=control)
    assert called.loc[cells.well == "A2", "dna_log2_rel"].isna().all()
    assert called.loc[cells.well == "A1", "dna_log2_rel"].notna().all()


def test_phase_profiles_centre_each_phase_on_its_wells_controls():
    rng = np.random.default_rng(3)
    cells = _cells(seed=3)
    control = cells["gene"] == "nontargeting"
    cells["phase"] = cell_cycle_phases(cells, dna="dna", control=control)["phase"]
    shift = {"G1": 0.0, "SG2": 2.0, "M": 5.0}
    cells["f"] = cells["phase"].map(shift) + np.where(cells.well == "A2", 1.0, 0.0) \
        + np.where(cells.gene == "GENE1", 3.0, 0.0) + rng.normal(0, 0.1, len(cells))
    cells.loc[0, "gene"] = "nontargeting_drop"
    guides, genes = phase_profiles(cells, ["f"], control=control)
    assert "nontargeting_drop" not in set(guides["gene"])
    for p in ("G1", "SG2", "M"):
        g = genes.set_index("gene")
        assert g.loc["GENE1", f"{p}_f"] == pytest.approx(3.0, abs=0.2)
        assert abs(g.loc["GENE2", f"{p}_f"]) < 0.2
        assert f"{p}_n_cells" in guides
    assert phase_columns(genes, "concat") == ["G1_f", "SG2_f", "M_f"]
    assert phase_columns(genes, "M") == ["M_f"]
    with pytest.raises(ValueError, match="cell-cycle profile"):
        phase_columns(genes, "S")


def test_the_aggregate_phase_writes_phase_resolved_tables(tmp_path):
    from tests.test_ops_contrastive_weighting_907 import _run

    report = _run(tmp_path, ops_profile_cell_cycle="concat",
                  ops_profile_g1_sg2_split_log2=0.3)
    options = report["profile_options"]
    fractions = options["cell_cycle"]["fractions"]
    assert fractions["SG2"] == pytest.approx(0.28, abs=0.05)
    assert fractions["M"] == pytest.approx(0.06, abs=0.02)
    genes = pd.read_parquet(options["outputs"]["genes_phase_parquet"])
    guides = pd.read_parquet(options["outputs"]["guides_phase_parquet"])
    assert {"G1_PC_0", "SG2_PC_0", "M_PC_0"} <= set(genes.columns)
    assert {"G1_n_cells", "SG2_n_cells", "M_n_cells"} <= set(guides.columns)


def test_phase_options_are_checked(tmp_path):
    from tests.test_ops_contrastive_weighting_907 import _run

    with pytest.raises(ValueError, match="needs ops_mitotic_model"):
        _run(tmp_path, ops_profile_cell_cycle="G1", ops_mitotic_source="classifier")
    with pytest.raises(ValueError, match="ops_mitotic_source must be"):
        _run(tmp_path, ops_profile_cell_cycle="G1", ops_mitotic_source="other")
