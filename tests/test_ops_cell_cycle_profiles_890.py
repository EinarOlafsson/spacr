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


def _interphase_fit_reference(wells, sample, seed=0):
    """build_cells.py's TVN: fitted on interphase cells, applied to every cell.

    ``wells``: list of (name, X all cells, gene, mitotic). Each well is centred
    and scaled on its interphase cells, the PCA sample and the control cells
    are interphase cells, and every cell goes through the fitted transform.
    """
    from scipy import linalg
    from sklearn.decomposition import PCA

    nonfin, n, meds = None, 0, []
    for _w, x, _g, _m in wells:
        x = x.astype(np.float32)
        nonfin = (~np.isfinite(x)).sum(0) if nonfin is None else nonfin + (~np.isfinite(x)).sum(0)
        n += len(x)
        meds.append(np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0))
    keep = nonfin / n <= 0.05
    med = np.nanmedian(np.vstack(meds), axis=0)[keep]
    rng = np.random.default_rng(seed)

    def well_matrix(i):
        """All cells of a well, centred and scaled on its interphase cells."""
        _w, x, _g, m = wells[i]
        x = x[:, keep].astype(np.float32)
        bad = ~np.isfinite(x)
        if bad.any():
            x[bad] = np.broadcast_to(med, x.shape)[bad]
        mu, sd = x[~m].mean(0), x[~m].std(0)
        sd[sd == 0] = 1
        return (x - mu) / sd

    sizes = [int((~m).sum()) for _w, _x, _g, m in wells]
    take = [int(round(sample * s / sum(sizes))) for s in sizes]
    samp = []
    for i, k in enumerate(take):
        x = well_matrix(i)[~wells[i][3]]
        samp.append(x[rng.choice(len(x), min(k, len(x)), replace=False)])
    pca = PCA(n_components=0.99, random_state=seed).fit(np.vstack(samp))
    es, nt, ntb = [], [], []
    for i, (name, _x, g, m) in enumerate(wells):
        e = pca.transform(well_matrix(i)).astype(np.float32)
        es.append(e)
        c = pd.Series(g).str.lower().str.startswith("nontargeting").to_numpy() & ~m
        nt.append(e[c])
        ntb.append(np.full(int(c.sum()), name, dtype=object))
    NT = np.vstack(nt).astype(np.float64)
    NTb = np.concatenate(ntb)
    mu1, sd1 = NT.mean(0), NT.std(0)
    sd1[sd1 == 0] = 1
    pca2 = PCA().fit((NT - mu1) / sd1)
    NT2 = pca2.transform((NT - mu1) / sd1)
    stats = {}
    for b in np.unique(NTb):
        m_, s_ = NT2[NTb == b].mean(0), NT2[NTb == b].std(0)
        s_[s_ == 0] = 1
        stats[b] = (m_, s_)
    NT3 = np.vstack([(NT2[NTb == b] - stats[b][0]) / stats[b][1] for b in np.unique(NTb)])
    d = NT3.shape[1]
    tgt = linalg.fractional_matrix_power(np.cov(NT3, rowvar=False, ddof=1) + 0.5 * np.eye(d), 0.5).real
    out = []
    for (name, _x, _g, _m), e in zip(wells, es):
        v = (NT2[NTb == name] - stats[name][0]) / stats[name][1]
        src = linalg.fractional_matrix_power(np.cov(v, rowvar=False, ddof=1) + 0.5 * np.eye(d), -0.5).real
        e = pca2.transform((e.astype(np.float64) - mu1) / sd1)
        out.append((((e - stats[name][0]) / stats[name][1]) @ (src @ tgt)).astype(np.float32))
    return np.vstack(out)


def _tvn_inputs(seed=4):
    """Three wells of cells in the stacked layout ``_aggregate_tvn`` reads."""
    from tests.test_ops_tvn import _cells

    wells = _cells(seed=seed, n=900)
    frames = {w: pd.DataFrame(x, columns=[f"f{j}" for j in range(x.shape[1])]) for w, x, *_ in wells}
    sources = [(w, list(frames[w].columns), len(frames[w])) for w, *_ in wells]

    def read_columns(chosen, columns):
        """The stored columns of the chosen wells, stacked."""
        return pd.concat([frames[s[0]][list(columns)] for s in chosen], ignore_index=True)

    cells = pd.DataFrame({"well": np.concatenate([[w] * len(x) for w, x, *_ in wells]),
                          "sgRNA": np.concatenate([g for *_, g in wells]),
                          "gene": np.concatenate([g for _w, _x, g, _m, _s in wells])})
    mitotic = np.concatenate([m for _w, _x, _g, m, _s in wells])
    control = cells["gene"].str.startswith("nontargeting").to_numpy()
    return wells, sources, read_columns, cells, mitotic, control, list(frames[wells[0][0]].columns)


def test_a_tvn_fitted_on_interphase_cells_is_applied_to_every_cell_as_the_reference_does():
    from spacr.ops_tvn import _aggregate_tvn

    wells, sources, read_columns, cells, mitotic, control, features = _tvn_inputs()
    settings = {"ops_tvn_mode": "streamed", "ops_tvn_pca_sample": 600, "ops_tvn_whiten_epsilon": 0.0,
                "_keep_embedding": True}
    rows = np.arange(len(cells))
    every = np.ones(len(cells), bool)
    _g, _genes, info = _aggregate_tvn(sources, rows, cells, every, control, features, settings,
                                      read_columns, fit=~mitotic)
    reference = _interphase_fit_reference([(w, x, g, m) for w, x, g, m, _s in wells], sample=600)
    np.testing.assert_allclose(info["_embedding"], reference, rtol=2e-5, atol=5e-5)
    assert info["fitted_cells"] == int((~mitotic).sum()) and info["cells"] == len(cells)
    _g, _genes, alone = _aggregate_tvn(sources, rows, cells, ~mitotic, control, features, settings,
                                       read_columns)
    np.testing.assert_allclose(info["_embedding"][~mitotic], alone["_embedding"], rtol=1e-5, atol=2e-5)
    _g, _genes, pooled = _aggregate_tvn(sources, rows, cells, every, control, features, settings,
                                        read_columns)
    assert not np.allclose(pooled["_embedding"], info["_embedding"])
    with pytest.raises(ValueError, match="ops_tvn_mode streamed"):
        _aggregate_tvn(sources, rows, cells, every, control, features,
                       {**settings, "ops_tvn_mode": "in_memory"}, read_columns, fit=~mitotic)


def test_the_aggregate_phase_fits_tvn_on_interphase_cells_when_asked(tmp_path):
    from tests.test_ops_contrastive_weighting_907 import _run

    report = _run(tmp_path, ops_profile_cell_cycle="concat", ops_tvn_fit_cells="interphase")
    tvn = report["tvn"]
    assert tvn["fitted_cells"] < tvn["cells"] == report["cells_pooled"]
    genes = pd.read_parquet(report["profile_options"]["outputs"]["genes_phase_parquet"])
    assert genes["M_PC_0"].notna().any()
    (tmp_path / "pooled").mkdir()
    (tmp_path / "bad").mkdir()
    pooled = _run(tmp_path / "pooled", ops_profile_cell_cycle="concat")
    assert "fitted_cells" not in pooled["tvn"]
    with pytest.raises(ValueError, match="ops_tvn_fit_cells must be"):
        _run(tmp_path / "bad", ops_profile_cell_cycle="concat", ops_tvn_fit_cells="mitotic")


def test_the_tvn_fit_setting_is_alpha_with_its_item():
    from spacr.ops_settings import OPS_DEFAULTS, OPS_TOOLTIPS
    from spacr.settings import ALPHA_FEATURES

    assert "ops_tvn_fit_cells" in ALPHA_FEATURES[890]["settings"]
    assert OPS_DEFAULTS["ops_tvn_fit_cells"] == "pooled"
    assert "Alpha" in OPS_TOOLTIPS["ops_tvn_fit_cells"] and len(OPS_TOOLTIPS["ops_tvn_fit_cells"]) <= 600
