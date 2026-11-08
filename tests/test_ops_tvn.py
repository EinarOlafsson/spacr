"""TVN profiles (item 854) match the code they reimplement.

Two references are kept here verbatim:

* Brieflow's ``centerscale_by_batch``, ``centerscale_on_controls`` and
  ``tvn_on_controls`` (cheeseman-lab/brieflow,
  ``workflow/lib/aggregate/align.py``; copyright 2025 Massachusetts Institute
  of Technology, MIT licence), driven as the plate A pilot's in-memory TVN
  (``aggregate_final.py tvn``) drove them;
* the plate A pilot's streamed TVN (``aggregate_plate.py tvn``), with its
  file reads replaced by in-memory batches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from spacr import ops_tvn


# --- Brieflow align.py (MIT), the functions the pilot imported ------------

def _control_mask(series, key):
    return series.astype(str).str.startswith(key)


def _bf_centerscale_by_batch(features, metadata, batch_col):
    for batch in metadata[batch_col].unique():
        ind = metadata[batch_col] == batch
        features[ind, :] = StandardScaler(copy=False).fit_transform(features[ind, :])
    return features


def _bf_centerscale_on_controls(embeddings, metadata, pert_col, control_key,
                                batch_col=None):
    ctrl = _control_mask(metadata[pert_col], control_key)
    if batch_col is not None:
        for batch in metadata[batch_col].unique():
            bm = metadata[batch_col] == batch
            bc = bm & ctrl
            if bc.sum() == 0:
                continue
            embeddings[bm] = StandardScaler(copy=False).fit(
                embeddings[bc]).transform(embeddings[bm])
        return embeddings
    return StandardScaler(copy=False).fit(embeddings[ctrl]).transform(embeddings)


def _bf_tvn_on_controls(embeddings, metadata, pert_col, control_key, batch_col):
    ctrl_ind = _control_mask(metadata[pert_col], control_key).to_list()
    embeddings = _bf_centerscale_on_controls(embeddings, metadata, pert_col, control_key)
    embeddings = PCA().fit(embeddings[ctrl_ind]).transform(embeddings)
    embeddings = _bf_centerscale_on_controls(embeddings, metadata, pert_col,
                                             control_key, batch_col)
    target_cov = np.cov(embeddings[ctrl_ind], rowvar=False, ddof=1) + 0.5 * np.eye(
        embeddings.shape[1])
    target_cov_sqrt = linalg.fractional_matrix_power(target_cov, 0.5)
    for batch in metadata[batch_col].unique():
        batch_ind = metadata[batch_col] == batch
        batch_control_ind = batch_ind & np.array(_control_mask(metadata[pert_col], control_key).to_list())
        if np.sum(batch_control_ind) < embeddings.shape[1]:
            continue
        source_cov = np.cov(embeddings[batch_control_ind], rowvar=False, ddof=1) + 0.5 * np.eye(
            embeddings.shape[1])
        source_cov_inv_sqrt = linalg.fractional_matrix_power(source_cov, -0.5)
        embeddings[batch_ind] = np.real(np.matmul(embeddings[batch_ind], source_cov_inv_sqrt))
        embeddings[batch_ind] = np.real(np.matmul(embeddings[batch_ind], target_cov_sqrt))
    return embeddings


def _pilot_in_memory(X, meta, sample=100000, seed=0):
    """aggregate_final.py tvn(), from the matrix on."""
    X = X.astype(np.float32)
    keep = np.isfinite(X).mean(0) >= 0.95
    X = np.ascontiguousarray(X[:, keep])
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.median(col[~bad])
    sd = X.std(0)
    if (sd == 0).any():
        X = np.ascontiguousarray(X[:, sd > 0])
    meta = meta.copy()
    meta["batch_values"] = meta.well
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), size=min(sample, len(X)), replace=False))
    pca = PCA(n_components=0.99).fit(_bf_centerscale_by_batch(
        X[idx].copy(), meta.iloc[idx].reset_index(drop=True), "batch_values"))
    X = _bf_centerscale_by_batch(X, meta, "batch_values")
    E = pca.transform(X)
    return _bf_tvn_on_controls(E, meta, "gene", "nontargeting", "batch_values").astype(np.float32)


# --- the pilot's streamed TVN (aggregate_plate.py tvn), reads in memory ---

def _pilot_streamed(wells, sample=100000, seed=0):
    """``wells``: list of (name, X all cells, gene, mitotic)."""
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
        _w, x, g, m = wells[i]
        x = x[~m][:, keep].astype(np.float32)
        bad = ~np.isfinite(x)
        if bad.any():
            x[bad] = np.broadcast_to(med, x.shape)[bad]
        mu, sd = x.mean(0), x.std(0)
        sd[sd == 0] = 1
        return g[~m], (x - mu) / sd

    sizes = [int((~m).sum()) for _w, _x, _g, m in wells]
    tot = sum(sizes)
    take = [int(round(sample * s / tot)) for s in sizes]
    samp = []
    for i, k in enumerate(take):
        _, x = well_matrix(i)
        samp.append(x[rng.choice(len(x), min(k, len(x)), replace=False)])
    pca = PCA(n_components=0.99, random_state=seed).fit(np.vstack(samp))
    es, nt, ntb = [], [], []
    for i in range(len(wells)):
        g, x = well_matrix(i)
        e = pca.transform(x).astype(np.float32)
        es.append(e)
        c = pd.Series(g).str.lower().str.startswith("nontargeting").to_numpy()
        nt.append(e[c])
        ntb.append(np.full(int(c.sum()), wells[i][0], dtype=object))
    NT = np.vstack(nt).astype(np.float64)
    NTb = np.concatenate(ntb)
    mu1, sd1 = NT.mean(0), NT.std(0)
    sd1[sd1 == 0] = 1
    NT1 = (NT - mu1) / sd1
    pca2 = PCA().fit(NT1)
    NT2 = pca2.transform(NT1)
    stats = {}
    for b in np.unique(NTb):
        k = NTb == b
        m_, s_ = NT2[k].mean(0), NT2[k].std(0)
        s_[s_ == 0] = 1
        stats[b] = (m_, s_)
    NT3 = np.vstack([(NT2[NTb == b] - stats[b][0]) / stats[b][1] for b in np.unique(NTb)])
    d = NT3.shape[1]
    tgt = linalg.fractional_matrix_power(np.cov(NT3, rowvar=False, ddof=1) + 0.5 * np.eye(d), 0.5).real
    coral = {}
    for b in np.unique(NTb):
        v = (NT2[NTb == b] - stats[b][0]) / stats[b][1]
        if len(v) < d:
            coral[b] = None
            continue
        src = linalg.fractional_matrix_power(np.cov(v, rowvar=False, ddof=1) + 0.5 * np.eye(d), -0.5).real
        coral[b] = src @ tgt if np.all(np.isfinite(src)) else None
    out = []
    for i, e in enumerate(es):
        e = e.astype(np.float64)
        e = pca2.transform((e - mu1) / sd1)
        b = wells[i][0]
        if b in stats:
            e = (e - stats[b][0]) / stats[b][1]
        if coral.get(b) is not None:
            e = e @ coral[b]
        out.append(e.astype(np.float32))
    return out


# --- data ---------------------------------------------------------------

def _cells(seed=0, n=700, wells=("A1", "A2", "B1"), features=14):
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(features, features)) * np.geomspace(1, 1e-3, features)[:, None]
    rows = []
    for w, well in enumerate(wells):
        latent = rng.normal(size=(n, features))
        x = latent @ mix + rng.normal(0, 0.01, (n, features)) + 0.3 * w
        x[:, 0] *= 1 + 0.2 * w
        gene = np.where(rng.random(n) < 0.3, "nontargeting",
                        rng.choice(["G1", "G2", "G3", "G4"], n))
        x[gene == "G1", 1] += 2.0
        x[gene == "G2", 2] -= 1.5
        mitotic = rng.random(n) < 0.05
        x[rng.random(n) < 0.002, 3] = np.nan
        guide = np.char.add(gene.astype(str), rng.choice(["_a", "_b"], n))
        rows.append((well, x.astype(np.float32), gene, mitotic, guide))
    return rows


def _stack(rows):
    x = np.vstack([r[1][~r[3]] for r in rows])
    meta = pd.DataFrame({"well": np.concatenate([[r[0]] * int((~r[3]).sum()) for r in rows]),
                         "gene": np.concatenate([r[2][~r[3]] for r in rows]),
                         "sgRNA": np.concatenate([r[4][~r[3]] for r in rows])})
    return x, meta


def _reader(rows):
    by = {r[0]: r for r in rows}

    def read(well):
        _w, x, gene, mitotic, _g = by[well]
        return x, pd.Series(gene).str.startswith("nontargeting").to_numpy(), ~mitotic
    return read


# --- tests --------------------------------------------------------------

@pytest.mark.parametrize("sample", [500, 100000])
def test_in_memory_tvn_is_brieflows_tvn_as_the_pilot_ran_it(sample):
    x, meta = _stack(_cells())
    reference = _pilot_in_memory(x.copy(), meta, sample=sample)
    ours, info = ops_tvn.tvn_in_memory(x, meta.well, meta.gene.str.startswith("nontargeting"),
                                       pca_sample=sample)
    assert ours.dtype == np.float32
    assert ours.shape == reference.shape
    np.testing.assert_allclose(ours, reference, rtol=0, atol=2e-5)
    assert info["pca_sample"] == min(sample, len(x))
    assert info["coral_skipped"] == []


@pytest.mark.parametrize("sample", [400, 100000])
def test_streamed_tvn_is_the_pilots_streamed_tvn(sample):
    rows = _cells(seed=1)
    reference = _pilot_streamed([(r[0], r[1], r[2], r[3]) for r in rows], sample=sample)
    model = ops_tvn.fit_tvn_streamed([r[0] for r in rows], _reader(rows), pca_sample=sample)
    for r, ref in zip(rows, reference):
        got = model.transform(r[1][~r[3]], r[0])
        np.testing.assert_array_equal(got, ref)
    assert model.info["axes"] == reference[0].shape[1]


def test_streamed_and_in_memory_agree_when_the_pca_sees_every_cell():
    rows = _cells(seed=2)
    x, meta = _stack(rows)
    whole, _ = ops_tvn.tvn_in_memory(x, meta.well, meta.gene.str.startswith("nontargeting"))
    model = ops_tvn.fit_tvn_streamed([r[0] for r in rows], _reader(rows))
    streamed = np.vstack([model.transform(r[1][~r[3]], r[0]) for r in rows])
    # Same model, different arithmetic (float32 vs float64, one CORAL matrix
    # vs two products): equal up to sign-free rounding.
    assert np.corrcoef(whole.ravel(), streamed.ravel())[0, 1] > 0.9999


def test_tvn_whitens_the_controls_of_every_batch():
    x, meta = _stack(_cells(seed=3))
    control = meta.gene.str.startswith("nontargeting").to_numpy()
    e, _ = ops_tvn.tvn_in_memory(x, meta.well, control)
    for well in meta.well.unique():
        rows = (meta.well == well).to_numpy() & control
        cov = np.cov(e[rows], rowvar=False)
        assert np.allclose(np.diag(cov), np.diag(cov).mean(), rtol=0.35)


def test_whiten_epsilon_zero_is_brieflow_and_above_zero_shrinks_the_weak_axes():
    x, meta = _stack(_cells(seed=4))
    control = meta.gene.str.startswith("nontargeting").to_numpy()
    base, _ = ops_tvn.tvn_in_memory(x, meta.well, control)
    zero, _ = ops_tvn.tvn_in_memory(x, meta.well, control, whiten_epsilon=0.0)
    np.testing.assert_array_equal(base, zero)
    # One batch, no CORAL, independent axes: epsilon enters both scalings, so
    # an axis of control variance v ends at l / (l + eps) with l = v / (v + eps).
    # Weak axes shrink and strong ones barely move.
    rng = np.random.default_rng(0)
    e = rng.normal(size=(2000, 4)) * np.array([3.0, 1.0, 0.1, 0.01])
    control = np.ones(len(e), bool)
    out, _ = ops_tvn.tvn_on_controls(e.copy(), control, whiten_epsilon=0.1)
    v = np.sort(e.var(0))[::-1]
    lam = v / (v + 0.1)
    np.testing.assert_allclose(out.var(0), lam / (lam + 0.1), rtol=1e-2)
    plain, _ = ops_tvn.tvn_on_controls(e.copy(), control)
    np.testing.assert_allclose(plain.var(0), 1, rtol=1e-9)


@pytest.mark.parametrize("components, expected", [(3, 3), (0.5, None)])
def test_components_keep_only_the_leading_control_axes(components, expected):
    rows = _cells(seed=5)
    x, meta = _stack(rows)
    control = meta.gene.str.startswith("nontargeting").to_numpy()
    full, _ = ops_tvn.tvn_in_memory(x, meta.well, control)
    cut, info = ops_tvn.tvn_in_memory(x, meta.well, control, components=components)
    assert cut.shape[1] == info["axes"] < full.shape[1]
    if expected:
        assert cut.shape[1] == expected
    model = ops_tvn.fit_tvn_streamed([r[0] for r in rows], _reader(rows), components=components)
    assert model.info["axes"] == cut.shape[1]


def test_the_pca_sample_and_seed_change_only_through_the_pca():
    x, meta = _stack(_cells(seed=6))
    control = meta.gene.str.startswith("nontargeting").to_numpy()
    a, _ = ops_tvn.tvn_in_memory(x, meta.well, control, pca_sample=300, seed=0)
    b, _ = ops_tvn.tvn_in_memory(x, meta.well, control, pca_sample=300, seed=0)
    c, _ = ops_tvn.tvn_in_memory(x, meta.well, control, pca_sample=300, seed=1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("kwargs", [
    {"pca_variance": 0}, {"pca_variance": 1.5}, {"pca_sample": 1},
    {"components": 0}, {"components": 2.5}, {"whiten_epsilon": -1},
    {"coral_ridge": -0.1}, {"max_nonfinite": 2}])
def test_impossible_options_are_refused(kwargs):
    x, meta = _stack(_cells(seed=7, n=100))
    with pytest.raises(ValueError):
        ops_tvn.tvn_in_memory(x, meta.well, meta.gene.str.startswith("nontargeting"), **kwargs)


def test_tvn_needs_control_cells():
    x, meta = _stack(_cells(seed=8, n=100))
    with pytest.raises(ValueError, match="control"):
        ops_tvn.tvn_in_memory(x, meta.well, np.zeros(len(x), bool))


def test_profiles_are_medians_of_cells_then_of_guides():
    x, meta = _stack(_cells(seed=9))
    e, _ = ops_tvn.tvn_in_memory(x, meta.well, meta.gene.str.startswith("nontargeting"))
    guides, genes = ops_tvn.tvn_profiles(meta, e)
    first = guides.iloc[0]
    rows = (meta.sgRNA == first.guide).to_numpy()
    np.testing.assert_allclose(first["PC_0"], np.median(e[rows, 0]), rtol=1e-6)
    g1 = guides[guides.gene == "G1"]
    assert genes.set_index("gene").loc["G1", "PC_1"] == pytest.approx(g1["PC_1"].median())
    cos = ops_tvn.gene_pair_cosines(genes, [c for c in genes if c.startswith("PC_")])
    assert len(cos) == len(genes) * (len(genes) - 1) // 2
    assert cos.abs().max() <= 1 + 1e-9


def test_settings_map_onto_the_options():
    opts = ops_tvn.tvn_options({"ops_tvn_pca_sample": "5000", "ops_tvn_seed": 3,
                                "ops_tvn_components": 0.9, "ops_tvn_whiten_epsilon": ""})
    assert opts == {"pca_sample": 5000, "seed": 3, "components": 0.9, "whiten_epsilon": 0.0}
    assert ops_tvn.tvn_options({"ops_tvn_components": 40.0})["components"] == 40


# --- the aggregate phase -------------------------------------------------

@pytest.mark.parametrize("mode", ["in_memory", "streamed"])
def test_the_aggregate_phase_writes_tvn_tables_beside_the_robust_z_ones(tmp_path, mode):
    from spacr import ops_engine
    from tests.test_ops_aggregate_blockwise import PLATE, _plate, _settings, _tables

    db, wells, _ = _plate(tmp_path, n=600)
    plain = ops_engine._ph_aggregate(db, PLATE, wells, _settings(tmp_path, ops_bootstrap=0),
                                     str(tmp_path))
    base_guides, base_genes = _tables(plain)
    report = ops_engine._ph_aggregate(
        db, PLATE, wells, _settings(tmp_path, ops_bootstrap=0, ops_profile_normalisation="tvn",
                                    ops_tvn_mode=mode, ops_tvn_pca_sample=500),
        str(tmp_path))
    guides, genes = _tables(report)
    pd.testing.assert_frame_equal(guides, base_guides, check_exact=True)
    pd.testing.assert_frame_equal(genes, base_genes, check_exact=True)
    tvn_guides = pd.read_parquet(report["outputs"]["guides_tvn_parquet"])
    tvn_genes = pd.read_parquet(report["outputs"]["genes_tvn_parquet"])
    assert report["outputs"]["genes_tvn_parquet"].endswith(f"ops_genes_{PLATE}_tvn.parquet")
    assert report["profile_normalisation"] == "tvn"
    # The streamed sample is split between wells by size and rounded per well.
    assert report["tvn"]["mode"] == mode and abs(report["tvn"]["pca_sample"] - 500) <= len(wells)
    pcs = [c for c in tvn_genes if c.startswith("PC_")]
    assert pcs and len(pcs) == report["tvn"]["axes"]
    assert set(tvn_guides.guide) == set(base_guides.guide)
    assert tvn_guides.n_cells.sum() == report["tvn"]["cells"] == report["cells_pooled"]
    assert np.isfinite(tvn_genes[pcs].to_numpy()).all()


def test_the_two_modes_give_the_same_profiles_in_the_aggregate_phase(tmp_path):
    from spacr import ops_engine
    from tests.test_ops_aggregate_blockwise import PLATE, _plate, _settings

    db, wells, _ = _plate(tmp_path, n=600)
    genes = {}
    for mode in ("in_memory", "streamed"):
        report = ops_engine._ph_aggregate(
            db, PLATE, wells, _settings(tmp_path, ops_bootstrap=0,
                                        ops_profile_normalisation="tvn", ops_tvn_mode=mode),
            str(tmp_path))
        genes[mode] = pd.read_parquet(report["outputs"]["genes_tvn_parquet"])
    pcs = [c for c in genes["streamed"] if c.startswith("PC_")]
    a, b = genes["in_memory"][pcs].to_numpy(), genes["streamed"][pcs].to_numpy()
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.999


@pytest.mark.parametrize("key, value", [("ops_profile_normalisation", "zscore"),
                                        ("ops_tvn_mode", "gpu")])
def test_unknown_normalisation_choices_are_refused(tmp_path, key, value):
    from spacr import ops_engine
    from tests.test_ops_aggregate_blockwise import PLATE, _plate, _settings

    db, wells, _ = _plate(tmp_path, n=200)
    settings = _settings(tmp_path, ops_bootstrap=0, ops_profile_normalisation="tvn")
    settings[key] = value
    with pytest.raises(ValueError, match=key):
        ops_engine._ph_aggregate(db, PLATE, wells, settings, str(tmp_path))
