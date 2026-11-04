"""TVN (typical variation normalisation) and Harmony in spacr.batch_correction.

Synthetic plates with a known batch effect and known biology: every batch
gets an additive shift, a per-feature gain and a correlated nuisance
direction; three treatments move the profile along their own directions.
A correction passes when the controls' batch centroids collapse and a
treatment is still found among the same treatment's wells *in other
batches* (leave-batch-out 1-NN, the NSCB idea).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from spacr import batch_correction as bc
from spacr.batch_correction import (
    HARMONY_MISSING_MESSAGE,
    METHODS,
    TVN_WHITENINGS,
    correct_batch_effects,
    correct_from_metadata,
    spherize_features,
)


def _plates(seed=0, n_batches=4, n_ctrl=40, n_per_treat=20, n_feat=12,
            shift_only=False):
    """Plates whose batches differ by shift, gain and a nuisance direction.

    ``shift_only`` keeps just the additive shift, which is Harmony's model.
    """
    rng = np.random.default_rng(seed)
    treat_dirs = {f"t{k}": rng.normal(size=n_feat) * 1.5 for k in range(3)}
    rows, meta = [], []
    for b in range(n_batches):
        shift = rng.normal(scale=4.0, size=n_feat)
        gain = rng.uniform(0.5, 2.0, size=n_feat)
        nuisance = rng.normal(size=n_feat)
        if shift_only:
            shift, gain, nuisance = shift / 2.0, np.ones(n_feat), np.zeros(n_feat)
        groups = ["DMSO"] * n_ctrl + [t for t in treat_dirs for _ in range(n_per_treat)]
        for g in groups:
            x = rng.normal(size=n_feat) + rng.normal(scale=3.0) * nuisance
            if g != "DMSO":
                x = x + treat_dirs[g]
            rows.append(x * gain + shift)
            meta.append((f"plate{b}", g))
    feats = pd.DataFrame(rows, columns=[f"f{i}" for i in range(n_feat)])
    meta = pd.DataFrame(meta, columns=["plate", "treatment"])
    return feats, meta


def _ctrl_spread(values, meta):
    ctrl = meta.treatment == "DMSO"
    centers = values[ctrl.values].groupby(meta.plate[ctrl].values).mean()
    return float(centers.std(axis=0, ddof=0).mean() / values[ctrl.values].std(axis=0).mean())


def _cross_batch_nn(values, meta):
    """1-NN accuracy (cosine, treatments only), neighbour must be in another batch."""
    keep = (meta.treatment != "DMSO").values
    x = values.values[keep]
    x = x - values.values[(meta.treatment == "DMSO").values].mean(axis=0)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = meta.treatment.values[keep]
    b = meta.plate.values[keep]
    sim = x @ x.T
    sim[b[:, None] == b[None, :]] = -np.inf
    return float((y[sim.argmax(1)] == y).mean())


def test_tvn_and_harmony_are_registered_methods():
    assert "tvn" in METHODS and "harmony" in METHODS
    assert set(TVN_WHITENINGS) == {"ZCA-cor", "ZCA", "PCA-cor", "PCA"}


def test_tvn_removes_the_batch_effect_and_keeps_the_biology():
    feats, meta = _plates()
    before_spread = _ctrl_spread(feats, meta)
    before_nn = _cross_batch_nn(feats, meta)
    out, report = correct_batch_effects(
        feats, meta.plate, method="tvn", control=meta.treatment,
        control_values="DMSO",
    )
    assert list(out.columns) == list(feats.columns)
    assert out.index.equals(feats.index)
    assert report.controls == 160
    # one global whitening cannot undo per-batch gains, so some spread is left
    assert _ctrl_spread(out, meta) < 0.35 * before_spread
    assert _cross_batch_nn(out, meta) > before_nn + 0.2


def test_tvn_whitens_the_controls_to_identity():
    feats, meta = _plates(seed=1)
    out, _ = correct_batch_effects(
        feats, meta.plate, method="tvn", control=meta.treatment,
        control_values="DMSO", tvn_center_batches=False, tvn_epsilon=0.0,
    )
    ctrl = out[(meta.treatment == "DMSO").values].to_numpy()
    cov = np.cov(ctrl - ctrl.mean(0), rowvar=False)
    assert np.allclose(cov, np.eye(feats.shape[1]), atol=1e-8)


def _reference_spherize(x, ref, method, eps):
    """pycytominer Spherize, written out independently."""
    mu = ref.mean(0)
    xr = ref - mu
    v_inv = np.eye(x.shape[1])
    if method.endswith("-cor"):
        v_inv = np.diag(1.0 / xr.std(0, ddof=1))
    c = np.cov(xr @ v_inv, rowvar=False)
    s, u = np.linalg.eigh(c)
    d = np.diag(1.0 / np.sqrt(s + eps))
    w = (u @ d @ u.T if method.startswith("ZCA") else d @ u.T) @ v_inv
    return (x - mu) @ w.T


@pytest.mark.parametrize("method", TVN_WHITENINGS)
def test_spherize_matches_the_pycytominer_formula(method):
    feats, meta = _plates(seed=2)
    ref = (meta.treatment == "DMSO")
    out, info = spherize_features(feats, ref, method=method, epsilon=1e-6)
    expected = _reference_spherize(feats.to_numpy(), feats[ref.values].to_numpy(),
                                   method, 1e-6)
    got = out.to_numpy()
    if method.startswith("PCA"):
        # eigenvector order and sign are not unique: compare up to them
        assert list(out.columns) == [f"tvn_PC{i + 1}" for i in range(feats.shape[1])]
        assert np.allclose(got @ got.T, expected @ expected.T, atol=1e-6)
    else:
        assert np.allclose(got, expected, atol=1e-8)
    assert info["rank"] == feats.shape[1]


def test_all_full_rank_whitenings_give_the_same_cosines():
    feats, meta = _plates(seed=3)
    ref = (meta.treatment == "DMSO")
    grams = []
    for method in ("ZCA", "PCA"):
        out, _ = spherize_features(feats, ref, method=method, epsilon=0.0)
        x = out.to_numpy()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        grams.append(x @ x.T)
    assert np.allclose(grams[0], grams[1], atol=1e-8)


def test_too_few_controls_warns_and_auto_truncates():
    rng = np.random.default_rng(4)
    feats = pd.DataFrame(rng.normal(size=(60, 30)))
    ref = pd.Series(np.arange(60) < 10)
    full, info = spherize_features(feats, ref, method="ZCA", n_components=None)
    assert info["rank"] == 9
    assert any("span 9 of 30" in note for note in info["warnings"])
    auto, info = spherize_features(feats, ref, method="ZCA", n_components="auto")
    assert info["n_components"] == 9 and not info["warnings"]
    assert np.isfinite(auto.to_numpy()).all()
    # null-space directions are amplified ~1000x without truncation
    assert np.abs(full.to_numpy()[~ref.values]).max() > 50 * np.abs(auto.to_numpy()[~ref.values]).max()
    pcs, info = spherize_features(feats, ref, method="PCA", n_components=5)
    assert pcs.shape == (60, 5)


def test_tvn_per_batch_scope_and_aliases():
    feats, meta = _plates(seed=5)
    out, report = correct_batch_effects(
        feats, meta.plate, method="sphere", control=meta.treatment,
        control_values=["DMSO"], tvn_scope="batch", tvn_method="ZCA",
    )
    assert report.method == "tvn"
    assert _ctrl_spread(out, meta) < 0.05 * _ctrl_spread(feats, meta)
    assert _cross_batch_nn(out, meta) > _cross_batch_nn(feats, meta) + 0.3
    with pytest.raises(ValueError, match="needs a ZCA whitening"):
        correct_batch_effects(feats, meta.plate, method="tvn", control=meta.treatment,
                              control_values="DMSO", tvn_scope="batch", tvn_method="PCA")


def test_tvn_pca_output_renames_columns_in_the_report():
    feats, meta = _plates(seed=6)
    out, report = correct_batch_effects(
        feats, meta.plate, method="tvn", control=meta.treatment,
        control_values="DMSO", tvn_method="PCA", tvn_n_components=4,
    )
    assert list(out.columns) == ["tvn_PC1", "tvn_PC2", "tvn_PC3", "tvn_PC4"]
    assert report.features == list(out.columns)


@pytest.mark.parametrize("kwargs, message", [
    (dict(control=None, control_values=None), "requires batch_control_column"),
    (dict(control_values="nothing"), "reference-control row"),
    (dict(control_values="DMSO", tvn_scope="plate"), "Unknown tvn scope"),
    (dict(control_values="DMSO", tvn_method="cholesky"), "Unknown tvn whitening"),
    (dict(control_values="DMSO", tvn_n_components="most"), "n_components must be"),
    (dict(control_values="DMSO", tvn_n_components=0), "n_components must be >= 1"),
    (dict(control_values="DMSO", tvn_epsilon=-1), "epsilon must be >= 0"),
])
def test_tvn_refuses_what_it_cannot_do(kwargs, message):
    feats, meta = _plates(seed=7)
    kwargs.setdefault("control", meta.treatment)
    with pytest.raises(ValueError, match=message):
        correct_batch_effects(feats, meta.plate, method="tvn", **kwargs)


def test_tvn_needs_controls_in_every_batch():
    feats, meta = _plates(seed=8)
    treatment = meta.treatment.copy()
    treatment[(meta.plate == "plate2") & (treatment == "DMSO")] = "t0"
    with pytest.raises(ValueError, match=r"\['plate2'\] have fewer"):
        correct_batch_effects(feats, meta.plate, method="tvn", control=treatment,
                              control_values="DMSO")
    # without centring and with a global fit, one control-free batch is allowed
    out, _ = correct_batch_effects(feats, meta.plate, method="tvn", control=treatment,
                                   control_values="DMSO", tvn_center_batches=False)
    assert np.isfinite(out.to_numpy()).all()


def test_tvn_refuses_missing_values_and_constant_controls():
    feats, meta = _plates(seed=9)
    holed = feats.copy()
    holed.iloc[3, 2] = np.nan
    with pytest.raises(ValueError, match="missing or infinite"):
        correct_batch_effects(holed, meta.plate, method="tvn", control=meta.treatment,
                              control_values="DMSO")
    flat = feats.copy()
    flat.loc[(meta.treatment == "DMSO").values, "f0"] = 1.0
    with pytest.raises(ValueError, match="constant in the reference"):
        spherize_features(flat, meta.treatment == "DMSO", method="ZCA-cor")
    with pytest.raises(ValueError, match="at least 2 reference rows"):
        spherize_features(feats, pd.Series(np.arange(len(feats)) == 0))
    with pytest.raises(ValueError, match="zero variance and epsilon=0"):
        spherize_features(flat, meta.treatment == "DMSO", method="ZCA", epsilon=0.0)


def test_tvn_one_batch_is_a_warned_no_op():
    feats, meta = _plates(seed=10, n_batches=1)
    out, report = correct_batch_effects(feats, meta.plate, method="tvn",
                                        control=meta.treatment, control_values="DMSO")
    assert np.allclose(out.to_numpy(), feats.to_numpy())
    assert "no-op" in report.warnings[0]


def test_tvn_through_correct_from_metadata():
    feats, meta = _plates(seed=11)
    out, report = correct_from_metadata(
        feats, meta, batch_correction="tvn", batch_column="plate",
        batch_control_column="treatment", batch_control_values="DMSO",
    )
    assert report.method == "tvn" and report.controls == 160
    assert _ctrl_spread(out, meta) < 0.35 * _ctrl_spread(feats, meta)


def test_harmony_without_harmonypy_says_how_to_get_it(monkeypatch):
    feats, meta = _plates(seed=12)
    monkeypatch.setitem(sys.modules, "harmonypy", None)
    with pytest.raises(ImportError) as caught:
        correct_batch_effects(feats, meta.plate, method="harmony")
    assert str(caught.value) == HARMONY_MISSING_MESSAGE
    assert "pip install harmonypy" in HARMONY_MISSING_MESSAGE


def _batch_mixing(values, meta, k=10):
    """Fraction of each row's k nearest neighbours from another batch."""
    x = values.to_numpy()
    d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    b = meta.plate.to_numpy()
    return float((b[nn] != b[:, None]).mean())


def test_harmony_mixes_batches_and_keeps_the_biology():
    pytest.importorskip("harmonypy")
    feats, meta = _plates(seed=13, shift_only=True)
    out, report = correct_batch_effects(feats, meta.plate, method="harmony",
                                        harmony_n_components=8)
    assert list(out.columns) == [f"harmony_{i + 1}" for i in range(8)]
    assert report.features == list(out.columns)
    assert any("embedding" in note for note in report.warnings)
    assert _batch_mixing(out, meta) > _batch_mixing(feats, meta) + 0.3
    assert _cross_batch_nn(out, meta) > 0.9
    again, _ = correct_batch_effects(feats, meta.plate, method="harmony",
                                     harmony_n_components=8)
    assert np.allclose(out.to_numpy(), again.to_numpy())


def test_harmony_refuses_bad_input():
    pytest.importorskip("harmonypy")
    feats, meta = _plates(seed=14)
    holed = feats.copy()
    holed.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="missing or infinite"):
        bc.harmony_embedding(holed, meta.plate)
    with pytest.raises(ValueError, match="at least two batches"):
        bc.harmony_embedding(feats, pd.Series(["one"] * len(feats)))


def test_tvn_is_an_alpha_dropdown_entry_and_harmony_is_not_offered():
    from spacr.settings import ALPHA_FEATURES, _is_alpha
    from spacr.qt.screens import settings_model

    assert set(ALPHA_FEATURES[770]["choices"]["batch_correction"]) == {"tvn"}
    assert _is_alpha("choices", "batch_correction", "tvn")
    assert not _is_alpha("choices", "batch_correction", "combat")
    assert "tvn" in settings_model._BATCH_CORRECTION_OPTIONS
    assert "harmony" not in settings_model._BATCH_CORRECTION_OPTIONS
    assert bc.EMBEDDING_METHODS == ("harmony",)
