"""Held behaviour for TVN input checks and the Harmony embedding in spacr.batch_correction.

Harmony is exercised through a stand-in ``harmonypy`` module so the tests
run without the optional dependency: the stand-in returns the PCA scores
unchanged, which lets the shapes, column names and report be asserted.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from spacr import batch_correction as bc
from spacr.batch_correction import correct_batch_effects, spherize_features


def _frame(n=30, k=5, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, k)), columns=[f"f{i}" for i in range(k)])


def _fake_harmonypy(monkeypatch, transposed=True, version="9.9"):
    calls = {}

    class Result:
        def __init__(self, z):
            self.Z_corr = z.T if transposed else z

    def run_harmony(scores, meta, vars_use, **kwargs):
        calls.update(scores=scores, meta=meta, vars_use=vars_use, kwargs=kwargs)
        return Result(np.asarray(scores))

    module = types.ModuleType("harmonypy")
    module.run_harmony = run_harmony
    if version is not None:
        module.__version__ = version
    monkeypatch.setitem(sys.modules, "harmonypy", module)
    return calls


def test_spherize_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown tvn whitening"):
        spherize_features(_frame(), method="nonsense")


def test_spherize_rejects_empty_or_non_frame_features():
    with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
        spherize_features(_frame().to_numpy())
    with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
        spherize_features(pd.DataFrame())


def test_spherize_rejects_missing_values_and_counts_them():
    feats = _frame()
    feats.iloc[0, 0] = np.nan
    feats.iloc[1, 1] = np.inf
    with pytest.raises(ValueError, match=r"missing or infinite values.*2 found"):
        spherize_features(feats)


def test_spherize_reindexes_a_reference_with_a_different_index():
    feats = _frame(n=20)
    feats.index = [f"w{i}" for i in range(20)]
    reference = pd.Series([True] * 10 + [False] * 10, index=list(feats.index[::-1]))
    out, info = spherize_features(feats, reference, method="ZCA")
    assert info["n_reference"] == 10
    assert out.shape == feats.shape


def test_tvn_rejects_unknown_scope():
    feats = _frame(n=40)
    batch = pd.Series(["a", "b"] * 20)
    control = pd.Series(["DMSO"] * 40)
    with pytest.raises(ValueError, match="Unknown tvn scope"):
        correct_batch_effects(feats, batch, method="tvn", control=control,
                              control_values="DMSO", tvn_scope="galaxy")


def test_tvn_reindexes_a_control_series_with_another_index():
    feats = _frame(n=40)
    feats.index = [f"w{i}" for i in range(40)]
    batch = pd.Series(["a", "b"] * 20, index=feats.index)
    control = pd.Series(["DMSO"] * 40, index=list(feats.index[::-1]))
    out, report = correct_batch_effects(feats, batch, method="tvn", control=control,
                                        control_values="DMSO")
    assert report.controls == 40
    assert out.shape[0] == 40


def test_harmony_embedding_returns_columns_and_info(monkeypatch):
    calls = _fake_harmonypy(monkeypatch, transposed=True)
    feats = _frame(n=30, k=6)
    batch = pd.Series(["a", "b", "c"] * 10)
    out, info = bc.harmony_embedding(feats, batch, n_components=3, theta=1.5, seed=4)
    assert list(out.columns) == ["harmony_1", "harmony_2", "harmony_3"]
    assert out.shape == (30, 3)
    assert info["n_components"] == 3
    assert info["theta"] == 1.5
    assert 0 < info["explained_variance_ratio"] <= 1
    assert info["harmonypy_version"] == "9.9"
    assert calls["vars_use"] == ["batch"]
    assert calls["kwargs"]["random_state"] == 4
    assert calls["scores"].shape == (30, 3)


def test_harmony_embedding_accepts_untransposed_result_and_missing_version(monkeypatch):
    _fake_harmonypy(monkeypatch, transposed=False, version=None)
    feats = _frame(n=20, k=4)
    batch = pd.Series(["a", "b"] * 10)
    out, info = bc.harmony_embedding(feats, batch, n_components=50, standardize=False)
    assert out.shape == (20, 4)
    assert info["harmonypy_version"] == ""


def test_harmony_embedding_aligns_batch_index_and_rejects_bad_input(monkeypatch):
    _fake_harmonypy(monkeypatch)
    feats = _frame(n=20, k=4)
    feats.index = [f"w{i}" for i in range(20)]
    batch = pd.Series(["a", "b"] * 10, index=list(feats.index[::-1]))
    out, _ = bc.harmony_embedding(feats, batch, n_components=2)
    assert list(out.index) == list(feats.index)
    with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
        bc.harmony_embedding(feats.to_numpy(), batch)
    holed = feats.copy()
    holed.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing or infinite"):
        bc.harmony_embedding(holed, batch)
    with pytest.raises(ValueError, match="at least two batches"):
        bc.harmony_embedding(feats, pd.Series(["x"] * 20, index=feats.index))


def test_correct_batch_effects_harmony_reports_the_embedding(monkeypatch):
    _fake_harmonypy(monkeypatch)
    feats = _frame(n=30, k=6)
    batch = pd.Series(["a", "b", "c"] * 10)
    out, report = correct_batch_effects(feats, batch, method="harmony",
                                        harmony_n_components=2)
    assert list(out.columns) == ["harmony_1", "harmony_2"]
    assert report.features == ["harmony_1", "harmony_2"]
    assert any("2-component corrected embedding" in note for note in report.warnings)


def test_spherize_without_a_reference_fits_on_every_row():
    feats = _frame(n=25)
    out, info = spherize_features(feats, method="ZCA")
    assert info["n_reference"] == 25
    assert np.allclose(np.cov(out.to_numpy(), rowvar=False), np.eye(5), atol=1e-3)
