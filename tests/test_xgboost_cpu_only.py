"""Item 794: the xgboost paths fit on a machine without a GPU.

The CUDA build of xgboost 3.4.1 (the default Linux wheel, and the one in the
spacr conda env) raises ``cudaErrorNoDevice`` under
``CUDA_VISIBLE_DEVICES=''`` as soon as a multiclass model sets up its
``mlogloss``/``merror`` metric, even with ``device='cpu'``.
"""

import numpy as np
import pytest

from spacr import ml


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def test_the_probe_sees_no_device_when_every_gpu_is_hidden(no_gpu):
    assert ml._cuda_device_usable() is False


def test_a_cuda_build_without_a_gpu_drops_the_device_bound_metrics(
        no_gpu, monkeypatch):
    monkeypatch.setattr(ml, "_xgboost_cuda_build", lambda: True)
    params = {"eval_metric": "mlogloss", "n_estimators": 3, "device": "cuda"}
    safe = ml._xgboost_cpu_safe(params)
    assert safe["device"] == "cpu"
    assert "eval_metric" not in safe
    assert safe["disable_default_eval_metric"] is True
    assert params["eval_metric"] == "mlogloss"
    assert ml._xgboost_cpu_safe({"eval_metric": ["logloss", "merror@0.5"]})[
        "eval_metric"] == ["logloss"]


def test_a_cpu_build_keeps_its_metrics(no_gpu, monkeypatch):
    monkeypatch.setattr(ml, "_xgboost_cuda_build", lambda: False)
    safe = ml._xgboost_cpu_safe({"eval_metric": "mlogloss"})
    assert safe == {"eval_metric": "mlogloss", "device": "cpu"}


def test_a_gpu_request_on_a_gpu_machine_is_kept(monkeypatch):
    monkeypatch.setattr(ml, "_cuda_device_usable", lambda: True)
    assert ml._xgboost_cpu_safe({"device": "cuda"}) == {"device": "cuda"}
    assert ml._xgboost_cpu_safe({}) == {"device": "cpu"}


@pytest.mark.parametrize("n_classes", [2, 4])
def test_the_classifier_fits_and_predicts_without_a_gpu(no_gpu, n_classes):
    xgboost = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    x = rng.random((60, 4)).astype(np.float32)
    y = np.arange(60) % n_classes
    model = xgboost.XGBClassifier(**ml._xgboost_cpu_safe(dict(
        n_estimators=3, max_depth=2, tree_method="hist",
        eval_metric="logloss" if n_classes == 2 else "mlogloss")))
    proba = model.fit(x, y).predict_proba(x)
    assert proba.shape == (60, n_classes)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_the_cell_cycle_phase_classifier_fits_without_a_gpu(no_gpu):
    pytest.importorskip("xgboost")
    import pandas as pd

    from spacr import measure as M

    rng = np.random.default_rng(1)
    phases = ["G1", "S", "G2", "M"]
    n = 160
    labels = pd.Series([phases[i % 4] for i in range(n)])
    shift = np.array([phases.index(p) for p in labels], dtype=float)
    features = pd.DataFrame({
        "dna": shift + rng.normal(0, 0.1, n),
        "area": rng.random(n),
    })
    groups = np.array([f"w{i % 8}" for i in range(n)])
    phase, confidence, report, _model = M._phases_by_xgboost(
        features, labels, groups, seed=0, n_jobs=1)
    assert len(phase) == n
    assert set(report["classes"]) == set(phases)
