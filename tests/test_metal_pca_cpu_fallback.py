"""Exercise real NumPy/sklearn PCA fallback and pre-allocation input refusal."""

import numpy as np
import pytest

from spacr.mps_reduce import MetalPCA


def test_full_pca_preserves_small_components_and_whitening():
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(9)
    u = np.linalg.qr(rng.normal(size=(200, 2)))[0]
    v = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    data = (u @ np.diag([100, 0.001]) @ v.T).astype("float32")
    model = MetalPCA(n_components=2, svd_solver="full", whiten=True)
    result = model.fit_transform(data)
    reference = PCA(n_components=2, svd_solver="full", whiten=True).fit(data.astype("float64"))
    np.testing.assert_allclose(model.explained_variance_, reference.explained_variance_, rtol=1e-8)
    np.testing.assert_allclose(np.cov(result.T), np.eye(2), atol=1e-8)
    assert model.projection_backend_ == "cpu"
    assert model.decomposition_backend_ == "cpu"
    assert result.dtype == model.components_.dtype == np.float64
    np.testing.assert_array_equal(result, reference.transform(data.astype(np.float64)))
    np.testing.assert_allclose(model.inverse_transform(result),
        reference.inverse_transform(reference.transform(data.astype(np.float64))),
        rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(model.transform(data), result)


def test_pca_wrong_feature_count_refuses_before_metal_allocation(monkeypatch):
    import torch

    data = np.random.default_rng(7).normal(size=(30, 3))
    model = MetalPCA(n_components=2).fit(data)
    assert model.decomposition_backend_ == "cpu"
    assert model.projection_backend_ == "mps"
    original = model.mean_.copy(), model.components_.copy()

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("invalid feature count reached tensor allocation")

    monkeypatch.setattr(torch, "as_tensor", unexpected_allocation)
    with pytest.raises(ValueError, match="Feature count differs from fitted reducer"):
        model.transform(data[:, :1])
    np.testing.assert_array_equal(model.mean_, original[0])
    np.testing.assert_array_equal(model.components_, original[1])


def test_unsupported_pca_solver_refuses_before_tensor_allocation(monkeypatch):
    import torch

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("unsupported solver reached tensor allocation")

    monkeypatch.setattr(torch, "as_tensor", unexpected_allocation)
    with pytest.raises(ValueError, match="This PCA solver remains on the CPU"):
        MetalPCA(n_components=2, svd_solver="randomized")


def test_unwhitened_cpu_pca_projection_and_inverse_match_sklearn():
    """A rank-deficient CPU projection retains float64 coordinates and the unwhitened inverse."""
    from sklearn.decomposition import PCA

    coordinate = np.arange(-5, 6, dtype=np.float64)
    data = np.column_stack((coordinate, coordinate * 2, coordinate * -3)) + [17, -11, 23]
    before = data.copy()
    model = MetalPCA(n_components=3, whiten=False, svd_solver="full").fit(data)
    reference = PCA(n_components=3, whiten=False, svd_solver="full").fit(data)
    assert model.projection_backend_ == model.decomposition_backend_ == "cpu"
    scores = model.transform(data)
    np.testing.assert_array_equal(scores, reference.transform(data))
    np.testing.assert_array_equal(model.inverse_transform(scores), reference.inverse_transform(scores))
    np.testing.assert_allclose(model.inverse_transform(scores), data, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(data, before)
