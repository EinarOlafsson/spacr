"""Real Metal reducer checks against scientific CPU references."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available(),
    reason="A working Metal device is required",
)


def test_metal_pca_covariance_projection_and_whitening():
    from sklearn.decomposition import PCA

    from spacr.mps_reduce import MetalPCA

    x = np.random.default_rng(12).normal(size=(200, 8)).astype("float32")
    x *= np.arange(1, 9)
    reference = PCA(n_components=4, whiten=True).fit(x)
    model = MetalPCA(n_components=4, whiten=True)
    output = model.fit_transform(x)
    np.testing.assert_allclose(model.explained_variance_, reference.explained_variance_, rtol=1e-4)
    np.testing.assert_allclose(np.cov(output.T), np.eye(4), atol=1e-4)
    np.testing.assert_allclose(
        model.inverse_transform(output),
        reference.inverse_transform(reference.transform(x)),
        atol=2e-4,
    )


def test_metal_kmeans_matches_separated_cpu_clusters():
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    from spacr.mps_reduce import MetalKMeans

    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(k * 12, 0.3, (50, 6)) for k in range(3)]).astype("float32")
    model = MetalKMeans(n_clusters=3, random_state=1, n_init=2)
    labels = model.fit_predict(x)
    reference = KMeans(n_clusters=3, random_state=1, n_init=2).fit(x)
    assert adjusted_rand_score(labels, reference.labels_) == 1
    np.testing.assert_array_equal(model.predict(x), labels)
    np.testing.assert_allclose(model.inertia_, reference.inertia_, rtol=0.01)


def test_metal_backend_is_opt_in_and_unsupported_options_keep_cpu():
    from spacr.gpu_reduce import make_reducer

    assert make_reducer("pca", prefer_gpu=False)[1] == "cpu"
    assert make_reducer("pca", prefer_gpu=True)[1] == "mps"
    assert make_reducer("kmeans", prefer_gpu=True)[1] == "mps"
    assert make_reducer("pca", prefer_gpu=True, svd_solver="randomized")[1] == "cpu"
    assert make_reducer("tsne", prefer_gpu=True)[1] == "cpu"


def test_image_reduction_and_clustering_runs_metal_end_to_end():
    from sklearn.metrics import adjusted_rand_score

    from spacr.utils import reduction_and_clustering

    rng = np.random.default_rng(23)
    data = np.concatenate([rng.normal(k * 10, 0.2, (20, 6)) for k in range(3)])
    embedding, labels, reducer = reduction_and_clustering(
        data,
        5,
        0.1,
        "euclidean",
        0.5,
        3,
        "kmeans",
        reduction_method="pca",
        prefer_gpu=True,
        n_jobs=1,
    )
    assert reducer._spacr_backend == "mps"
    assert embedding.shape == (60, 2)
    assert adjusted_rand_score(np.repeat(np.arange(3), 20), labels) == 1


def test_cellpose_training_context_restores_references_after_failure():
    from cellpose import train

    from spacr.metal_cellpose import training_context

    original = train.torch, train.dynamics, train.random_rotate_and_resize
    with pytest.raises(RuntimeError, match="cancelled"):
        with training_context():
            assert train.torch is not original[0]
            with train.torch.autocast(device_type="mps"):
                assert torch.ones(1, device="mps").dtype == torch.float32
            raise RuntimeError("cancelled")
    assert (train.torch, train.dynamics, train.random_rotate_and_resize) == original


def test_kmeans_translation_does_not_change_clusters_or_inertia():
    from sklearn.metrics import adjusted_rand_score

    from spacr.mps_reduce import MetalKMeans

    rng = np.random.default_rng(2)
    data = np.concatenate([rng.normal(k, 0.1, (30, 2)) for k in (-2, 2)])
    original = MetalKMeans(n_clusters=2, random_state=42).fit(data)
    shifted = MetalKMeans(n_clusters=2, random_state=42).fit(data + 10000)
    assert adjusted_rand_score(original.labels_, shifted.labels_) == 1
    np.testing.assert_allclose(original.inertia_, shifted.inertia_, rtol=1e-4)
    np.testing.assert_array_equal(shifted.predict(data + 10000), shifted.labels_)
    distances = shifted.fit_transform(data + 10000)
    np.testing.assert_allclose(np.square(distances).min(axis=1).sum(), shifted.inertia_, rtol=1e-4)




def test_pca_whitening_preserves_small_absolute_feature_scales():
    from sklearn.decomposition import PCA

    from spacr.mps_reduce import MetalPCA

    data = np.random.default_rng(17).normal(size=(100, 3)) * 1e-9
    reference = PCA(n_components=3, whiten=True).fit(data)
    model = MetalPCA(n_components=3, whiten=True)
    result = model.fit_transform(data)
    assert model.projection_backend_ == "mps"
    np.testing.assert_allclose(result, reference.transform(data), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.cov(result.T), np.eye(3), atol=1e-5)


def test_metal_prediction_rejects_broadcastable_missing_features():
    from spacr.mps_reduce import MetalKMeans, MetalPCA

    data = np.random.default_rng(7).normal(size=(30, 3))
    pca = MetalPCA(n_components=2).fit(data)
    kmeans = MetalKMeans(n_clusters=2, random_state=7).fit(data)
    with pytest.raises(ValueError, match="Feature count"):
        pca.transform(data[:, :1])
    with pytest.raises(ValueError, match="Feature count"):
        kmeans.predict(data[:, :1])


def test_cellpose_training_budget_keeps_fitting_models_on_metal(monkeypatch, capsys):
    from spacr import metal_cellpose

    net = torch.nn.Linear(4, 2).to("mps")
    monkeypatch.setattr(metal_cellpose, "_radeon_vram_bytes", lambda: 1024**3)
    metal_cellpose._fit_training_memory(net)
    assert next(net.parameters()).device.type == "mps"
    monkeypatch.setattr(metal_cellpose, "_radeon_vram_bytes", lambda: 1)
    metal_cellpose._fit_training_memory(net)
    assert next(net.parameters()).device.type == "cpu"
    assert "Training on CPU in float32" in capsys.readouterr().out
