"""Native Metal coverage for attribution, counterfactuals, OPS and motion models."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available(),
    reason="A working Metal device is required",
)
from spacr import attribution as cf  # noqa: E402


def cnn():
    return (
        torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 2),
        )
        .to("mps")
        .eval()
    )


@pytest.mark.parametrize("method", ["hirescam", "ablation_cam", "gradient_shap", "deeplift_shap"])
def test_new_attribution_methods(method):
    if method in ("gradient_shap", "deeplift_shap"):
        pytest.importorskip("captum")
    model = cnn()
    x = torch.randn(1, 3, 16, 16, device="mps")
    result = cf.attribute(
        model,
        x,
        method=method,
        target=1,
        layer="0",
        n_samples=4,
        baselines=torch.zeros(2, 3, 16, 16, device="mps"),
    )
    assert result.map.shape == (16, 16) and np.isfinite(result.map).all()


@pytest.mark.parametrize("method", ["saliency", "integrated_gradients", "gradcam", "eigencam"])
def test_smoothgrad_wrappers(method):
    if method == "gradcam":
        pytest.importorskip("torchcam")
    if method in ("saliency", "integrated_gradients"):
        pytest.importorskip("captum")
    result = cf.smoothgrad(
        cnn(),
        torch.randn(1, 3, 16, 16, device="mps"),
        base_method=method,
        n_samples=2,
        n_steps=4,
        target=1,
        layer="0",
        seed=2,
    )
    assert np.isfinite(result.map).all()


def test_chefer_vit_on_radeon():
    from torchvision.models.vision_transformer import VisionTransformer

    model = (
        VisionTransformer(
            image_size=32,
            patch_size=8,
            num_layers=2,
            num_heads=2,
            hidden_dim=32,
            mlp_dim=64,
            num_classes=2,
        )
        .to("mps")
        .eval()
    )
    result = cf.attribute(model, torch.randn(1, 3, 32, 32, device="mps"), method="chefer", target=1)
    assert result.map.shape == (32, 32) and np.isfinite(result.map).all()


@pytest.mark.parametrize("kind", ["autoencoder", "diffusion"])
def test_counterfactual_generator_training_and_sequences(kind, tmp_path):
    model = cnn()
    crops = torch.linspace(-1, 1, 12 * 3 * 16 * 16).reshape(12, 3, 16, 16)
    options = (
        {"latent": 4, "batch_size": 4}
        if kind == "autoencoder"
        else {
            "width": 8,
            "timesteps": 8,
            "sample_steps": 2,
            "strength": 0.5,
            "batch_size": 4,
            "augment": False,
        }
    )
    summary, rows, frames = cf._counterfactual_report(
        model,
        crops,
        epochs=1,
        steps=3,
        holdout=0.25,
        show=1,
        seed=2,
        device="mps",
        generator=kind,
        generator_options=options,
        out_dir=str(tmp_path),
    )
    assert len(rows) > 0 and len(frames) > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_active_learning_prediction_bridge():
    from spacr.active_learning import predict_probabilities

    model = cnn()
    images = torch.randn(4, 3, 16, 16)
    result = predict_probabilities(model, images.split(2), device="mps")
    assert result.shape == (4, 2) and np.isfinite(result).all()
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-5)


def test_ops_gpu_primitives_match_cpu():
    from scipy.ndimage import maximum_filter

    from spacr import ops_accel as ops

    rng = np.random.default_rng(8)
    image = rng.normal(size=(48, 48)).astype("float32")
    np.testing.assert_allclose(
        ops.maximum_filter(image, 5, gpu=True, backend="torch"),
        maximum_filter(image, 5, mode="nearest"),
        atol=1e-6,
    )
    a = rng.normal(size=(128, 4)).astype("float32")
    b = rng.normal(size=(4, 8)).astype("float32")
    np.testing.assert_allclose(ops.matmul(a, b, gpu=True, backend="torch"), a @ b, atol=1e-5)
    points = rng.normal(size=(40, 2)).astype("float32")
    target = rng.normal(size=(50, 2)).astype("float32")
    indices, distances = ops.nearest_neighbours(points, target, gpu=True, chunk=16)
    squared = ((points[:, None] - target[None]) ** 2).sum(-1)
    np.testing.assert_array_equal(indices, squared.argmin(1))
    np.testing.assert_allclose(distances, np.sqrt(squared.min(1)), atol=1e-5)


def test_ops_registration_has_working_metal_or_cpu_fallback():
    from spacr.ops_register import phase_correlate

    image = np.random.default_rng(4).normal(size=(64, 64)).astype("float32")
    moved = np.roll(image, (3, -4), axis=(0, 1))
    cpu = phase_correlate(image, moved, gpu=False)
    available = phase_correlate(image, moved, gpu=True)
    assert (available.dy, available.dx) == (cpu.dy, cpu.dx)


def test_timeflows_training_and_prediction_on_metal():
    from spacr import timeflows_model as tm

    class Backbone(torch.nn.Module):
        channels = 4
        ps = 1

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    net = tm.TimeflowsNet(Backbone())
    labels = np.zeros((2, 32, 32), dtype="int32")
    labels[0, 8:16, 8:16] = 1
    labels[1, 8:16, 10:18] = 1
    pair = tm._Pair(
        (labels[0] > 0).astype("float32"), (labels[1] > 0).astype("float32"), labels[0], labels[1]
    )
    history = tm.train_timeflows(net, [pair], head_steps=1, full_steps=1, device="mps")
    assert len(history) == 2 and np.isfinite(history).all()
    result = tm.predict_pair(net, pair.frame_t, pair.frame_t1, device="mps")
    assert result and next(net.parameters()).device.type == "mps"
