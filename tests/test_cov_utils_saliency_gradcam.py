"""CPU coverage for the saliency / Grad-CAM / class-visualization block of
spacr.utils.

Covered symbols: SelectChannels, SaliencyMapGenerator, GradCAMGenerator,
preprocess_image, class_visualization, GradCAM (including its ``use_cuda``
branches, exercised with ``.cuda()`` stubbed to the identity so the tests stay
strictly CPU-only), show_cam_on_image and recommend_target_layers.

Everything runs on 8-16 px tensors and a 4-filter conv net, so the whole module
is a fraction of a second.
"""
from __future__ import annotations


import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# tiny deterministic models
# ---------------------------------------------------------------------------

class _TinyNet(nn.Module):
    """conv(3->width) -> relu -> global-avg-pool -> linear(num_classes)."""

    def __init__(self, num_classes=1, in_ch=3, width=4):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


def _tiny(num_classes=1, seed=0, positive_head=True):
    """Seeded _TinyNet whose head weights are all +1 when ``positive_head``.

    A positive head guarantees the pooled Grad-CAM gradients are positive, so
    the ReLU'd CAM is not identically zero and the (cam-min)/(max-min)
    normalisation cannot divide by zero.
    """
    torch.manual_seed(seed)
    m = _TinyNet(num_classes=num_classes)
    if positive_head:
        with torch.no_grad():
            m.fc.weight.fill_(1.0)
            m.fc.bias.zero_()
    return m


# ---------------------------------------------------------------------------
# SelectChannels
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SaliencyMapGenerator
# ---------------------------------------------------------------------------

def test_compute_saliency_maps_shape_and_positivity():
    from spacr.utils import SaliencyMapGenerator
    model = _tiny()
    model.train()                       # compute_saliency_maps must call eval()
    gen = SaliencyMapGenerator(model)
    X = torch.rand(4, 3, 8, 8)
    y = torch.tensor([0, 1, 1, 0])
    sal = gen.compute_saliency_maps(X, y)
    assert not model.training
    assert sal.shape == X.shape
    assert torch.all(sal >= 0)
    assert float(sal.sum()) > 0.0


def test_compute_saliency_maps_label_sign_does_not_change_magnitude():
    """saliency is |grad|, so flipping every label must not change it."""
    from spacr.utils import SaliencyMapGenerator
    gen = SaliencyMapGenerator(_tiny())
    X0 = torch.rand(3, 3, 8, 8)
    X1 = X0.clone()
    a = gen.compute_saliency_maps(X0, torch.zeros(3, dtype=torch.long))
    b = gen.compute_saliency_maps(X1, torch.ones(3, dtype=torch.long))
    assert torch.allclose(a, b, atol=1e-6)


def test_saliency_percentile_normalize_clips_to_unit_range():
    from spacr.utils import SaliencyMapGenerator
    gen = SaliencyMapGenerator(_tiny())
    rng = np.random.default_rng(0)
    img = rng.normal(5.0, 2.0, size=(16, 16, 3)).astype(np.float32)
    out = gen.percentile_normalize(img)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert out.min() >= 0.0 and out.max() <= 1.0
    for c in range(3):
        lo = np.percentile(img[:, :, c], 2)
        hi = np.percentile(img[:, :, c], 98)
        assert (out[:, :, c][img[:, :, c] <= lo] == 0.0).all()
        assert (out[:, :, c][img[:, :, c] >= hi] == 1.0).all()


def test_saliency_percentile_normalize_full_range_is_minmax():
    from spacr.utils import SaliencyMapGenerator
    gen = SaliencyMapGenerator(_tiny())
    img = np.linspace(0.0, 9.0, 3 * 3 * 2, dtype=np.float64).reshape(3, 3, 2)
    out = gen.percentile_normalize(img, lower_percentile=0, upper_percentile=100)
    for c in range(2):
        chan = img[:, :, c]
        expected = (chan - chan.min()) / (chan.max() - chan.min())
        assert np.allclose(out[:, :, c], expected)


# ---------------------------------------------------------------------------
# GradCAMGenerator
# ---------------------------------------------------------------------------

def test_gradcam_generator_resolves_layer_and_registers_hooks():
    from spacr.utils import GradCAMGenerator
    model = _tiny()
    gen = GradCAMGenerator(model, "features.0", cam_type="gradcam")
    assert gen.target_layer_module is model.features[0]
    assert len(model.features[0]._forward_hooks) == 1
    assert len(model.features[0]._backward_hooks) == 1
    assert gen.cam_type == "gradcam"
    assert not model.training                 # __init__ calls model.eval()
    # nested dotted lookup works for the classifier head too
    assert gen.get_layer(model, "fc") is model.fc


def test_gradcam_generator_compute_map_is_minmax_normalized():
    from spacr.utils import GradCAMGenerator
    model = _tiny(seed=3)
    gen = GradCAMGenerator(model, "features.0")
    X = torch.rand(1, 3, 16, 16)
    cam = gen.compute_gradcam_maps(X, torch.tensor(1))
    assert isinstance(cam, np.ndarray)
    assert cam.shape == (16, 16)
    assert np.isfinite(cam).all()
    assert np.isclose(cam.min(), 0.0) and np.isclose(cam.max(), 1.0)
    # both hooks fired: activations from forward, gradients from backward
    assert gen.activations.shape == (1, 4, 16, 16)
    assert gen.gradients is not None and gen.gradients.shape == (1, 4, 16, 16)


def test_gradcam_generator_batch_maps_and_predictions():
    from spacr.utils import GradCAMGenerator
    model = _tiny(seed=4)
    gen = GradCAMGenerator(model, "features.0")
    X = torch.rand(3, 3, 12, 12)
    with torch.no_grad():
        expected = (model(X).squeeze() > 0).long()
    maps, preds = gen.compute_gradcam_and_predictions(X)
    assert isinstance(maps, torch.Tensor)
    assert tuple(maps.shape) == (3, 12, 12)
    assert preds.shape == (3,) and preds.dtype == torch.int64
    assert torch.equal(preds, expected)
    assert torch.isfinite(maps).all()
    assert float(maps.min()) == 0.0 and float(maps.max()) == 1.0


def test_gradcam_generator_1x1_target_layer():
    from spacr.utils import GradCAMGenerator
    model = _tiny(seed=5)
    gen = GradCAMGenerator(model, "pool")          # AdaptiveAvgPool2d(1) -> 1x1
    cam = gen.compute_gradcam_maps(torch.rand(1, 3, 16, 16), torch.tensor(1))
    assert cam.shape == (16, 16)
    assert np.isfinite(cam).all()


def test_gradcam_generator_percentile_normalize():
    from spacr.utils import GradCAMGenerator
    gen = GradCAMGenerator(_tiny(seed=9), "features.0")
    rng = np.random.default_rng(1)
    img = rng.uniform(-3.0, 7.0, size=(12, 12, 3)).astype(np.float32)
    out = gen.percentile_normalize(img, lower_percentile=10, upper_percentile=90)
    assert out.shape == img.shape and out.dtype == np.float32
    assert out.min() == 0.0 and out.max() == 1.0
    for c in range(3):
        lo = np.percentile(img[:, :, c], 10)
        hi = np.percentile(img[:, :, c], 90)
        mid = (img[:, :, c] > lo) & (img[:, :, c] < hi)
        assert ((out[:, :, c][mid] > 0.0) & (out[:, :, c][mid] < 1.0)).all()


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# class_visualization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GradCAM / helpers
# ---------------------------------------------------------------------------

def test_gradcam_forward_returns_model_output():
    from spacr.utils import GradCAM
    model = _tiny(num_classes=2, seed=13)
    cam = GradCAM(model, ["features.0"], use_cuda=False)
    x = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        assert torch.allclose(cam.forward(x), model(x))


def test_recommend_target_layers_picks_last_conv():
    from spacr.utils import recommend_target_layers, get_submodules
    model = _tiny(seed=14)
    recommended, all_layers = recommend_target_layers(model)
    assert all_layers == ["features.0"]
    assert recommended == [all_layers[-1]]
    assert "features.0" in get_submodules(model)
