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

import random
import warnings

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

def test_select_channels_zeroes_red_and_green():
    from spacr.utils import SelectChannels
    img = torch.ones(3, 4, 4)
    out = SelectChannels([3])(img)
    assert torch.equal(out[0], torch.zeros(4, 4))   # red zeroed
    assert torch.equal(out[1], torch.zeros(4, 4))   # green zeroed
    assert torch.equal(out[2], torch.ones(4, 4))    # blue kept
    # the transform must not mutate its input (it clones)
    assert torch.equal(img, torch.ones(3, 4, 4))


def test_select_channels_keeps_only_green():
    from spacr.utils import SelectChannels
    img = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2)
    out = SelectChannels([2])(img)
    assert float(out[0].abs().sum()) == 0.0
    assert torch.equal(out[1], img[1])
    assert float(out[2].abs().sum()) == 0.0


def test_select_channels_all_channels_is_identity():
    from spacr.utils import SelectChannels
    img = torch.rand(3, 5, 5)
    out = SelectChannels([1, 2, 3])(img)
    assert torch.equal(out, img)
    assert out is not img


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


def test_compute_saliency_and_predictions_matches_logit_sign():
    from spacr.utils import SaliencyMapGenerator
    model = _tiny(seed=1)
    gen = SaliencyMapGenerator(model)
    X = torch.rand(6, 3, 8, 8)
    sal, preds = gen.compute_saliency_and_predictions(X)
    with torch.no_grad():
        expected = (model(X.detach()).squeeze() > 0).long()
    assert torch.equal(preds, expected)
    assert preds.dtype == torch.int64
    assert sal.shape == X.shape and torch.all(sal >= 0)


def test_saliency_plot_activation_grid_overlay_normalized():
    """3-channel saliency is transposed to HWC and drawn over the input."""
    from spacr.utils import SaliencyMapGenerator
    gen = SaliencyMapGenerator(_tiny())
    n = 9                                    # >8 so subplots() returns a 2-D axes grid
    X = torch.rand(n, 3, 8, 8)
    sal = torch.rand(n, 3, 8, 8)
    preds = torch.arange(n) % 2
    fig = gen.plot_activation_grid(X, sal, preds, overlay=True, normalize=True)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 16               # 2 rows x 8 cols
    for i in range(n):
        ax = fig.axes[i]
        assert len(ax.images) == 2           # image + saliency overlay
        assert ax.images[1].get_alpha() == 0.5
        assert ax.images[1].get_cmap().name == "jet"
        assert ax.images[1].get_array().shape == (8, 8, 3)
        assert [t.get_text() for t in ax.texts] == [str(int(preds[i]))]
        assert ax.axison is False
    assert all(len(ax.images) == 0 for ax in fig.axes[n:])


def test_saliency_plot_activation_grid_2d_saliency_not_transposed():
    from spacr.utils import SaliencyMapGenerator
    gen = SaliencyMapGenerator(_tiny())
    n = 9
    X = torch.rand(n, 3, 8, 8)
    sal = torch.rand(n, 8, 8)                # (N, H, W): no channel transpose
    preds = torch.zeros(n, dtype=torch.long)
    fig = gen.plot_activation_grid(X, sal, preds, overlay=True, normalize=False)
    assert fig.axes[0].images[1].get_array().shape == (8, 8)
    assert all(t.get_text() == "0" for ax in fig.axes[:n] for t in ax.texts)


def test_saliency_plot_activation_grid_without_overlay_draws_the_map_alone():
    """overlay=False draws the MAP, not nothing.

    Renamed from ``..._draws_labels_only``, which pinned a defect: the
    ``imshow`` calls sat inside ``if overlay``, so overlay=False produced a
    grid of bare class labels on empty axes -- no input and no saliency map.
    A parameter called "overlay" means "draw the input underneath", not
    "draw nothing at all".
    """
    from spacr.utils import SaliencyMapGenerator
    gen = SaliencyMapGenerator(_tiny())
    n = 9
    X = torch.rand(n, 3, 8, 8)
    sal = torch.rand(n, 3, 8, 8)
    preds = torch.ones(n, dtype=torch.long)
    fig = gen.plot_activation_grid(X, sal, preds, overlay=False)
    # exactly one image per populated panel: the map
    assert sum(len(ax.images) for ax in fig.axes) == n
    assert all(len(fig.axes[i].images) == 1 for i in range(n))
    # and two per panel with the overlay, the input plus the map
    overlaid = gen.plot_activation_grid(X, sal, preds, overlay=True)
    assert sum(len(ax.images) for ax in overlaid.axes) == 2 * n
    assert [t.get_text() for t in fig.axes[0].texts] == ["1"]
    assert len(fig.axes[n].texts) == 0


def test_saliency_plot_activation_grid_single_row_batch():
    from spacr.utils import SaliencyMapGenerator
    gen = SaliencyMapGenerator(_tiny())
    X = torch.rand(4, 3, 8, 8)
    sal = torch.rand(4, 3, 8, 8)
    preds = torch.zeros(4, dtype=torch.long)
    fig = gen.plot_activation_grid(X, sal, preds, overlay=True)
    assert len(fig.axes) == 8


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


def test_gradcam_generator_plot_activation_grid_overlay_normalized():
    from spacr.utils import GradCAMGenerator
    gen = GradCAMGenerator(_tiny(seed=6), "features.0")
    n = 9
    X = torch.rand(n, 3, 8, 8)
    cams = torch.rand(n, 8, 8)
    preds = torch.arange(n) % 2
    fig = gen.plot_activation_grid(X, cams, preds, overlay=True, normalize=True)
    assert len(fig.axes) == 16
    for i in range(n):
        ax = fig.axes[i]
        assert len(ax.images) == 2
        assert ax.images[1].get_cmap().name == "jet"
        assert ax.images[1].get_alpha() == 0.5
        assert [t.get_text() for t in ax.texts] == [str(int(preds[i]))]
        assert ax.axison is False
    assert all(len(ax.images) == 0 for ax in fig.axes[n:])


def test_gradcam_generator_plot_activation_grid_without_overlay():
    """The Grad-CAM twin honours overlay the same way the saliency one does.

    Both used to draw NOTHING when overlay was false. They are now one
    contract: the map always draws, and overlay decides whether the input is
    drawn beneath it.
    """
    from spacr.utils import GradCAMGenerator
    gen = GradCAMGenerator(_tiny(seed=7), "features.0")
    n = 9
    X = torch.rand(n, 3, 8, 8)
    cams = torch.rand(n, 8, 8)
    preds = torch.zeros(n, dtype=torch.long)
    fig = gen.plot_activation_grid(X, cams, preds, overlay=False, normalize=False)
    assert sum(len(ax.images) for ax in fig.axes) == n
    assert [t.get_text() for t in fig.axes[0].texts] == ["0"]


def test_both_activation_grids_accept_the_same_map_shapes():
    """The twins disagreed on shape: (3, H, W) rendered in one and raised in
    the other, and (1, H, W) raised in both. One helper now serves both."""
    from spacr.utils import GradCAMGenerator, SaliencyMapGenerator
    n = 4
    X = torch.rand(n, 3, 8, 8)
    preds = torch.zeros(n, dtype=torch.long)
    pairs = [
        (SaliencyMapGenerator(_tiny(seed=11)), "saliency"),
        (GradCAMGenerator(_tiny(seed=12), "features.0"), "gradcam"),
    ]
    for gen, name in pairs:
        for shape in ((n, 8, 8), (n, 1, 8, 8), (n, 3, 8, 8)):
            fig = gen.plot_activation_grid(
                X, torch.rand(*shape), preds, overlay=False)
            assert sum(len(ax.images) for ax in fig.axes) == n, (
                f"{name} drew nothing for a {shape} map")


def test_gradcam_generator_plot_activation_grid_single_row_batch():
    from spacr.utils import GradCAMGenerator
    gen = GradCAMGenerator(_tiny(seed=8), "features.0")
    fig = gen.plot_activation_grid(
        torch.rand(2, 3, 8, 8), torch.rand(2, 8, 8),
        torch.zeros(2, dtype=torch.long), overlay=True)
    assert len(fig.axes) == 8


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

def _write_png(path, rng, size=(30, 40, 3)):
    from PIL import Image
    arr = (rng.random(size) * 255).astype(np.uint8)
    Image.fromarray(arr).save(str(path))
    return arr


def test_preprocess_image_normalizes_with_imagenet_stats(tmp_path, rng):
    from spacr.utils import preprocess_image
    p = tmp_path / "img.png"
    _write_png(p, rng)
    image, tensor = preprocess_image(str(p), normalize=True, image_size=16)
    assert image.mode == "RGB"
    assert image.size == (40, 30)                    # PIL (w, h), unresized
    assert tuple(tensor.shape) == (1, 3, 16, 16)
    assert tensor.dtype == torch.float32

    _, raw = preprocess_image(str(p), normalize=False, image_size=16)
    assert float(raw.min()) >= 0.0 and float(raw.max()) <= 1.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    assert torch.allclose(tensor[0], (raw[0] - mean) / std, atol=1e-5)


def test_preprocess_image_explicit_channels_and_grayscale(tmp_path):
    from spacr.utils import preprocess_image
    from PIL import Image
    p = tmp_path / "gray.png"
    Image.fromarray(np.full((8, 8), 128, dtype=np.uint8), mode="L").save(str(p))
    image, tensor = preprocess_image(str(p), normalize=False, image_size=8,
                                     channels=[1, 2])
    assert image.mode == "RGB"                       # convert('RGB') applied
    assert tuple(tensor.shape) == (1, 3, 8, 8)
    # grayscale -> all three channels identical
    assert torch.allclose(tensor[0, 0], tensor[0, 1])
    assert torch.allclose(tensor[0, 1], tensor[0, 2])
    assert pytest.approx(float(tensor.mean()), abs=1e-3) == 128 / 255


# ---------------------------------------------------------------------------
# class_visualization
# ---------------------------------------------------------------------------

def _save_tiny_model(tmp_path, seed=0):
    model = _tiny(seed=seed)
    path = tmp_path / "model.pth"
    torch.save(model, str(path))
    return str(path)


def _patch_torch_load(monkeypatch):
    """Make torch.load behave like torch < 2.6 (weights_only=False).

    class_visualization calls torch.load(model_path) with no weights_only
    argument, which cannot unpickle a whole nn.Module on torch >= 2.6 (see the
    xfail test below). Patching it here lets the rest of the function body run.
    """
    orig = torch.load

    def _load(f, *a, **kw):
        kw.setdefault("weights_only", False)
        return orig(f, *a, **kw)

    monkeypatch.setattr(torch, "load", _load)


def test_class_visualization_target_class_one(tmp_path, monkeypatch):
    from spacr.utils import class_visualization
    path = _save_tiny_model(tmp_path)
    _patch_torch_load(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(
        (plt.gca().get_title(), tuple(plt.gcf().get_size_inches()))))
    random.seed(0)
    out = class_visualization(
        1, path, dtype=torch.FloatTensor, img_size=8, l2_reg=1e-3,
        learning_rate=1.0, num_iterations=3, blur_every=2, max_jitter=2,
        show_every=25)
    assert isinstance(out, np.ndarray)
    assert out.shape == (8, 8, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.std() > 0.0                       # not a constant image
    # shown at t == 0 and at t == num_iterations - 1, never in between
    assert len(shown) == 2
    assert [t for t, _ in shown] == ["pc\nIteration 1 / 3", "pc\nIteration 3 / 3"]
    assert shown[0][1] == (4.0, 4.0)


def test_class_visualization_target_class_zero_uses_custom_names(tmp_path, monkeypatch):
    from spacr.utils import class_visualization
    path = _save_tiny_model(tmp_path, seed=2)
    _patch_torch_load(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    titles = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: titles.append(plt.gca().get_title()))
    random.seed(1)
    out = class_visualization(
        0, path, dtype=torch.FloatTensor, img_size=8, channels=[0, 1, 2],
        class_names=["neg", "pos"], learning_rate=2.0, num_iterations=1,
        blur_every=10, max_jitter=4, show_every=1)
    assert out.shape == (8, 8, 3)
    assert titles == ["neg\nIteration 1 / 1"]
    # deprocess un-normalizes into the displayable [0, 1] range
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


def test_class_visualization_loads_a_pickled_model(tmp_path, monkeypatch):
    from spacr.utils import class_visualization
    if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 6):
        pytest.skip("torch < 2.6 still defaults to weights_only=False")
    path = _save_tiny_model(tmp_path, seed=3)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    random.seed(2)
    out = class_visualization(1, path, dtype=torch.FloatTensor, img_size=8,
                              num_iterations=1, show_every=1)
    assert out.shape == (8, 8, 3)


# ---------------------------------------------------------------------------
# GradCAM / helpers
# ---------------------------------------------------------------------------

def test_gradcam_index_none_uses_argmax_and_removes_hooks():
    from spacr.utils import GradCAM
    model = _tiny(num_classes=2, seed=11)
    with torch.no_grad():
        model.fc.weight[1].fill_(2.0)            # class 1 wins the argmax
    cam = GradCAM(model, ["features.0"], use_cuda=False)
    x = torch.rand(1, 3, 16, 16)
    out = cam(x, index=None)
    assert out.shape == (16, 16)
    assert np.isfinite(out).all()
    assert np.isclose(out.min(), 0.0) and np.isclose(out.max(), 1.0)
    # every forward hook registered inside __call__ is removed again
    assert len(model.features[0]._forward_hooks) == 0


def test_gradcam_use_cuda_branch_with_identity_cuda(monkeypatch):
    """Exercise the three `if self.cuda:` branches without touching a GPU."""
    from spacr.utils import GradCAM
    seen = {"module": 0, "tensor": 0}

    def _module_cuda(self, *a, **k):
        seen["module"] += 1
        return self

    def _tensor_cuda(self, *a, **k):
        seen["tensor"] += 1
        return self

    monkeypatch.setattr(nn.Module, "cuda", _module_cuda)
    monkeypatch.setattr(torch.Tensor, "cuda", _tensor_cuda)

    model = _tiny(num_classes=2, seed=12)
    cam = GradCAM(model, ["features.0"], use_cuda=True)
    assert seen["module"] == 1 and cam.model is model
    out = cam(torch.rand(1, 3, 16, 16), index=0)
    assert seen["tensor"] == 2                   # input x and the one-hot vector
    assert out.shape == (16, 16)
    assert out.dtype == np.float32
    assert np.isclose(out.min(), 0.0) and np.isclose(out.max(), 1.0)


def test_gradcam_forward_returns_model_output():
    from spacr.utils import GradCAM
    model = _tiny(num_classes=2, seed=13)
    cam = GradCAM(model, ["features.0"], use_cuda=False)
    x = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        assert torch.allclose(cam.forward(x), model(x))


def test_show_cam_on_image_blends_jet_heatmap():
    from spacr.utils import show_cam_on_image
    mask = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float32), (4, 1))
    img = np.zeros((4, 64, 3), dtype=np.float32)
    out = show_cam_on_image(img, mask)
    assert out.shape == (4, 64, 3) and out.dtype == np.uint8
    # cv2 works in BGR: the cold end of jet is blue, the hot end is red
    assert int(out[0, 0].argmax()) == 0
    assert int(out[0, -1].argmax()) == 2
    # cam / np.max(cam) always saturates the strongest channel
    assert out.max() == 255
    # adding image intensity lifts the whole blend off zero
    bright = show_cam_on_image(np.full((4, 64, 3), 0.5, np.float32), mask)
    assert out.min() == 0 and bright.min() > 0


@pytest.mark.parametrize("value", [1.1, 1.5, 2.0, 7.0])
def test_show_cam_on_image_clips_an_unnormalized_mask_instead_of_wrapping(value):
    """A CAM above 1.0 saturates at the hot end of jet, and says so.

    Before the clip, ``np.uint8(255 * mask)`` WRAPPED: 1.1 landed at 24 (the
    cold end), 1.5 at 126 (mid-jet), 2.0 back at 254 -- so the hottest region
    of an un-normalized attribution map could render as the coldest colour,
    inverting what the picture means.
    """
    from spacr.utils import show_cam_on_image
    img = np.zeros((4, 4, 3), dtype=np.float32)
    saturated = show_cam_on_image(img, np.ones((4, 4), dtype=np.float32))

    with pytest.warns(RuntimeWarning, match="clipping"):
        out = show_cam_on_image(img, np.full((4, 4), value, dtype=np.float32))

    assert np.array_equal(out, saturated)
    assert int(out[0, 0].argmax()) == 2          # hot end of jet, in BGR


def test_show_cam_on_image_clips_a_negative_mask_to_the_cold_end():
    from spacr.utils import show_cam_on_image
    img = np.zeros((4, 4, 3), dtype=np.float32)
    coldest = show_cam_on_image(img, np.zeros((4, 4), dtype=np.float32))
    with pytest.warns(RuntimeWarning, match="clipping"):
        out = show_cam_on_image(img, np.full((4, 4), -3.0, dtype=np.float32))
    assert np.array_equal(out, coldest)


def test_show_cam_on_image_leaves_a_normalized_mask_alone():
    """The clip must not warn on the input the function documents."""
    from spacr.utils import show_cam_on_image
    mask = np.tile(np.linspace(0.0, 1.0, 8, dtype=np.float32), (8, 1))
    img = np.full((8, 8, 3), 0.25, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = show_cam_on_image(img, mask)
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_show_cam_on_image_refuses_a_non_finite_mask(bad):
    """A non-finite CAM used to come back as an all-black overlay, which is
    indistinguishable from 'the model looked nowhere'."""
    from spacr.utils import show_cam_on_image
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[1, 1] = bad
    with pytest.raises(ValueError, match="NaN or infinity"):
        show_cam_on_image(np.zeros((4, 4, 3), dtype=np.float32), mask)


def test_show_cam_on_image_refuses_a_non_finite_image():
    from spacr.utils import show_cam_on_image
    img = np.zeros((4, 4, 3), dtype=np.float32)
    img[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        show_cam_on_image(img, np.zeros((4, 4), dtype=np.float32))


def test_show_cam_on_image_refuses_an_image_that_cancels_the_heatmap():
    """The old ``else: cam.fill(0.0)`` branch returned a black frame here."""
    from spacr.utils import show_cam_on_image
    img = np.full((4, 4, 3), -10.0, dtype=np.float32)
    with pytest.raises(ValueError, match="nothing to normalize against"):
        show_cam_on_image(img, np.zeros((4, 4), dtype=np.float32))


def test_show_cam_on_image_never_wraps_the_output_cast():
    """A partly negative image is clipped, not wrapped, on the way out."""
    from spacr.utils import show_cam_on_image
    img = np.zeros((4, 4, 3), dtype=np.float32)
    img[0, 0] = -0.5
    mask = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    out = show_cam_on_image(img, mask)
    assert out.dtype == np.uint8
    assert int(out.min()) == 0 and int(out.max()) == 255


def test_integrated_gradients_default_baseline_shape_and_values():
    from spacr.utils import IntegratedGradients
    model = _tiny(num_classes=2, seed=15)
    model.train()
    ig = IntegratedGradients(model)
    assert not model.training                    # __init__ calls eval()
    x = torch.rand(1, 3, 8, 8)
    attr = ig.generate_integrated_gradients(x, target_label_idx=1, num_steps=8)
    assert isinstance(attr, np.ndarray)
    assert attr.shape == (1, 3, 8, 8)
    assert np.isfinite(attr).all()
    assert np.abs(attr).sum() > 0.0


def test_integrated_gradients_zero_when_baseline_equals_input():
    """input - baseline == 0 makes every attribution exactly zero."""
    from spacr.utils import IntegratedGradients
    ig = IntegratedGradients(_tiny(num_classes=2, seed=16))
    x = torch.rand(1, 3, 8, 8)
    attr = ig.generate_integrated_gradients(x, target_label_idx=0,
                                            baseline=x.clone(), num_steps=4)
    assert np.array_equal(attr, np.zeros_like(attr))


def test_integrated_gradients_rejects_mismatched_baseline():
    from spacr.utils import IntegratedGradients
    ig = IntegratedGradients(_tiny(num_classes=2, seed=17))
    with pytest.raises(AssertionError):
        ig.generate_integrated_gradients(torch.rand(1, 3, 8, 8), 0,
                                         baseline=torch.zeros(1, 3, 4, 4),
                                         num_steps=2)


def test_recommend_target_layers_picks_last_conv():
    from spacr.utils import recommend_target_layers, get_submodules
    model = _tiny(seed=14)
    recommended, all_layers = recommend_target_layers(model)
    assert all_layers == ["features.0"]
    assert recommended == [all_layers[-1]]
    assert "features.0" in get_submodules(model)
