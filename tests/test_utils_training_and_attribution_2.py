"""Attribution helpers of ``spacr.utils`` (part 2 of the training/attribution
consolidation; part 1 is ``test_utils_training_and_attribution.py``).

One table-driven test per unit (two or three for the large ones), each row run
as a subtest so a failing case names itself.  Everything is CPU-only, offline
and works on 8-64 px tensors: tiny stand-in backbones replace TorchVision
wherever a real one would download weights or cost seconds.

Units: SelectChannels, SaliencyMapGenerator, GradCAMGenerator, GradCAM,
IntegratedGradients, preprocess_image, class_visualization,
show_cam_on_image and suggest_training_changes.
"""
from __future__ import annotations

import inspect
import os
import random
import warnings

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from spacr import utils as U  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    plt.close("all")
    yield
    plt.close("all")


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


# ===========================================================================
# Attribution: tiny models
# ===========================================================================

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
    """Seeded _TinyNet whose head weights are all +1 when ``positive_head``."""
    torch.manual_seed(seed)
    m = _TinyNet(num_classes=num_classes)
    if positive_head:
        with torch.no_grad():
            m.fc.weight.fill_(1.0)
            m.fc.bias.zero_()
    return m


class _TinyCNN(nn.Module):
    """Two conv layers over small images; trainable in well under a second."""

    def __init__(self, classes=2, channels=1):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(channels, 4, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(4, 8, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, classes)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


def _square_images(n=2, size=16):
    """Even indices hold a bright central square, odd ones an empty frame."""
    generator = _gen(0)
    out = []
    for index in range(n):
        image = torch.rand((1, size, size), generator=generator) * 0.1
        if index % 2 == 0:
            image[:, 4:12, 4:12] += 0.9
        out.append(image)
    return out


@pytest.fixture(scope="module")
def trained_square_model():
    """A _TinyCNN trained for real to tell the square from the empty frame."""
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(0)
    model = _TinyCNN()
    x = torch.stack(_square_images(16))
    y = torch.tensor([0 if i % 2 == 0 else 1 for i in range(16)])
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _epoch in range(8):
        for batch_x, batch_y in loader:
            optimiser.zero_grad()
            loss_fn(model(batch_x), batch_y).backward()
            optimiser.step()
    model.eval()
    return model


# ===========================================================================
# SelectChannels
# ===========================================================================

def test_select_channels_zeroes_every_unselected_channel(subtests):
    """1-based channel numbers are kept, the rest zeroed, on a clone: the
    input is never mutated and selecting all three is an equal copy."""
    from spacr.utils import SelectChannels
    cases = [
        ("blue-only", [3], torch.ones(3, 4, 4)),
        ("green-only", [2], torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)),
        ("red-and-blue", [1, 3], torch.rand(3, 5, 5, generator=_gen(1))),
        ("all-is-identity", [1, 2, 3], torch.rand(3, 5, 5, generator=_gen(2))),
    ]
    for cid, keep, img in cases:
        with subtests.test(case=cid):
            before = img.clone()
            out = SelectChannels(keep)(img)
            assert out is not img
            assert torch.equal(img, before)
            for ch in range(3):
                expected = img[ch] if (ch + 1) in keep else torch.zeros_like(img[ch])
                assert torch.equal(out[ch], expected), ch


# ===========================================================================
# SaliencyMapGenerator / GradCAMGenerator
# ===========================================================================

def test_saliency_generator_explains_the_class_the_model_chose(
        subtests, trained_square_model):
    """compute_saliency_and_predictions returns one non-negative, non-flat map
    per image at the image's size, predictions equal to the model's own
    (logit sign for one logit, argmax otherwise), weight on the object rather
    than the corners, and one distinct plane per input channel."""
    from PIL import Image
    from spacr.utils import SaliencyMapGenerator

    with subtests.test(case="single-logit-sign"):
        model = _tiny(seed=1)
        X = torch.rand(6, 3, 8, 8, generator=_gen(5))
        sal, preds = SaliencyMapGenerator(model).compute_saliency_and_predictions(X)
        with torch.no_grad():
            expected = (model(X.detach()).squeeze() > 0).long()
        assert torch.equal(preds, expected) and preds.dtype == torch.int64
        assert sal.shape == X.shape and torch.all(sal >= 0)

    gen = SaliencyMapGenerator(trained_square_model)
    for n in (1, 2, 4):
        with subtests.test(case=f"trained-{n}-images"):
            images = torch.stack(_square_images(n))
            maps, predicted = gen.compute_saliency_and_predictions(images)
            assert maps.shape[0] == n and predicted.shape[0] == n
            assert maps.shape[-2:] == images.shape[-2:]
            with torch.no_grad():
                expected = trained_square_model(images).argmax(dim=1)
            assert predicted.flatten().tolist() == expected.tolist()
            values = maps.detach().cpu().numpy()
            assert (values >= 0).all()
            first = values[0].reshape(16, 16)
            assert float(first.max() - first.min()) > 0.0
            centre = float(first[4:12, 4:12].mean())
            corners = float(np.concatenate([first[:4, :4].ravel(), first[:4, -4:].ravel(),
                                            first[-4:, :4].ravel(),
                                            first[-4:, -4:].ravel()]).mean())
            assert centre > corners

    with subtests.test(case="one-plane-per-channel"):
        torch.manual_seed(0)
        model = _TinyCNN(channels=3).eval()
        image = torch.zeros((1, 3, 16, 16))
        image[0, 0, 4:12, 4:12] = 0.9
        image[0, 1, :, 6:10] = 0.6
        image[0, 2] = 0.1
        maps, _ = SaliencyMapGenerator(model).compute_saliency_and_predictions(image)
        activation_map = maps[0].detach().cpu().numpy()
        assert activation_map.shape[0] == 3
        assert not np.allclose(activation_map[0], activation_map[1])
        # Pillow 10 raises ValueError, 11 TypeError; either refusal is the point.
        with pytest.raises((ValueError, TypeError)):
            Image.fromarray((activation_map * 255).astype(np.uint8), mode="L")


def test_activation_grids_draw_the_map_with_or_without_the_input(subtests):
    """Both generators lay maps out 8 per row; each populated panel draws the
    map (jet, alpha 0.5 over the input when overlaid) plus its class label and
    no axis, empty panels stay blank, and (N,H,W), (N,1,H,W) and (N,3,H,W)
    maps all render."""
    from spacr.utils import GradCAMGenerator, SaliencyMapGenerator

    gens = [("saliency", lambda: SaliencyMapGenerator(_tiny(seed=11))),
            ("gradcam", lambda: GradCAMGenerator(_tiny(seed=12), "features.0"))]
    layout = [
        # (id, n, map shape suffix, overlay, normalize, expected axes)
        ("overlay-normalized-3ch", 9, (3, 8, 8), True, True, 16),
        ("overlay-2d", 9, (8, 8), True, False, 16),
        ("overlay-normalized-2d", 9, (8, 8), True, True, 16),
        ("no-overlay-3ch", 9, (3, 8, 8), False, False, 16),
        ("no-overlay-2d", 9, (8, 8), False, False, 16),
        ("no-overlay-1ch", 4, (1, 8, 8), False, False, 8),
        ("single-row-3ch", 4, (3, 8, 8), True, False, 8),
        ("single-row-2d", 2, (8, 8), True, False, 8),
    ]
    for gname, make in gens:
        gen = make()
        for cid, n, shape, overlay, normalize, n_axes in layout:
            with subtests.test(generator=gname, case=cid):
                X = torch.rand(n, 3, 8, 8, generator=_gen(n))
                maps = torch.rand(n, *shape, generator=_gen(n + 1))
                preds = torch.arange(n) % 2
                fig = gen.plot_activation_grid(X, maps, preds, overlay=overlay,
                                               normalize=normalize)
                assert isinstance(fig, plt.Figure)
                assert len(fig.axes) == n_axes
                for i in range(n):
                    ax = fig.axes[i]
                    assert len(ax.images) == (2 if overlay else 1)
                    if overlay:
                        top = ax.images[1]
                        assert top.get_alpha() == 0.5
                        assert top.get_cmap().name == "jet"
                        if gname == "saliency" and shape == (3, 8, 8):
                            assert top.get_array().shape == (8, 8, 3)
                        elif shape == (8, 8):
                            assert top.get_array().shape == (8, 8)
                    assert [t.get_text() for t in ax.texts] == [str(int(preds[i]))]
                    assert ax.axison is False
                for ax in fig.axes[n:]:
                    assert len(ax.images) == 0 and len(ax.texts) == 0
                plt.close(fig)


# ===========================================================================
# GradCAM / IntegratedGradients
# ===========================================================================

def test_gradcam_returns_a_normalised_map_and_fails_loudly_without_gradient(
        subtests, monkeypatch):
    """GradCAM gives an (H,W) float32 map scaled to [0,1], uses the argmax
    class when index is None, removes its hooks, moves model / input / one-hot
    to CUDA when asked (stubbed here), and refuses a frozen target layer."""
    from spacr.utils import GradCAM

    with subtests.test(case="index-none-is-argmax"):
        model = _tiny(num_classes=2, seed=11)
        with torch.no_grad():
            model.fc.weight[1].fill_(2.0)            # class 1 wins the argmax
        cam = GradCAM(model, ["features.0"], use_cuda=False)
        x = torch.rand(1, 3, 16, 16, generator=_gen(6))
        out = cam(x, index=None)
        assert out.shape == (16, 16) and np.isfinite(out).all()
        assert np.isclose(out.min(), 0.0) and np.isclose(out.max(), 1.0)
        assert np.allclose(out, cam(x, index=1))
        assert len(model.features[0]._forward_hooks) == 0

    def build():
        return nn.Sequential(nn.Conv2d(3, 2, 3, padding=1), nn.ReLU(),
                             nn.Conv2d(2, 2, 3, padding=1),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten())

    x = torch.randn(1, 3, 16, 16, generator=_gen(7))
    with subtests.test(case="sequential-layer-by-index"):
        torch.manual_seed(0)
        cam = GradCAM(build(), target_layers=["0"], use_cuda=False)(x)
        assert cam.shape == (16, 16)
        assert 0.0 <= float(cam.min()) and float(cam.max()) == pytest.approx(1.0)

    with subtests.test(case="frozen-layer-refused"):
        frozen = build()
        for parameter in frozen[0].parameters():
            parameter.requires_grad_(False)
        with pytest.raises(AttributeError, match="NoneType"):
            GradCAM(frozen, target_layers=["0"], use_cuda=False)(x)

    with subtests.test(case="use-cuda-branches"):
        seen = {"module": 0, "tensor": 0}

        def _module_cuda(self, *a, **k):
            seen["module"] += 1
            return self

        def _tensor_cuda(self, *a, **k):
            seen["tensor"] += 1
            return self

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(nn.Module, "cuda", _module_cuda)
            mp.setattr(torch.Tensor, "cuda", _tensor_cuda)
            model = _tiny(num_classes=2, seed=12)
            cam = GradCAM(model, ["features.0"], use_cuda=True)
            assert seen["module"] == 1 and cam.model is model
            out = cam(torch.rand(1, 3, 16, 16, generator=_gen(8)), index=0)
        assert seen["tensor"] == 2                   # input x and the one-hot vector
        assert out.shape == (16, 16) and out.dtype == np.float32
        assert np.isclose(out.min(), 0.0) and np.isclose(out.max(), 1.0)


def test_integrated_gradients_attribute_against_the_baseline(subtests):
    """IntegratedGradients puts the model in eval mode, returns a finite,
    non-zero (N,C,H,W) array against the default zero baseline, exactly zero
    when baseline == input, and asserts on a baseline of another shape."""
    from spacr.utils import IntegratedGradients
    x = torch.rand(1, 3, 8, 8, generator=_gen(9))

    with subtests.test(case="default-baseline"):
        model = _tiny(num_classes=2, seed=15)
        model.train()
        ig = IntegratedGradients(model)
        assert not model.training
        attr = ig.generate_integrated_gradients(x, target_label_idx=1, num_steps=8)
        assert isinstance(attr, np.ndarray) and attr.shape == (1, 3, 8, 8)
        assert np.isfinite(attr).all() and np.abs(attr).sum() > 0.0

    with subtests.test(case="baseline-equals-input"):
        ig = IntegratedGradients(_tiny(num_classes=2, seed=16))
        attr = ig.generate_integrated_gradients(x, target_label_idx=0,
                                                baseline=x.clone(), num_steps=4)
        assert np.array_equal(attr, np.zeros_like(attr))

    with subtests.test(case="mismatched-baseline"):
        ig = IntegratedGradients(_tiny(num_classes=2, seed=17))
        with pytest.raises(AssertionError):
            ig.generate_integrated_gradients(x, 0, baseline=torch.zeros(1, 3, 4, 4),
                                             num_steps=2)


# ===========================================================================
# preprocess_image
# ===========================================================================

def test_preprocess_image_returns_rgb_and_a_resized_tensor(subtests, tmp_path, rng):
    """The image comes back RGB at its own size and the tensor (1,3,S,S) in
    [0,1], ImageNet-normalised on request; greyscale fills all three channels
    and the reserved ``channels`` argument changes nothing, here or in
    visualize_smooth_grad."""
    from PIL import Image
    from spacr.utils import preprocess_image

    with subtests.test(case="rgb-imagenet-normalised"):
        p = tmp_path / "img.png"
        Image.fromarray((rng.random((30, 40, 3)) * 255).astype(np.uint8)).save(str(p))
        image, tensor = preprocess_image(str(p), normalize=True, image_size=16)
        assert image.mode == "RGB" and image.size == (40, 30)
        assert tuple(tensor.shape) == (1, 3, 16, 16) and tensor.dtype == torch.float32
        _, raw = preprocess_image(str(p), normalize=False, image_size=16)
        assert float(raw.min()) >= 0.0 and float(raw.max()) <= 1.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        assert torch.allclose(tensor[0], (raw[0] - mean) / std, atol=1e-5)

    with subtests.test(case="greyscale-explicit-channels"):
        p = tmp_path / "gray.png"
        Image.fromarray(np.full((8, 8), 128, dtype=np.uint8), mode="L").save(str(p))
        image, tensor = preprocess_image(str(p), normalize=False, image_size=8,
                                         channels=[1, 2])
        assert image.mode == "RGB" and tuple(tensor.shape) == (1, 3, 8, 8)
        assert torch.allclose(tensor[0, 0], tensor[0, 1])
        assert torch.allclose(tensor[0, 1], tensor[0, 2])
        assert float(tensor.mean()) == pytest.approx(128 / 255, abs=1e-3)

    with subtests.test(case="channels-make-no-difference"):
        from spacr.deep_spacr import visualize_smooth_grad
        size = 8
        src = tmp_path / "src"
        src.mkdir()
        pixels = np.random.default_rng(11).integers(
            0, 256, (size, size, 3), dtype=np.uint16).astype(np.uint8)
        Image.fromarray(pixels).save(src / "a.png")
        _, default_tensor = preprocess_image(str(src / "a.png"), image_size=size,
                                             channels=None)
        _, chosen_tensor = preprocess_image(str(src / "a.png"), image_size=size,
                                            channels=[0, 2])
        assert torch.equal(default_tensor, chosen_tensor)
        torch.manual_seed(3)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * size * size, 2))
        model_path = tmp_path / "m.pth"
        torch.save(model, str(model_path))
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda *a, **k: None)
            for channels, folder in ((None, "default"), ([0, 2], "explicit")):
                visualize_smooth_grad(str(src), str(model_path), 1, image_size=size,
                                      channels=channels, save_smooth_grad=True,
                                      save_dir=str(tmp_path / folder))
        default_map = np.array(Image.open(tmp_path / "default" / "smooth_grad_a.png"))
        explicit_map = np.array(Image.open(tmp_path / "explicit" / "smooth_grad_a.png"))
        assert default_map.shape == (size, size)
        assert np.array_equal(default_map, explicit_map)


# ===========================================================================
# class_visualization
# ===========================================================================

def test_class_visualization_optimises_a_displayable_class_image(subtests, tmp_path):
    """class_visualization loads a pickled model, returns an (S,S,3) image in
    [0,1] that is not constant, and shows it at the first and last iteration
    (and every show_every) titled with the class name."""
    from spacr.utils import class_visualization

    orig_load = torch.load

    def _load_full(f, *a, **kw):
        kw.setdefault("weights_only", False)
        return orig_load(f, *a, **kw)

    cases = [
        # (id, seed, patch load, call kwargs, random seed, expected titles)
        ("class-one-default-names", 0, True,
         dict(target_y=1, l2_reg=1e-3, learning_rate=1.0, num_iterations=3,
              blur_every=2, max_jitter=2, show_every=25),
         0, ["pc\nIteration 1 / 3", "pc\nIteration 3 / 3"]),
        ("class-zero-custom-names", 2, True,
         dict(target_y=0, channels=[0, 1, 2], class_names=["neg", "pos"],
              learning_rate=2.0, num_iterations=1, blur_every=10, max_jitter=4,
              show_every=1),
         1, ["neg\nIteration 1 / 1"]),
        ("default-torch-load", 3, False,
         dict(target_y=1, num_iterations=1, show_every=1), 2,
         ["pc\nIteration 1 / 1"]),
    ]
    for cid, seed, patch_load, kw, rseed, titles in cases:
        with subtests.test(case=cid):
            d = tmp_path / cid
            d.mkdir()
            path = d / "model.pth"
            torch.save(_tiny(seed=seed), str(path))
            shown = []
            with pytest.MonkeyPatch.context() as mp:
                if patch_load:
                    mp.setattr(torch, "load", _load_full)
                mp.setattr(torch.cuda, "is_available", lambda: False)
                mp.setattr(plt, "show", lambda *a, **k: shown.append(
                    (plt.gca().get_title(), tuple(plt.gcf().get_size_inches()))))
                random.seed(rseed)
                target_y = kw.pop("target_y")
                out = class_visualization(target_y, str(path), dtype=torch.FloatTensor,
                                          img_size=8, **kw)
            assert isinstance(out, np.ndarray) and out.shape == (8, 8, 3)
            assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0
            assert out.std() > 0.0
            assert [t for t, _ in shown] == titles
            assert shown[0][1] == (4.0, 4.0)


# ===========================================================================
# show_cam_on_image
# ===========================================================================

def test_show_cam_on_image_blends_clips_and_refuses_what_it_cannot_draw(subtests):
    """The CAM is blended as a BGR jet heatmap saturating at 255; a mask
    outside [0,1] is clipped with a RuntimeWarning (never wrapped), a
    normalised one is drawn silently, a partly negative image is clipped on
    the way out, and non-finite input or an image that cancels the heatmap is
    refused by message."""
    from spacr.utils import show_cam_on_image

    with subtests.test(case="jet-blend"):
        mask = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float32), (4, 1))
        out = show_cam_on_image(np.zeros((4, 64, 3), dtype=np.float32), mask)
        assert out.shape == (4, 64, 3) and out.dtype == np.uint8
        assert int(out[0, 0].argmax()) == 0 and int(out[0, -1].argmax()) == 2
        assert out.max() == 255 and out.min() == 0
        bright = show_cam_on_image(np.full((4, 64, 3), 0.5, np.float32), mask)
        assert bright.min() > 0

    img = np.zeros((4, 4, 3), dtype=np.float32)
    hottest = show_cam_on_image(img, np.ones((4, 4), dtype=np.float32))
    coldest = show_cam_on_image(img, np.zeros((4, 4), dtype=np.float32))
    for value, expected, hot in [(1.1, hottest, True), (1.5, hottest, True),
                                 (2.0, hottest, True), (7.0, hottest, True),
                                 (-3.0, coldest, False)]:
        with subtests.test(case=f"clip-{value}"):
            with pytest.warns(RuntimeWarning, match="clipping"):
                out = show_cam_on_image(img, np.full((4, 4), value, dtype=np.float32))
            assert np.array_equal(out, expected)
            assert int(out[0, 0].argmax()) == (2 if hot else 0)

    with subtests.test(case="normalised-mask-no-warning"):
        mask = np.tile(np.linspace(0.0, 1.0, 8, dtype=np.float32), (8, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = show_cam_on_image(np.full((8, 8, 3), 0.25, dtype=np.float32), mask)
        assert out.shape == (8, 8, 3) and out.dtype == np.uint8

    with subtests.test(case="negative-image-clipped-not-wrapped"):
        neg = np.zeros((4, 4, 3), dtype=np.float32)
        neg[0, 0] = -0.5
        out = show_cam_on_image(neg, np.linspace(0.0, 1.0, 16,
                                                 dtype=np.float32).reshape(4, 4))
        assert out.dtype == np.uint8 and int(out.min()) == 0 and int(out.max()) == 255

    bad_img = np.zeros((4, 4, 3), dtype=np.float32)
    bad_img[0, 0, 0] = np.nan
    refusals = [("nan-mask", np.nan, None, "NaN or infinity"),
                ("inf-mask", np.inf, None, "NaN or infinity"),
                ("neg-inf-mask", -np.inf, None, "NaN or infinity"),
                ("nan-image", None, bad_img, "NaN or infinity"),
                ("cancelling-image", None, np.full((4, 4, 3), -10.0, dtype=np.float32),
                 "nothing to normalize against")]
    for cid, bad, image, match in refusals:
        with subtests.test(case=cid):
            mask = np.zeros((4, 4), dtype=np.float32)
            if bad is not None:
                mask[1, 1] = bad
            with pytest.raises(ValueError, match=match):
                show_cam_on_image(img if image is None else image, mask)


# ===========================================================================
# suggest_training_changes
# ===========================================================================

def _frames(n=30, train_loss=None, val_loss=None, train_acc=None, val_acc=None,
            train_f1=None, val_f1=None, val_f1_col="macro_f1"):
    """A pair of well-formed progress frames with ``n`` epochs."""
    epoch = np.arange(1, n + 1)
    tr = pd.DataFrame({"epoch": epoch,
                       "loss": np.linspace(0.70, 0.10, n) if train_loss is None
                       else train_loss})
    va = pd.DataFrame({"epoch": epoch,
                       "loss": np.linspace(0.70, 0.20, n) if val_loss is None
                       else val_loss})
    if train_acc is not None:
        tr["accuracy"] = train_acc
    if val_acc is not None:
        va["accuracy"] = val_acc
    if train_f1 is not None:
        tr["f1_macro"] = train_f1
    if val_f1 is not None:
        va[val_f1_col] = val_f1
    return tr, va


def _advise(dst, tr, va, train_name="train_progress.csv",
            val_name="validation_progress.csv", explicit=True, **kw):
    """Write the frames into ``dst`` and run the advisor on them."""
    from spacr.utils import suggest_training_changes
    os.makedirs(dst, exist_ok=True)
    tp = os.path.join(dst, train_name)
    vp = os.path.join(dst, val_name)
    if tr is not None:
        tr.to_csv(tp, index=False)
    if va is not None:
        va.to_csv(vp, index=False)
    if explicit:
        return suggest_training_changes(str(dst), train_csv=tp, val_csv=vp, **kw)
    return suggest_training_changes(str(dst), **kw)


def _non_unique_reader(real_read_csv, relabel):
    def _read(path, *args, **kwargs):
        df = real_read_csv(path, *args, **kwargs)
        if os.path.basename(str(path)).startswith("validation"):
            df.index = relabel(len(df))
        return df
    return _read


def test_training_advisor_reads_the_progress_files_into_a_summary(subtests, tmp_path):
    """The advisor bails out with exactly one flag when a CSV or the epoch /
    loss column is missing; otherwise it normalises aliased, case-folded and
    duplicate columns (first wins), fits loss slopes with 0.0 for flat or
    <2-finite windows, reports final metrics and macro-F1 from either column
    spelling, and survives a non-unique index."""
    n = 20
    ep = np.arange(1, n + 1)
    bail = [
        # (id, train frame, val frame, explicit paths, flags, exact suggestions or fragment)
        ("missing-val-csv", _frames()[0], None, False, ["missing_val_csv"], "val CSV"),
        ("explicit-val-path-missing", _frames()[0], None, True, ["missing_val_csv"],
         "val CSV"),
        ("missing-train-csv-first", None, None, False, ["missing_train_csv"], None),
        ("missing-loss", pd.DataFrame({"epoch": ep, "accuracy": np.linspace(0.6, 0.9, n)}),
         pd.DataFrame({"epoch": ep, "accuracy": np.linspace(0.6, 0.9, n)}), True,
         ["missing_required_col:loss"],
         ["Progress CSVs lack 'loss'. Ensure _save_progress writes epoch and loss."]),
        ("missing-epoch-checked-first", pd.DataFrame({"loss": np.linspace(0.7, 0.1, n)}),
         pd.DataFrame({"loss": np.linspace(0.7, 0.2, n)}), True,
         ["missing_required_col:epoch"], "epoch"),
        ("val-only-missing-loss",
         pd.DataFrame({"epoch": ep, "loss": np.linspace(0.7, 0.1, n)}),
         pd.DataFrame({"epoch": ep, "accuracy": np.linspace(0.6, 0.9, n)}), True,
         ["missing_required_col:loss"], "lack 'loss'"),
        ("train-csv-with-in-row-val-columns",
         pd.DataFrame([{"epoch": i, "accuracy": 0.5, "loss": 1.0, "val_accuracy": 0.5,
                        "val_loss": 1.0} for i in range(1, 31)]), None, False,
         ["missing_val_csv"],
         ["Could not locate val CSV; enable validation logging in _save_progress."]),
    ]
    for cid, tr, va, explicit, flags, sugg in bail:
        with subtests.test(case=cid):
            d = tmp_path / cid
            name = "train.csv" if cid == "train-csv-with-in-row-val-columns" else \
                "train_progress.csv"
            out = _advise(d, tr, va, train_name=name, explicit=explicit)
            assert out["flags"] == flags
            assert out["summary"] == {}
            assert not (d / "validation_progress.csv").exists() or va is not None
            if isinstance(sugg, list):
                assert out["suggestions"] == sugg
            elif sugg is not None:
                assert len(out["suggestions"]) == 1 and sugg in out["suggestions"][0]

    nan_window = np.full(30, np.nan)
    nan_window[0] = 0.60
    one_point = np.full(12, np.nan)
    one_point[-1] = 0.25
    f1_tr = np.linspace(0.50, 0.88, 30)
    f1_va = np.linspace(0.45, 0.81, 30)
    f1_half_nan = f1_tr.copy()
    f1_half_nan[::2] = np.nan
    f1_tenth_nan = f1_tr.copy()
    f1_tenth_nan[:3] = np.nan
    summaries = [
        # (id, frames, advisor kwargs, expected summary paths, flags in, flags out)
        ("all-nan-train-window", _frames(n=30, train_loss=nan_window), {"last_k": 25},
         {("slope_train_loss_last_k",): 0.0, ("epochs",): 30,
          ("final_metrics", "train_loss"): "nan"}, [], []),
        ("single-finite-point", _frames(n=12, train_loss=one_point), {"last_k": 25},
         {("slope_train_loss_last_k",): 0.0, ("final_metrics", "train_loss"): 0.25},
         [], []),
        ("constant-val-window", _frames(n=30, val_loss=np.full(30, 0.42)), {},
         {("slope_val_loss_last_k",): 0.0, ("val_loss_std_last_k",): 0.0},
         ["val_plateau"], []),
        ("f1-both-spellings", _frames(n=30, train_acc=np.linspace(0.60, 0.90, 30),
                                      val_acc=np.linspace(0.58, 0.86, 30),
                                      train_f1=f1_tr, val_f1=f1_va), {},
         {("final_metrics", "train_f1_macro"): f1_tr[-1],
          ("final_metrics", "val_f1_macro"): f1_va[-1]}, [], ["f1_nan_detected"]),
        ("f1-train-half-nan", _frames(n=30, train_f1=f1_half_nan), {},
         {("final_metrics", "train_f1_macro"): f1_half_nan[-1]}, ["f1_nan_detected"], []),
        ("f1-val-all-nan", _frames(n=30, val_f1=np.full(30, np.nan)), {},
         {("final_metrics", "val_f1_macro"): "nan"}, ["f1_nan_detected"], []),
        ("f1-train-tenth-nan", _frames(n=30, train_f1=f1_tenth_nan), {},
         {("final_metrics", "train_f1_macro"): f1_tenth_nan[-1]}, [], ["f1_nan_detected"]),
        ("aliased-and-case-folded",
         (pd.DataFrame({"Epoch": ep, "Train_Loss": np.linspace(0.70, 0.10, n),
                        "ACC": np.linspace(0.60, 0.95, n)}),
          pd.DataFrame({" epoch ": ep, "Val_Loss": np.linspace(0.70, 0.25, n),
                        "acc": np.linspace(0.58, 0.90, n)})), {},
         {("final_metrics", "train_loss"): 0.10, ("final_metrics", "val_loss"): 0.25,
          ("final_metrics", "train_accuracy"): 0.95, ("best_epoch",): n,
          ("gen_gap_acc",): 0.05}, [], []),
        ("duplicates-keep-first",
         (pd.DataFrame({"epoch": ep, "loss": np.linspace(0.70, 0.10, n),
                        "train_loss": np.full(n, 999.0)}),
          pd.DataFrame({"epoch": ep, "loss": np.linspace(0.70, 0.30, n),
                        "Loss": np.full(n, -999.0)})), {},
         {("final_metrics", "train_loss"): 0.10, ("final_metrics", "val_loss"): 0.30,
          ("best_val_loss",): 0.30, ("best_epoch",): n}, [], []),
    ]
    for cid, (tr, va), kw, expected, flags_in, flags_out in summaries:
        with subtests.test(case=cid):
            out = _advise(tmp_path / cid, tr, va, **kw)
            summary = out["summary"]
            for path, value in expected.items():
                got = summary
                for key in path:
                    got = got[key]
                if value == "nan":
                    assert np.isnan(got), path
                else:
                    assert got == pytest.approx(value), path
            for flag in flags_in:
                assert flag in out["flags"]
            for flag in flags_out:
                assert flag not in out["flags"]
            if cid == "all-nan-train-window":
                # the val window is untouched: a real negative fit, not a fallback
                assert summary["slope_val_loss_last_k"] < -1e-3
            if cid == "f1-both-spellings":
                assert isinstance(summary["final_metrics"]["train_f1_macro"], float)
            if cid == "f1-train-half-nan":
                assert any("F1(macro) shows NaN" in s for s in out["suggestions"])
                assert any("stratified sampling" in s for s in out["suggestions"])
            if cid == "f1-val-all-nan":
                assert "train_f1_macro" not in summary["final_metrics"]

    non_unique = [
        ("duplicated-last-row", 20, lambda tr, va: (tr, pd.concat([va, va.iloc[[-1]]],
                                                                  ignore_index=True)),
         lambda k: list(range(k - 1)) + [k - 2], 0.20, 20),
        ("one-shared-label", 12,
         lambda tr, va: (tr.assign(loss=np.linspace(1.0, 0.4, 12),
                                   accuracy=np.linspace(0.5, 0.9, 12)),
                         va.assign(loss=np.linspace(1.0, 0.5, 12),
                                   accuracy=np.linspace(0.5, 0.8, 12))),
         lambda k: [0] * k, 0.5, 12),
    ]
    for cid, n_ep, shape, relabel, best_loss, best_epoch in non_unique:
        with subtests.test(case=cid):
            tr, va = shape(*_frames(n=n_ep))
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(pd, "read_csv", _non_unique_reader(pd.read_csv, relabel))
                out = _advise(tmp_path / cid, tr, va)
            assert isinstance(out["summary"]["best_val_loss"], float)
            assert out["summary"]["best_val_loss"] == pytest.approx(best_loss)
            assert isinstance(out["summary"]["best_epoch"], int)
            assert out["summary"]["best_epoch"] == best_epoch


def test_training_advisor_flags_each_failure_mode_and_never_repeats_advice(
        subtests, tmp_path):
    """Imbalance (accuracy beating macro-F1 by >0.10), plateau, overfitting by
    accuracy gap or diverging slopes, underfitting, short runs and noisy
    validation each raise their flag and advice; advice is de-duplicated in
    order via dict.fromkeys and no two rules share a sentence."""
    n = 30

    def _imbalance(val_acc, val_f1, val_loss=None):
        tr = pd.DataFrame({"epoch": np.arange(1, n + 1),
                           "loss": np.linspace(0.70, 0.05, n),
                           "accuracy": np.linspace(0.60, 0.98, n),
                           "f1_macro": np.linspace(0.60, 0.97, n)})
        va = pd.DataFrame({"epoch": np.arange(1, n + 1),
                           "loss": np.linspace(0.70, 0.20, n) if val_loss is None
                           else val_loss,
                           "accuracy": np.full(n, val_acc),
                           "f1_macro": np.full(n, val_f1)})
        return tr, va

    epochs = list(range(1, 31))
    many_rules = (
        pd.DataFrame({"epoch": epochs, "loss": [1.0 / (e ** 2) for e in epochs],
                      "accuracy": [min(0.99, 0.5 + e * 0.02) for e in epochs],
                      "f1_macro": [min(0.99, 0.5 + e * 0.02) for e in epochs]}),
        pd.DataFrame({"epoch": epochs, "loss": [1.0 + 0.01 * e for e in epochs],
                      "accuracy": [0.60] * 30, "f1_macro": [0.20] * 30}))

    cases = [
        # (id, frames, advisor kwargs, write kwargs, flags in, flags out,
        #  suggestion fragments, first suggestion prefix, summary checks)
        ("imbalance", _imbalance(0.95, 0.60), {}, {}, ["class_imbalance_suspected"], [],
         ["macro-F1", "per-class metrics"], None,
         {("final_metrics", "val_accuracy"): 0.95, ("final_metrics", "val_f1_macro"): 0.60}),
        ("small-gap-no-imbalance", _imbalance(0.90, 0.87), {}, {}, [],
         ["class_imbalance_suspected"], [], None, {}),
        ("imbalance-plus-regression",
         _imbalance(0.95, 0.40, np.concatenate([np.full(5, 0.30), np.full(n - 5, 0.50)])),
         {}, {}, ["class_imbalance_suspected", "past_best_regression"], [], [], None, {}),
        ("plateau-and-overfit-ordered",
         _frames(n=n, val_loss=np.full(n, 0.50) + np.linspace(0, 1e-5, n),
                 train_acc=np.linspace(0.80, 0.99, n), val_acc=np.linspace(0.55, 0.60, n)),
         {}, {}, ["val_plateau", "overfitting"], [], [],
         "Validation loss plateau detected", {}),
        ("short-run", _frames(n=6), {"min_epochs": 10}, {}, ["few_epochs"], [], [],
         "Only 6 epochs logged (<10)",
         {("epochs",): 6, ("best_epoch",): 6, ("best_val_loss",): 0.20}),
        ("diverging-slopes-overfit",
         _frames(n=n, train_loss=np.linspace(0.70, 0.05, n),
                 val_loss=np.linspace(0.40, 0.90, n)),
         {}, {}, ["overfitting"], [], ["early stopping"], None, {("gen_gap_acc",): None}),
        ("accuracy-gap-overfit",
         _frames(n=n, train_loss=np.linspace(0.70, 0.01, n),
                 val_loss=np.linspace(0.70, 0.65, n), train_acc=np.linspace(0.5, 0.99, n),
                 val_acc=np.linspace(0.5, 0.55, n)),
         {}, {"explicit": False}, ["overfitting"], [], [], None, {}),
        ("underfitting",
         _frames(n=n, train_loss=np.full(n, 0.90), val_loss=np.full(n, 0.92),
                 train_acc=np.full(n, 0.55), val_acc=np.full(n, 0.54)),
         {}, {}, ["underfitting"], ["overfitting"], ["Underfitting signs"], None,
         {("final_metrics", "train_accuracy"): 0.55, ("slope_train_loss_last_k",): 0.0}),
        ("healthy-run-noisy-val", _frames(n=n), {}, {"explicit": False},
         ["unstable_training"], ["overfitting", "underfitting"],
         ["Validation loss is noisy", "consistent image normalization"], None, {}),
        ("many-rules-auto-discovered", many_rules, {},
         {"explicit": False, "val_name": "val_progress.csv"},
         ["overfitting", "unstable_training", "past_best_regression",
          "class_imbalance_suspected"], [], [], None, {}),
    ]
    for cid, (tr, va), kw, wkw, flags_in, flags_out, frags, first, checks in cases:
        with subtests.test(case=cid):
            out = _advise(tmp_path / cid, tr, va, **wkw, **kw)
            sugg = out["suggestions"]
            assert len(sugg) == len(set(sugg)), "the advisor repeated itself"
            for flag in flags_in:
                assert flag in out["flags"], flag
            for flag in flags_out:
                assert flag not in out["flags"], flag
            joined = " ".join(sugg)
            for frag in frags:
                assert frag in joined, frag
            if flags_in:
                assert sugg
            if first is not None:
                assert sugg[0].startswith(first)
            for path, value in checks.items():
                got = out["summary"]
                for key in path:
                    got = got[key]
                if value is None:
                    assert got is None
                else:
                    assert got == pytest.approx(value), path
            if cid == "diverging-slopes-overfit":
                assert out["summary"]["slope_train_loss_last_k"] < 0
                assert out["summary"]["slope_val_loss_last_k"] > 0
            if cid == "many-rules-auto-discovered":
                assert out["flags"] == flags_in and len(sugg) == 10
            if cid == "healthy-run-noisy-val":
                assert out["flags"] == ["unstable_training"] and len(sugg) == 2

    with subtests.test(case="dedup-is-order-preserving-dict-fromkeys"):
        source = inspect.getsource(U.suggest_training_changes)
        assert "dict.fromkeys" in source
        assert "if s not in seen:" not in source
