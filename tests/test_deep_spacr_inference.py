"""CPU coverage for spacr.deep_spacr inference / activation-map helpers.

Uses a tiny real TorchModel saved to disk plus synthetic PNGs and tar
archives, so everything runs on CPU without a GPU or a trained checkpoint.
"""
from __future__ import annotations

import os
import sqlite3
import tarfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


def _png(path, rng, size=32):
    Image.fromarray(rng.integers(0, 255, (size, size, 3)).astype(np.uint8)).save(path)
    return str(path)


def _save_model(path, num_classes=2):
    import torch
    from spacr.utils import TorchModel
    m = TorchModel(model_name="resnet18", pretrained=False,
                   num_classes=num_classes, image_size=32)
    torch.save(m, path)
    return str(path)


def _tar_of_pngs(tmp_path, rng, n=6):
    d = tmp_path / "imgs"; d.mkdir(exist_ok=True)
    names = []
    for i in range(n):
        p = d / f"plate1_A01_f1_o{i}.png"
        _png(p, rng)
        names.append(p.name)
    tar_path = tmp_path / "ds.tar"
    with tarfile.open(tar_path, "w") as t:
        for nm in names:
            t.add(d / nm, arcname=nm)
    return str(tar_path), names


# ---------------------------------------------------------------------------
# apply_model
# ---------------------------------------------------------------------------

def test_apply_model_on_folder(tmp_path, rng):
    from spacr.deep_spacr import apply_model
    src = tmp_path / "crops"; src.mkdir()
    for i in range(6):
        _png(src / f"o{i}.png", rng)
    model_path = _save_model(tmp_path / "m.pth")
    out = apply_model(str(src), model_path, image_size=32, batch_size=2,
                      normalize=True, n_jobs=0)
    assert out is not None
    if isinstance(out, pd.DataFrame):
        assert len(out) == 6


# ---------------------------------------------------------------------------
# apply_model_to_tar
# ---------------------------------------------------------------------------

def test_apply_model_to_tar(tmp_path, rng):
    from spacr.deep_spacr import apply_model_to_tar
    tar_path, _ = _tar_of_pngs(tmp_path, rng)
    model_path = _save_model(tmp_path / "m.pth")
    settings = {"tar_path": tar_path, "model_path": model_path,
                "image_size": 32, "batch_size": 2, "normalize": True,
                "n_jobs": 0, "verbose": True, "score_threshold": 0.5}
    out = apply_model_to_tar(settings)
    assert out is not None
    csvs = []
    for root, _d, files in os.walk(tmp_path):
        csvs += [f for f in files if f.endswith(".csv")]
    assert csvs, "expected a results CSV"


# ---------------------------------------------------------------------------
# save_top_class_examples
# ---------------------------------------------------------------------------

def test_save_top_class_examples(tmp_path, rng):
    from spacr.deep_spacr import save_top_class_examples
    tar_path, names = _tar_of_pngs(tmp_path, rng, n=8)
    df = pd.DataFrame({"path": names,
                       "pred": np.linspace(0.05, 0.95, len(names))})
    dst = tmp_path / "top"
    save_top_class_examples(df, tar_path, str(dst), n=2)
    produced = list(dst.rglob("*.png"))
    assert produced, "no top-class examples extracted"


def test_save_top_class_examples_explicit_classes(tmp_path, rng):
    from spacr.deep_spacr import save_top_class_examples
    tar_path, names = _tar_of_pngs(tmp_path, rng, n=6)
    df = pd.DataFrame({"path": names,
                       "pred": np.linspace(0.1, 0.9, len(names))})
    dst = tmp_path / "top2"
    save_top_class_examples(df, tar_path, str(dst), n=1,
                            classes=["neg", "pos"])
    assert list(dst.rglob("*.png"))


# ---------------------------------------------------------------------------
# activation maps / gradients
# ---------------------------------------------------------------------------

def _activation_settings(tar_path, model_path, **over):
    s = {"dataset": tar_path, "model_path": model_path,
         "cam_type": "saliency_image", "target_layer": None,
         "image_size": 32, "batch_size": 2, "channels": [0, 1, 2],
         "normalize": True, "save": False, "plot": False,
         "correlation": False, "shuffle": False, "n_jobs": 0,
         "overlay": False, "class_names": None, "manders_thresholds": [15, 85, 95]}
    s.update(over)
    return s


def test_generate_activation_map_saliency(tmp_path, rng):
    from spacr.deep_spacr import generate_activation_map
    tar_path, _ = _tar_of_pngs(tmp_path, rng)
    model_path = _save_model(tmp_path / "m.pth")
    # No try/skip: this crashed on the DEFAULT two-class model for as long as
    # the multi-logit prediction bug existed, and the swallow reported it as a
    # tidy "skipped" instead of a failure.
    generate_activation_map(_activation_settings(tar_path, model_path))


def test_generate_activation_map_gradcam(tmp_path, rng):
    # No try/skip: like the saliency sibling above, this ran head-first into the
    # multi-logit prediction bug on the default two-class model and the swallow
    # reported it as "skipped". Running clean IS the assertion here.
    from spacr.deep_spacr import generate_activation_map
    tar_path, _ = _tar_of_pngs(tmp_path, rng)
    model_path = _save_model(tmp_path / "m.pth")
    from spacr.utils import recommend_target_layers, TorchModel
    import torch as _t
    model = _t.load(model_path, weights_only=False)
    recommended, _all = recommend_target_layers(model)
    generate_activation_map(_activation_settings(
        tar_path, model_path, cam_type="gradcam",
        target_layer=recommended[0]))


def test_recommend_target_layers():
    from spacr.utils import recommend_target_layers, TorchModel
    m = TorchModel(model_name="resnet18", pretrained=False, num_classes=2)
    recommended, all_layers = recommend_target_layers(m)
    assert recommended and all_layers
    assert recommended[0] == all_layers[-1]      # last conv layer


def test_recommend_target_layers_without_conv_raises():
    import torch, pytest as _pt
    from spacr.utils import recommend_target_layers
    with _pt.raises(ValueError):
        recommend_target_layers(torch.nn.Sequential(torch.nn.Linear(4, 2)))


def test_get_submodules():
    from spacr.utils import get_submodules, TorchModel
    m = TorchModel(model_name="resnet18", pretrained=False, num_classes=2)
    subs = get_submodules(m)
    assert isinstance(subs, (list, dict)) and len(subs) > 0


def test_gradcam_class_runs():
    """GradCAM must return a 2-D heatmap matching the input's spatial size.

    Regression guard for two fixes: the forward hook now calls retain_grad()
    (PyTorch only populates .grad on leaf tensors, so features[0].grad was
    None), and the CAM is atleast_2d'd before cv2.resize (it collapses to
    0-d when the target layer's spatial dims are 1x1).
    """
    import torch
    from spacr.utils import GradCAM, TorchModel, recommend_target_layers
    m = TorchModel(model_name="resnet18", pretrained=False, num_classes=2).eval()
    recommended, _all = recommend_target_layers(m)
    cam = GradCAM(m, recommended, use_cuda=False)
    out = cam(torch.rand(1, 3, 224, 224), index=0)
    assert out is not None and out.ndim == 2
    assert out.shape == (224, 224)


def test_gradcam_small_input_1x1_feature_map():
    """A 32x32 input collapses resnet18's last conv to 1x1 — must still work."""
    import torch
    from spacr.utils import GradCAM, TorchModel, recommend_target_layers
    m = TorchModel(model_name="resnet18", pretrained=False, num_classes=2).eval()
    recommended, _all = recommend_target_layers(m)
    cam = GradCAM(m, recommended, use_cuda=False)
    out = cam(torch.rand(1, 3, 32, 32), index=0)
    assert out.ndim == 2 and out.shape == (32, 32)
    assert np.isfinite(out).all()


def test_integrated_gradients_class():
    import torch
    from spacr.utils import IntegratedGradients, TorchModel
    m = TorchModel(model_name="resnet18", pretrained=False, num_classes=2).eval()
    ig = IntegratedGradients(m)
    out = ig.generate_integrated_gradients(
        torch.rand(1, 3, 32, 32), target_label_idx=0, baseline=None,
        num_steps=3)
    # attributions come back shaped like the input, and are finite
    out = np.asarray(out)
    assert out.shape[-2:] == (32, 32)
    assert np.isfinite(out).all()


def test_show_cam_on_image():
    from spacr.utils import show_cam_on_image
    img = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    mask = np.random.default_rng(1).random((32, 32)).astype(np.float32)
    out = show_cam_on_image(img, mask)
    assert out.shape[:2] == (32, 32)


# ---------------------------------------------------------------------------
# deep_spacr orchestrator
# ---------------------------------------------------------------------------

def test_deep_spacr_train_only(tmp_path, rng):
    """The orchestrator wires generate_training_dataset -> train_test_model."""
    from spacr.deep_spacr import deep_spacr
    src = tmp_path / "ds"
    for split in ("train", "test"):
        for cls in ("nc", "pc"):
            d = src / split / cls
            d.mkdir(parents=True)
            for i in range(4):
                _png(d / f"{cls}_{i}.png", rng)
    settings = {
        "src": str(src), "classes": ["nc", "pc"], "model_type": "resnet18",
        "epochs": 1, "batch_size": 2, "image_size": 32,
        "train_channels": ["r", "g", "b"], "train": True, "test": False,
        "val_split": 0.25, "n_jobs": 0, "augment": False,
        "init_weights": False, "use_checkpoint": False, "pin_memory": False,
        "normalize": True, "intermedeate_save": False,
        "gradient_accumulation": False, "gradient_accumulation_steps": 1,
        "loss_type": "cross_entropy", "optimizer_type": "adamw",
        "schedule": "cosine", "plot": False, "verbose": False,
        "dropout_rate": 0.0, "early_stopping_patience": 0,
        "generate_training_dataset": False, "apply_model_to_dataset": False,
    }
    deep_spacr(settings)
    # `assert ... or True` under a swallowed skip could not fail either way.
    # Check the orchestrator actually produced a checkpoint and its logs.
    assert list(src.rglob("*.pth")), "no checkpoint written"
    assert (src / "settings" / "DL_model.csv").is_file()
    assert list(src.rglob("train.csv")) and list(src.rglob("validation.csv"))
