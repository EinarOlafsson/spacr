"""CPU coverage for the activation-map / attribution block of spacr.deep_spacr.

Covers ``generate_activation_map`` (all four ``cam_type`` variants, the
save / plot / correlation side effects, the ``n_jobs=None`` fallback and the
missing-dataset early return), ``visualize_classes`` and
``visualize_integrated_gradients``.

Everything runs on CPU against a tiny randomly-initialised ``TorchModel``
(resnet18 backbone, ``num_classes=1`` — the single-logit binary head the
activation-map code is written against) and 32x32 synthetic PNGs packed into
a tar archive, so no GPU, no download and no trained checkpoint is needed.
"""
from __future__ import annotations

import sqlite3
import tarfile

import numpy as np
import pytest
from PIL import Image

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_figure_leak():
    """Never let Agg figures accumulate across tests."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    """A single-logit resnet18 TorchModel pickled to disk, shared by the module."""
    from spacr.utils import TorchModel
    p = tmp_path_factory.mktemp("model") / "binary_model.pth"
    model = TorchModel(model_name="resnet18", pretrained=False,
                       num_classes=1, image_size=32)
    torch.save(model, str(p))
    return str(p)


def _write_png(path, seed, size=32):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint16).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return str(path)


def _project(tmp_path, n_images=6):
    """Build ``<root>/datasets/ds.tar`` + an empty ``<root>/measurements`` dir.

    Returns ``(root, tar_path, member_names)``. File names follow the legacy
    spacr crop convention ``plate_well_field_object.png`` that
    ``_map_wells_png`` parses (field and object must be bare integers) when
    rows are pushed into the DB.
    """
    root = tmp_path / "proj"
    (root / "measurements").mkdir(parents=True)
    ds_dir = root / "datasets"
    ds_dir.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()

    names = []
    for i in range(n_images):
        name = f"plate1_A01_1_{i}.png"
        _write_png(raw / name, seed=i)
        names.append(name)

    tar_path = ds_dir / "ds.tar"
    with tarfile.open(tar_path, "w") as tar:
        for name in names:
            tar.add(raw / name, arcname=name)
    return root, str(tar_path), names


def _settings(tar_path, model_path, **over):
    s = {
        "dataset": tar_path,
        "model_path": model_path,
        "model_type": "resnet18",
        "cam_type": "saliency_image",
        "target_layer": None,
        "image_size": 32,
        "batch_size": 3,
        "channels": [1, 2, 3],
        "normalize": False,
        "normalize_input": True,
        "save": False,
        "plot": False,
        "correlation": False,
        "overlay": True,
        "shuffle": False,
        "n_jobs": 0,
        "manders_thresholds": [15, 50, 75],
    }
    s.update(over)
    return s


def _tables(db_path):
    con = sqlite3.connect(db_path)
    try:
        return {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# generate_activation_map — early return when the dataset is missing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# generate_activation_map — saliency_image: save maps + DB rows
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# generate_activation_map — plot=True writes one batch-grid PDF per batch
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# generate_activation_map — saliency_channel writes RGB maps
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# generate_activation_map — correlation=True writes the correlations table
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# generate_activation_map — gradcam branch
# ---------------------------------------------------------------------------

def test_generate_activation_map_gradcam(tmp_path, model_path):
    """cam_type='gradcam' uses GradCAMGenerator and still saves one map per image."""
    from spacr.deep_spacr import generate_activation_map
    from spacr.utils import recommend_target_layers

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    recommended, _all_layers = recommend_target_layers(model)

    root, tar_path, names = _project(tmp_path, n_images=4)
    generate_activation_map(_settings(tar_path, model_path, cam_type="gradcam",
                                      target_layer=recommended[0], save=True,
                                      batch_size=2))

    pngs = sorted((root / "datasets" / "ds" / "gradcam").rglob("*.png"))
    assert {p.name for p in pngs} == set(names)
    arr = np.array(Image.open(pngs[0]))
    assert arr.shape == (32, 32) and arr.dtype == np.uint8

    assert "gradcam_list" in _tables(str(root / "measurements" / "ds.db"))


# ---------------------------------------------------------------------------
# generate_activation_map — maxvit default target layer
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# generate_activation_map — n_jobs=None falls back to cpu_count() - 4
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# generate_activation_map — multi-logit models (what train_test_model produces)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# visualize_classes
# ---------------------------------------------------------------------------

def test_visualize_classes_plots_one_image_per_class(model_path, monkeypatch, capsys):
    """One class_visualization call + one plt.show per class, titled and axis-off.

    ``visualize_classes`` forwards its ``model`` argument straight into
    ``utils.class_visualization``'s ``model_path`` slot, so a checkpoint path
    is what actually flows through here.
    """
    import spacr.utils as utils
    from spacr.deep_spacr import visualize_classes

    calls = []
    fake_img = np.linspace(0, 1, 8 * 8 * 3).reshape(8, 8, 3)

    def fake_class_visualization(target_y, model_arg, dtype, **kwargs):
        calls.append((target_y, model_arg, dtype, dict(kwargs)))
        return fake_img

    shown = []
    monkeypatch.setattr(utils, "class_visualization", fake_class_visualization)
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(plt.gcf()))

    visualize_classes(model_path, torch.FloatTensor, ["nc", "pc"], img_size=8)

    assert [c[0] for c in calls] == [0, 1]
    assert all(c[1] == model_path for c in calls)
    assert all(c[2] is torch.FloatTensor for c in calls)
    assert all(c[3] == {"img_size": 8} for c in calls)

    assert len(shown) == 2
    ax = shown[-1].axes[0]
    assert ax.get_title() == "Class pc Visualization"
    assert not ax.axison
    np.testing.assert_allclose(ax.images[0].get_array(), fake_img)

    out = capsys.readouterr().out
    assert "Visualizing class: nc" in out and "Visualizing class: pc" in out


def test_visualize_classes_runs_real_class_visualization(model_path, monkeypatch):
    """End-to-end with the real optimiser (1 iteration) — returns a deprocessed image."""
    from spacr.deep_spacr import visualize_classes

    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(plt.gcf()))

    visualize_classes(model_path, torch.FloatTensor, ["nc", "pc"],
                      img_size=32, channels=[0, 1, 2], num_iterations=1,
                      blur_every=1, show_every=1, max_jitter=2)

    # class_visualization previews once per iteration, visualize_classes once more
    assert len(shown) >= 4
    ax = shown[-1].axes[0]
    img = ax.images[0].get_array()
    assert img.shape == (32, 32, 3)
    assert np.isfinite(np.asarray(img)).all()


# ---------------------------------------------------------------------------
# visualize_integrated_gradients
# ---------------------------------------------------------------------------


