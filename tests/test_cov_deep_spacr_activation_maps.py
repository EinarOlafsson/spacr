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

import os
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

def test_generate_activation_map_missing_dataset_returns_early(tmp_path, capsys):
    """A non-existent dataset must return before the model is ever loaded."""
    from spacr.deep_spacr import generate_activation_map

    root = tmp_path / "proj"
    (root / "datasets").mkdir(parents=True)
    missing = str(root / "datasets" / "nope.tar")

    settings = _settings(missing, str(root / "no_such_model.pth"), save=True)
    assert generate_activation_map(settings) is None

    out = capsys.readouterr().out
    assert f"Dataset not found at {missing}" in out
    # returned before os.makedirs(save_dir) -> nothing created for the dataset
    assert not (root / "datasets" / "nope").exists()
    # ...but save_settings ran first, so the settings snapshot exists
    assert (root / "settings" / "saliency_image_settings.csv").is_file()


# ---------------------------------------------------------------------------
# generate_activation_map — saliency_image: save maps + DB rows
# ---------------------------------------------------------------------------

def test_generate_activation_map_saliency_image_saves_pngs_and_db(tmp_path, model_path, capsys):
    """save=True writes one uint8 'L' map per image and one DB row per map."""
    from spacr.deep_spacr import generate_activation_map

    root, tar_path, names = _project(tmp_path, n_images=6)
    generate_activation_map(_settings(tar_path, model_path, save=True))

    save_dir = root / "datasets" / "ds" / "saliency_image"
    assert save_dir.is_dir()
    pngs = sorted(save_dir.rglob("*.png"))
    assert len(pngs) == 6
    assert {p.name for p in pngs} == set(names)

    # class_<pred>/<plate>/<well>/<file>
    for p in pngs:
        assert p.parent.name == "A01"
        assert p.parent.parent.name == "plate1"
        assert p.parent.parent.parent.name in ("class_0", "class_1")

    img = Image.open(pngs[0])
    assert img.mode == "L"
    assert img.size == (32, 32)
    arr = np.array(img)
    assert arr.dtype == np.uint8
    # min-max normalisation before the uint8 cast must saturate both ends
    assert arr.min() == 0 and arr.max() == 255

    db = root / "measurements" / "ds.db"
    assert db.is_file()
    assert "saliency_image_list" in _tables(str(db))
    con = sqlite3.connect(str(db))
    try:
        rows = con.execute(
            "SELECT png_path, plateID, rowID, columnID, fieldID, object "
            "FROM saliency_image_list ORDER BY png_path").fetchall()
    finally:
        con.close()
    assert len(rows) == 6
    assert {r[1] for r in rows} == {"plate1"}
    assert {r[2] for r in rows} == {"r1"}
    assert {r[3] for r in rows} == {"c1"}
    assert {r[4] for r in rows} == {"f1"}
    assert {r[5] for r in rows} == {f"o{i}" for i in range(6)}
    assert all(os.path.isfile(r[0]) for r in rows)

    out = capsys.readouterr().out
    assert "Activation maps will be saved in" in out
    assert "Activation map generation complete." in out


# ---------------------------------------------------------------------------
# generate_activation_map — plot=True writes one batch-grid PDF per batch
# ---------------------------------------------------------------------------

def test_generate_activation_map_plot_writes_batch_grids(tmp_path, model_path, capsys):
    """plot=True renders a grid PDF per batch; save=False writes no PNG."""
    from spacr.deep_spacr import generate_activation_map

    root, tar_path, _names = _project(tmp_path, n_images=6)
    generate_activation_map(_settings(tar_path, model_path, plot=True, save=False,
                                      batch_size=3, normalize=True))

    grid_dir = root / "datasets" / "ds" / "saliency_image" / "batch_grids"
    pdfs = sorted(grid_dir.glob("*.pdf"))
    assert [p.name for p in pdfs] == ["batch_0_grid.pdf", "batch_1_grid.pdf"]
    for p in pdfs:
        assert p.stat().st_size > 0
        assert p.read_bytes()[:4] == b"%PDF"

    # save=False: the per-class folders are still created, but stay empty
    save_dir = root / "datasets" / "ds" / "saliency_image"
    assert (save_dir / "class_0").is_dir() or (save_dir / "class_1").is_dir()
    assert list(save_dir.rglob("*.png")) == []
    # ...and nothing was pushed to the database
    assert not (root / "measurements" / "ds.db").is_file()

    assert "Batch grid maps will be saved in" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# generate_activation_map — saliency_channel writes RGB maps
# ---------------------------------------------------------------------------

def test_generate_activation_map_saliency_channel_writes_rgb(tmp_path, model_path):
    """cam_type='saliency_channel' keeps the channels and saves RGB PNGs."""
    from spacr.deep_spacr import generate_activation_map

    root, tar_path, names = _project(tmp_path, n_images=4)
    generate_activation_map(_settings(tar_path, model_path,
                                      cam_type="saliency_channel",
                                      save=True, batch_size=2))

    save_dir = root / "datasets" / "ds" / "saliency_channel"
    pngs = sorted(save_dir.rglob("*.png"))
    assert len(pngs) == len(names)
    img = Image.open(pngs[0])
    assert img.mode == "RGB"
    assert img.size == (32, 32)
    arr = np.array(img)
    assert arr.shape == (32, 32, 3) and arr.dtype == np.uint8
    # every channel is independently min-max scaled to the full uint8 range
    for c in range(3):
        assert arr[:, :, c].min() == 0 and arr[:, :, c].max() == 255

    assert "saliency_channel_list" in _tables(str(root / "measurements" / "ds.db"))


# ---------------------------------------------------------------------------
# generate_activation_map — correlation=True writes the correlations table
# ---------------------------------------------------------------------------

def test_generate_activation_map_correlations_to_database(tmp_path, model_path):
    """correlation=True computes per-image stats and stores them next to the maps."""
    from spacr.deep_spacr import generate_activation_map

    root, tar_path, names = _project(tmp_path, n_images=4)
    generate_activation_map(_settings(tar_path, model_path, save=True, plot=True,
                                      correlation=True, batch_size=2,
                                      manders_thresholds=[50]))

    db = str(root / "measurements" / "ds.db")
    tables = _tables(db)
    assert {"saliency_image_list", "saliency_image_correlations"} <= tables

    con = sqlite3.connect(db)
    try:
        cur = con.execute("SELECT * FROM saliency_image_correlations")
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    finally:
        con.close()

    assert len(rows) == len(names)
    # 3 input channels x 1 summed activation channel, pearson + M1/M2 per threshold
    for in_c in range(3):
        assert f"channel_{in_c}_activation_0_pearsons" in cols
        assert f"channel_{in_c}_activation_0_50_M1" in cols
        assert f"channel_{in_c}_activation_0_50_M2" in cols
    assert "file_name" in cols and "png_path" in cols

    fn_idx, p_idx = cols.index("file_name"), cols.index("channel_0_activation_0_pearsons")
    assert {r[fn_idx] for r in rows} == set(names)
    for r in rows:
        assert -1.0 <= float(r[p_idx]) <= 1.0


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

def test_generate_activation_map_maxvit_default_target_layer(tmp_path, model_path, monkeypatch):
    """model_type='maxvit' + target_layer=None fills in the MaxViT block path."""
    import spacr.utils as utils
    from spacr.deep_spacr import generate_activation_map

    seen = {}

    class _RecordingGradCAM:
        """Stand-in for GradCAMGenerator that records the resolved target layer."""

        def __init__(self, model, target_layer, cam_type="gradcam"):
            seen["target_layer"] = target_layer
            seen["cam_type"] = cam_type
            self.model = model

        def compute_gradcam_and_predictions(self, X):
            return X.detach().mean(dim=1), torch.zeros(X.shape[0], dtype=torch.long)

    monkeypatch.setattr(utils, "GradCAMGenerator", _RecordingGradCAM)

    root, tar_path, names = _project(tmp_path, n_images=2)
    settings = _settings(tar_path, model_path, model_type="maxvit",
                         cam_type="gradcam", target_layer=None, save=True,
                         batch_size=2)
    generate_activation_map(settings)

    assert seen["cam_type"] == "gradcam"
    assert seen["target_layer"] == (
        "base_model.blocks.3.layers.1.layers.MBconv.layers.conv_b")
    assert settings["target_layer"] == seen["target_layer"]
    pngs = sorted((root / "datasets" / "ds" / "gradcam").rglob("*.png"))
    assert {p.name for p in pngs} == set(names)


def test_generate_activation_map_saliency_clears_target_layer(tmp_path, model_path):
    """A saliency cam_type always wipes target_layer, even for maxvit defaults."""
    from spacr.deep_spacr import generate_activation_map

    _root, tar_path, _names = _project(tmp_path, n_images=2)
    settings = _settings(tar_path, model_path, model_type="maxvit",
                         cam_type="saliency_image",
                         target_layer="some.layer", batch_size=2)
    generate_activation_map(settings)
    assert settings["target_layer"] is None


def test_generate_activation_map_normalize_input_false(tmp_path, model_path):
    """normalize_input=False must simply drop the Normalize step.

    The transform pipeline used to be built with an inline
    ``Normalize(...) if normalize_input else None``, which left a literal None
    inside the Compose, so this documented setting raised
    "TypeError: 'NoneType' object is not callable" on the first image.
    """
    from spacr.deep_spacr import generate_activation_map

    root, tar_path, names = _project(tmp_path, n_images=2)
    generate_activation_map(_settings(tar_path, model_path, normalize_input=False,
                                      save=True, batch_size=2))

    pngs = sorted((root / "datasets" / "ds" / "saliency_image").rglob("*.png"))
    assert {p.name for p in pngs} == set(names)


# ---------------------------------------------------------------------------
# generate_activation_map — n_jobs=None falls back to cpu_count() - 4
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cpus,expected_workers", [(6, 2), (3, 1)])
def test_generate_activation_map_n_jobs_defaults_to_cpu_count(tmp_path, model_path,
                                                              monkeypatch, cpus,
                                                              expected_workers):
    """n_jobs=None -> max(1, cpu_count() - 4) workers are requested."""
    import spacr.deep_spacr as ds
    from spacr.deep_spacr import generate_activation_map

    real_loader = ds.DataLoader
    captured = {}

    def fake_loader(dataset, **kwargs):
        captured.update(kwargs)
        # never actually fork worker processes inside the test suite
        kwargs["num_workers"] = 0
        kwargs["pin_memory"] = False
        return real_loader(dataset, **kwargs)

    monkeypatch.setattr(ds, "cpu_count", lambda: cpus)
    monkeypatch.setattr(ds, "DataLoader", fake_loader)

    _root, tar_path, _names = _project(tmp_path, n_images=2)
    generate_activation_map(_settings(tar_path, model_path, n_jobs=None,
                                      batch_size=2))

    assert captured["num_workers"] == expected_workers
    assert captured["batch_size"] == 2
    assert captured["shuffle"] is False


# ---------------------------------------------------------------------------
# generate_activation_map — multi-logit models (what train_test_model produces)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "BUG: generate_activation_map only handles single-logit models. For a "
    "2-class TorchModel (what train_test_model builds for classes=['nc','pc']) "
    "predicted_classes is (B, 2), so deep_spacr.py:1127 "
    "predicted_classes[i].item() raises 'a Tensor with 2 elements cannot be "
    "converted to Scalar'. The per-image class should be the argmax logit."))
def test_generate_activation_map_multiclass_model(tmp_path):
    """A 2-class model must still yield exactly one activation map per image."""
    from spacr.deep_spacr import generate_activation_map
    from spacr.utils import TorchModel

    mp = tmp_path / "m2.pth"
    torch.save(TorchModel(model_name="resnet18", pretrained=False,
                          num_classes=2, image_size=32), str(mp))

    root, tar_path, names = _project(tmp_path, n_images=4)
    generate_activation_map(_settings(tar_path, str(mp), save=True, batch_size=2))

    pngs = sorted((root / "datasets" / "ds" / "saliency_image").rglob("*.png"))
    assert {p.name for p in pngs} == set(names)
    assert {p.parent.parent.parent.name for p in pngs} <= {"class_0", "class_1"}


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

def test_visualize_integrated_gradients_saves_maps(tmp_path, model_path, monkeypatch):
    """PNG inputs produce one saved IG map each; non-PNG files are skipped."""
    from spacr.deep_spacr import visualize_integrated_gradients

    src = tmp_path / "pngs"
    src.mkdir()
    _write_png(src / "a.png", seed=1)
    _write_png(src / "b.png", seed=2)
    (src / "notes.txt").write_text("ignore me")

    out_dir = tmp_path / "igs"
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(plt.gcf()))

    visualize_integrated_gradients(str(src), model_path, target_label_idx=0,
                                   image_size=32, channels=[1, 2, 3],
                                   normalize=True, save_integrated_grads=True,
                                   save_dir=str(out_dir))

    saved = sorted(p.name for p in out_dir.glob("*"))
    assert saved == ["integrated_grads_a.png", "integrated_grads_b.png"]
    assert not (out_dir / "integrated_grads_notes.txt").exists()

    arr = np.array(Image.open(out_dir / "integrated_grads_a.png"))
    assert arr.shape == (32, 32) and arr.dtype == np.uint8

    # one 1x3 figure per PNG
    assert len(shown) == 2
    assert [ax.get_title() for ax in shown[-1].axes] == [
        "Original Image", "Integrated Gradients", "Overlay"]


def test_visualize_integrated_gradients_no_save_default_channels(tmp_path, model_path, monkeypatch):
    """save_integrated_grads=False writes nothing but still builds the overlay."""
    from spacr.deep_spacr import visualize_integrated_gradients

    src = tmp_path / "pngs"
    src.mkdir()
    _write_png(src / "only.png", seed=7)

    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(plt.gcf()))

    before = set(os.listdir(tmp_path))
    visualize_integrated_gradients(str(src), model_path, image_size=32)
    assert set(os.listdir(tmp_path)) == before
    assert not os.path.exists("integrated_grads")
    assert list(src.iterdir()) == [src / "only.png"]

    assert len(shown) == 1
    axes = shown[0].axes
    assert len(axes) == 3
    assert all(not ax.axison for ax in axes)

    original = np.asarray(axes[0].images[0].get_array())
    assert original.shape == (32, 32, 3)

    ig_map = np.asarray(axes[1].images[0].get_array())
    assert ig_map.shape == (32, 32)          # channel-averaged, squeezed
    assert np.isfinite(ig_map).all()

    overlay = np.asarray(axes[2].images[0].get_array())
    assert overlay.shape == (32, 32, 3)      # IG map broadcast back to RGB
    assert overlay.min() >= 0.0 and overlay.max() <= 1.0
