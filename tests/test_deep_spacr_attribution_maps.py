"""Activation-map and attribution drivers of :mod:`spacr.deep_spacr`.

One table-driven test per unit, each row run as a subtest so a failing case
names itself.  Units: ``generate_activation_map`` (outputs, then the input
pipeline and hand-offs), ``analyze_activation_maps``, ``SmoothGrad``,
``visualize_smooth_grad`` and ``visualize_integrated_gradients``.

The models are linear (``Flatten -> Linear``) wherever the attribution can
then be written down: the input gradient of a linear head is exactly its
weight row, so saliency and SmoothGrad maps are ``|W|`` and integrated
gradients are ``x * W``.  Weights are multiples of 1/64, which keeps every sum
and average exact in float32, so the saved PNGs are compared bit for bit.
Everything is CPU-only, offline and works on 8-32 px crops.
"""
from __future__ import annotations

import os
import sqlite3
import tarfile

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from spacr import deep_spacr as D  # noqa: E402

#: what ``pick_device`` says when a busy card sends the run to the CPU
_NOTE = ("The GPU has 300 MiB free of 24576 and this run needs about 2000 "
         "MiB, so it is running on the CPU instead.")


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def shown(monkeypatch):
    """Capture every figure spaCR shows as ``[{title, data, cmap}, ...]``."""
    captured = []

    def capture(*_a, **_k):
        captured.append([
            {"title": ax.get_title(),
             "data": np.asarray(ax.images[0].get_array()) if ax.images else None,
             "cmap": ax.images[0].get_cmap().name if ax.images else None,
             "axis_on": ax.axison}
            for ax in plt.gcf().axes])

    monkeypatch.setattr(plt, "show", capture)
    return captured


def _dyadic_weight(n_out, n_in, signed=True):
    """Weights that are multiples of 1/64, so sums and means stay exact."""
    k = torch.arange(n_out * n_in, dtype=torch.float32).reshape(n_out, n_in)
    w = ((k * 37) % 17 - 8) / 64.0 if signed else (k % 64) / 64.0
    return w


def _linear_model(n_in, n_out, weight=None, bias=0.0):
    """``Flatten -> Linear`` with fully deterministic parameters."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(n_in, n_out))
    with torch.no_grad():
        model[1].weight.copy_(_dyadic_weight(n_out, n_in) if weight is None else weight)
        model[1].bias.fill_(bias)
    return model


class _TinyConv(nn.Module):
    """A spatial layer, so the Grad-CAM family applies."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1)
        self.head = nn.Linear(2, 1)

    def forward(self, x):
        return self.head(self.conv(x).mean(dim=(2, 3)))


def _write_png(path, seed, size=32):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (size, size, 3)).astype(np.uint8)).save(path)
    return str(path)


def _expected_map(values):
    """The uint8 map spaCR saves for float32 ``values`` (min-max to 0..255)."""
    a = np.asarray(values, dtype=np.float32)
    lo = a.min()
    rng = a.max() - lo
    a = (a - lo) / rng if rng > 0 else np.zeros_like(a)
    return (a * 255).astype(np.uint8)


# ===========================================================================
# generate_activation_map
# ===========================================================================

_SIZE = 32
_N_IN = 3 * _SIZE * _SIZE


def _project(tmp_path, tag, n_images=4):
    """``<root>/datasets/ds.tar`` of random crops + an empty measurements dir."""
    root = tmp_path / tag
    (root / "measurements").mkdir(parents=True)
    (root / "datasets").mkdir()
    raw = tmp_path / f"{tag}_raw"
    raw.mkdir()
    names = [f"plate1_A01_1_{i}.png" for i in range(n_images)]
    tar_path = root / "datasets" / "ds.tar"
    with tarfile.open(tar_path, "w") as tar:
        for i, name in enumerate(names):
            tar.add(_write_png(raw / name, seed=i), arcname=name)
    return root, str(tar_path), names


def _settings(tar_path, model_path, **over):
    s = {"dataset": tar_path, "model_path": model_path, "model_type": "resnet18",
         "cam_type": "saliency_image", "target_layer": None, "image_size": _SIZE,
         "batch_size": 2, "channels": [1, 2, 3], "normalize": False,
         "normalize_input": True, "save": False, "plot": False,
         "correlation": False, "overlay": True, "shuffle": False, "n_jobs": 0,
         "manders_thresholds": [15, 50, 75]}
    s.update(over)
    return s


def _tables(db):
    with sqlite3.connect(db) as con:
        return {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """Linear single-logit checkpoints that predict class 1 / class 0 for every
    crop (|W.x| <= 384 < |bias|), and a tiny conv model."""
    d = tmp_path_factory.mktemp("models")
    paths = {}
    for name, bias in (("positive", 1000.0), ("negative", -1000.0)):
        paths[name] = str(d / f"{name}.pth")
        torch.save(_linear_model(_N_IN, 1, bias=bias), paths[name])
    paths["conv"] = str(d / "conv.pth")
    torch.save(_TinyConv(), paths["conv"])
    return paths


def test_generate_activation_map_saves_exact_maps_and_their_records(
        tmp_path, models, capsys, subtests):
    """For a linear head the saliency map is |W| summed over channels (image)
    or min-max scaled per channel (RGB): every crop's PNG equals it bit for
    bit under class_<pred>/plate/well, the map list and correlations reach the
    database only with save=True, and plot=True writes one PDF grid per batch."""
    w = np.abs(_dyadic_weight(1, _N_IN).numpy().reshape(3, _SIZE, _SIZE))
    image_map = _expected_map(w.sum(axis=0))
    channel_map = np.stack([_expected_map(w[c]) for c in range(3)], axis=-1)

    cases = [
        # (id, model, overrides, png mode, expected array, class folder)
        ("saliency-image-positive", "positive", dict(save=True), "L", image_map, "class_1"),
        ("saliency-image-negative-no-normalize", "negative",
         dict(save=True, normalize_input=False), "L", image_map, "class_0"),
        ("saliency-channel-rgb", "positive", dict(save=True, cam_type="saliency_channel"),
         "RGB", channel_map, "class_1"),
    ]
    for cid, model, over, mode, expected, folder in cases:
        with subtests.test(case=cid):
            root, tar_path, names = _project(tmp_path, cid)
            capsys.readouterr()
            assert D.generate_activation_map(_settings(tar_path, models[model], **over)) is None
            cam = over.get("cam_type", "saliency_image")
            save_dir = root / "datasets" / "ds" / cam
            pngs = sorted(save_dir.rglob("*.png"))
            assert [p.relative_to(save_dir).as_posix() for p in pngs] == [
                f"{folder}/plate1/A01/{n}" for n in names]
            for png in pngs:
                img = Image.open(png)
                assert img.mode == mode
                np.testing.assert_array_equal(np.array(img), expected)
            with sqlite3.connect(root / "measurements" / "ds.db") as con:
                rows = con.execute(
                    f"SELECT png_path, plateID, rowID, columnID, fieldID, object "
                    f"FROM {cam}_list ORDER BY png_path").fetchall()
            assert rows == [(str(p), "plate1", "r1", "c1", "f1", f"o{i}")
                            for i, p in enumerate(pngs)]
            out = capsys.readouterr().out
            assert f"Activation maps will be saved in: {save_dir}" in out
            assert out.rstrip().endswith("Activation map generation complete.")

    with subtests.test(case="plot-writes-one-grid-per-batch-and-nothing-else"):
        root, tar_path, _names = _project(tmp_path, "plot", n_images=6)
        D.generate_activation_map(_settings(tar_path, models["positive"], plot=True,
                                            batch_size=3, normalize=True))
        cam_dir = root / "datasets" / "ds" / "saliency_image"
        grids = sorted((cam_dir / "batch_grids").iterdir())
        assert [p.name for p in grids] == ["batch_0_grid.pdf", "batch_1_grid.pdf"]
        assert all(p.read_bytes()[:4] == b"%PDF" for p in grids)
        assert list(cam_dir.rglob("*.png")) == []
        assert not (root / "measurements" / "ds.db").exists()

    with subtests.test(case="correlations-computed-but-neither-shown-nor-stored"):
        root, tar_path, _names = _project(tmp_path, "quiet")
        D.generate_activation_map(_settings(tar_path, models["positive"],
                                            correlation=True))
        cam_dir = root / "datasets" / "ds" / "saliency_image"
        assert not (cam_dir / "batch_grids").exists()
        assert list(cam_dir.rglob("*.png")) == []
        assert not (root / "measurements" / "ds.db").exists()

    with subtests.test(case="correlations-stored-with-the-maps"):
        root, tar_path, names = _project(tmp_path, "corr")
        D.generate_activation_map(_settings(tar_path, models["positive"], save=True,
                                            plot=True, correlation=True,
                                            manders_thresholds=[50]))
        db = str(root / "measurements" / "ds.db")
        assert {"saliency_image_list", "saliency_image_correlations"} <= _tables(db)
        with sqlite3.connect(db) as con:
            cur = con.execute("SELECT * FROM saliency_image_correlations")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
        measured = {f"channel_{c}_activation_0_{suffix}"
                    for c in range(3) for suffix in ("pearsons", "50_M1", "50_M2")}
        assert measured | {"file_name", "png_path"} <= set(cols)
        assert sorted(r[cols.index("file_name")] for r in rows) == names
        for r in rows:
            assert -1.0 <= float(r[cols.index("channel_0_activation_0_pearsons")]) <= 1.0


def test_generate_activation_map_builds_its_inputs_and_hands_them_on(
        tmp_path, models, monkeypatch, capsys, subtests):
    """The settings decide the transform (Normalize only for real statistics),
    the worker count (cpu_count()-4, at least 1, through the RAM guard), the
    target layer and generator, the counterfactual crops and contract; a
    missing dataset or an unusable cam_type stops the run with a reason."""
    from torchvision import transforms as tv

    import spacr.resource_log as resource_log
    from spacr import io as io_module

    seen = []
    real_dataset = io_module.TarImageDataset

    def spy(path, transform=None, **kwargs):
        seen.append([type(step).__name__ if not isinstance(step, tv.Normalize)
                     else ("Normalize", tuple(step.mean), tuple(step.std))
                     for step in transform.transforms])
        return real_dataset(path, transform=transform, **kwargs)

    monkeypatch.setattr(io_module, "TarImageDataset", spy)
    root, tar_path, names = _project(tmp_path, "inputs")
    base = ["ToTensor", "CenterCrop"]
    transform_cases = [
        ("symmetric", dict(input_statistics="symmetric"),
         base + [("Normalize", (0.5,) * 3, (0.5,) * 3), "SelectChannels"]),
        ("imagenet", dict(input_statistics="imagenet"),
         base + [("Normalize", (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                 "SelectChannels"]),
        ("statistics-none", dict(input_statistics="none"), base + ["SelectChannels"]),
        ("normalize-input-off", dict(normalize_input=False), base + ["SelectChannels"]),
    ]
    for cid, over, steps in transform_cases:
        with subtests.test(case=f"transform-{cid}"):
            seen.clear()
            D.generate_activation_map(_settings(tar_path, models["positive"], **over))
            assert seen == [steps]

    for cpus, workers in ((6, 2), (3, 1)):
        with subtests.test(case=f"n-jobs-none-on-{cpus}-cpus"):
            real_loader = D.DataLoader
            loader_kwargs, guard_calls = {}, []

            def loader(dataset, **kwargs):
                loader_kwargs.update(kwargs)
                kwargs.update(num_workers=0, pin_memory=False, worker_init_fn=None)
                return real_loader(dataset, **kwargs)

            monkeypatch.setattr(D, "cpu_count", lambda: cpus)
            monkeypatch.setattr(D, "DataLoader", loader)
            monkeypatch.setattr(resource_log, "_guard_workers",
                                lambda task, n, unit, **k: (guard_calls.append((task, n)), n)[1])
            D.generate_activation_map(_settings(tar_path, models["positive"], n_jobs=None))
            assert guard_calls == [("classify", workers)]
            assert (loader_kwargs["num_workers"], loader_kwargs["batch_size"],
                    loader_kwargs["shuffle"]) == (workers, 2, False)
            monkeypatch.setattr(D, "DataLoader", real_loader)

    with subtests.test(case="maxvit-default-target-layer-reaches-gradcam"):
        import spacr.utils as utils
        built = {}

        class RecordingGradCAM:
            def __init__(self, model, target_layer, cam_type="gradcam"):
                built.update(target_layer=target_layer, cam_type=cam_type)

            def compute_gradcam_and_predictions(self, x):
                return x.detach().mean(dim=1), torch.zeros(x.shape[0], dtype=torch.long)

        monkeypatch.setattr(utils, "GradCAMGenerator", RecordingGradCAM)
        settings = _settings(tar_path, models["conv"], model_type="maxvit",
                             cam_type="gradcam_pp", save=True)
        D.generate_activation_map(settings)
        layer = "base_model.blocks.3.layers.1.layers.MBconv.layers.conv_b"
        assert built == {"target_layer": layer, "cam_type": "gradcam_pp"}
        assert settings["target_layer"] == layer
        pngs = sorted((root / "datasets" / "ds" / "gradcam_pp").rglob("*.png"))
        assert [p.name for p in pngs] == names

    with subtests.test(case="saliency-clears-the-target-layer"):
        settings = _settings(tar_path, models["positive"], model_type="maxvit",
                             target_layer="some.layer")
        D.generate_activation_map(settings)
        assert settings["target_layer"] is None

    refusals = [
        ("unknown-cam-type", dict(cam_type="not_a_cam"), "^unknown cam_type 'not_a_cam'"),
        ("gradcam-needs-a-spatial-layer", dict(cam_type="gradcam"),
         r"^cam_type 'gradcam' does not apply to the loaded Sequential "
         r"\(model_type='resnet18'\): "),
    ]
    for cid, over, pattern in refusals:
        with subtests.test(case=cid):
            with pytest.raises(ValueError, match=pattern):
                D.generate_activation_map(_settings(tar_path, models["positive"], **over))

    with subtests.test(case="missing-dataset-returns-after-the-device-note"):
        monkeypatch.setattr(D, "pick_device", lambda *a, **k: (torch.device("cpu"), _NOTE))
        missing = tmp_path / "nothing" / "datasets" / "nope.tar"
        missing.parent.mkdir(parents=True)
        capsys.readouterr()
        assert D.generate_activation_map(_settings(
            str(missing), str(tmp_path / "no_such_model.pth"), save=True)) is None
        out = capsys.readouterr().out
        assert out.index(_NOTE) < out.index(f"Dataset not found at {missing}")
        assert not (missing.parent / "nope").exists()
        assert (missing.parent.parent / "settings" / "saliency_image_settings.csv").is_file()
        monkeypatch.undo()
        monkeypatch.setattr(io_module, "TarImageDataset", spy)

    with subtests.test(case="counterfactual-crops-are-collected-up-to-the-limit"):
        handed = []
        monkeypatch.setattr(D, "_run_counterfactuals",
                            lambda settings, model, crops, names_, out, device, *,
                            input_contract: handed.append(
                                ([tuple(c.shape) for c in crops], list(names_), out,
                                 input_contract)))
        D.generate_activation_map(_settings(tar_path, models["positive"],
                                            counterfactuals=True, counterfactual_crops=3))
        assert handed == [([(2, 3, _SIZE, _SIZE), (1, 3, _SIZE, _SIZE)], names[:3],
                           str(root / "datasets" / "ds" / "saliency_image" / "counterfactuals"),
                           None)]

    with subtests.test(case="counterfactual-contract-carries-saved-classes"):
        from spacr.classification_pixels import DECLARED_UINT8

        monkeypatch.setenv("SPACR_DEVICE", "cpu")  # forced CPU, also on Metal Macs

        local = tmp_path / "local-classifier.pth"
        local.write_bytes(b"local fixture handle; deserialization is stubbed")
        two_class = _linear_model(_N_IN, 2).eval()
        metadata = {"classes": ["positive", "negative"], "preprocessing": {
            "crop_loading_policy": DECLARED_UINT8, "input_statistics": "symmetric"}}
        monkeypatch.setattr(D, "_load_inference_model",
                            lambda path, device: (two_class, metadata))
        handed.clear()
        D.generate_activation_map(_settings(
            tar_path, str(local), model_type="tiny_local", counterfactuals=True,
            counterfactual_crops=1, counterfactual_condition="CLASS"))
        (shapes, cf_names, out, contract), = handed
        assert (shapes, cf_names) == ([(1, 3, _SIZE, _SIZE)], names[:1])
        assert not os.path.exists(out)
        assert contract == {
            "schema": 1,
            "channel_ids": ["stored_rgb_plane_0", "stored_rgb_plane_1", "stored_rgb_plane_2"],
            "class_labels": ["positive", "negative"], "condition_key": "class",
            "preprocessing": {
                "pipeline": "activation_map_model_input_v1",
                "crop_loading_policy": DECLARED_UINT8,
                "center_crop": [_SIZE, _SIZE],
                "normalization": {"mean": [.5, .5, .5], "std": [.5, .5, .5]},
                "selected_channels_1based": [1, 2, 3],
                "selection_semantics": "zero_unselected_stored_rgb_planes",
                "classifier_preprocessing": metadata["preprocessing"],
            },
        }


# ===========================================================================
# analyze_activation_maps
# ===========================================================================

class _TinyCNN(nn.Module):
    """Two conv layers over 1 x 16 x 16, global pool, two logits."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(4, 8, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


def _square_images(n):
    """Alternating bright-square / empty frames a trained model separates."""
    generator = torch.Generator().manual_seed(0)
    out = []
    for index in range(n):
        image = torch.rand((1, 16, 16), generator=generator) * 0.1
        if index % 2 == 0:
            image[:, 4:12, 4:12] += 0.9
        out.append(image)
    return out


@pytest.fixture(scope="module")
def trained_cnn():
    """Eight real epochs: an untrained head gives flat, meaningless maps."""
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(0)
    model = _TinyCNN()
    x = torch.stack(_square_images(16))
    y = torch.tensor([i % 2 for i in range(16)])
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
    model.train()
    for _epoch in range(8):
        for bx, by in loader:
            optimiser.zero_grad()
            nn.functional.cross_entropy(model(bx), by).backward()
            optimiser.step()
    model.eval()
    with torch.no_grad():
        assert model(torch.stack(_square_images(4))).argmax(1).tolist() == [0, 1, 0, 1]
    return model


def test_analyze_activation_maps_returns_a_sorted_panel_with_its_caveats(
        trained_cnn, capsys, subtests):
    """One row per (image, method), sorted by image then deletion AUC with
    NaN last; a method that cannot run is a failed NaN row; masks add the
    pointing game; the notes say what was skipped; verbose only prints."""
    from spacr.attribution import NOT_AN_EXPLANATION

    no_mask = ("No object masks were given, so the pointing game was not scored. "
               "spaCR's merged/*.npy files carry the label planes if you want it.")
    no_sanity = ("The model-randomisation sanity check was skipped. Without it, a "
                 "method that returns the same map for a randomised model is "
                 "indistinguishable here from one that does not.")
    square = torch.zeros((1, 16, 16))
    square[:, 4:12, 4:12] = 1.0
    cases = [
        # (id, images, kwargs, methods in table, notes)
        ("one-image-one-method", _square_images(2)[:1], dict(methods=["saliency"]),
         {(0, "saliency")}, [NOT_AN_EXPLANATION, no_mask, no_sanity]),
        ("two-images-two-methods", _square_images(2),
         dict(methods=["saliency", "integrated_gradients"]),
         {(i, m) for i in (0, 1) for m in ("saliency", "integrated_gradients")},
         [NOT_AN_EXPLANATION, no_mask, no_sanity]),
        ("a-method-that-cannot-run", _square_images(1)[0],
         dict(methods=["saliency", "not_a_method"]),
         {(0, "saliency"), (0, "not_a_method")}, [NOT_AN_EXPLANATION, no_mask, no_sanity]),
        ("empty-method-list-uses-the-defaults", _square_images(1), dict(methods=[]),
         {(0, m) for m in ("gradcam", "saliency", "integrated_gradients", "occlusion")},
         [NOT_AN_EXPLANATION, no_mask, no_sanity]),
        ("a-mask-adds-the-pointing-game", _square_images(1),
         dict(methods=["saliency"], masks=[square]), {(0, "saliency")},
         [NOT_AN_EXPLANATION, no_sanity]),
    ]
    for cid, images, kwargs, pairs, notes in cases:
        with subtests.test(case=cid):
            result = D.analyze_activation_maps(trained_cnn, images, n_steps=4,
                                               sanity_check=False, verbose=False,
                                               **kwargs)
            assert capsys.readouterr().out == ""
            table = result["table"]
            assert set(zip(table["image"], table["method"])) == pairs
            assert len(table) == len(pairs)
            assert result["notes"] == notes
            assert result["sanity"] == {}
            assert list(table["image"]) == sorted(table["image"])
            for _image, block in table.groupby("image", sort=False):
                auc = block["deletion_auc"].tolist()
                finite = [a for a in auc if not np.isnan(a)]
                assert auc[:len(finite)] == sorted(finite)  # NaN last
            failed = table[table["failed"]]
            assert list(failed["method"]) == (
                ["not_a_method"] if cid == "a-method-that-cannot-run" else [])
            assert failed["deletion_auc"].isna().all()
            assert failed["pointing_game"].isna().all()
            if "masks" in kwargs:
                assert table["pointing_game"].notna().all()

    with subtests.test(case="no-images-is-refused"):
        with pytest.raises(ValueError, match="needs at least one image"):
            D.analyze_activation_maps(trained_cnn, [], methods=["saliency"])

    with subtests.test(case="verbose-prints-what-the-return-already-holds"):
        quiet = D.analyze_activation_maps(trained_cnn, _square_images(1),
                                          methods=["saliency", "occlusion"], n_steps=3,
                                          sanity_check=True, verbose=False)
        assert capsys.readouterr().out == ""
        loud = D.analyze_activation_maps(trained_cnn, _square_images(1),
                                         methods=["saliency", "occlusion"], n_steps=3,
                                         sanity_check=True, verbose=True)
        spoken = capsys.readouterr().out
        assert set(quiet["sanity"]) == set(loud["sanity"]) == {"saliency", "occlusion"}
        assert quiet["notes"] == [NOT_AN_EXPLANATION, no_mask]
        lines = spoken.strip().splitlines()
        assert lines[0] == NOT_AN_EXPLANATION
        assert lines[1:3] == [loud["sanity"][m].verdict() for m in ("saliency", "occlusion")]
        assert loud["agreement"] is not None  # two methods can be compared
        assert lines[3:] == [loud["agreement"].verdict()]

    with subtests.test(case="verbose-with-one-method-has-no-agreement-line"):
        loud = D.analyze_activation_maps(trained_cnn, _square_images(1),
                                         methods=["saliency"], n_steps=3,
                                         sanity_check=True, verbose=True)
        assert loud["agreement"] is None
        assert capsys.readouterr().out.strip().splitlines() == [
            NOT_AN_EXPLANATION, loud["sanity"]["saliency"].verdict()]


# ===========================================================================
# SmoothGrad
# ===========================================================================

def test_smooth_grad_averages_the_abs_gradient_of_the_requested_logit(subtests):
    """For a linear head every noisy gradient is the weight row, so the map is
    exactly |W[target]| for each target and every row of a batch; a flat input
    (zero noise) gives the plain |gradient| of a ReLU net; noise changes it."""
    from spacr.deep_spacr import SmoothGrad

    n_in = 3 * 4 * 4
    weight = _dyadic_weight(3, n_in)
    linear = _linear_model(n_in, 3, weight=weight)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 4, 4)
    batch = torch.rand(3, 3, 4, 4)

    with subtests.test(case="defaults"):
        sg = SmoothGrad(linear)
        assert (sg.model, sg.n_samples, sg.stdev_spread) == (linear, 50, 0.15)

    cases = [
        # (id, input, target, n_samples)
        ("target-0", x, 0, 5), ("target-1", x, 1, 3), ("target-2", x, 2, 4),
        ("batch-every-row", batch, 1, 2),
    ]
    for cid, inp, target, n_samples in cases:
        with subtests.test(case=cid):
            linear.train()
            before = inp.clone()
            out = SmoothGrad(linear, n_samples=n_samples).compute_smooth_grad(inp, target)
            expected = weight[target].abs().reshape(1, 3, 4, 4).expand_as(inp)
            assert torch.equal(out, expected)
            assert linear.training is False
            assert torch.equal(inp, before) and inp.grad is None
            assert inp.requires_grad is False

    torch.manual_seed(0)
    relu_net = nn.Sequential(nn.Flatten(), nn.Linear(n_in, 6), nn.ReLU(), nn.Linear(6, 2))
    with torch.no_grad():
        relu_net[1].weight.copy_(torch.linspace(-1, 1, 6 * n_in).reshape(6, n_in))
        relu_net[1].bias.zero_()
        relu_net[3].weight.copy_(torch.linspace(-1, 1, 12).reshape(2, 6))
        relu_net[3].bias.zero_()

    with subtests.test(case="flat-input-is-the-plain-gradient"):
        flat = torch.full((1, 3, 4, 4), 0.7)
        ref = flat.clone().requires_grad_()
        relu_net(ref)[0, 1].backward()
        out = SmoothGrad(relu_net, n_samples=6).compute_smooth_grad(flat, 1)
        torch.testing.assert_close(out, ref.grad.abs(), rtol=0, atol=1e-6)
        assert float(out.sum()) > 0.0

    with subtests.test(case="noise-changes-a-relu-map"):
        straddle = torch.linspace(-0.05, 0.05, n_in).reshape(1, 3, 4, 4)
        torch.manual_seed(1)
        quiet = SmoothGrad(relu_net, n_samples=8, stdev_spread=0.0).compute_smooth_grad(straddle, 1)
        torch.manual_seed(1)
        noisy = SmoothGrad(relu_net, n_samples=8, stdev_spread=2.0).compute_smooth_grad(straddle, 1)
        assert not torch.allclose(quiet, noisy, atol=1e-4)
        assert bool((noisy >= 0).all())


# ===========================================================================
# visualize_smooth_grad / visualize_integrated_gradients
# ===========================================================================

def _png_folder(tmp_path, tag, files):
    """``files``: {name: size or text}; PNGs get random pixels."""
    src = tmp_path / tag
    src.mkdir()
    for i, (name, spec) in enumerate(files.items()):
        if isinstance(spec, int):
            _write_png(src / name, seed=i, size=spec)
        else:
            (src / name).write_text(spec, encoding="utf-8")
    return src


def _overlay(png, size, attribution_map):
    """The 50/50 blend both drivers draw, at the attribution's resolution."""
    small = np.array(Image.open(png).convert("RGB").resize((size, size)))
    small = small / small.max()
    return (small * 0.5 + np.stack([attribution_map] * 3, axis=-1) * 0.5).clip(0, 1)


def test_visualize_smooth_grad_draws_and_saves_the_exact_map(tmp_path, monkeypatch,
                                                             capsys, shown, subtests):
    """Each PNG (and only PNGs) gives a three-panel figure whose SmoothGrad
    panel is mean_c |W[target]| at image_size, overlaid at that resolution
    even for larger sources, and saves it as smooth_grad_<file> on request."""
    size = 8
    n_in = 3 * size * size
    weight = _dyadic_weight(2, n_in, signed=False)
    model_path = str(tmp_path / "linear.pth")
    torch.save(_linear_model(n_in, 2, weight=weight), model_path)

    def expected(target):
        return np.mean(weight[target].abs().numpy().reshape(1, 3, size, size), axis=1).squeeze()

    cases = [
        # (id, folder files, target, kwargs, saved names or None, source size)
        ("saves-into-a-new-folder", {"a.png": size, "notes.txt": "not an image"}, 1,
         dict(save_smooth_grad=True), ["smooth_grad_a.png"], size),
        ("plots-without-saving", {"a.png": size}, 0,
         dict(save_smooth_grad=False, normalize=False), None, size),
        ("reuses-an-existing-folder", {"b.png": size}, 0,
         dict(save_smooth_grad=True, keep=True), ["keep_me.txt", "smooth_grad_b.png"], size),
        ("larger-source-overlaid-at-map-size", {"big.png": 16}, 1,
         dict(save_smooth_grad=False), None, 16),
    ]
    for cid, files, target, kwargs, saved, source in cases:
        with subtests.test(case=cid):
            src = _png_folder(tmp_path, cid, files)
            out_dir = tmp_path / f"{cid}_out"
            if kwargs.pop("keep", False):
                out_dir.mkdir()
                (out_dir / "keep_me.txt").write_text("previous run", encoding="utf-8")
            shown.clear()
            assert D.visualize_smooth_grad(str(src), model_path, target, image_size=size,
                                           save_dir=str(out_dir), **kwargs) is None
            png = next(src.glob("*.png"))
            (panels,) = shown
            assert [(p["title"], p["cmap"], p["axis_on"]) for p in panels] == [
                ("Original Image", "viridis", False), ("SmoothGrad", "hot", False),
                ("Overlay", "viridis", False)]
            original, smooth, overlay = (p["data"] for p in panels)
            assert original.shape == (source, source, 3)
            np.testing.assert_array_equal(smooth, expected(target))
            np.testing.assert_allclose(overlay, _overlay(png, size, expected(target)),
                                       rtol=0, atol=1e-12)
            if saved is None:
                assert not out_dir.exists()
            else:
                assert sorted(p.name for p in out_dir.iterdir()) == saved
                img = Image.open(out_dir / f"smooth_grad_{png.name}")
                assert img.mode == "L"
                np.testing.assert_array_equal(
                    np.array(img), (expected(target) * 255).astype(np.uint8))

    with subtests.test(case="no-pngs-is-a-no-op"):
        src = _png_folder(tmp_path, "empty", {"readme.md": "x", "table.csv": "a,b\n1,2\n"})
        shown.clear()
        D.visualize_smooth_grad(str(src), model_path, 0, image_size=size,
                                save_smooth_grad=False, save_dir=str(tmp_path / "none"))
        assert shown == []
        assert not (tmp_path / "none").exists()
        assert sorted(p.name for p in src.iterdir()) == ["readme.md", "table.csv"]

    with subtests.test(case="cpu-fallback-is-announced-first"):
        monkeypatch.setattr(D, "pick_device", lambda *a, **k: (torch.device("cpu"), _NOTE))
        src = _png_folder(tmp_path, "note", {"only.png": size})
        capsys.readouterr()
        D.visualize_smooth_grad(str(src), model_path, 0, image_size=size)
        assert capsys.readouterr().out.splitlines()[0] == _NOTE


def test_visualize_integrated_gradients_draws_and_saves_x_times_w(
        tmp_path, monkeypatch, capsys, shown, subtests):
    """From a zero baseline a linear head's integrated gradients are x * W, so
    each PNG's middle panel is mean_c(x_c * W_c) at image_size, overlaid at
    that resolution, and saved as integrated_grads_<file> on request."""
    from spacr.utils import preprocess_image

    size = 8
    n_in = 3 * size * size
    weight = _dyadic_weight(2, n_in, signed=False)
    model_path = str(tmp_path / "linear.pth")
    torch.save(_linear_model(n_in, 2, weight=weight), model_path)

    def expected(png, target, normalize):
        _img, x = preprocess_image(str(png), normalize=normalize, image_size=size)
        avg = np.mean([weight[target].numpy().reshape(1, 3, size, size)] * 50, axis=0)
        return np.mean(x.numpy() * avg, axis=1).squeeze()

    cases = [
        # (id, files, target, kwargs, saved names or None)
        ("saves-every-png-and-skips-text", {"a.png": size, "b.png": size, "notes.txt": "x"},
         0, dict(normalize=False, save_integrated_grads=True, channels=[1, 2, 3]),
         ["integrated_grads_a.png", "integrated_grads_b.png"]),
        ("plots-without-saving-imagenet-normalised", {"only.png": size}, 1,
         dict(normalize=True), None),
    ]
    for cid, files, target, kwargs, saved in cases:
        with subtests.test(case=cid):
            src = _png_folder(tmp_path, cid, files)
            out_dir = tmp_path / f"{cid}_out"
            shown.clear()
            D.visualize_integrated_gradients(str(src), model_path, target_label_idx=target,
                                             image_size=size, save_dir=str(out_dir),
                                             **kwargs)
            pngs = sorted(src.glob("*.png"))
            assert len(shown) == len(pngs)
            by_name = {}
            for panels in shown:
                assert [(p["title"], p["axis_on"]) for p in panels] == [
                    ("Original Image", False), ("Integrated Gradients", False),
                    ("Overlay", False)]
                assert panels[1]["cmap"] == "hot"
                match = [p for p in pngs if np.array_equal(
                    panels[0]["data"], np.array(Image.open(p).convert("RGB")))]
                by_name[match[0].name] = panels
            assert sorted(by_name) == [p.name for p in pngs]
            for png in pngs:
                ig = expected(png, target, kwargs["normalize"])
                _orig, middle, overlay = (p["data"] for p in by_name[png.name])
                np.testing.assert_allclose(middle, ig, rtol=0, atol=1e-6)
                np.testing.assert_allclose(overlay, _overlay(png, size, middle),
                                           rtol=0, atol=1e-12)
                if saved:
                    np.testing.assert_array_equal(
                        np.array(Image.open(out_dir / f"integrated_grads_{png.name}")),
                        (middle * 255).astype(np.uint8))
            if saved is None:
                assert not out_dir.exists()
            else:
                assert sorted(p.name for p in out_dir.iterdir()) == saved

    with subtests.test(case="cpu-fallback-is-announced-first"):
        monkeypatch.setattr(D, "pick_device", lambda *a, **k: (torch.device("cpu"), _NOTE))
        src = _png_folder(tmp_path, "note", {"only.png": size})
        capsys.readouterr()
        D.visualize_integrated_gradients(str(src), model_path, image_size=size,
                                         normalize=False)
        assert capsys.readouterr().out.splitlines()[0] == _NOTE
