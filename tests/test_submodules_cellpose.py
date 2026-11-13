"""Table-driven tests of the Cellpose train / test / apply entry points in ``spacr.submodules``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture:

* ``CellposeLazyDataset``: the exact image / label pair for every one of the
  eight dihedral variants, grayscale and unit scaling, and the refusals;
* ``train_cellpose``: what reaches ``train_seg`` (every annotated pair, the
  resolved settings, the model name and folder) and the settings snapshot;
* ``test_cellpose_model``: the per-image metric table for perfect, empty and
  unpaired predictions, the eval settings, progress and the saved artefacts;
* ``apply_cellpose_model``: per-object measurements and per-image summaries,
  with and without circularisation, empty predictions and the diagnostics.

``cellpose.models.CellposeModel`` and ``cellpose.train.train_seg`` are
replaced by recording doubles declaring the installed 4.0.7 signatures
(``channel_axis=MISSING_CHANNEL_AXIS``, no ``**kwargs``) whose ``eval`` passes
every call through :func:`tests.conftest.check_cellpose_eval_call`.  The
dataset, metrics, CSVs and figures run for real on tiny TIFFs.
"""
from __future__ import annotations

import os
import re
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import spacr.submodules as SUB  # noqa: E402
from tests.cellpose_api_contract import (  # noqa: E402
    DEPRECATED_EVAL_ARGUMENTS,
    configured_eval_arguments,
    emulate_pretrained_model,
    eval_arguments,
    init_arguments,
)
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call  # noqa: E402

_NET = object()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _case_dir(tmp_path, name):
    d = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", name)
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# the Cellpose doubles
# ---------------------------------------------------------------------------

def _install_cellpose(mp, folder, rec, mask_fn=None, pin_kwargs=False):
    """Record ``CellposeModel`` / ``train_seg``; the stock checkpoint is a local file.

    ``eval`` returns ``rec['preds']`` in dataset order, or ``mask_fn()`` per
    image when given.
    """
    from spacr import accelerator, model_zoo

    cached = folder / "cached-cpsam"
    cached.write_bytes(b"recording model; never loaded")
    entry = model_zoo.ModelEntry(key="cpsam", name="cpsam", kind="cellpose",
                                 path=str(cached), uri="")
    mp.setattr(model_zoo, "catalogue", lambda remote=True: [entry])
    mp.setattr(model_zoo, "fetch", lambda *a, **k: pytest.fail("stock weights fetched"))
    rec.update(training_base=str(cached), models=[], eval_calls=[], eval_configured=[],
               train_calls=[], n_predicted=0)
    rec.setdefault("preds", [])

    class _Model:
        def __init__(self, gpu=False, pretrained_model="cpsam", model_type=None,
                     diam_mean=None, device=None, nchan=None, use_bfloat16=True):
            self.gpu = gpu
            self.pretrained_model = pretrained_model
            self.init = init_arguments(locals())
            self.loaded_model = emulate_pretrained_model(pretrained_model, model_type)
            self.net = _NET
            rec["models"].append(self)

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
                 invert=False, rescale=None, diameter=None, flow_threshold=0.4,
                 cellprob_threshold=0.0, do_3D=False, anisotropy=None,
                 flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
                 max_size_fraction=0.4, niter=None, augment=False,
                 tile_overlap=0.1, bsize=256, compute_masks=True, progress=None):
            # plain 2-D planes: the axis is cellpose's to detect
            check_cellpose_eval_call(x, channel_axis, require_channel_axis=False)
            bound = locals()
            rec["eval_configured"].append(configured_eval_arguments(bound))
            rec["eval_calls"].append({"x": list(x), **eval_arguments(bound)})
            masks = []
            for _ in x:
                if mask_fn is not None:
                    masks.append(mask_fn())
                else:
                    masks.append(np.asarray(rec["preds"][rec["n_predicted"]], np.uint16))
                rec["n_predicted"] += 1
            flows = [[np.zeros(m.shape + (3,), np.float32), np.zeros((2,) + m.shape, np.float32),
                      np.linspace(-3, 3, m.size, dtype=np.float32).reshape(m.shape)]
                     for m in masks]
            return masks, flows, [None] * len(masks)

    def _train_seg(net, **kwargs):
        rec["train_calls"].append({"net": net, **kwargs})
        return "cp_model_path_sentinel", [0.5], [0.4]

    mp.setattr(SUB.cp_models, "CellposeModel", _Model)
    mp.setattr(SUB, "cp_models", types.SimpleNamespace(CellposeModel=_Model))
    mp.setattr(SUB.train_cp, "train_seg", _train_seg)
    if pin_kwargs:
        mp.setattr(accelerator, "cellpose_kwargs", lambda: {"gpu": True, "device": None})


# ---------------------------------------------------------------------------
# synthetic TIFFs
# ---------------------------------------------------------------------------

def _label(size, blocks):
    """uint16 label image; ``blocks`` = [(value, (r0, r1, c0, c1)), ...]."""
    lbl = np.zeros((size, size), np.uint16)
    for value, (r0, r1, c0, c1) in blocks:
        lbl[r0:r1, c0:c1] = value
    return lbl


TWO = [(1, (2, 10, 2, 10)), (2, (18, 28, 18, 28))]       # 64 px and 100 px
ONE = [(1, (2, 10, 2, 10))]


def _write_pairs(root, split, labels, extra_image_only=(), extra_mask_only=(), junk=None):
    """Matched ``<root>/<split>/{images,masks}`` TIFFs (bright objects on dim ground)."""
    img_dir, mask_dir = root / split / "images", root / split / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    names = []
    for i, lbl in enumerate(labels):
        name = f"img_{i:03d}.tif"
        tifffile.imwrite(str(img_dir / name), (lbl > 0).astype(np.uint16) * 40000 + 100)
        tifffile.imwrite(str(mask_dir / name), lbl)
        names.append(name)
    for name in extra_image_only:
        tifffile.imwrite(str(img_dir / name), np.full((32, 32), 7, np.uint16))
    for name in extra_mask_only:
        tifffile.imwrite(str(mask_dir / name), np.zeros((32, 32), np.uint16))
    if junk:
        (img_dir / junk).write_bytes(b"not a tif")
        (mask_dir / junk).write_bytes(b"not a tif")
    return {"names": names, "images": [str(img_dir / n) for n in names],
            "masks": [str(mask_dir / n) for n in names]}


def _dataset_labels(written, size):
    ds = SUB.CellposeLazyDataset(written["images"], written["masks"],
                                 {"target_size": size, "normalize": True,
                                  "percentiles": (2, 98)}, randomize=False)
    return [ds[i][1] for i in range(len(ds))]


# ---------------------------------------------------------------------------
# CellposeLazyDataset
# ---------------------------------------------------------------------------

def _dihedral(a, k):
    """The eight variants in the dataset's order (0 = original), built from
    skimage's ``rotate`` the way the dataset documents them."""
    from skimage.transform import rotate

    def rot(x, angle):
        return rotate(x, angle, resize=False, preserve_range=True)
    return [a, rot(a, 90), rot(a, 180), rot(a, 270), np.fliplr(a), np.flipud(a),
            np.fliplr(rot(a, 90)), np.flipud(rot(a, 90))][k]


def test_cellpose_lazy_dataset_yields_resized_dihedral_pairs(subtests, tmp_path):
    """Item ``i`` is variant ``i % 8`` of file ``i // 8``: the image grayscaled
    and unit-scaled (percentile-stretched when asked), the label transformed
    identically and resized by nearest neighbour."""
    rng = np.random.default_rng(0)
    gray = (rng.random((32, 32)) * 1000).astype(np.uint16)
    rgb = (rng.random((32, 32, 3)) * 1000).astype(np.uint16)
    lbl = _label(32, [(1, (1, 6, 1, 20)), (2, (24, 30, 2, 8))])
    paths = {}
    for name, arr in (("gray", gray), ("rgb", rgb), ("lbl", lbl)):
        paths[name] = str(tmp_path / f"{name}.tif")
        tifffile.imwrite(paths[name], arr)
    unit_gray = gray.astype(np.float32) / gray.max()
    unit_rgb = rgb.mean(axis=-1).astype(np.float32)
    unit_rgb = unit_rgb / unit_rgb.max()
    cases = [
        # (id, images, settings, augment, expected len, expected image of each item or None)
        ("plain, no normalisation", [paths["gray"]], {"normalize": False, "target_size": 32},
         False, 1, unit_gray),
        ("RGB is averaged to one plane", [paths["rgb"]], {"normalize": False, "target_size": 32},
         False, 1, unit_rgb),
        ("eight variants of every file", [paths["gray"], paths["gray"]],
         {"normalize": False, "target_size": 32}, True, 16, unit_gray),
        ("percentile stretch then resize", [paths["gray"]] * 3,
         {"normalize": True, "percentiles": (2, 99), "target_size": 16}, False, 3, None),
        ("percentile stretch with augmentation", [paths["gray"]] * 2,
         {"normalize": True, "percentiles": [2, 99], "target_size": 16}, True, 16, None),
    ]
    for case_id, images, settings, augment, n, image in cases:
        with subtests.test(msg=case_id):
            ds = SUB.CellposeLazyDataset(images, [paths["lbl"]] * len(images), settings,
                                         randomize=False, augment=augment)
            assert len(ds) == n
            size = settings["target_size"]
            for idx in range(n):
                img, out_lbl = ds[idx]
                assert img.shape == out_lbl.shape == (size, size)
                assert img.dtype == np.float32 and out_lbl.dtype == np.uint16
                k = idx % 8 if augment else 0
                if image is not None:
                    np.testing.assert_allclose(img, _dihedral(image, k), atol=1e-5)
                    np.testing.assert_array_equal(out_lbl, _dihedral(lbl, k).astype(np.uint16))
                    assert sorted(np.unique(out_lbl)) == [0, 1, 2]
                else:
                    assert set(np.unique(out_lbl)) <= {0, 1, 2}
                    assert 0.0 <= img.min() and img.max() <= 1.0

    with subtests.test(msg="randomize keeps every image with its own label"):
        ds = SUB.CellposeLazyDataset([paths["gray"], paths["rgb"]], [paths["lbl"], paths["gray"]],
                                     {"target_size": 8}, randomize=True)
        assert sorted(zip(ds.image_files, ds.label_files)) == sorted(
            [(paths["gray"], paths["lbl"]), (paths["rgb"], paths["gray"])])

    for case_id, images, labels, message in (
            ("unequal lengths are refused", ["a"], ["b", "c"],
             "image_files and label_files must have the same length."),
            ("no files are refused", [], [], "image_files and label_files cannot be empty.")):
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError, match=re.escape(message)):
                SUB.CellposeLazyDataset(images, labels, {"target_size": 8})


# ---------------------------------------------------------------------------
# train_cellpose
# ---------------------------------------------------------------------------

def _train(**over):
    s = {"model_name": "mymodel", "n_epochs": 20, "target_size": 16, "augment": False,
         "batch_size": 2, "learning_rate": 0.05, "weight_decay": 1e-4}
    s.update(over)
    return s


def test_train_cellpose_hands_every_annotated_pair_to_train_seg(subtests, tmp_path, capsys):
    """Every matched pair reaches ``train_seg`` with the resolved settings,
    under ``<name>_cpsam_e<epochs>.CP_model`` in ``models/cellpose_model``;
    unpaired files stop the run before training."""
    from spacr import accelerator
    from spacr.settings import get_train_cellpose_default_settings

    three = [_label(32, TWO), _label(32, [(1, (4, 14, 4, 14))]), _label(32, [(1, (6, 16, 20, 30))])]
    two = [_label(32, [(1, (4, 14, 4, 14))])] * 2
    asym = [_label(32, [(1, (1, 6, 1, 20)), (2, (24, 30, 2, 8))]),
            _label(32, [(1, (2, 8, 3, 26)), (2, (20, 31, 1, 6))])]
    cases = [
        # (id, labels, settings or None for the documented defaults, extras, expected)
        ("a CUDA box trains on every pair", three, _train(), "cuda",
         {"n": 3, "name": "mymodel_cpsam_e20.CP_model", "batch_size": 2, "lr": 0.05,
          "wd": 1e-4, "epochs": 20}),
        ("augment asks cellpose for online augmentation", asym,
         _train(model_name="aug", n_epochs=5, target_size=32, augment=True, batch_size=1,
                learning_rate=0.1, weight_decay=1e-5), None,
         {"n": 2, "name": "aug_cpsam_e5.CP_model", "batch_size": 1, "lr": 0.1, "wd": 1e-5,
          "epochs": 5, "labels": [lbl for lbl in asym]}),
        ("the preview plots the very arrays it trains on", two,
         _train(model_name="plotted", n_epochs=10, learning_rate=0.2, weight_decay=1e-5),
         "record_plot", {"n": 2, "name": "plotted_cpsam_e10.CP_model", "batch_size": 2,
                         "lr": 0.2, "wd": 1e-5, "epochs": 10}),
        ("a failing preview does not stop training", two,
         _train(model_name="plotfail", n_epochs=30, learning_rate=0.2, weight_decay=1e-5),
         "fail_plot", {"n": 2, "name": "plotfail_cpsam_e30.CP_model", "batch_size": 2,
                       "lr": 0.2, "wd": 1e-5, "epochs": 30,
                       "printed": "could not print batch images"}),
        ("the documented defaults are enough", two, None, None, {"n": 2, "epochs": 2}),
        ("unpaired images are refused before training", three,
         _train(model_name="matched", batch_size=50), "unpaired",
         {"raises": "Missing masks.*orphan_image"}),
    ]
    for case_id, labels, settings, extra, expected in cases:
        with subtests.test(msg=case_id):
            folder = _case_dir(tmp_path, case_id)
            if extra == "unpaired":
                _write_pairs(folder, "train", labels, extra_image_only=("orphan_image.tif",),
                             extra_mask_only=("orphan_mask.tif",), junk="notes.png")
            else:
                _write_pairs(folder, "train", labels)
            if settings is None:
                settings = get_train_cellpose_default_settings({})
                settings["n_epochs"] = 2
            settings = dict(settings, src=str(folder))
            rec, seen = {}, {}
            capsys.readouterr()
            with pytest.MonkeyPatch.context() as mp:
                _install_cellpose(mp, folder, rec)
                if extra == "cuda":
                    mp.setattr(accelerator, "resolve", lambda: accelerator.Accelerator(
                        kind="cuda", device="cuda:0", label="recording CUDA", bfloat16=False))
                if extra == "record_plot":
                    mp.setattr(SUB, "plot_cellpose_batch", lambda images, labels: seen.update(
                        images=list(images), labels=list(labels)))
                if extra == "fail_plot":
                    def _boom(images, labels):
                        raise RuntimeError("no display")
                    mp.setattr(SUB, "plot_cellpose_batch", _boom)
                if "raises" in expected:
                    with pytest.raises(ValueError, match=expected["raises"]):
                        SUB.train_cellpose(settings)
                    assert rec["train_calls"] == []
                    continue
                SUB.train_cellpose(settings)
            assert len(rec["train_calls"]) == 1 and len(rec["models"]) == 1
            call, model = rec["train_calls"][0], rec["models"][0]
            assert model.pretrained_model == rec["training_base"]
            assert call["net"] is _NET
            assert len(call["train_data"]) == len(call["train_labels"]) == expected["n"]
            assert (call["channel_axis"], call["rescale"], call["save_every"]) == (0, False, 100)
            assert call["n_epochs"] == expected["epochs"]
            assert call["normalize"] == {"normalize": True, "percentile": [1, 99]}
            save_path = os.path.join(str(folder), "models", "cellpose_model")
            assert call["save_path"] == save_path and os.path.isdir(save_path)
            for img, lbl in zip(call["train_data"], call["train_labels"]):
                assert img.shape == lbl.shape == (32, 32)
                assert img.dtype == np.float32 and np.isfinite(img).all()
                assert lbl.dtype == np.uint16
            if "name" in expected:
                assert call["model_name"] == expected["name"]
                assert (call["batch_size"], call["learning_rate"], call["weight_decay"]) == (
                    expected["batch_size"], expected["lr"], expected["wd"])
                snapshot = pd.read_csv(folder / "settings" / f"{expected['name']}.csv")
                values = dict(zip(snapshot["Key"], snapshot["Value"]))
                assert {"src", "model_name", "n_epochs", "target_size", "learning_rate",
                        "weight_decay", "batch_size"} <= set(values)
                assert values["base_model"] == "cpsam"
            if extra == "cuda":
                assert model.gpu is True
                assert str(model.init["device"]) == "cuda:0"
                assert model.init["use_bfloat16"] is False
            if "labels" in expected:
                for got, want in zip(call["train_labels"], expected["labels"]):
                    np.testing.assert_array_equal(got, want)
            if extra == "record_plot":
                assert [id(a) for a in seen["images"]] == [id(a) for a in call["train_data"]]
                assert [id(a) for a in seen["labels"]] == [id(a) for a in call["train_labels"]]
            if "printed" in expected:
                assert expected["printed"] in capsys.readouterr().out


# ---------------------------------------------------------------------------
# test_cellpose_model
# ---------------------------------------------------------------------------

COLUMNS = ["label_image", "Jaccard", "n_objects_true", "n_objects_pred", "mean_area_true",
           "mean_area_pred", "TP", "FP", "FN", "Precision", "Recall", "F1", "Accuracy", "n_error"]
PERFECT_TWO = [1.0, 2, 2, 82.0, 82.0, 2, 0, 0, 1.0, 1.0, 1.0, 1.0, 0]
PERFECT_ONE = [1.0, 1, 1, 64.0, 64.0, 1, 0, 0, 1.0, 1.0, 1.0, 1.0, 0]


def _test_settings(folder, **over):
    checkpoint = folder / "custom_model.CP_model"
    checkpoint.write_bytes(b"recording model; never loaded")
    s = {"src": str(folder), "model_path": str(checkpoint), "save": False, "normalize": True,
         "percentiles": (2, 98), "batch_size": 50, "CP_probability": 0.5, "FT": 0.6,
         "target_size": 32}
    s.update(over)
    return s


def test_test_cellpose_model_scores_each_image_against_its_mask(subtests, tmp_path):
    """Each matched test image gets one exact metric row (zero-division cases
    included); eval gets the scoring settings, progress counts every image,
    and ``save`` writes the CSV and one diagnostic per image."""
    import spacr.plot as P
    import spacr.utils as U

    cases = [
        # (id, labels, predictions ("truth" / "empty"), settings overrides, write kwargs,
        #  expected rows (without label_image) or None)
        ("a perfect prediction", [_label(32, TWO)] * 2, "truth", {}, {},
         [PERFECT_TWO, PERFECT_TWO]),
        ("nothing predicted, and an empty ground truth",
         [_label(32, TWO), _label(32, [])], "empty", {}, {},
         [[0.0, 2, 0, 82.0, 0, 0, 0, 2, 0, 0.0, 0, 0.0, 2],
          [np.nan, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        ("unpaired files are not scored", [_label(32, ONE)] * 2, "truth", {},
         {"extra_image_only": ("only_image.tif",), "extra_mask_only": ("only_mask.tif",),
          "junk": "readme.png"}, [PERFECT_ONE, PERFECT_ONE]),
        ("eval gets FT and CP_probability", [_label(32, ONE)], "truth",
         {"FT": 0.25, "CP_probability": -1.5}, {}, [PERFECT_ONE]),
        ("four images in batches of two, saved", [_label(32, [(1, (2 + i, 10 + i, 2, 10))])
                                                 for i in range(4)], "truth",
         {"save": True, "batch_size": 2}, {}, [PERFECT_ONE] * 4),
        ("saved diagnostics", [_label(32, ONE), _label(32, [(1, (4, 12, 4, 12)),
                                                           (2, (20, 30, 20, 30))])],
         "truth", {"save": True}, {}, [PERFECT_ONE, [1.0, 2, 2, 82.0, 82.0, 2, 0, 0, 1.0, 1.0,
                                                     1.0, 1.0, 0]]),
        ("only the documented defaults", [_label(32, ONE)], "truth", "defaults", {},
         [PERFECT_ONE]),
    ]
    for case_id, labels, preds, over, write_kw, rows in cases:
        with subtests.test(msg=case_id):
            folder = _case_dir(tmp_path, case_id)
            written = _write_pairs(folder, "test", labels, **write_kw)
            rec = {}
            if over == "defaults":
                checkpoint = folder / "cp.CP_model"
                checkpoint.write_bytes(b"settings snapshot checkpoint; never decoded")
                settings = {"src": str(folder), "model_path": str(checkpoint), "save": False,
                            "target_size": 32, "batch_size": 10}
            else:
                settings = _test_settings(folder, **over)
            inputs = {p: Path(p).read_bytes() for p in written["images"] + written["masks"]}
            shown, progress, saved = [], [], []
            with pytest.MonkeyPatch.context() as mp:
                _install_cellpose(mp, folder, rec)
                rec["preds"] = (_dataset_labels(written, 32) if preds == "truth"
                                else [np.zeros((32, 32), np.uint16)] * len(labels))
                mp.setattr(SUB, "display", shown.append)
                mp.setattr(U, "print_progress", lambda done, total, **kw: progress.append(
                    (done, total)))
                mp.setattr(P, "figure_output_preferences", lambda: ("png", 200))
                real_save = SUB.save_figure
                mp.setattr(SUB, "save_figure", lambda fig, path, **kw: saved.append(
                    str(real_save(fig, path, **kw))) or saved[-1])
                assert SUB.test_cellpose_model(settings) is None
            df = shown[0]
            assert list(df.columns) == COLUMNS
            assert df["label_image"].tolist() == written["names"]
            np.testing.assert_array_equal(df[COLUMNS[1:]].to_numpy(dtype=float),
                                          np.array(rows, dtype=float))
            assert rec["n_predicted"] == len(labels)
            assert rec["models"][0].pretrained_model == settings["model_path"]
            batch = settings["batch_size"]
            assert [len(c["x"]) for c in rec["eval_calls"]] == [
                min(batch, len(labels) - i) for i in range(0, len(labels), batch)]
            assert progress == [(min(batch * (i + 1), len(labels)), len(labels))
                                for i in range(len(rec["eval_calls"]))]
            for call, configured in zip(rec["eval_calls"], rec["eval_configured"]):
                assert not set(configured) & set(DEPRECATED_EVAL_ARGUMENTS)
                assert (call["flow_threshold"], call["cellprob_threshold"]) == (
                    settings["FT"], settings["CP_probability"])
                assert (call["normalize"], call["diameter"], call["rescale"], call["resample"],
                        call["anisotropy"], call["min_size"], call["augment"],
                        call["tile_overlap"], call["bsize"]) == (
                            False, 30, None, True, None, 5, True, 0.2, 256)
                assert all(x.shape == (32, 32) and x.dtype == np.float32 for x in call["x"])
            results = folder / "results"
            if settings["save"]:
                expected = [f"cellpose_result_{i:03d}.png" for i in range(len(labels))]
                assert sorted(p.name for p in results.glob("*.png")) == expected
                assert len(saved) == len(labels)
                for name in expected:
                    assert (results / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
                csv = pd.read_csv(results / "test_results.csv")
                assert csv["label_image"].tolist() == written["names"]
                assert csv["n_objects_true"].tolist() == [r[1] for r in rows]
            else:
                assert list(results.iterdir()) == [] and saved == []
            assert plt.get_fignums() == []
            if over == "defaults":
                values = dict(zip(*pd.read_csv(folder / "settings" / "test_cellpose_model.csv")[
                    ["Key", "Value"]].values.T))
                assert (values["model_path"], values["FT"], values["normalize"]) == (
                    settings["model_path"], "0.4", "True")
                assert settings["CP_probability"] == 0
                assert Path(settings["model_path"]).read_bytes() == (
                    b"settings snapshot checkpoint; never decoded")
            assert {p: Path(p).read_bytes() for p in inputs} == inputs


# ---------------------------------------------------------------------------
# apply_cellpose_model
# ---------------------------------------------------------------------------

def _four_objects(size=64):
    """Areas 100, 100, 64 inside the inscribed circle and 25 in the corner."""
    m = np.zeros((size, size), np.int32)
    m[10:20, 10:20] = 1
    m[28:38, 28:38] = 2
    m[44:52, 20:28] = 3
    m[0:5, 0:5] = 4
    return m


def _write_images(folder, n):
    rng = np.random.default_rng(7)
    for k in range(n):
        tifffile.imwrite(str(folder / f"img_{k:02d}.tif"),
                         rng.integers(100, 4000, size=(64, 64)).astype(np.uint16))


def test_apply_cellpose_model_measures_and_summarises_each_image(subtests, tmp_path):
    """Every predicted object becomes a measurement row and every image a
    summary row (circularisation drops what lies outside the inscribed
    circle); ``save`` adds one diagnostic per image."""
    import spacr.plot as P

    empty = lambda: np.zeros((64, 64), np.int32)  # noqa: E731
    cases = [
        # (id, n images, settings overrides, mask sequence or None, expected areas per image)
        ("three images in batches of 2 and 1", 3, {}, None,
         {f"img_{k:02d}.tif": [25, 64, 100, 100] for k in range(3)}),
        ("circularize drops the corner object, saved", 3, {"circularize": True, "save": True},
         None, {f"img_{k:02d}.tif": [64, 100, 100] for k in range(3)}),
        ("no objects anywhere", 2, {}, [empty, empty], {}),
        ("one empty image of two", 2, {}, [_four_objects, empty],
         {"img_00.tif": [25, 64, 100, 100]}),
    ]
    for case_id, n, over, seq, areas in cases:
        with subtests.test(msg=case_id):
            folder = _case_dir(tmp_path, case_id)
            src = folder / "src"
            src.mkdir()
            _write_images(src, n)
            checkpoint = folder / "cyto_fake.pt"
            checkpoint.write_bytes(b"never decoded")
            settings = {"src": str(src), "model_path": str(checkpoint), "batch_size": 2,
                        "FT": 0.4, "CP_probability": 0.5, "circularize": False, "save": False,
                        "normalize": True, "percentiles": (2, 98), "target_size": 64}
            settings.update(over)
            masks = iter(seq) if seq else None
            rec, shows = {}, []
            with pytest.MonkeyPatch.context() as mp:
                _install_cellpose(mp, folder, rec,
                                  mask_fn=(lambda: next(masks)()) if seq else _four_objects,
                                  pin_kwargs=True)
                mp.setattr(P, "figure_output_preferences", lambda: ("png", 200))
                mp.setattr(plt, "show", lambda *a, **k: shows.append(1))
                assert SUB.apply_cellpose_model(settings) is None
            assert [(m.gpu, m.pretrained_model) for m in rec["models"]] == [(True, str(checkpoint))]
            assert [len(c["x"]) for c in rec["eval_calls"]] == [min(2, n - i)
                                                                for i in range(0, n, 2)]
            for call, configured in zip(rec["eval_calls"], rec["eval_configured"]):
                assert not set(configured) & set(DEPRECATED_EVAL_ARGUMENTS)
                assert "channels" not in configured
                assert (call["normalize"], call["diameter"], call["flow_threshold"],
                        call["cellprob_threshold"], call["rescale"], call["resample"],
                        call["anisotropy"], call["min_size"], call["augment"],
                        call["tile_overlap"], call["bsize"]) == (
                            False, 30, 0.4, 0.5, None, True, None, 5, True, 0.2, 256)
                assert all(x.shape == (64, 64) and x.dtype == np.float32 for x in call["x"])
            results = src / "results"
            meas = pd.read_csv(results / "measurements.csv")
            summary = pd.read_csv(results / "summary.csv")
            assert list(meas.columns) == ["image", "object_id", "area"]
            assert list(summary.columns) == ["image", "object_count", "average_area"]
            got = {name: sorted(g["area"]) for name, g in meas.groupby("image")}
            assert got == areas
            for name, g in meas.groupby("image"):
                assert sorted(g["object_id"]) == list(range(1, len(areas[name]) + 1))
            assert summary.set_index("image")["object_count"].to_dict() == {
                k: len(v) for k, v in areas.items()}
            if areas:
                np.testing.assert_allclose(summary["average_area"],
                                           [np.mean(areas[k]) for k in summary["image"]])
            else:
                assert len(meas) == len(summary) == 0
            pngs = sorted(p.name for p in results.glob("*.png"))
            if settings["save"]:
                assert pngs == [f"cellpose_result_{i:03d}.png" for i in range(n)]
                assert len(shows) == n
            else:
                assert pngs == [] and shows == []
            assert plt.get_fignums() == []
            assert (src / "settings" / "apply_cellpose_model.csv").is_file()
