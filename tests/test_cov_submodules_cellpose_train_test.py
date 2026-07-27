"""
Coverage tests for the Cellpose train/test entry points of
``spacr.submodules`` — ``train_cellpose`` and ``test_cellpose_model``.

Both functions wrap a real GPU Cellpose model, so the two heavyweight
seams (``cellpose.models.CellposeModel`` and ``cellpose.train.train_seg``)
are replaced by recording fakes.  Everything else — the settings
resolution, the filename intersection, the lazy dataset, the augmentation
fan-out, the metric computation, the diagnostic PNGs and the result
DataFrame — is the real product code running on real synthetic TIFFs.

Known defects are asserted as ``xfail(strict=True)`` against the CORRECT
behaviour so they flip to a failure the moment they are fixed. The only
one still outstanding lives in ``spacr.settings``:
``get_train_cellpose_default_settings`` supplies neither ``target_size``
nor ``augment``, both of which ``train_cellpose`` indexes unconditionally.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import spacr.submodules as SUB

_NET_SENTINEL = object()


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def _close_figures():
    """No matplotlib window/figure may survive a test."""
    import matplotlib.pyplot as plt
    yield
    plt.close("all")


@pytest.fixture
def cp_stub(monkeypatch):
    """Replace the two cellpose seams with recording fakes.

    ``rec['preds']`` is the queue of predicted masks handed back by
    ``model.eval`` in dataset order; tests fill it before calling the
    function under test.
    """
    rec = {
        "models": [],
        "eval_calls": [],
        "train_calls": [],
        "preds": [],
        "n_predicted": 0,
    }

    class _FakeCellposeModel:
        def __init__(self, gpu=False, pretrained_model=None, **kwargs):
            self.gpu = gpu
            self.pretrained_model = pretrained_model
            self.extra_kwargs = kwargs
            self.net = _NET_SENTINEL
            rec["models"].append(self)

        def eval(self, x, **kwargs):
            rec["eval_calls"].append({"x": list(x), **kwargs})
            masks = []
            for _ in x:
                pred = np.asarray(rec["preds"][rec["n_predicted"]], dtype=np.uint16)
                rec["n_predicted"] += 1
                masks.append(pred)
            flows = []
            for m in masks:
                rgb = np.zeros(m.shape + (3,), dtype=np.uint8)
                dP = np.zeros((2,) + m.shape, dtype=np.float32)
                cellprob = np.zeros(m.shape, dtype=np.float32)
                flows.append([rgb, dP, cellprob])
            styles = [np.zeros(8, dtype=np.float32) for _ in masks]
            return masks, flows, styles

    def _fake_train_seg(net, **kwargs):
        rec["train_calls"].append({"net": net, **kwargs})
        return "cp_model_path_sentinel", [0.5], [0.4]

    monkeypatch.setattr(SUB.cp_models, "CellposeModel", _FakeCellposeModel)
    monkeypatch.setattr(SUB.train_cp, "train_seg", _fake_train_seg)
    return rec


# ===========================================================================
# Synthetic image/mask helpers
# ===========================================================================

def _label_image(size, blocks):
    """Build a uint16 label image; ``blocks`` = [(value, (r0, r1, c0, c1)), ...]."""
    lbl = np.zeros((size, size), dtype=np.uint16)
    for value, (r0, r1, c0, c1) in blocks:
        lbl[r0:r1, c0:c1] = value
    return lbl


def _write_pairs(root, split, labels, size=32, extra_image_only=(),
                 extra_mask_only=(), junk_ext=None):
    """Write matched ``images``/``masks`` TIFFs under ``<root>/<split>/``.

    Returns dict with the matched filenames and their absolute paths.
    """
    import tifffile

    img_dir = root / split / "images"
    mask_dir = root / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    names, img_paths, mask_paths = [], [], []
    for i, lbl in enumerate(labels):
        name = f"img_{i:03d}.tif"
        # Foreground bright, background dim -> a real dynamic range so the
        # dataset's percentile rescale actually does something.
        img = ((lbl > 0).astype(np.uint16) * 40000) + 100
        tifffile.imwrite(str(img_dir / name), img)
        tifffile.imwrite(str(mask_dir / name), lbl)
        names.append(name)
        img_paths.append(str(img_dir / name))
        mask_paths.append(str(mask_dir / name))

    blank = np.zeros((size, size), dtype=np.uint16)
    for name in extra_image_only:
        import tifffile as _tf
        _tf.imwrite(str(img_dir / name), blank + 7)
    for name in extra_mask_only:
        import tifffile as _tf
        _tf.imwrite(str(mask_dir / name), blank)
    if junk_ext:
        (img_dir / junk_ext).write_bytes(b"not a tif")
        (mask_dir / junk_ext).write_bytes(b"not a tif")

    return {"names": names, "images": img_paths, "masks": mask_paths}


def _dataset_labels(img_paths, mask_paths, target_size):
    """The exact label arrays CellposeLazyDataset yields (dataset order)."""
    ds = SUB.CellposeLazyDataset(
        img_paths, mask_paths,
        {"target_size": target_size, "normalize": True, "percentiles": (2, 98)},
        randomize=False, augment=False,
    )
    return [ds[i][1] for i in range(len(ds))]


# ===========================================================================
# train_cellpose
# ===========================================================================

def test_train_cellpose_builds_batch_and_calls_train_seg(tmp_path, cp_stub):
    """One full train_cellpose pass: batch build, model dir, train_seg kwargs."""
    from spacr.submodules import train_cellpose

    labels = [
        _label_image(32, [(1, (2, 10, 2, 10)), (2, (18, 28, 18, 28))]),
        _label_image(32, [(1, (4, 14, 4, 14))]),
        _label_image(32, [(1, (6, 16, 20, 30))]),
    ]
    _write_pairs(tmp_path, "train", labels)

    settings = {
        "src": str(tmp_path),
        "model_name": "mymodel",
        "n_epochs": 20,
        "target_size": 16,
        "augment": False,
        "batch_size": 2,
        "learning_rate": 0.05,
        "weight_decay": 1e-4,
    }
    train_cellpose(settings)

    # -- the model was built for the SAM checkpoint on the GPU path
    assert len(cp_stub["models"]) == 1
    assert cp_stub["models"][0].gpu is True
    assert cp_stub["models"][0].pretrained_model == "cpsam"

    # -- train_seg got the resolved settings
    assert len(cp_stub["train_calls"]) == 1
    call = cp_stub["train_calls"][0]
    assert call["net"] is _NET_SENTINEL
    assert call["channel_axis"] is None
    assert call["rescale"] is False
    assert call["n_epochs"] == 20
    assert call["batch_size"] == 2
    assert call["learning_rate"] == 0.05
    assert call["weight_decay"] == 1e-4
    assert call["save_every"] == 2                     # n_epochs // 10
    # train_cellpose fine-tunes cpsam; `_cyto_` was a Cellpose-3 leftover.
    assert call["model_name"] == "mymodel_cpsam_e20_X16_Y16.CP_model"
    assert call["save_path"] == os.path.join(str(tmp_path), "models", "cellpose_model")
    assert os.path.isdir(call["save_path"])

    # -- batch_size caps the number of base images pulled from the dataset
    assert len(call["train_data"]) == 2
    assert len(call["train_labels"]) == 2
    for img in call["train_data"]:
        assert img.shape == (16, 16)
        assert img.dtype == np.float32
        assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0 + 1e-6
    for lbl in call["train_labels"]:
        assert lbl.shape == (16, 16)
        assert lbl.dtype == np.uint16

    # -- the resolved settings were snapshotted next to the data
    saved = tmp_path / "settings" / "mymodel_cpsam_e20_X16_Y16.CP_model.csv"
    assert saved.exists()
    saved_df = pd.read_csv(saved)
    assert set(saved_df["Key"]) >= {"src", "model_name", "n_epochs", "target_size",
                                    "learning_rate", "weight_decay", "batch_size"}
    # Defaults that get_train_cellpose_default_settings injects.
    assert dict(zip(saved_df["Key"], saved_df["Value"]))["model_type"] == "cpsam"


def test_train_cellpose_augment_expands_one_base_image_to_eight(tmp_path, cp_stub):
    """augment=True turns each selected base image into its 8 dihedral variants."""
    from spacr.submodules import train_cellpose

    # Deliberately asymmetric so the 8 variants are genuinely different.
    labels = [
        _label_image(32, [(1, (1, 6, 1, 20)), (2, (24, 30, 2, 8))]),
        _label_image(32, [(1, (2, 8, 3, 26)), (2, (20, 31, 1, 6))]),
    ]
    _write_pairs(tmp_path, "train", labels)

    settings = {
        "src": str(tmp_path),
        "model_name": "aug",
        "n_epochs": 5,          # -> save_every = max(1, 0) = 1
        "target_size": 32,
        "augment": True,
        "batch_size": 1,        # only one base image survives the min()
        "learning_rate": 0.1,
        "weight_decay": 1e-5,
    }
    train_cellpose(settings)

    call = cp_stub["train_calls"][0]
    assert call["save_every"] == 1
    assert len(call["train_data"]) == 8
    assert len(call["train_labels"]) == 8

    distinct = {lbl.tobytes() for lbl in call["train_labels"]}
    assert len(distinct) >= 4, "augmentation produced near-identical labels"
    # Every variant keeps the object count of its base image.
    for lbl in call["train_labels"]:
        assert lbl.shape == (32, 32)
        assert len(np.unique(lbl)) == 3      # background + 2 objects


def test_train_cellpose_uses_only_filenames_present_in_both_folders(tmp_path, cp_stub):
    """Unpaired TIFFs and non-TIFF junk are dropped by the set intersection."""
    from spacr.submodules import train_cellpose

    labels = [_label_image(32, [(1, (4 + i, 14 + i, 4, 14))]) for i in range(3)]
    _write_pairs(
        tmp_path, "train", labels,
        extra_image_only=("orphan_image.tif",),
        extra_mask_only=("orphan_mask.tif",),
        junk_ext="notes.png",
    )

    settings = {
        "src": str(tmp_path),
        "model_name": "matched",
        "n_epochs": 10,
        "target_size": 16,
        "augment": False,
        "batch_size": 50,       # larger than the dataset -> min() clamps
        "learning_rate": 0.2,
        "weight_decay": 1e-5,
    }
    train_cellpose(settings)

    call = cp_stub["train_calls"][0]
    assert len(call["train_data"]) == 3, "unpaired / non-tif files leaked into training"
    assert len(call["train_labels"]) == 3


def test_train_cellpose_survives_a_failing_batch_plot(tmp_path, cp_stub, monkeypatch,
                                                      capsys):
    """A crash inside plot_cellpose_batch must not abort training."""
    from spacr.submodules import train_cellpose

    def _boom(images, labels):
        raise RuntimeError("no display")

    monkeypatch.setattr(SUB, "plot_cellpose_batch", _boom)

    labels = [_label_image(32, [(1, (4, 14, 4, 14))]) for _ in range(2)]
    _write_pairs(tmp_path, "train", labels)

    settings = {
        "src": str(tmp_path),
        "model_name": "plotfail",
        "n_epochs": 30,
        "target_size": 16,
        "augment": False,
        "batch_size": 2,
        "learning_rate": 0.2,
        "weight_decay": 1e-5,
    }
    train_cellpose(settings)

    out = capsys.readouterr().out
    assert "could not print batch images" in out
    assert len(cp_stub["train_calls"]) == 1, "training was skipped after a plot failure"
    assert cp_stub["train_calls"][0]["save_every"] == 3


def test_train_cellpose_plots_the_batch_it_trains_on(tmp_path, cp_stub, monkeypatch):
    """plot_cellpose_batch is handed exactly the images/labels sent to train_seg."""
    from spacr.submodules import train_cellpose

    seen = {}

    def _record(images, labels):
        seen["images"] = list(images)
        seen["labels"] = list(labels)

    monkeypatch.setattr(SUB, "plot_cellpose_batch", _record)

    labels = [_label_image(32, [(1, (4, 14, 4, 14))]) for _ in range(2)]
    _write_pairs(tmp_path, "train", labels)

    train_cellpose({
        "src": str(tmp_path), "model_name": "plotted", "n_epochs": 10,
        "target_size": 16, "augment": False, "batch_size": 2,
        "learning_rate": 0.2, "weight_decay": 1e-5,
    })

    call = cp_stub["train_calls"][0]
    assert len(seen["images"]) == 2
    assert [id(a) for a in seen["images"]] == [id(a) for a in call["train_data"]]
    assert [id(a) for a in seen["labels"]] == [id(a) for a in call["train_labels"]]


@pytest.mark.xfail(strict=True, reason=(
    "BUG: get_train_cellpose_default_settings supplies neither 'target_size' "
    "nor 'augment', so train_cellpose(settings) raises KeyError for the exact "
    "settings dict the GUI builds from that defaults function."))
def test_train_cellpose_runs_on_documented_defaults(tmp_path, cp_stub):
    """train_cellpose must work with only the keys the defaults helper defines."""
    from spacr.settings import get_train_cellpose_default_settings
    from spacr.submodules import train_cellpose

    labels = [_label_image(32, [(1, (4, 14, 4, 14))]) for _ in range(2)]
    _write_pairs(tmp_path, "train", labels)

    settings = get_train_cellpose_default_settings({})
    settings["src"] = str(tmp_path)
    settings["n_epochs"] = 2

    train_cellpose(settings)
    assert len(cp_stub["train_calls"]) == 1


# ===========================================================================
# test_cellpose_model
# ===========================================================================

def _test_settings(src, **over):
    settings = {
        "src": str(src),
        "model_path": "/nonexistent/custom_model.CP_model",
        "save": False,
        "normalize": True,
        "percentiles": (2, 98),
        "batch_size": 50,
        "CP_probability": 0.5,
        "FT": 0.6,
        "target_size": 32,
    }
    settings.update(over)
    return settings


def test_test_cellpose_model_perfect_prediction_metrics(tmp_path, cp_stub, monkeypatch):
    """Predicting the ground truth exactly gives Jaccard/precision/recall == 1."""
    from spacr.submodules import test_cellpose_model

    labels = [
        _label_image(32, [(1, (2, 10, 2, 10)), (2, (18, 28, 18, 28))]),
        _label_image(32, [(1, (2, 10, 2, 10)), (2, (18, 28, 18, 28))]),
    ]
    written = _write_pairs(tmp_path, "test", labels)
    expected = _dataset_labels(written["images"], written["masks"], 32)
    cp_stub["preds"] = [lbl.copy() for lbl in expected]

    shown = {}
    monkeypatch.setattr(SUB, "display", lambda df: shown.setdefault("df", df))

    settings = _test_settings(tmp_path, save=False)
    test_cellpose_model(settings)

    df = shown["df"]
    assert list(df.columns) == [
        "label_image", "Jaccard", "n_objects_true", "n_objects_pred",
        "mean_area_true", "mean_area_pred", "TP", "FP", "FN",
        "Precision", "Recall", "F1", "Accuracy", "n_error",
    ]
    assert df["label_image"].tolist() == written["names"]
    assert df["Jaccard"].tolist() == [1.0, 1.0]
    assert df["n_objects_true"].tolist() == [2, 2]
    assert df["n_objects_pred"].tolist() == [2, 2]
    # blocks are 8x8=64 px and 10x10=100 px -> mean 82
    assert df["mean_area_true"].tolist() == [82.0, 82.0]
    assert df["mean_area_pred"].tolist() == [82.0, 82.0]
    assert df["TP"].tolist() == [2, 2]
    assert df["FP"].tolist() == [0, 0]
    assert df["FN"].tolist() == [0, 0]
    assert df["Precision"].tolist() == [1.0, 1.0]
    assert df["Recall"].tolist() == [1.0, 1.0]
    assert df["F1"].tolist() == [1.0, 1.0]
    assert df["Accuracy"].tolist() == [1.0, 1.0]
    assert df["n_error"].tolist() == [0, 0]

    # save=False -> results dir exists but nothing written into it
    results_dir = tmp_path / "results"
    assert results_dir.is_dir()
    assert list(results_dir.iterdir()) == []

    # the custom checkpoint was handed to CellposeModel, not 'cpsam'
    assert cp_stub["models"][0].pretrained_model == settings["model_path"]


def test_test_cellpose_model_forwards_eval_parameters(tmp_path, cp_stub, monkeypatch):
    """FT / CP_probability and the fixed eval tuning reach model.eval."""
    from spacr.submodules import test_cellpose_model

    labels = [_label_image(32, [(1, (2, 10, 2, 10))])]
    written = _write_pairs(tmp_path, "test", labels)
    cp_stub["preds"] = _dataset_labels(written["images"], written["masks"], 32)
    monkeypatch.setattr(SUB, "display", lambda df: None)

    test_cellpose_model(_test_settings(tmp_path, save=False, FT=0.25,
                                       CP_probability=-1.5))

    assert len(cp_stub["eval_calls"]) == 1
    call = cp_stub["eval_calls"][0]
    assert call["flow_threshold"] == 0.25
    assert call["cellprob_threshold"] == -1.5
    assert call["normalize"] is False
    assert call["diameter"] == 30
    assert call["rescale"] is None
    assert call["resample"] is True
    assert call["anisotropy"] is None
    assert call["min_size"] == 5
    assert call["augment"] is True
    assert call["tile_overlap"] == 0.2
    assert call["bsize"] == 224
    assert len(call["x"]) == 1
    assert call["x"][0].shape == (32, 32)
    assert call["x"][0].dtype == np.float32


def test_test_cellpose_model_handles_empty_masks(tmp_path, cp_stub, monkeypatch):
    """Empty predictions / empty ground truth take every zero-division branch."""
    from spacr.submodules import test_cellpose_model

    labels = [
        _label_image(32, [(1, (2, 10, 2, 10)), (2, (18, 28, 18, 28))]),
        _label_image(32, []),        # nothing at all in the ground truth
    ]
    written = _write_pairs(tmp_path, "test", labels)
    cp_stub["preds"] = [np.zeros((32, 32), dtype=np.uint16) for _ in labels]

    shown = {}
    monkeypatch.setattr(SUB, "display", lambda df: shown.setdefault("df", df))

    test_cellpose_model(_test_settings(tmp_path, save=False))

    df = shown["df"]
    assert len(df) == 2

    # Row 0: real objects, nothing predicted -> all false negatives.
    assert df.loc[0, "n_objects_true"] == 2
    assert df.loc[0, "n_objects_pred"] == 0
    assert df.loc[0, "mean_area_pred"] == 0
    assert df.loc[0, "mean_area_true"] == 82.0
    assert (df.loc[0, "TP"], df.loc[0, "FP"], df.loc[0, "FN"]) == (0, 0, 2)
    assert df.loc[0, "Precision"] == 0      # tp + fp == 0 -> guarded
    assert df.loc[0, "Recall"] == 0.0
    assert df.loc[0, "F1"] == 0
    assert df.loc[0, "Accuracy"] == 0.0
    assert df.loc[0, "Jaccard"] == 0.0
    assert df.loc[0, "n_error"] == 2

    # Row 1: empty vs empty -> every denominator is zero.
    assert df.loc[1, "n_objects_true"] == 0
    assert df.loc[1, "n_objects_pred"] == 0
    assert df.loc[1, "mean_area_true"] == 0
    assert df.loc[1, "mean_area_pred"] == 0
    assert (df.loc[1, "TP"], df.loc[1, "FP"], df.loc[1, "FN"]) == (0, 0, 0)
    assert df.loc[1, "Precision"] == 0
    assert df.loc[1, "Recall"] == 0
    assert df.loc[1, "F1"] == 0
    assert df.loc[1, "Accuracy"] == 0
    assert np.isnan(df.loc[1, "Jaccard"])   # cellpose AJI is undefined here
    assert df.loc[1, "n_error"] == 0


def test_test_cellpose_model_saves_csv_and_diagnostic_pngs(tmp_path, cp_stub,
                                                           monkeypatch):
    """save=True writes one PNG per image plus test_results.csv."""
    import matplotlib.pyplot as plt
    from spacr.submodules import test_cellpose_model

    labels = [
        _label_image(32, [(1, (2, 10, 2, 10))]),
        _label_image(32, [(1, (4, 12, 4, 12)), (2, (20, 30, 20, 30))]),
    ]
    written = _write_pairs(tmp_path, "test", labels)
    cp_stub["preds"] = _dataset_labels(written["images"], written["masks"], 32)
    monkeypatch.setattr(SUB, "display", lambda df: None)

    test_cellpose_model(_test_settings(tmp_path, save=True))

    results_dir = tmp_path / "results"
    for idx in (0, 1):
        png = results_dir / f"cellpose_result_{idx:03d}.png"
        assert png.exists(), f"missing diagnostic {png.name}"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    csv = results_dir / "test_results.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert len(df) == 2
    assert df["label_image"].tolist() == written["names"]
    assert df["n_objects_true"].tolist() == [1, 2]
    assert df["Jaccard"].tolist() == [1.0, 1.0]

    # Every diagnostic figure was closed again.
    assert plt.get_fignums() == []


def test_test_cellpose_model_multi_batch_reports_every_image(tmp_path, cp_stub,
                                                             monkeypatch):
    """With 4 images and batch_size=2 the CSV must still hold all 4 rows."""
    from spacr.submodules import test_cellpose_model

    labels = [_label_image(32, [(1, (2 + i, 10 + i, 2, 10))]) for i in range(4)]
    written = _write_pairs(tmp_path, "test", labels)
    cp_stub["preds"] = _dataset_labels(written["images"], written["masks"], 32)
    monkeypatch.setattr(SUB, "display", lambda df: None)

    test_cellpose_model(_test_settings(tmp_path, save=True, batch_size=2))

    df = pd.read_csv(tmp_path / "results" / "test_results.csv")
    assert len(df) == 4
    assert df["label_image"].tolist() == written["names"]


def test_test_cellpose_model_progress_total_is_the_image_count(tmp_path, cp_stub,
                                                               monkeypatch):
    from spacr.submodules import test_cellpose_model
    import spacr.utils as UTILS

    calls = []
    monkeypatch.setattr(
        UTILS, "print_progress",
        lambda files_processed, files_to_process, **kw: calls.append(
            (files_processed, files_to_process)),
    )

    labels = [_label_image(32, [(1, (2, 10, 2, 10))]) for _ in range(3)]
    written = _write_pairs(tmp_path, "test", labels)
    cp_stub["preds"] = _dataset_labels(written["images"], written["masks"], 32)
    monkeypatch.setattr(SUB, "display", lambda df: None)

    test_cellpose_model(_test_settings(tmp_path, save=False))

    assert len(calls) == 1
    assert calls[0][1] == 3


def test_test_cellpose_model_progress_counts_processed_images(tmp_path, cp_stub,
                                                              monkeypatch):
    from spacr.submodules import test_cellpose_model
    import spacr.utils as UTILS

    calls = []
    monkeypatch.setattr(
        UTILS, "print_progress",
        lambda files_processed, files_to_process, **kw: calls.append(
            (files_processed, files_to_process)),
    )

    labels = [_label_image(32, [(1, (2, 10, 2, 10))]) for _ in range(3)]
    written = _write_pairs(tmp_path, "test", labels)
    cp_stub["preds"] = _dataset_labels(written["images"], written["masks"], 32)
    monkeypatch.setattr(SUB, "display", lambda df: None)

    test_cellpose_model(_test_settings(tmp_path, save=False, batch_size=50))

    assert len(calls) == 1
    assert calls[0][0] == 3


def test_test_cellpose_model_renders_each_diagnostic_once(tmp_path, cp_stub,
                                                          monkeypatch):
    from spacr.submodules import test_cellpose_model

    saved = []
    monkeypatch.setattr(SUB.plt, "savefig",
                        lambda path, **kw: saved.append(str(path)))

    labels = [_label_image(32, [(1, (2, 10, 2, 10))])]
    written = _write_pairs(tmp_path, "test", labels)
    cp_stub["preds"] = _dataset_labels(written["images"], written["masks"], 32)
    monkeypatch.setattr(SUB, "display", lambda df: None)

    test_cellpose_model(_test_settings(tmp_path, save=True))

    assert len(saved) == 1


def test_test_cellpose_model_only_scores_matched_filenames(tmp_path, cp_stub,
                                                           monkeypatch):
    """Images without a mask (and vice versa) are excluded from the report."""
    from spacr.submodules import test_cellpose_model

    labels = [_label_image(32, [(1, (2, 10, 2, 10))]) for _ in range(2)]
    written = _write_pairs(
        tmp_path, "test", labels,
        extra_image_only=("only_image.tif",),
        extra_mask_only=("only_mask.tif",),
        junk_ext="readme.png",
    )
    cp_stub["preds"] = _dataset_labels(written["images"], written["masks"], 32)

    shown = {}
    monkeypatch.setattr(SUB, "display", lambda df: shown.setdefault("df", df))

    test_cellpose_model(_test_settings(tmp_path, save=False))

    assert cp_stub["n_predicted"] == 2, "unpaired files were fed to the model"
    assert shown["df"]["label_image"].tolist() == written["names"]


def test_test_cellpose_model_writes_settings_snapshot(tmp_path, cp_stub, monkeypatch):
    """The resolved settings are persisted before any inference happens."""
    from spacr.submodules import test_cellpose_model

    labels = [_label_image(32, [(1, (2, 10, 2, 10))])]
    written = _write_pairs(tmp_path, "test", labels)
    cp_stub["preds"] = _dataset_labels(written["images"], written["masks"], 32)
    monkeypatch.setattr(SUB, "display", lambda df: None)

    settings = {"src": str(tmp_path), "model_path": "cp.CP_model",
                "save": False, "target_size": 32, "batch_size": 10}
    test_cellpose_model(settings)

    snapshot = tmp_path / "settings" / "test_cellpose_model.csv"
    assert snapshot.exists()
    values = dict(zip(*pd.read_csv(snapshot)[["Key", "Value"]].values.T))
    assert values["model_path"] == "cp.CP_model"
    # defaults filled in by get_default_test_cellpose_model_settings
    assert values["FT"] == "100"
    assert values["normalize"] == "True"
    assert settings["CP_probability"] == 0
