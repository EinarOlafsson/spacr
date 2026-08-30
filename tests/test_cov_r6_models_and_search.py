"""Round-6 coverage for the training/report helpers of ``spacr.deep_spacr``
and the sweep helpers of ``spacr.hyperparam``.

One theme: what a model report or a hyperparameter sweep does when a piece
of the usual input is missing or turned off.

Pinned here:

* ``deep_spacr.format_per_class_accuracy`` names the worst class only when
  there is more than one class to be worst *of*.
* ``deep_spacr._print_cv_report`` prints the fold-to-fold spread even for a
  run whose summary carries no accuracy row.
* ``deep_spacr._log_tensorboard_epoch`` writes a scalar per finite class
  accuracy and skips a non-finite one.
* ``deep_spacr._dataset_class_counts`` skips a split folder holding no class
  folders, and ``_imbalance_note`` skips a class counted at zero.
* ``deep_spacr.write_model_card`` writes the Markdown twin only on request,
  and ``model_card`` still returns a card and its path when the registry
  refuses the row.
* ``deep_spacr.generate_activation_map`` applies no Normalize step under
  ``input_statistics='none'``, and writes correlations to the database with
  the on-screen table turned off.
* ``hyperparam.walk_search`` resumes a checkpoint whose stored centre names
  only some of the axes, and one written before the walk went N-dimensional
  that names none of them.
* ``hyperparam.build_folds`` accepts an empty exclusion list,
  ``format_search`` prints the noise yardstick only when there is one, and
  ``build_sklearn_model`` builds the XGBoost estimator.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import deep_spacr as D  # noqa: E402
from spacr import hyperparam as hp  # noqa: E402


# ===========================================================================
# spacr.deep_spacr -- per-class reporting
# ===========================================================================

def test_the_worst_class_is_only_named_when_there_is_more_than_one():
    """deep_spacr.py:801 -- ``if len(finite) > 1`` False.

    A class whose accuracy is not finite has no place in a "worst class"
    comparison, so a two-class head with one NaN leaves only one comparable
    class and the line stops at the breakdown. The same head with both
    accuracies finite and far apart does name the worst one.
    """
    one_finite = {"per_class_accuracy": [float("nan"), 0.5],
                  "class_support": [0, 20]}
    both_finite = {"per_class_accuracy": [0.40, 0.95],
                   "class_support": [10, 20]}

    line_one = D.format_per_class_accuracy(one_finite, ["neg", "pos"])
    line_both = D.format_per_class_accuracy(both_finite, ["neg", "pos"])

    assert "pos 0.500 (n=20)" in line_one
    assert "WORST" not in line_one
    assert "WORST: neg at 0.400" in line_both
    assert "0.550 below the best class" in line_both


def test_a_cv_summary_without_an_accuracy_row_still_prints_the_spread(capsys):
    """deep_spacr.py:1166 -- ``if not acc.empty`` False, exiting the report.

    A search that scored folds on f1 alone has no ``accuracy`` row. The
    per-fold table and the spread table are still the report; only the
    accuracy sentence is dropped.
    """
    fold_df = pd.DataFrame({"fold": [0, 1], "f1_macro": [0.7, 0.9]})
    no_acc = pd.DataFrame({"metric": ["f1_macro"], "mean": [0.8],
                           "std": [0.1], "min": [0.7], "max": [0.9]})
    with_acc = pd.DataFrame({"metric": ["accuracy"], "mean": [0.8],
                             "std": [0.1], "min": [0.7], "max": [0.9]})

    D._print_cv_report(fold_df, no_acc, 2)
    without = capsys.readouterr().out
    D._print_cv_report(fold_df, with_acc, 2)
    withit = capsys.readouterr().out

    assert "Fold-to-fold spread" in without and "f1_macro" in without
    assert "accuracy across folds" not in without
    assert "accuracy across folds: 0.8000 +/- 0.1000" in withit


class _RecordingWriter:
    """The two SummaryWriter calls ``_log_tensorboard_epoch`` makes."""

    def __init__(self):
        self.scalars = {}
        self.flushed = 0

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = (float(value), int(step))

    def flush(self):
        self.flushed += 1


def test_a_class_with_no_finite_accuracy_gets_no_scalar():
    """deep_spacr.py:2134 -- ``if np.isfinite(acc)`` False, back to the loop.

    A class with no support in a split has an undefined accuracy, and a NaN
    written into TensorBoard draws a gap that reads as a dead run. The class
    beside it, which does have an accuracy, still gets its scalar.
    """
    writer = _RecordingWriter()
    train = {"loss": 0.3, "accuracy": 0.8,
             "per_class_accuracy": [float("nan"), 0.9],
             "class_support": [0, 30]}

    D._log_tensorboard_epoch(writer, train, {}, epoch=4,
                             classes=["absent", "present"])

    assert "accuracy_present/train" in writer.scalars
    assert writer.scalars["accuracy_present/train"] == (0.9, 4)
    assert "accuracy_absent/train" not in writer.scalars
    assert writer.scalars["loss/train"] == (0.3, 4)
    assert writer.flushed == 1


# ===========================================================================
# spacr.deep_spacr -- class balance
# ===========================================================================

def test_a_split_folder_with_no_class_folders_is_not_a_split(tmp_path):
    """deep_spacr.py:2225 -- ``if counts:`` False, on to the next split.

    ``test/`` exists but was never populated, so it contributes no counts at
    all rather than an empty ``{}`` that downstream code would read as "this
    split has zero of everything". ``train/`` in the same tree still counts.
    """
    for name in ("a", "b"):
        d = tmp_path / "train" / name
        d.mkdir(parents=True)
        (d / "1.png").write_bytes(b"x")
    (tmp_path / "test").mkdir()

    counts = D.dataset_class_balance(str(tmp_path))

    assert counts == {"train": {"a": 1, "b": 1}}
    assert "test" not in counts


def test_a_class_counted_at_zero_is_not_a_class():
    """deep_spacr.py:2259 -- ``if count > 0`` False.

    A class with no examples cannot be the smallest class in an imbalance
    sentence -- the ratio would be infinite. It is dropped, so the note
    compares the two classes that are actually there.
    """
    note = D._imbalance_note({"gone": 0, "big": 900, "small": 100})
    assert "big" in note and "small" in note
    assert "gone" not in note
    # ... and with only one surviving class there is nothing to compare.
    assert D._imbalance_note({"gone": 0, "big": 900}) == ""


# ===========================================================================
# spacr.deep_spacr -- the model card
# ===========================================================================

def test_the_markdown_twin_is_written_only_when_asked(tmp_path):
    """deep_spacr.py:2455 -- ``if markdown:`` False.

    The JSON card is the record; the Markdown is the human-readable twin.
    Both calls write the JSON to the same path, and only the second leaves a
    ``.card.md`` beside it.
    """
    model_path = tmp_path / "weights.pth"
    card = D.build_model_card(str(model_path), classes=["a", "b"],
                              split_rule="grouped by well")

    json_path = D.write_model_card(str(model_path), card, markdown=False)
    assert os.path.isfile(json_path)
    assert not os.path.isfile(str(tmp_path / ("weights" + D.MODEL_CARD_MD_SUFFIX)))

    D.write_model_card(str(model_path), card, markdown=True)
    md = str(tmp_path / ("weights" + D.MODEL_CARD_MD_SUFFIX))
    assert os.path.isfile(md)
    assert "grouped by well" in open(md).read()


def test_a_registry_that_refuses_the_row_still_leaves_the_card(tmp_path,
                                                               capsys):
    """deep_spacr.py:2548 -- ``if artifact is not None`` False.

    Losing the registry must not lose the weights or the card. With a
    registry object that cannot register, the card is on disk, the artifact
    is None and the card carries no ``artifact_id`` -- against a real
    registry it carries one.
    """
    model_path = tmp_path / "refused" / "weights.pth"
    model_path.parent.mkdir()

    class _Broken:
        def register(self, **kwargs):
            raise RuntimeError("registry is read-only")

    card, card_path, artifact = D.model_card(
        str(model_path), registry=_Broken(), classes=["a", "b"])

    assert artifact is None
    assert "artifact_id" not in card
    assert os.path.isfile(card_path)
    assert "not registered" in capsys.readouterr().out

    ok_card, _ok_path, ok_artifact = D.model_card(
        str(tmp_path / "kept" / "weights.pth"), classes=["a", "b"])
    assert ok_artifact is not None
    assert ok_card["artifact_id"] == ok_artifact.artifact_id


# ===========================================================================
# spacr.hyperparam -- resuming a walk
# ===========================================================================

def _axes():
    return [hp.WalkAxis("n_neighbors", step=5.0, minimum=2.0, integer=True),
            hp.WalkAxis("min_dist", step=0.05, minimum=0.0, maximum=1.0)]


def _store(tmp_path, *, resume=False):
    return hp._UmapCheckpoint(str(tmp_path / "walk.json"),
                              {"features": "matrix-a"}, resume, False)


def test_a_stored_centre_that_names_only_one_axis_moves_only_that_one(
        tmp_path):
    """hyperparam.py:1556 -- ``if axis.name in stored`` False.

    A checkpoint written by a build with a different axis set names some of
    this walk's axes and not others. The named one is restored; the unnamed
    one keeps the caller's starting value rather than being reset or raising.
    """
    store = _store(tmp_path)
    store.record(hp.Trial(params={"n_neighbors": 30, "min_dist": 0.05},
                          score=0.5, index=0), round_index=0)
    store.finish({"rounds_completed": 1,
                  "centre": {"n_neighbors": 30, "an_axis_we_dropped": 7},
                  "best_score": 0.5})

    seen = []
    hp.walk_search(lambda params: (seen.append(dict(params)), 0.4)[1],
                   {"n_neighbors": 5, "min_dist": 0.5}, _axes(),
                   n_trials=2, checkpoint=_store(tmp_path, resume=True))

    assert seen, "the resumed walk still has trials to run"
    # n_neighbors came from the checkpoint (30 +/- 5), min_dist from the call.
    assert {row["n_neighbors"] for row in seen} <= {25, 30, 35}
    assert {round(row["min_dist"], 3) for row in seen} <= {0.45, 0.5, 0.55}


def test_a_legacy_checkpoint_that_names_no_axis_leaves_the_centre_alone(
        tmp_path):
    """hyperparam.py:1564 -- ``if value is not None`` False.

    The pre-N-dimensional reader looks for ``centre_n``/``centre_d``. A
    checkpoint that carries neither leaves every axis at the caller's
    starting point, and still reports that it resumed.
    """
    store = _store(tmp_path)
    store.record(hp.Trial(params={"n_neighbors": 10, "min_dist": 0.1},
                          score=0.5, index=0), round_index=0)
    store.finish({"rounds_completed": 1, "centre": "not a mapping",
                  "best_score": 0.5})

    seen = []
    result = hp.walk_search(
        lambda params: (seen.append(dict(params)), 0.4)[1],
        {"n_neighbors": 20, "min_dist": 0.5}, _axes(),
        n_trials=2, checkpoint=_store(tmp_path, resume=True))

    assert any("Resumed" in note for note in result.notes)
    assert {row["n_neighbors"] for row in seen} <= {15, 20, 25}


# ===========================================================================
# spacr.hyperparam -- folds, report, estimator
# ===========================================================================

def test_an_empty_exclusion_list_excludes_nothing():
    """hyperparam.py:3037 -- ``if ex.size`` False.

    ``exclude=[]`` is not ``exclude=None``: it still takes the branch that
    would range-check the indices, and must leave every sample in the pool.
    A non-empty exclusion in the same test does remove its rows.
    """
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    groups = ["A1", "A1", "A2", "A2", "B1", "B1", "B2", "B2"]

    folds_empty, _ = hp.build_folds(labels, n_folds=2, groups=groups,
                                    exclude=[])
    folds_some, _ = hp.build_folds(labels, n_folds=2, groups=groups,
                                   exclude=[0, 4])

    used_empty = set()
    for train_idx, val_idx in folds_empty:
        used_empty |= set(np.asarray(train_idx).tolist())
        used_empty |= set(np.asarray(val_idx).tolist())
    used_some = set()
    for train_idx, val_idx in folds_some:
        used_some |= set(np.asarray(train_idx).tolist())
        used_some |= set(np.asarray(val_idx).tolist())

    assert used_empty == set(range(8))
    assert used_some == set(range(8)) - {0, 4}


def _result(n, with_fold_std):
    trials = []
    for i in range(n):
        extra = {"fold_std": 0.02} if with_fold_std else {}
        trials.append(hp.Trial(params={"n_neighbors": 5 * (i + 1)},
                               score=0.4 + 0.2 * i, index=i,
                               extra_metrics=extra))
    return hp.SearchResult(trials=trials, best=trials[-1], metric="score")


def test_the_noise_yardstick_is_printed_only_when_there_is_one():
    """hyperparam.py:3310 -- ``if noise is not None``.

    The yardstick is the best trial's fold-to-fold spread, or failing that
    the spread across trials -- and a single successful trial has neither.
    Printing a bare "Noise yardstick:" with nothing behind it would read as
    zero noise, so the line is dropped entirely.
    """
    without = hp.format_search(_result(1, False))
    across = hp.format_search(_result(2, False))
    per_config = hp.format_search(_result(2, True))

    assert "Spread over 1 successful trials" in without
    assert "Noise yardstick" not in without
    assert "Noise yardstick: 0.1000 (standard deviation across trials)" \
        in across
    assert "Noise yardstick: 0.0200 (fold-to-fold standard deviation" \
        in per_config


def test_the_xgboost_estimator_is_built_with_the_trial_parameters():
    """hyperparam.py:3538 -- the ``xgboost`` arm of the model ladder."""
    pytest.importorskip("xgboost")

    model = hp.build_sklearn_model(
        "xgboost", {"n_estimators": 7, "learning_rate": 0.3,
                    "reg_alpha": 0.5, "reg_lambda": 2.0}, seed=11)

    assert type(model).__name__ == "XGBClassifier"
    params = model.get_params()
    assert params["n_estimators"] == 7
    assert params["learning_rate"] == pytest.approx(0.3)
    assert params["reg_alpha"] == pytest.approx(0.5)
    assert params["reg_lambda"] == pytest.approx(2.0)


def test_a_walk_over_other_axes_writes_no_two_axis_compatibility_keys(
        tmp_path):
    """hyperparam.py:1546 / 1548 -- ``"n_neighbors" in centre`` False.

    ``centre_n``/``centre_d`` exist only so a checkpoint from this build
    stays readable by the 1.5.x two-axis reader. A walk whose axes are not
    the UMAP pair has nothing to write there, and writing an axis under the
    wrong legacy name would resume the old reader at the wrong centre.
    """
    import json

    def _run(folder, axes, start):
        path = str(folder / "walk.json")
        store = hp._UmapCheckpoint(path, {"features": "matrix-a"}, False,
                                   False)
        hp.walk_search(lambda params: 0.5, start, axes, n_trials=2,
                       checkpoint=store)
        return json.load(open(path))["meta"]

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    umap_dir = tmp_path / "umap"
    umap_dir.mkdir()

    other = _run(other_dir,
                 [hp.WalkAxis("alpha", step=1.0, minimum=0.0),
                  hp.WalkAxis("beta", step=0.5, minimum=0.0)],
                 {"alpha": 2.0, "beta": 1.0})
    umap = _run(umap_dir, _axes(), {"n_neighbors": 10, "min_dist": 0.1})

    assert set(other["centre"]) == {"alpha", "beta"}
    assert "centre_n" not in other and "centre_d" not in other
    # The pair that DOES get the compatibility keys, in the same test.
    assert umap["centre_n"] == umap["centre"]["n_neighbors"]
    assert umap["centre_d"] == umap["centre"]["min_dist"]


# ===========================================================================
# spacr.deep_spacr.generate_activation_map -- the two switched-off arms
# ===========================================================================

torch = pytest.importorskip("torch")


def _activation_project(tmp_path, n_images=4):
    """``<root>/datasets/ds.tar`` of tiny PNGs plus an empty measurements dir."""
    import tarfile
    from PIL import Image

    root = tmp_path / "proj"
    (root / "measurements").mkdir(parents=True)
    ds_dir = root / "datasets"
    ds_dir.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    names = []
    for i in range(n_images):
        name = f"plate1_A01_1_{i}.png"
        rng = np.random.default_rng(i)
        arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint16).astype(np.uint8)
        Image.fromarray(arr).save(raw / name)
        names.append(name)
    tar_path = ds_dir / "ds.tar"
    with tarfile.open(tar_path, "w") as tar:
        for name in names:
            tar.add(raw / name, arcname=name)
    return root, str(tar_path)


def _activation_model(tmp_path):
    from spacr.utils import TorchModel
    p = tmp_path / "binary_model.pth"
    torch.manual_seed(0)
    model = TorchModel(model_name="resnet18", pretrained=False,
                       num_classes=1, image_size=32)
    torch.save(model, str(p))
    return str(p)


def _activation_settings(tar_path, model_path, **over):
    s = {
        "dataset": tar_path, "model_path": model_path,
        "model_type": "resnet18", "cam_type": "saliency_image",
        "target_layer": None, "image_size": 32, "batch_size": 2,
        "channels": [1, 2, 3], "normalize": False, "normalize_input": True,
        "save": False, "plot": False, "correlation": False, "overlay": True,
        "shuffle": False, "n_jobs": 0, "manders_thresholds": [15, 50, 75],
    }
    s.update(over)
    return s


def test_input_statistics_none_leaves_the_pixels_unnormalised(tmp_path):
    """deep_spacr.py:3196 -- ``if stats is not None`` False.

    ``normalize_input`` stays the on/off it always was; ``input_statistics``
    says WHICH statistics, and ``'none'`` means no Normalize step at all.
    The model then sees different inputs, so the saliency it reports is a
    different map -- which is the whole reason the setting exists.
    """
    from spacr.deep_spacr import generate_activation_map
    from PIL import Image

    model_path = _activation_model(tmp_path)

    def _maps(tag, statistics):
        root, tar = _activation_project(tmp_path / tag)
        generate_activation_map(_activation_settings(
            tar, model_path, save=True, input_statistics=statistics))
        out = sorted((root / "datasets" / "ds" / "saliency_image")
                     .rglob("*.png"))
        return [np.array(Image.open(p)) for p in out]

    plain = _maps("none", "none")
    imagenet = _maps("imagenet", "imagenet")

    assert len(plain) == 4 and len(imagenet) == 4
    assert all(m.dtype == np.uint8 and m.shape == (32, 32) for m in plain)
    assert any(not np.array_equal(a, b) for a, b in zip(plain, imagenet)), \
        "dropping the Normalize step must change what the model was shown"


def test_correlations_can_be_computed_without_being_stored_or_shown(tmp_path):
    """deep_spacr.py:3332 / 3334 -- ``plot`` and ``save`` both False.

    ``correlation=True`` with both outputs off still walks every batch and
    computes the table; it simply keeps nothing. The same settings with
    ``save=True`` do write the correlations table, so the absence is a
    choice and not a run that stopped early.
    """
    from spacr.deep_spacr import generate_activation_map
    import sqlite3

    model_path = _activation_model(tmp_path)

    quiet_root, quiet_tar = _activation_project(tmp_path / "quiet")
    generate_activation_map(_activation_settings(
        quiet_tar, model_path, correlation=True, save=False, plot=False))

    quiet_maps = quiet_root / "datasets" / "ds" / "saliency_image"
    assert not list(quiet_maps.rglob("*.png"))
    assert not (quiet_root / "measurements" / "ds.db").exists()

    kept_root, kept_tar = _activation_project(tmp_path / "kept")
    generate_activation_map(_activation_settings(
        kept_tar, model_path, correlation=True, save=True, plot=False))

    db = kept_root / "measurements" / "ds.db"
    assert db.is_file()
    con = sqlite3.connect(str(db))
    try:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()
    assert "saliency_image_correlations" in tables
