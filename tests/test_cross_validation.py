"""Cross-validation mode: k folds, group awareness, and per-fold spread.

What is actually being defended here:

  * the k folds *partition* the data -- every crop is validated exactly once,
    so "k-fold" is not a k-times-repeated random split;
  * no group id spans two folds. Crops from the same well share focus,
    illumination and seeding density, so a well straddling the split lets the
    model recognise the well instead of the phenotype and inflates every fold;
  * folds are class-stratified, including under grouping;
  * the report carries per-fold numbers *and* a spread statistic, because the
    whole point of k-fold is seeing how lucky a single split could have been;
  * k = 0 and k = 1 fall back to the single split rather than erroring;
  * a class too rare to reach every fold is reported, not crashed on.

All splitting is pure index arithmetic, and the loader tests use 8x8 PNGs, so
the file runs on the CPU in seconds.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spacr import io as IO


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _png(path, rng, size=8):
    Image.fromarray(rng.integers(0, 255, (size, size, 3)).astype(np.uint8)).save(path)
    return str(path)


def _crop_tree(root, rng, per_class=(24, 24), classes=("nc", "pc"),
               n_wells=6, split="train"):
    """Write crops named ``plate1_<well>_<field>_<object>.png``.

    Each class is spread over ``n_wells`` wells so grouping has something to
    hold onto, and the well index is recoverable from the filename.
    """
    for ci, (cls, n) in enumerate(zip(classes, per_class)):
        d = root / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            well = f"A{(i % n_wells) + 1:02d}"
            field = (i % 2) + 1
            _png(d / f"plate1_{well}_{field}_{ci}{i:03d}.png", rng)
    return root


def _labels(n_per_class):
    out = []
    for c, n in enumerate(n_per_class):
        out.extend([c] * n)
    return out


# ---------------------------------------------------------------------------
# make_cv_folds -- the partition property
# ---------------------------------------------------------------------------

def test_k_folds_cover_every_sample_exactly_once():
    labels = _labels([30, 30])
    folds = IO.make_cv_folds(labels, 5, seed=0)
    assert len(folds) == 5
    seen = np.concatenate([v for _, v in folds])
    assert sorted(seen.tolist()) == list(range(60))
    assert len(set(seen.tolist())) == 60


def test_each_fold_train_is_the_complement_of_its_validation():
    labels = _labels([17, 13])
    folds = IO.make_cv_folds(labels, 3, seed=1)
    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(set(val_idx))
        assert sorted(train_idx.tolist() + val_idx.tolist()) == list(range(30))


def test_folds_are_class_stratified():
    """A 5-fold split of 50/10 puts ~2 minority samples in every fold."""
    labels = _labels([50, 10])
    folds = IO.make_cv_folds(labels, 5, seed=0)
    y = np.asarray(labels)
    minority_per_fold = [int((y[v] == 1).sum()) for _, v in folds]
    assert sum(minority_per_fold) == 10
    assert set(minority_per_fold) == {2}


def test_folds_are_deterministic_for_a_seed_and_vary_across_seeds():
    labels = _labels([20, 20])
    a = IO.make_cv_folds(labels, 4, seed=7)
    b = IO.make_cv_folds(labels, 4, seed=7)
    c = IO.make_cv_folds(labels, 4, seed=8)
    assert [v.tolist() for _, v in a] == [v.tolist() for _, v in b]
    assert [v.tolist() for _, v in a] != [v.tolist() for _, v in c]


def test_fold_sizes_are_within_one_of_each_other():
    """No fold quietly collects every class remainder."""
    labels = _labels([23, 19])
    folds = IO.make_cv_folds(labels, 4, seed=3)
    sizes = sorted(len(v) for _, v in folds)
    assert sizes[-1] - sizes[0] <= 2


# ---------------------------------------------------------------------------
# make_cv_folds -- grouping
# ---------------------------------------------------------------------------

def test_no_group_appears_in_two_folds():
    """The leak this whole feature exists to prevent."""
    labels = _labels([40, 40])
    groups = [f"well_{i % 10}" for i in range(80)]
    folds = IO.make_cv_folds(labels, 5, groups=groups, seed=0)
    g = np.asarray(groups)
    fold_of_group = {}
    for f, (_, val_idx) in enumerate(folds):
        for gid in set(g[val_idx].tolist()):
            assert gid not in fold_of_group, (
                f"group {gid} is in fold {fold_of_group[gid]} and fold {f}")
            fold_of_group[gid] = f
    assert len(fold_of_group) == 10


def test_grouped_folds_keep_train_and_validation_groups_disjoint():
    labels = _labels([30, 30])
    groups = [f"w{i % 12}" for i in range(60)]
    folds = IO.make_cv_folds(labels, 4, groups=groups, seed=0)
    g = np.asarray(groups)
    for train_idx, val_idx in folds:
        assert set(g[train_idx]).isdisjoint(set(g[val_idx]))


def test_grouped_folds_still_stratify_by_class():
    """Groups are placed to keep each class evenly spread across folds."""
    # 12 wells; wells 0-7 are class 0, wells 8-11 are class 1
    labels, groups = [], []
    for w in range(12):
        cls = 0 if w < 8 else 1
        for _ in range(10):
            labels.append(cls)
            groups.append(f"w{w}")
    folds = IO.make_cv_folds(labels, 4, groups=groups, seed=0)
    y = np.asarray(labels)
    minority = [int((y[v] == 1).sum()) for _, v in folds]
    # 40 minority samples in 4 whole wells -> exactly one well per fold
    assert minority == [10, 10, 10, 10]


def test_grouped_folds_partition_the_data_too():
    labels = _labels([25, 25])
    groups = [f"w{i % 7}" for i in range(50)]
    folds = IO.make_cv_folds(labels, 3, groups=groups, seed=2)
    seen = np.concatenate([v for _, v in folds])
    assert sorted(seen.tolist()) == list(range(50))


def test_more_folds_than_groups_is_an_actionable_error():
    labels = _labels([10, 10])
    groups = ["w0"] * 10 + ["w1"] * 10
    with pytest.raises(ValueError) as e:
        IO.make_cv_folds(labels, 5, groups=groups)
    msg = str(e.value)
    assert "2 distinct group" in msg
    assert "cv_group_by='field'" in msg


def test_mismatched_group_length_is_rejected():
    with pytest.raises(ValueError, match="groups has 3 entries"):
        IO.make_cv_folds([0, 1, 0, 1], 2, groups=["a", "b", "c"])


# ---------------------------------------------------------------------------
# make_cv_folds -- guard rails
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [0, 1, -3])
def test_fewer_than_two_folds_is_rejected_by_the_splitter(k):
    with pytest.raises(ValueError, match="must be >= 2"):
        IO.make_cv_folds([0, 1, 0, 1], k)


def test_more_folds_than_samples_is_rejected():
    with pytest.raises(ValueError, match="cannot build 5 folds from 3 samples"):
        IO.make_cv_folds([0, 1, 0], 5)


def test_a_class_too_rare_to_reach_every_fold_is_split_not_crashed():
    """2 minority samples across 5 folds: two folds get one, three get none."""
    labels = _labels([50, 2])
    folds = IO.make_cv_folds(labels, 5, seed=0)
    y = np.asarray(labels)
    per_fold = [int((y[v] == 1).sum()) for _, v in folds]
    assert sum(per_fold) == 2
    assert sorted(per_fold) == [0, 0, 0, 1, 1]


# ---------------------------------------------------------------------------
# fold reporting
# ---------------------------------------------------------------------------

def test_fold_table_lists_sizes_and_per_class_validation_counts():
    labels = _labels([20, 20])
    folds = IO.make_cv_folds(labels, 4, seed=0)
    table = IO.summarize_cv_folds(labels, folds, classes=["nc", "pc"])
    assert list(table["fold"]) == [1, 2, 3, 4]
    assert table["n_val"].sum() == 40
    assert (table["n_train"] + table["n_val"] == 40).all()
    assert table["val_nc"].sum() == 20 and table["val_pc"].sum() == 20


def test_fold_table_counts_distinct_groups_per_fold():
    labels = _labels([20, 20])
    groups = [f"w{i % 8}" for i in range(40)]
    folds = IO.make_cv_folds(labels, 4, groups=groups, seed=0)
    table = IO.summarize_cv_folds(labels, folds, classes=["nc", "pc"],
                                  groups=groups)
    assert table["val_groups"].sum() == 8      # every well counted once


def test_report_warns_when_a_class_is_missing_from_a_fold(capsys):
    labels = _labels([50, 2])
    folds = IO.make_cv_folds(labels, 5, seed=0)
    table, warnings = IO.report_cv_folds(labels, folds, classes=["nc", "pc"],
                                         groups=None, group_by="none")
    out = capsys.readouterr().out
    assert (table["val_classes_missing"] == "pc").sum() == 3
    assert any("no validation samples for class(es) pc" in w for w in warnings)
    assert "undefined in this fold" in out


def test_report_warns_loudly_when_folds_are_not_group_aware(capsys):
    labels = _labels([10, 10])
    folds = IO.make_cv_folds(labels, 2, seed=0)
    _, warnings = IO.report_cv_folds(labels, folds, classes=["nc", "pc"],
                                     group_by="none")
    assert any("NOT group-aware" in w for w in warnings)
    assert "leaks and inflates" in capsys.readouterr().out


def test_grouped_report_raises_no_leak_warning():
    labels = _labels([20, 20])
    groups = [f"w{i % 8}" for i in range(40)]
    folds = IO.make_cv_folds(labels, 4, groups=groups, seed=0)
    _, warnings = IO.report_cv_folds(labels, folds, classes=["nc", "pc"],
                                     groups=groups, group_by="well",
                                     verbose=False)
    assert not any("NOT group-aware" in w for w in warnings)


# ---------------------------------------------------------------------------
# group ids from filenames
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level,expected", [
    ("plate", "plate1"),
    ("well", "plate1_A01"),
    ("field", "plate1_A01_2"),
])
def test_group_id_is_parsed_from_the_crop_filename(level, expected):
    assert IO._png_group_id("/data/plate1_A01_2_17.png", level) == expected


def test_group_id_returns_none_when_the_name_is_too_short():
    assert IO._png_group_id("lonely.png", "well") is None
    assert IO._png_group_id("plate1_A01.png", "field") is None
    assert IO._png_group_id("plate1_A01.png", "well") == "plate1_A01"


def test_group_id_rejects_an_unknown_level():
    with pytest.raises(ValueError, match="not one of"):
        IO._png_group_id("plate1_A01_2_1.png", "row")


def test_unparseable_names_become_their_own_group_and_are_reported(capsys):
    names = ["plate1_A01_1_1.png", "plate1_A01_1_2.png", "weird.png"]
    ids, unparsed = IO._cv_group_ids(names, "well")
    assert unparsed == 1
    assert ids[0] == ids[1] == "plate1_A01"
    assert ids[2] == "weird"
    out = capsys.readouterr().out
    assert "do not encode a well" in out
    assert "independence is not enforced" in out


def test_group_by_none_returns_no_groups():
    assert IO._cv_group_ids(["a.png"], "none") == (None, 0)


def test_cv_group_ids_rejects_an_unknown_level():
    with pytest.raises(ValueError, match="cv_group_by"):
        IO._cv_group_ids(["a.png"], "column")


# ---------------------------------------------------------------------------
# generate_cv_loaders
# ---------------------------------------------------------------------------

def test_cv_loaders_build_one_pair_per_fold_covering_every_crop(tmp_path, rng):
    _crop_tree(tmp_path, rng, per_class=(24, 24), n_wells=6)
    fold_loaders, info = IO.generate_cv_loaders(
        str(tmp_path), n_splits=3, image_size=8, batch_size=4,
        classes=["nc", "pc"], n_jobs=0, group_by="well")

    assert len(fold_loaders) == 3
    total_val = sum(len(v.dataset) for _, v in fold_loaders)
    assert total_val == 48
    for train_loader, val_loader in fold_loaders:
        assert train_loader.num_workers == 0
        assert val_loader.num_workers == 0
        assert len(train_loader.dataset) + len(val_loader.dataset) == 48


def test_cv_loaders_never_put_a_well_in_two_folds(tmp_path, rng):
    _crop_tree(tmp_path, rng, per_class=(24, 24), n_wells=6)
    fold_loaders, info = IO.generate_cv_loaders(
        str(tmp_path), n_splits=3, image_size=8, batch_size=4,
        classes=["nc", "pc"], n_jobs=0, group_by="well")

    seen = {}
    for f, (_, val_loader) in enumerate(fold_loaders):
        for name in IO.dataset_filenames(val_loader.dataset):
            gid = IO._png_group_id(name, "well")
            assert seen.setdefault(gid, f) == f, (
                f"well {gid} appears in fold {seen[gid]} and fold {f}")
    assert len(seen) == 6


def test_cv_loaders_do_not_resample_the_validation_fold(tmp_path, rng):
    from torch.utils.data import WeightedRandomSampler

    _crop_tree(tmp_path, rng, per_class=(36, 12), n_wells=6)
    fold_loaders, _ = IO.generate_cv_loaders(
        str(tmp_path), n_splits=3, image_size=8, batch_size=4,
        classes=["nc", "pc"], n_jobs=0, group_by="well",
        class_balance="weighted_sampler")
    for train_loader, val_loader in fold_loaders:
        assert isinstance(train_loader.sampler, WeightedRandomSampler)
        assert not isinstance(val_loader.sampler, WeightedRandomSampler)
        assert list(val_loader.sampler) == list(range(len(val_loader.dataset)))


def test_cv_loaders_report_folds_and_skew(tmp_path, rng, capsys):
    _crop_tree(tmp_path, rng, per_class=(36, 12), n_wells=6)
    IO.generate_cv_loaders(str(tmp_path), n_splits=3, image_size=8, batch_size=4,
                           classes=["nc", "pc"], n_jobs=0, group_by="well")
    out = capsys.readouterr().out
    assert "Cross-validation folds (k=3, grouping=well)" in out
    assert "Grouping folds by well: 6 distinct well(s)" in out
    assert "Class balance (train, n=48)" in out
    assert "imbalance ratio" in out


def test_cv_loaders_with_grouping_off_warn_about_the_leak(tmp_path, rng, capsys):
    _crop_tree(tmp_path, rng, per_class=(12, 12), n_wells=4)
    _, info = IO.generate_cv_loaders(
        str(tmp_path), n_splits=2, image_size=8, batch_size=4,
        classes=["nc", "pc"], n_jobs=0, group_by="none")
    assert info["groups"] is None
    assert any("NOT group-aware" in w for w in info["warnings"])


def test_cv_loaders_augment_only_the_train_side(tmp_path, rng):
    _crop_tree(tmp_path, rng, per_class=(12, 12), n_wells=4)
    fold_loaders, _ = IO.generate_cv_loaders(
        str(tmp_path), n_splits=2, image_size=8, batch_size=4,
        classes=["nc", "pc"], n_jobs=0, group_by="well", augment=True)
    for train_loader, val_loader in fold_loaders:
        assert len(train_loader.dataset) == 8 * (24 - len(val_loader.dataset))


@pytest.mark.parametrize("k", [0, 1])
def test_cv_loaders_refuse_a_degenerate_k(tmp_path, rng, k):
    _crop_tree(tmp_path, rng, per_class=(4, 4), n_wells=2)
    with pytest.raises(ValueError, match="must be >= 2"):
        IO.generate_cv_loaders(str(tmp_path), n_splits=k, image_size=8,
                               classes=["nc", "pc"], n_jobs=0)


def test_cv_loaders_reuse_generate_loaders_missing_folder_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Generate Training Data"):
        IO.generate_cv_loaders(str(tmp_path), n_splits=2, classes=["nc", "pc"],
                               n_jobs=0)


# ---------------------------------------------------------------------------
# per-fold metrics and their spread
# ---------------------------------------------------------------------------

def test_summary_reports_mean_and_spread_not_just_the_mean():
    from spacr.deep_spacr import summarize_cv_metrics

    fold_df = pd.DataFrame({
        "fold": [1, 2, 3, 4, 5],
        "accuracy": [0.90, 0.60, 0.75, 0.70, 0.80],
        "loss": [0.20, 0.80, 0.50, 0.60, 0.40],
    })
    out = summarize_cv_metrics(fold_df)
    acc = out.set_index("metric").loc["accuracy"]
    assert acc["n_folds"] == 5
    assert acc["mean"] == pytest.approx(0.75)
    assert acc["std"] == pytest.approx(np.std([0.9, 0.6, 0.75, 0.7, 0.8], ddof=1))
    assert acc["min"] == pytest.approx(0.60) and acc["max"] == pytest.approx(0.90)
    assert acc["range"] == pytest.approx(0.30)
    assert acc["cv_percent"] == pytest.approx(acc["std"] / 0.75 * 100)


def test_summary_skips_metrics_no_fold_produced():
    from spacr.deep_spacr import summarize_cv_metrics

    fold_df = pd.DataFrame({"fold": [1, 2],
                            "accuracy": [0.8, 0.6],
                            "prauc": [np.nan, np.nan]})
    out = summarize_cv_metrics(fold_df)
    assert list(out["metric"]) == ["accuracy"]


def test_summary_of_a_single_fold_has_no_standard_deviation():
    from spacr.deep_spacr import summarize_cv_metrics

    out = summarize_cv_metrics(pd.DataFrame({"fold": [1], "accuracy": [0.8]}))
    row = out.iloc[0]
    assert row["n_folds"] == 1
    assert np.isnan(row["std"])
    assert row["range"] == 0.0


def test_report_prints_every_fold_and_the_spread(capsys):
    from spacr.deep_spacr import _print_cv_report, summarize_cv_metrics

    fold_df = pd.DataFrame({"fold": [1, 2, 3],
                            "accuracy": [0.9, 0.6, 0.75]})
    _print_cv_report(fold_df, summarize_cv_metrics(fold_df), 3)
    out = capsys.readouterr().out
    assert "Cross-validation results (3 folds)" in out
    assert "Fold-to-fold spread" in out
    assert "accuracy across folds: 0.7500 +/- 0.1500" in out
    assert "range 0.6000-0.9000" in out
    assert "how lucky that number could have been" in out


def test_report_survives_a_run_where_no_metric_was_numeric(capsys):
    from spacr.deep_spacr import _print_cv_report

    _print_cv_report(pd.DataFrame({"fold": [1]}), pd.DataFrame(), 1)
    assert "no numeric metrics" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# train_test_model wiring
# ---------------------------------------------------------------------------

def _fake_train_model(**kwargs):
    import torch.nn as nn

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 8 * 8, 2)

        def forward(self, x):
            return self.fc(x.flatten(1))

    return _Head(), None


def test_cross_validation_branch_trains_every_fold_and_writes_csvs(
        tmp_path, rng, monkeypatch):
    import spacr.deep_spacr as DS

    _crop_tree(tmp_path, rng, per_class=(18, 18), n_wells=6)
    calls = []

    def _spy(**kwargs):
        calls.append(kwargs)
        return _fake_train_model(**kwargs)

    monkeypatch.setattr(DS, "train_model", _spy)

    out = DS.train_test_model({
        "src": str(tmp_path), "classes": ["nc", "pc"], "train": True,
        "test": False, "epochs": 1, "image_size": 8, "batch_size": 4,
        "n_jobs": 0, "augment": False, "pin_memory": False, "normalize": False,
        "cross_validation_folds": 3, "cv_group_by": "well",
    })

    assert len(calls) == 3
    # every fold got its own destination folder
    assert sorted(os.path.basename(c["dst"]) for c in calls) == [
        "fold_1", "fold_2", "fold_3"]
    assert out is not None and out.endswith("_per_fold.csv")

    per_fold = pd.read_csv(out)
    assert list(per_fold["fold"]) == [1, 2, 3]
    assert per_fold["n_val"].sum() == 36
    assert "accuracy" in per_fold.columns

    spread = pd.read_csv(out.replace("_per_fold.csv", "_spread.csv"))
    assert "std" in spread.columns and "range" in spread.columns
    assert (spread["n_folds"] == 3).all()

    composition = pd.read_csv(out.replace("_per_fold.csv",
                                          "_fold_composition.csv"))
    assert composition["n_val"].sum() == 36

    evaluation_dir = os.path.join(os.path.dirname(out), "evaluation")
    manifest_path = os.path.join(
        evaluation_dir, "evaluation_manifest.json",
    )
    assert os.path.isfile(manifest_path)
    predictions = pd.read_csv(
        os.path.join(evaluation_dir, "oof_predictions.csv"),
    )
    assert len(predictions) == 36
    assert set(predictions["fold"]) == {1, 2, 3}
    confusion = pd.read_csv(
        os.path.join(evaluation_dir, "confusion_counts.csv"),
        index_col=0,
    )
    assert confusion.to_numpy().sum() == 36


def test_nested_cv_uses_inner_validation_and_untouched_outer_folds(
        tmp_path, rng, monkeypatch):
    """Every inner model selects on inner data and scores one untouched outer fold."""
    import spacr.deep_spacr as DS
    from spacr.classifier_evaluation import audit_split_leakage
    from spacr.io import dataset_filenames

    _crop_tree(tmp_path, rng, per_class=(24, 24), n_wells=8)
    calls = []

    def _spy(**kwargs):
        calls.append(kwargs)
        return _fake_train_model(**kwargs)

    monkeypatch.setattr(DS, "train_model", _spy)

    out = DS.train_test_model({
        "src": str(tmp_path), "classes": ["nc", "pc"], "train": True,
        "test": False, "epochs": 1, "image_size": 8, "batch_size": 4,
        "n_jobs": 0, "augment": False, "pin_memory": False, "normalize": False,
        "cross_validation_folds": 2, "nested_cv_inner_folds": 2,
        "cv_group_by": "well", "evaluation_calibration": "none",
    })

    assert len(calls) == 4
    assert sorted(
        os.path.relpath(call["dst"], os.path.dirname(out))
        for call in calls
    ) == [
        "fold_1/inner_1",
        "fold_1/inner_2",
        "fold_2/inner_1",
        "fold_2/inner_2",
    ]
    for call in calls:
        report = audit_split_leakage(
            dataset_filenames(call["train_loaders"].dataset),
            dataset_filenames(call["val_loaders"].dataset),
            group_by="well",
        )
        assert report.passed

    evaluation_dir = os.path.join(os.path.dirname(out), "evaluation")
    predictions = pd.read_csv(
        os.path.join(evaluation_dir, "oof_predictions.csv"),
    )
    assert len(predictions) == 48
    assert set(predictions["fold"]) == {1, 2}
    with open(
        os.path.join(evaluation_dir, "leakage.json"),
        encoding="utf-8",
    ) as handle:
        leakage = json.load(handle)
    assert leakage["passed"]
    # One whole-partition proof + two outer and four inner boundaries.
    assert len(leakage["folds"]) == 7
    assert leakage["folds"][0]["split_name"] == "all_cv_folds"
    assert all(report["passed"] for report in leakage["folds"])


@pytest.mark.parametrize("k", [0, 1, None])
def test_k_of_zero_or_one_falls_back_to_the_single_split(
        tmp_path, rng, monkeypatch, k):
    import spacr.deep_spacr as DS

    _crop_tree(tmp_path, rng, per_class=(12, 12), n_wells=4)
    calls = []

    def _spy(**kwargs):
        calls.append(kwargs)
        return _fake_train_model(**kwargs)

    monkeypatch.setattr(DS, "train_model", _spy)

    DS.train_test_model({
        "src": str(tmp_path), "classes": ["nc", "pc"], "train": True,
        "test": False, "epochs": 1, "image_size": 8, "batch_size": 4,
        "n_jobs": 0, "augment": False, "pin_memory": False, "normalize": False,
        "val_split": 0.25, "cross_validation_folds": k,
    })
    # exactly one training run, into the top-level dst (no fold_ subfolder)
    assert len(calls) == 1
    assert not os.path.basename(calls[0]["dst"]).startswith("fold_")


def test_k_of_one_says_it_is_falling_back(tmp_path, rng, monkeypatch, capsys):
    import spacr.deep_spacr as DS

    _crop_tree(tmp_path, rng, per_class=(12, 12), n_wells=4)
    monkeypatch.setattr(DS, "train_model", lambda **kw: _fake_train_model(**kw))
    DS.train_test_model({
        "src": str(tmp_path), "classes": ["nc", "pc"], "train": True,
        "test": False, "epochs": 1, "image_size": 8, "batch_size": 4,
        "n_jobs": 0, "augment": False, "pin_memory": False, "normalize": False,
        "val_split": 0.25, "cross_validation_folds": 1,
    })
    out = capsys.readouterr().out
    assert "cross_validation_folds=1 is not a cross-validation" in out


def test_a_fold_whose_model_cannot_be_built_is_skipped_not_fatal(
        tmp_path, rng, monkeypatch, capsys):
    import spacr.deep_spacr as DS

    _crop_tree(tmp_path, rng, per_class=(18, 18), n_wells=6)
    seq = iter([(None, None), _fake_train_model(), _fake_train_model()])
    monkeypatch.setattr(DS, "train_model", lambda **kw: next(seq))

    out = DS.train_test_model({
        "src": str(tmp_path), "classes": ["nc", "pc"], "train": True,
        "test": False, "epochs": 1, "image_size": 8, "batch_size": 4,
        "n_jobs": 0, "augment": False, "pin_memory": False, "normalize": False,
        "cross_validation_folds": 3, "cv_group_by": "well",
    })
    assert "Fold 1" in capsys.readouterr().out
    assert list(pd.read_csv(out)["fold"]) == [2, 3]


def test_no_usable_fold_returns_none(tmp_path, rng, monkeypatch, capsys):
    import spacr.deep_spacr as DS

    _crop_tree(tmp_path, rng, per_class=(12, 12), n_wells=4)
    monkeypatch.setattr(DS, "train_model", lambda **kw: (None, None))
    out = DS.train_test_model({
        "src": str(tmp_path), "classes": ["nc", "pc"], "train": True,
        "test": False, "epochs": 1, "image_size": 8, "batch_size": 4,
        "n_jobs": 0, "augment": False, "pin_memory": False, "normalize": False,
        "cross_validation_folds": 2, "cv_group_by": "well",
    })
    assert out is None
    assert "produced no fold results" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# settings surface
# ---------------------------------------------------------------------------

def test_cross_validation_folds_defaults_to_the_single_split():
    from spacr.settings import (get_train_test_model_settings,
                                set_default_train_test_model)

    assert get_train_test_model_settings({})["cross_validation_folds"] == 0
    assert set_default_train_test_model({})["cross_validation_folds"] == 0


def test_grouped_folds_are_the_recommended_default():
    from spacr.settings import get_train_test_model_settings

    assert get_train_test_model_settings({})["cv_group_by"] == "well"


def test_new_cv_settings_are_typed_and_documented():
    from spacr.settings import expected_types, tooltips

    assert expected_types["cross_validation_folds"] is int
    assert expected_types["cv_group_by"] is str
    for level in IO.CV_GROUP_LEVELS:
        assert level in tooltips["cv_group_by"]


def test_the_regression_cross_validation_toggle_is_untouched():
    """The pre-existing bool setting keeps its name, type and meaning."""
    from spacr.settings import expected_types

    assert expected_types["cross_validation"] is bool
    assert expected_types["cross_validation_folds"] is int
