"""Class-imbalance auto-handling: detection, the four modes, and the report.

The behaviour under test is deliberately narrow but load-bearing:

  * skew is *measured* from the labels the model will actually see and printed
    on every run, so an auto-fix is never silent and a non-fix is never
    invisible;
  * ``class_balance='none'`` on skewed data still names the modes that would
    have helped;
  * each sampler mode moves the *realised* draw frequencies toward balance by
    the amount it claims -- this is measured by drawing from the sampler, not
    asserted from the weight formula;
  * the sampler is attached to the TRAIN loader and to nothing else.
    Resampling validation or test data silently changes the class prior the
    metrics are measured against, which is the classic way a "balanced"
    accuracy stops describing the real screen.

Everything runs on 8x8 PNGs on the CPU in milliseconds.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spacr import io as IO


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _png(path, rng, size=8):
    Image.fromarray(rng.integers(0, 255, (size, size, 3)).astype(np.uint8)).save(path)
    return str(path)


def _skewed_tree(root, rng, counts=(60, 6), classes=("nc", "pc"), split="train"):
    """Write ``counts[i]`` crops into ``root/<split>/<classes[i]>``.

    Names follow the spacr crop convention ``plate_well_field_object.png`` so
    the same tree can be reused by the grouping tests.
    """
    for ci, (cls, n) in enumerate(zip(classes, counts)):
        d = root / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            well = f"A{(i % 3) + 1:02d}"
            _png(d / f"plate1_{well}_{i % 2 + 1}_{ci}{i}.png", rng)
    return root


def _realised_fractions(sampler, labels, n_draws=None):
    """Draw from ``sampler`` and return the class frequencies it produced."""
    labels = list(labels)
    n_classes = max(labels) + 1
    drawn = list(sampler)
    if n_draws is not None:
        drawn = drawn[:n_draws]
    counts = np.zeros(n_classes, dtype=float)
    for idx in drawn:
        counts[labels[int(idx)]] += 1
    return counts / max(counts.sum(), 1)


# ---------------------------------------------------------------------------
# skew detection
# ---------------------------------------------------------------------------

def test_summarize_reports_exact_counts_and_ratio():
    """A 90/10 split is reported as 90/10 with a ratio of exactly 9."""
    labels = [0] * 90 + [1] * 10
    s = IO.summarize_class_imbalance(labels, classes=["nc", "pc"])
    assert s["counts"] == [90, 10]
    assert s["fractions"] == pytest.approx([0.9, 0.1])
    assert s["imbalance_ratio"] == pytest.approx(9.0)
    assert s["majority"] == "nc" and s["minority"] == "pc"
    assert s["minority_fraction"] == pytest.approx(0.1)
    assert s["n"] == 100
    assert s["skewed"] is True
    assert s["severe"] is False  # 9 < 10


def test_summarize_flags_severe_skew_and_balanced_data():
    assert IO.summarize_class_imbalance([0] * 100 + [1] * 5)["severe"] is True
    balanced = IO.summarize_class_imbalance([0] * 50 + [1] * 48)
    assert balanced["skewed"] is False
    assert balanced["imbalance_ratio"] == pytest.approx(50 / 48)


def test_summarize_handles_an_empty_class_without_dividing_by_zero():
    """A declared class with no members gives an infinite ratio, not a crash."""
    s = IO.summarize_class_imbalance([0] * 10, classes=["nc", "pc", "extra"])
    assert s["counts"] == [10, 0, 0]
    assert s["imbalance_ratio"] == float("inf")
    assert s["empty_classes"] == ["pc", "extra"]


def test_summarize_counts_labels_outside_the_declared_classes():
    s = IO.summarize_class_imbalance([0, 1, 5, -1], classes=["nc", "pc"])
    assert s["counts"] == [1, 1]
    assert s["unknown_labels"] == 2


def test_summarize_on_empty_labels_is_a_no_op():
    s = IO.summarize_class_imbalance([])
    assert s["counts"] == [] and s["n"] == 0
    assert s["skewed"] is False and s["imbalance_ratio"] == 1.0


# ---------------------------------------------------------------------------
# the report -- "show the effect"
# ---------------------------------------------------------------------------

def test_report_prints_counts_ratio_and_action(capsys):
    IO.report_class_balance([0] * 60 + [1] * 6, classes=["nc", "pc"],
                            class_balance="weighted_sampler")
    out = capsys.readouterr().out
    assert "nc: 60" in out and "pc: 6" in out
    assert "imbalance ratio (majority/minority): 10.00" in out
    assert "WeightedRandomSampler on the train loader only" in out


def test_none_on_skewed_data_still_recommends_a_mode(capsys):
    """A silent non-fix is as bad as a silent fix."""
    s = IO.report_class_balance([0] * 60 + [1] * 6, classes=["nc", "pc"],
                                class_balance="none")
    out = capsys.readouterr().out
    assert s["action"] == "none (no rebalancing applied)"
    assert "recommendation:" in out
    assert "weighted_sampler" in out and "weighted_loss" in out
    assert "severe skew" in s["recommendation"]


def test_none_on_mildly_skewed_data_recommends_the_gentle_modes():
    s = IO.report_class_balance([0] * 30 + [1] * 15, classes=["nc", "pc"],
                                class_balance="none", verbose=False)
    assert s["skewed"] is True and s["severe"] is False
    assert "sqrt_weighted_sampler" in s["recommendation"]


def test_none_on_balanced_data_recommends_nothing():
    s = IO.report_class_balance([0] * 30 + [1] * 28, classes=["nc", "pc"],
                                class_balance="none", verbose=False)
    assert s["recommendation"] == ""
    assert "none needed" in s["action"]


def test_report_shows_the_expected_post_sampling_frequencies():
    """The report prints where the sampler will move each class to."""
    s = IO.report_class_balance([0] * 90 + [1] * 10, classes=["nc", "pc"],
                                class_balance="weighted_sampler", verbose=False)
    assert "-> sampled at ~50.0%" in s["report"]


def test_report_names_the_loss_switch_for_weighted_loss():
    s = IO.report_class_balance([0] * 90 + [1] * 10, classes=["nc", "pc"],
                                class_balance="weighted_loss", verbose=False)
    assert "ce_weighted" in s["action"]
    # weighted_loss does not touch sampling, so no frequencies move.
    assert "sampled at" not in s["report"]


def test_report_warns_about_a_class_with_no_samples():
    s = IO.report_class_balance([0] * 10, classes=["nc", "pc"],
                                class_balance="none", verbose=False)
    assert "classes with no train samples: ['pc']" in s["report"]


def test_report_for_non_train_splits_says_they_are_untouched():
    s = IO.report_class_balance([0] * 20 + [1] * 2, classes=["nc", "pc"],
                                class_balance="weighted_sampler",
                                split_name="validation", verbose=False)
    assert "never resampled" in s["action"]
    assert "sampled at" not in s["report"]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="not one of"):
        IO.report_class_balance([0, 1], class_balance="magic")


# ---------------------------------------------------------------------------
# the modes, measured
# ---------------------------------------------------------------------------

def test_weighted_sampler_realises_near_uniform_class_frequencies():
    """Draw from the sampler and check the frequencies actually moved."""
    labels = [0] * 90 + [1] * 10
    g = __import__("torch").Generator().manual_seed(0)
    sampler, weights = IO.make_class_balance_sampler(
        labels, "weighted_sampler", num_samples=20000, generator=g)
    frac = _realised_fractions(sampler, labels)
    assert frac[1] == pytest.approx(0.5, abs=0.02)   # from 0.10 to ~0.50
    # per-sample weights: a minority sample is drawn 9x as often as a majority one
    assert float(weights[-1] / weights[0]) == pytest.approx(9.0)


def test_sqrt_weighted_sampler_moves_partway_not_all_the_way():
    labels = [0] * 90 + [1] * 10
    g = __import__("torch").Generator().manual_seed(0)
    sampler, _ = IO.make_class_balance_sampler(
        labels, "sqrt_weighted_sampler", num_samples=20000, generator=g)
    frac = _realised_fractions(sampler, labels)
    # sqrt correction: n_c^0.5 mass -> sqrt(10)/(sqrt(10)+sqrt(90)) = 0.25
    assert frac[1] == pytest.approx(0.25, abs=0.02)
    assert 0.10 < frac[1] < 0.50


def test_expected_fractions_match_what_the_sampler_realises():
    counts = [90, 10]
    labels = [0] * 90 + [1] * 10
    for mode in ("weighted_sampler", "sqrt_weighted_sampler"):
        g = __import__("torch").Generator().manual_seed(1)
        sampler, _ = IO.make_class_balance_sampler(labels, mode,
                                                   num_samples=20000, generator=g)
        realised = _realised_fractions(sampler, labels)
        expected = IO.expected_sampled_fractions(counts, mode)
        assert realised == pytest.approx(np.array(expected), abs=0.02)


def test_none_and_weighted_loss_build_no_sampler():
    for mode in ("none", "weighted_loss"):
        sampler, weights = IO.make_class_balance_sampler([0, 0, 1], mode)
        assert sampler is None and weights is None


def test_expected_fractions_for_non_sampler_modes_are_the_observed_ones():
    assert IO.expected_sampled_fractions([90, 10], "none") == pytest.approx([0.9, 0.1])
    assert IO.expected_sampled_fractions([90, 10], "weighted_loss") == pytest.approx([0.9, 0.1])


def test_class_sampling_weights_rejects_a_non_sampler_mode():
    with pytest.raises(ValueError, match="does not build a sampler"):
        IO.class_sampling_weights([5, 5], "weighted_loss")


def test_sampler_on_empty_labels_returns_none():
    assert IO.make_class_balance_sampler([], "weighted_sampler") == (None, None)


def test_sampler_skips_an_empty_class_instead_of_dividing_by_zero():
    labels = [0] * 5 + [2] * 5           # class 1 has no members
    sampler, weights = IO.make_class_balance_sampler(labels, "weighted_sampler")
    assert sampler is not None
    assert float(weights.min()) > 0


def test_make_sampler_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="not one of"):
        IO.make_class_balance_sampler([0, 1], "oversample")


# ---------------------------------------------------------------------------
# generate_loaders wiring -- train only
# ---------------------------------------------------------------------------

def test_generate_loaders_attaches_the_sampler_to_train_only(tmp_path, rng):
    """The classic error is resampling the metric set; prove it does not happen."""
    _skewed_tree(tmp_path, rng, counts=(40, 8))
    train, val, _ = IO.generate_loaders(
        str(tmp_path), mode="train", image_size=8, batch_size=4,
        classes=["nc", "pc"], n_jobs=0, validation_split=0.25,
        class_balance="weighted_sampler")

    from torch.utils.data import WeightedRandomSampler
    assert isinstance(train.sampler, WeightedRandomSampler)
    assert not isinstance(val.sampler, WeightedRandomSampler)
    # ...and the val loader is still a plain sequential pass over its subset
    assert len(val.dataset) == 12
    assert len(list(val.sampler)) == len(val.dataset)


@pytest.mark.parametrize("mode", ["none", "weighted_sampler",
                                  "sqrt_weighted_sampler", "weighted_loss"])
def test_validation_loader_is_never_resampled(tmp_path, rng, mode):
    _skewed_tree(tmp_path, rng, counts=(40, 8))
    train, val, _ = IO.generate_loaders(
        str(tmp_path), mode="train", image_size=8, batch_size=4,
        classes=["nc", "pc"], n_jobs=0, validation_split=0.25,
        class_balance=mode)

    from torch.utils.data import WeightedRandomSampler
    assert not isinstance(val.sampler, WeightedRandomSampler)
    # every validation index appears exactly once, in order
    assert list(val.sampler) == list(range(len(val.dataset)))
    # the train loader carries a sampler only for the two sampler modes
    is_sampled = isinstance(train.sampler, WeightedRandomSampler)
    assert is_sampled == (mode in ("weighted_sampler", "sqrt_weighted_sampler"))


@pytest.mark.parametrize("mode", ["none", "weighted_sampler",
                                  "sqrt_weighted_sampler", "weighted_loss"])
def test_test_loader_is_never_resampled(tmp_path, rng, mode):
    """mode='test' must ignore class_balance entirely."""
    _skewed_tree(tmp_path, rng, counts=(20, 4), split="test")
    test, val, _ = IO.generate_loaders(
        str(tmp_path), mode="test", image_size=8, batch_size=4,
        classes=["nc", "pc"], n_jobs=0, class_balance=mode)

    from torch.utils.data import WeightedRandomSampler
    assert not isinstance(test.sampler, WeightedRandomSampler)
    assert val == []
    assert len(test.dataset) == 24


def test_test_mode_reports_the_skew_but_takes_no_action(tmp_path, rng, capsys):
    _skewed_tree(tmp_path, rng, counts=(20, 4), split="test")
    IO.generate_loaders(str(tmp_path), mode="test", image_size=8, batch_size=4,
                        classes=["nc", "pc"], n_jobs=0,
                        class_balance="weighted_sampler")
    out = capsys.readouterr().out
    assert "Class balance (test, n=24)" in out
    assert "never resampled" in out


def test_generate_loaders_reports_train_skew_on_every_run(tmp_path, rng, capsys):
    _skewed_tree(tmp_path, rng, counts=(40, 8))
    IO.generate_loaders(str(tmp_path), mode="train", image_size=8, batch_size=4,
                        classes=["nc", "pc"], n_jobs=0, validation_split=0.25,
                        class_balance="none")
    out = capsys.readouterr().out
    assert "Class balance (train," in out
    assert "Class balance (validation," in out
    assert "imbalance ratio" in out
    assert "recommendation:" in out          # skewed + mode 'none'


def test_generate_loaders_rejects_an_unknown_class_balance(tmp_path, rng):
    _skewed_tree(tmp_path, rng, counts=(4, 4))
    with pytest.raises(ValueError, match="not one of"):
        IO.generate_loaders(str(tmp_path), mode="train", image_size=8,
                            classes=["nc", "pc"], n_jobs=0,
                            validation_split=0.25, class_balance="smote")


def test_sampler_survives_the_augmentation_expansion(tmp_path, rng):
    """augment=True replaces the Subset with a list; labels must still resolve."""
    _skewed_tree(tmp_path, rng, counts=(8, 4))
    train, val, _ = IO.generate_loaders(
        str(tmp_path), mode="train", image_size=8, batch_size=4,
        classes=["nc", "pc"], n_jobs=0, validation_split=0.25, augment=True,
        class_balance="weighted_sampler")
    assert len(train.dataset) == 9 * 8          # 12 crops, 25% held out, x8
    assert train.sampler.num_samples == len(train.dataset)


def test_no_validation_split_still_samples_train_only(tmp_path, rng):
    _skewed_tree(tmp_path, rng, counts=(20, 4))
    train, val, _ = IO.generate_loaders(
        str(tmp_path), mode="train", image_size=8, batch_size=4,
        classes=["nc", "pc"], n_jobs=0, validation_split=0.0,
        class_balance="weighted_sampler")
    from torch.utils.data import WeightedRandomSampler
    assert isinstance(train.sampler, WeightedRandomSampler)
    assert val == []


# ---------------------------------------------------------------------------
# dataset label / filename extraction
# ---------------------------------------------------------------------------

def test_dataset_labels_and_filenames_follow_a_subset(tmp_path, rng):
    from torch.utils.data import Subset

    _skewed_tree(tmp_path, rng, counts=(3, 2))
    data = IO.spacrDataset(str(tmp_path / "train"), ["nc", "pc"], shuffle=False)
    assert IO.dataset_labels(data) == [0, 0, 0, 1, 1]

    sub = Subset(data, [4, 0])
    assert IO.dataset_labels(sub) == [1, 0]
    assert [f.endswith(".png") for f in IO.dataset_filenames(sub)] == [True, True]
    assert IO.dataset_filenames(sub)[0] == data.filenames[4]


def test_dataset_labels_reads_a_plain_tuple_list():
    import torch
    fake = [(torch.zeros(1), 1, "a.png"), (torch.zeros(1), 0, "b.png")]
    assert IO.dataset_labels(fake) == [1, 0]
    assert IO.dataset_filenames(fake) == ["a.png", "b.png"]


# ---------------------------------------------------------------------------
# loss steering
# ---------------------------------------------------------------------------

def test_weighted_loss_switches_loss_type_to_ce_weighted():
    from spacr.deep_spacr import resolve_class_balance_loss

    lt, msg = resolve_class_balance_loss("focal_loss", "weighted_loss", 2)
    assert lt == "ce_weighted"
    assert "focal_loss" in msg and "ce_weighted" in msg


def test_other_modes_leave_the_loss_alone():
    from spacr.deep_spacr import resolve_class_balance_loss

    for mode in ("none", "weighted_sampler", "sqrt_weighted_sampler"):
        assert resolve_class_balance_loss("focal_loss", mode, 2) == ("focal_loss", "")


@pytest.mark.parametrize("loss", ["ce_weighted", "logit_adjust_ce"])
@pytest.mark.parametrize("mode", ["weighted_sampler", "sqrt_weighted_sampler"])
def test_a_sampler_on_top_of_a_reweighting_loss_is_flagged(loss, mode):
    """Resampling and reweighting compound; the user is told, not silently doubled."""
    from spacr.deep_spacr import resolve_class_balance_loss

    lt, msg = resolve_class_balance_loss(loss, mode, 2)
    assert lt == loss                      # nothing is changed behind their back
    assert "compound" in msg and "WARNING" in msg


def test_weighted_loss_is_idempotent():
    from spacr.deep_spacr import resolve_class_balance_loss

    lt, msg = resolve_class_balance_loss("ce_weighted", "weighted_loss", 2)
    assert lt == "ce_weighted"
    assert "already" in msg


def test_weighted_loss_declines_on_a_single_logit_head():
    """ce_weighted is a multiclass loss; a 1-logit head must not be broken."""
    from spacr.deep_spacr import resolve_class_balance_loss

    lt, msg = resolve_class_balance_loss("binary_cross_entropy_with_logits",
                                         "weighted_loss", 1)
    assert lt == "binary_cross_entropy_with_logits"
    assert "focal_alpha" in msg


def test_ce_weighted_actually_upweights_the_rare_class():
    """The steering is only useful if build_loss does what the report claims."""
    import torch
    from spacr.utils import build_loss

    plain = build_loss("ce", num_classes=2)                     # no counts
    weighted = build_loss("ce_weighted", num_classes=2,
                          class_counts=torch.tensor([90, 10]))
    # one confident correct call on the common class, one confident error on
    # the rare one: reweighting makes that batch cost far more.
    logits = torch.tensor([[5.0, -5.0], [5.0, -5.0]])
    y = torch.tensor([0, 1])
    assert float(weighted(logits, y)) > float(plain(logits, y))
    # inverse-frequency weights are 0.2 / 1.8, so the rare error dominates
    assert float(weighted(logits, y)) == pytest.approx(9.0, rel=1e-3)


# ---------------------------------------------------------------------------
# settings surface
# ---------------------------------------------------------------------------

def test_class_balance_defaults_to_none_so_nothing_changes_silently():
    from spacr.settings import (get_train_test_model_settings,
                                set_default_train_test_model)

    assert get_train_test_model_settings({})["class_balance"] == "none"
    assert set_default_train_test_model({})["class_balance"] == "none"


def test_class_balance_is_typed_and_documented():
    from spacr.settings import expected_types, tooltips

    assert expected_types["class_balance"] is str
    for mode in IO.CLASS_BALANCE_MODES:
        assert mode in tooltips["class_balance"]


def test_train_test_model_rejects_an_unknown_class_balance(tmp_path, rng):
    from spacr.deep_spacr import train_test_model

    _skewed_tree(tmp_path, rng, counts=(4, 4))
    with pytest.raises(ValueError, match="class_balance"):
        train_test_model({"src": str(tmp_path), "classes": ["nc", "pc"],
                          "class_balance": "oversample", "train": False,
                          "test": False, "epochs": 1})
