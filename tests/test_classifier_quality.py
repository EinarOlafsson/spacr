"""The classifier's error, and what it does to a fraction.

Asked 2026-08-21: "you have the computer vision scores, cant this be
calculated from them? cant you bake the calculation of this into the
modula?"

THE ANSWER THESE TESTS ENCODE: not from scores alone, exactly from the
labelled split, and badly from an unlabelled column -- badly enough that
the module says so rather than returning the number quietly.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import classifier_quality as module


@pytest.fixture
def labelled():
    """4,000 cells at 20% prevalence, a classifier with real but imperfect
    separation."""
    rng = np.random.default_rng(0)
    n = 4000
    truth = rng.random(n) < 0.20
    scores = np.where(truth, rng.normal(1.6, 1.0, n), rng.normal(0.0, 1.0, n))
    return scores, truth


class TestAccuracyIsTheWrongNumber:
    """The argument that started this: 94% accuracy is one number where two
    are needed, and on an imbalanced well it is the majority class wearing a
    hat."""

    def test_calling_everything_negative_scores_well_and_is_useless(self):
        truth = np.array([True] * 10 + [False] * 90)
        scores = np.zeros(100)              # everything below any threshold

        got = module.confusion(scores, truth, threshold=0.5)

        assert got.accuracy == pytest.approx(0.90)
        assert got.sensitivity == pytest.approx(0.0)
        assert not got.usable

    def test_sensitivity_and_specificity_are_reported_apart(self, labelled):
        scores, truth = labelled
        got = module.confusion(scores, truth, 0.8)
        assert 0.0 < got.sensitivity < 1.0
        assert 0.0 < got.specificity < 1.0
        assert got.sensitivity != got.specificity


class TestTheOperatingPoint:

    def test_youden_maximises_the_correction_denominator(self, labelled):
        """Not an arbitrary criterion here: se + sp - 1 is exactly what
        Rogan-Gladen divides by, so this is the point where the correction
        is most stable."""
        scores, truth = labelled
        best = module.best_threshold(scores, truth)
        others = module.operating_points(scores, truth)

        mine = best.sensitivity + best.specificity
        assert all(mine >= p.sensitivity + p.specificity - 1e-9
                   for p in others)

    def test_an_unknown_criterion_is_refused(self, labelled):
        scores, truth = labelled
        with pytest.raises(ValueError):
            module.best_threshold(scores, truth, criterion="whatever")


class TestTheCorrectionMovesTheEstimate:

    def test_it_recovers_the_true_share(self, labelled):
        """The whole point: an observed share is not the share."""
        scores, truth = labelled
        best = module.best_threshold(scores, truth)
        observed = float((scores >= best.threshold).mean())

        out = module.rogan_gladen(observed, best.sensitivity,
                                  best.specificity, n=scores.size)

        assert observed > 0.30, "the raw share is badly inflated"
        assert out["corrected"] == pytest.approx(0.20, abs=0.03)

    def test_at_94_percent_an_observed_six_percent_is_nothing(self):
        """The example from the discussion, pinned: the whole observed
        signal is false positives."""
        out = module.rogan_gladen(0.06, 0.94, 0.94)
        assert out["corrected"] == pytest.approx(0.0, abs=1e-9)

    def test_it_inflates_the_variance_rather_than_leaving_it(self):
        out = module.rogan_gladen(0.20, 0.94, 0.94, n=1000)
        assert out["variance_inflation"] == pytest.approx(1.0 / 0.88 ** 2,
                                                          rel=1e-6)
        assert out["standard_error"] > np.sqrt(0.2 * 0.8 / 1000)

    def test_a_classifier_with_no_information_gets_no_correction(self):
        """se + sp = 1 is a coin, and the correction divides by zero."""
        out = module.rogan_gladen(0.4, 0.5, 0.5)
        assert out["usable"] == 0.0
        assert not np.isfinite(out["corrected"])


class TestPrevalenceStratification:
    """The number 214 asked for, and the reason it asked."""

    def test_a_context_reading_classifier_is_caught(self):
        """Sensitivity should not depend on prevalence. A model that has
        learned 'this well looks crowded, call more of it positive' shows
        rising sensitivity with prevalence, and every fraction it produces
        is then partly a copy of the fraction it was given."""
        rng = np.random.default_rng(1)
        scores, truth, wells = [], [], []
        for index, share in enumerate([0.05, 0.2, 0.5, 0.8]):
            for _ in range(600):
                positive = rng.random() < share
                # THE CHEAT: the score is shifted by the well's prevalence.
                centre = (1.2 if positive else 0.0) + 2.5 * share
                scores.append(rng.normal(centre, 0.7))
                truth.append(positive)
                wells.append(f"w{index}")

        bands = module.sensitivity_by_prevalence(scores, truth, wells,
                                                 threshold=1.5, bins=4)

        assert len(bands) >= 3
        rising = [b["sensitivity"] for b in bands]
        assert rising[-1] > rising[0] + 0.2, (
            f"the stratification did not expose the cheat: {rising}")

    def test_an_honest_classifier_looks_flat(self):
        rng = np.random.default_rng(2)
        scores, truth, wells = [], [], []
        for index, share in enumerate([0.05, 0.2, 0.5, 0.8]):
            for _ in range(800):
                positive = rng.random() < share
                scores.append(rng.normal(1.8 if positive else 0.0, 0.7))
                truth.append(positive)
                wells.append(f"w{index}")

        bands = module.sensitivity_by_prevalence(scores, truth, wells,
                                                 threshold=0.9, bins=4)
        values = [b["sensitivity"] for b in bands]
        assert max(values) - min(values) < 0.12, values


class TestTheUnlabelledFallbackKnowsItIsWeak:

    def test_it_flags_itself_when_the_classes_overlap(self, labelled):
        """It got 0.45 for a true 0.20 on this very fixture. The number is
        not the problem -- returning it without the warning would be."""
        scores, _truth = labelled
        out = module.deconvolve(scores)
        assert out["separation"] < 2.0
        assert out["trustworthy"] == 0.0

    def test_it_is_trusted_when_the_classes_really_do_separate(self):
        rng = np.random.default_rng(3)
        n = 4000
        truth = rng.random(n) < 0.30
        scores = np.where(truth, rng.normal(6.0, 1.0, n),
                          rng.normal(0.0, 1.0, n))

        out = module.deconvolve(scores)

        assert out["trustworthy"] == 1.0
        assert out["prevalence"] == pytest.approx(0.30, abs=0.05)

    def test_too_few_cells_is_an_error_not_a_guess(self):
        assert module.deconvolve([1.0, 2.0, 3.0]).get("error") == 1.0


class TestTheTrainingWells:
    """"columns one and 2 of each plate were the training wells"."""

    def test_it_finds_both_spellings(self):
        mask = module.training_wells(
            ["r1_c1", "r2_c2", "r3_c3", "plate1_r4_c02", "c1"])
        assert mask.tolist() == [True, True, False, True, True]

    def test_a_name_it_cannot_parse_is_not_training(self):
        """Guessing would silently drop real validation wells."""
        assert module.training_wells(["A03", "", "well"]).tolist() == \
            [False, False, False]

    def test_the_columns_are_a_parameter(self):
        mask = module.training_wells(["r1_c1", "r1_c5"], columns=(5,))
        assert mask.tolist() == [False, True]
