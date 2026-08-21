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


class TestReadingARealTestSplit:
    """The tsg101 splits, supplied 2026-08-21.

    THE NUMBER 214 ASKED FOR WAS ALREADY ON DISK. `pos_accuracy` is the
    sensitivity and `neg_accuracy` the specificity, written by the training
    code on every plate. The request was answered by a file that existed.
    """

    def test_a_summary_file_is_read(self, tmp_path):
        import pandas as pd

        path = tmp_path / "x_test_result.csv"
        pd.DataFrame([{"accuracy": 0.968, "neg_accuracy": 0.9817,
                       "pos_accuracy": 0.9553, "loss": 0.0008,
                       "prauc": 0.994, "optimal_threshold": 0.2955}]
                     ).to_csv(path, index=False)

        got = module.from_test_split(str(path))

        assert got["sensitivity"] == pytest.approx(0.9553)
        assert got["specificity"] == pytest.approx(0.9817)
        assert got["per_cell"] == 0.0

    def test_a_per_cell_file_is_read_and_can_be_re_asked(self, tmp_path):
        """Only a per-cell file supports a different threshold, which is why
        the two shapes are distinguished rather than merged."""
        import pandas as pd

        rng = np.random.default_rng(4)
        n = 600
        truth = np.arange(n) < n // 2
        scores = np.where(truth, rng.beta(8, 2, n), rng.beta(2, 8, n))
        path = tmp_path / "x_test_acc.csv"
        pd.DataFrame({"filename": [f"plate1_A0{i%9+1}_1_{i}.png"
                                   for i in range(n)],
                      "true_label": truth.astype(float),
                      "predicted_label": (scores >= 0.5).astype(float),
                      "class_1_probability": scores}).to_csv(path, index=False)

        loose = module.from_test_split(str(path), threshold=0.2)
        tight = module.from_test_split(str(path), threshold=0.8)

        assert loose["per_cell"] == 1.0
        assert loose["sensitivity"] > tight["sensitivity"]
        assert loose["specificity"] < tight["specificity"]

    def test_an_unreadable_shape_is_refused_by_name(self, tmp_path):
        import pandas as pd

        path = tmp_path / "wrong.csv"
        pd.DataFrame({"something": [1, 2]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="not a test split"):
            module.from_test_split(str(path))


class TestWhatTheClassifierDoesToARareGuide:
    """The result that decides whether any of this matters."""

    def test_it_is_a_no_op_for_a_common_class(self):
        rows = module.inflation_by_prevalence(0.9604, 0.9812,
                                              prevalences=(0.30,))
        assert rows[0]["inflation"] == pytest.approx(1.0, abs=0.02)

    def test_it_nearly_triples_a_one_percent_guide(self):
        """False positives are a share of the NEGATIVES, and when a guide is
        rare almost every cell in the well is a negative -- so a small
        false-positive rate on a large population swamps a large
        true-positive rate on a small one. Screen hits are rare."""
        rows = module.inflation_by_prevalence(0.9604, 0.9812,
                                              prevalences=(0.01,))
        assert rows[0]["inflation"] > 2.5

    def test_the_correction_undoes_it_at_every_prevalence(self):
        rows = module.inflation_by_prevalence(0.9604, 0.9812)
        for row in rows:
            assert row["corrected"] == pytest.approx(row["true"], abs=1e-6)


class TestNoScreenLivesInTheLibrary:
    """"wahtever information you use from my screen any calculated
    coefficients need to be recalculable for users whou do their own
    screens".

    A table of one screen's measured sensitivities was in this module for a
    single commit. It is gone, and this is the guard: a constant in a
    library becomes a default the moment somebody is in a hurry, and a
    sensitivity measured on one model, one stain and one microscope is
    wrong for every other screen in a way that produces plausible numbers
    rather than an error.
    """

    def test_the_module_carries_no_measured_constants(self):
        assert not hasattr(module, "TSG101_TEST_SPLIT")
        for name in dir(module):
            if name.startswith("_"):
                continue
            value = getattr(module, name)
            if isinstance(value, dict) and value:
                first = next(iter(value.values()))
                assert not (isinstance(first, dict)
                            and "sensitivity" in first), (
                    f"{name} looks like a screen's numbers baked in")

    def test_every_entry_point_demands_the_numbers(self):
        """Nothing has a sensitivity to fall back to, so nothing can quietly
        use somebody else's."""
        import inspect

        for name in ("rogan_gladen", "required_fraction" ) :
            if not hasattr(module, name):
                continue
            signature = inspect.signature(getattr(module, name))
            for parameter in signature.parameters.values():
                if parameter.name in ("sensitivity", "specificity"):
                    assert parameter.default is inspect.Parameter.empty, (
                        f"{name}.{parameter.name} has a default")

    def test_inflation_by_prevalence_demands_them_too(self):
        import inspect

        signature = inspect.signature(module.inflation_by_prevalence)
        for parameter in ("sensitivity", "specificity"):
            assert (signature.parameters[parameter].default
                    is inspect.Parameter.empty)


class TestDiscoveringAUsersOwnSplits:

    def test_it_finds_one_file_per_plate_folder(self, tmp_path):
        import pandas as pd

        for plate in ("plate1", "plate2"):
            folder = tmp_path / plate
            folder.mkdir()
            pd.DataFrame([{"accuracy": 0.9, "neg_accuracy": 0.98,
                           "pos_accuracy": 0.95,
                           "optimal_threshold": 0.3}]).to_csv(
                folder / f"model_time_x_test_result.csv", index=False)
            (folder / "unrelated.csv").write_text("a,b\n1,2\n")

        found = module.discover_test_splits(str(tmp_path))

        assert sorted(found) == ["plate1", "plate2"]
        assert all("test_result" in path for path in found.values())

    def test_measure_screen_reads_them_all(self, tmp_path):
        import pandas as pd

        for plate, se in (("plate1", 0.95), ("plate2", 0.96)):
            folder = tmp_path / plate
            folder.mkdir()
            pd.DataFrame([{"accuracy": 0.9, "neg_accuracy": 0.98,
                           "pos_accuracy": se,
                           "optimal_threshold": 0.3}]).to_csv(
                folder / "m_test_result.csv", index=False)

        got = module.measure_screen(str(tmp_path))

        assert got["plate1"]["sensitivity"] == pytest.approx(0.95)
        assert got["plate2"]["sensitivity"] == pytest.approx(0.96)

    def test_a_missing_folder_is_refused(self):
        with pytest.raises(ValueError, match="not a directory"):
            module.discover_test_splits("/nowhere/at/all")

    def test_plates_are_kept_apart_rather_than_pooled(self, tmp_path):
        """Their thresholds are not comparable, so one averaged number would
        be wrong for every plate."""
        import pandas as pd

        for plate, threshold in (("plate1", 0.29), ("plate2", 0.86)):
            folder = tmp_path / plate
            folder.mkdir()
            pd.DataFrame([{"accuracy": 0.9, "neg_accuracy": 0.98,
                           "pos_accuracy": 0.95,
                           "optimal_threshold": threshold}]).to_csv(
                folder / "m_test_result.csv", index=False)

        got = module.measure_screen(str(tmp_path))
        assert got["plate1"]["threshold"] != got["plate2"]["threshold"]
