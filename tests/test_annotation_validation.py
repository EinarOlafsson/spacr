"""The harness that decides whether an annotation strategy works.

Instruction 209: "we need a validation and test mechanism to show that this
actually works, we need that for each strategy".

A HARNESS NEEDS VALIDATING TOO, and this file exists because the first
version of it lied. The `no_effect` scenario -- features carrying no
information at all -- reported 85% precision for `assigned`, which is
impossible. The cause was in the generator: each guide's cells were emitted
as a contiguous block, so the ROW ORDER carried the answer and any method
handing out contiguous runs scored far above what the data could support.

That is the failure mode of every benchmark: it is easiest to believe when
it flatters. The tests here are mostly about the harness being unable to
flatter.
"""
from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from spacr import annotation_validation as validation


def test_the_simulated_screen_documents_every_stored_field():
    """Scenario metadata is public evidence, not an unexplained sidecar."""
    missing = [
        field.name for field in fields(validation.Screen)
        if f":param {field.name}:" not in (validation.Screen.__doc__ or "")
    ]
    assert not missing, f"undocumented Screen fields: {missing}"


class TestTheGeneratorCannotLeakTheAnswer:

    def test_the_cells_are_shuffled_within_a_well(self):
        """The bug that made the first benchmark meaningless."""
        screen = validation.synthesise(wells=4, cells_per_well=80,
                                       guides_per_well=4, seed=3)
        wells = np.asarray(screen.wells)
        truth = np.asarray(screen.truth)

        for well in sorted(set(wells.tolist())):
            here = truth[wells == well]
            # Count the runs. Contiguous blocks give exactly one run per
            # guide; a shuffle gives many more.
            runs = 1 + int((here[1:] != here[:-1]).sum())
            assert runs > len(set(here.tolist())) * 2, (
                f"{well} looks blocked: {runs} runs for "
                f"{len(set(here.tolist()))} guides")

    def test_position_predicts_nothing(self):
        """The general form: a method reading only the row index must be at
        chance."""
        screen = validation.synthesise(wells=6, cells_per_well=60, seed=4)
        wells = np.asarray(screen.wells)
        called = []
        for well in sorted(set(wells.tolist())):
            rows = np.flatnonzero(wells == well)
            names = sorted(screen.fractions[well])
            # Hand out contiguous runs, the strategy the leak rewarded.
            per = max(1, rows.size // max(len(names), 1))
            for position in range(rows.size):
                called.append(names[min(position // per, len(names) - 1)])
        verdict = validation.score_annotation(screen.truth, called,
                                              guides=screen.guides)
        floor = validation.score_annotation(
            screen.truth, validation.baseline_chance(screen),
            guides=screen.guides)
        assert verdict.precision <= floor.precision + 0.12


class TestTheScenariosMeanWhatTheySay:

    def test_no_effect_really_has_no_effect(self):
        screen = validation.synthesise(effect=0.0, seed=5)
        # Cells of different guides are drawn from the same distribution.
        truth = np.asarray(screen.truth)
        first, second = sorted(set(truth.tolist()))[:2]
        a = screen.features[truth == first].mean(axis=0)
        b = screen.features[truth == second].mean(axis=0)
        assert np.abs(a - b).max() < 0.6

    def test_a_strong_effect_separates_the_guides(self):
        screen = validation.synthesise(effect=4.0, seed=6)
        truth = np.asarray(screen.truth)
        first, second = sorted(set(truth.tolist()))[:2]
        a = screen.features[truth == first].mean(axis=0)
        b = screen.features[truth == second].mean(axis=0)
        assert np.abs(a - b).max() > 1.5

    def test_the_fraction_bias_reaches_the_reported_fractions(self):
        """The 207 mechanism: threshold, renormalise, inflate."""
        plain = validation.synthesise(seed=7)
        biased = validation.synthesise(fraction_threshold=0.10,
                                       fraction_bias=1.8, seed=7)
        plain_sum = np.mean([sum(v.values()) for v in plain.fractions.values()])
        biased_sum = np.mean([sum(v.values())
                              for v in biased.fractions.values()])
        assert biased_sum > plain_sum * 1.3

    def test_the_true_fractions_are_kept_apart_from_the_reported_ones(self):
        screen = validation.synthesise(fraction_bias=1.8, seed=8)
        well = sorted(screen.fractions)[0]
        assert screen.fractions[well] != screen.true_fractions[well]


class TestScoring:

    def test_abstaining_is_not_an_error_but_does_cost_coverage(self):
        truth = ["a", "a", "b", "b"]
        called = ["a", "Non_annotated", "b", "Non_annotated"]
        verdict = validation.score_annotation(truth, called)
        assert verdict.precision == 1.0          # nothing wrong was said
        assert verdict.coverage == 0.5
        assert verdict.recall == 0.5

    def test_a_method_cannot_buy_precision_by_saying_nothing(self):
        truth = ["a"] * 10
        silent = validation.score_annotation(truth, ["Non_annotated"] * 10)
        assert silent.coverage == 0.0
        assert silent.recall == 0.0

    def test_calibration_notices_an_overconfident_method(self):
        truth = ["a"] * 100
        called = ["a"] * 50 + ["b"] * 50
        confidence = [0.95] * 100        # sure about all of it, wrong on half
        bins = validation.calibration(truth, called, confidence, bins=5)
        assert bins
        mean_confidence, accuracy, _count = bins[-1]
        assert mean_confidence > 0.9
        assert accuracy == pytest.approx(0.5, abs=0.05)


class TestTheBaselinesAreAlwaysThere:

    def test_a_benchmark_includes_them_even_if_the_caller_forgot(self):
        """Without them a benchmark cannot tell 'the method works' from
        'the fractions work'."""
        scenes = {"clean": validation.synthesise(wells=4, cells_per_well=30,
                                                 seed=9)}
        out = validation.benchmark(
            {"silent": lambda s: ["Non_annotated"] * len(s)}, scenes)
        assert "baseline:majority" in out["clean"]
        assert "baseline:chance" in out["clean"]

    def test_the_majority_baseline_uses_no_measurement(self):
        screen = validation.synthesise(wells=4, cells_per_well=30, seed=10)
        scrambled = validation.Screen(
            features=np.random.default_rng(0).normal(
                size=screen.features.shape),
            scores=screen.scores, wells=screen.wells, truth=screen.truth,
            fractions=screen.fractions,
            true_fractions=screen.true_fractions, guides=screen.guides)
        assert (validation.baseline_majority(screen)
                == validation.baseline_majority(scrambled))


class TestThePermutationNull:

    def test_it_keeps_the_cells_and_moves_only_the_fractions(self):
        screen = validation.synthesise(wells=8, seed=11)
        null = validation.permuted(screen, seed=1)

        assert null.features is screen.features
        assert null.truth == screen.truth
        assert sorted(null.fractions) == sorted(screen.fractions)
        assert any(null.fractions[w] != screen.fractions[w]
                   for w in screen.fractions)


class TestMixedRatioWells:
    """The maintainer's own proposal, and the strongest check available --
    because it runs on real cells with a truth that is aggregate but exact.
    """

    @staticmethod
    def _series(bias=1.0, penetrance=1.0, n=200, seed=0):
        rng = np.random.default_rng(seed)
        proportions = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        features, wells, reported = [], [], {}
        for index, share in enumerate(proportions):
            label = f"m{index:02d}"
            positives = int(round(n * share))
            for _ in range(positives):
                shows = rng.random() < penetrance
                centre = np.array([3.0, 1.0]) if shows else np.zeros(2)
                features.append(centre + rng.normal(size=2, scale=0.6))
                wells.append(label)
            for _ in range(n - positives):
                features.append(rng.normal(size=2, scale=0.6))
                wells.append(label)
            reported[label] = min(1.0, share * bias)
        return np.asarray(features), wells, reported

    def test_an_unbiased_screen_gives_a_slope_of_one(self):
        features, wells, reported = self._series(seed=2)
        out = validation.mixed_ratio_calibration(
            features, wells, reported,
            pure_pc_wells=["m06"], pure_nc_wells=["m00"])
        assert out["slope"] == pytest.approx(1.0, abs=0.12)

    @pytest.mark.parametrize("penetrance", [1.0, 0.6, 0.3])
    def test_the_slope_recovers_the_bias_whatever_the_penetrance(
            self, penetrance):
        """THE CLAIM THAT MAKES THIS BETTER THAN INSTRUCTION 214's SINGLE
        POSITIVE CONTROL. A lone control gives penetrance x bias and cannot
        separate them. Here penetrance is absorbed into the PC reference --
        a non-penetrant PC cell is still a PC cell and is still in that
        reference -- so the slope is the bias alone."""
        features, wells, reported = self._series(
            bias=0.55, penetrance=penetrance, seed=3)
        out = validation.mixed_ratio_calibration(
            features, wells, reported,
            pure_pc_wells=["m06"], pure_nc_wells=["m00"])
        assert out["slope"] == pytest.approx(1 / 0.55, abs=0.25)

    def test_naming_the_pure_wells_by_their_reported_fraction_is_refused(self):
        """It is circular -- the reported fraction is the quantity under
        test -- and a bias large enough to matter moves a pure well below
        the cut-off, so the fit refuses on exactly the screens that need
        it."""
        features, wells, reported = self._series(bias=0.55, seed=4)
        out = validation.mixed_ratio_calibration(features, wells, reported)
        assert "error" in out or "warning" in out

    def test_indistinguishable_controls_give_no_number_rather_than_a_wrong_one(
            self):
        rng = np.random.default_rng(5)
        features = rng.normal(size=(400, 2))
        wells = ["a"] * 200 + ["b"] * 200
        out = validation.mixed_ratio_calibration(
            features, wells, {"a": 0.0, "b": 1.0},
            pure_pc_wells=["b"], pure_nc_wells=["a"])
        # Two wells is too few for a slope, and that is said rather than
        # fitted.
        assert "error" in out

    def test_count_agreement_scores_a_method_on_unlabelled_wells(self):
        called = (["PC"] * 90 + ["NC"] * 110) + (["PC"] * 10 + ["NC"] * 190)
        wells = ["m0"] * 200 + ["m1"] * 200
        out = validation.count_agreement(called, wells,
                                         {"m0": 0.45, "m1": 0.05}, "PC")
        assert out["median_absolute_error"] < 0.01
        assert out["per_well"]["m0"] == (0.45, 0.45)
