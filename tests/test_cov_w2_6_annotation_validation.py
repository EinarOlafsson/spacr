"""Validation that can tell a working method from working guide fractions.

The screens here are the module's own simulations, driven end to end: they
are the only place where cell-level guide truth exists, so a check that
stubbed them out would be checking the stub. Each test names the confound it
isolates -- penetrance, fraction inflation, classifier error, order
dependence -- rather than only that a number came back.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import annotation_validation as av


@pytest.fixture(scope="module")
def small_screen():
    """A clean, small screen: enough wells for a permutation to mean
    something, small enough to build many times."""
    return av.synthesise(wells=6, guides_per_well=2, cells_per_well=12,
                         guides=4, seed=3)


# --------------------------------------------------------------------------
# the simulation
# --------------------------------------------------------------------------

def test_a_screen_knows_how_many_cells_it_has(small_screen):
    assert len(small_screen) == 6 * 12
    assert small_screen.features.shape == (72, 6)
    assert len(small_screen.wells) == len(small_screen.truth) == 72


def test_a_perfect_classifier_never_flips_a_score_toward_the_wrong_tail():
    """The score tracks the phenotype and then the classifier errs; at
    accuracy 1.0 there is no second step."""
    honest = av.synthesise(wells=4, guides_per_well=2, cells_per_well=20,
                           effect=3.0, classifier_accuracy=1.0, seed=1)
    sloppy = av.synthesise(wells=4, guides_per_well=2, cells_per_well=20,
                           effect=3.0, classifier_accuracy=0.5, seed=1)
    assert (honest.scores < 0).mean() < (sloppy.scores < 0).mean()


def test_a_threshold_that_drops_everything_keeps_the_well_rather_than_empty_it():
    """A well whose every guide falls under the cut-off still contains cells,
    and reporting no guides for it would delete them from the benchmark."""
    screen = av.synthesise(wells=3, guides_per_well=2, cells_per_well=10,
                           fraction_threshold=1.1, seed=2)
    for well, reported in screen.fractions.items():
        assert reported, well
        assert set(reported) == set(screen.true_fractions[well])


def test_inflating_the_reported_fractions_leaves_the_true_ones_alone():
    """The true fractions are kept so a test can ask how much of a failure
    was the fraction rather than the method."""
    screen = av.synthesise(wells=4, guides_per_well=2, cells_per_well=10,
                           fraction_bias=1.8, seed=4)
    well = sorted(screen.fractions)[0]
    reported = sum(screen.fractions[well].values())
    honest = sum(screen.true_fractions[well].values())
    assert honest == pytest.approx(1.0, abs=1e-6)
    assert reported > honest


def test_a_guides_cells_are_not_emitted_as_one_contiguous_block():
    """Emitting them in blocks leaves the ROW ORDER carrying the answer, and
    any method handing out contiguous runs then scores far above what the
    data supports."""
    screen = av.synthesise(wells=2, guides_per_well=3, cells_per_well=40,
                           guides=5, seed=6)
    truth = list(screen.truth[:40])
    runs = 1 + sum(1 for a, b in zip(truth, truth[1:]) if a != b)
    assert runs > 3


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def test_a_verdict_says_coverage_and_precision_in_one_line():
    verdict = av.score_annotation(["a", "a", "b", "b"],
                                  ["a", "Non_annotated", "b", "a"])
    assert verdict.coverage == 0.75
    assert verdict.precision == pytest.approx(2 / 3)
    assert verdict.recall == 0.5
    assert verdict.summary() == "75% annotated, 67% of those correct " \
                                "(50% of all cells)"


def test_a_call_list_of_the_wrong_length_scores_nothing_rather_than_aligning():
    """Zipping mismatched lists would score cell i's truth against cell i's
    call in a different order and report a number that means nothing."""
    verdict = av.score_annotation(["a", "b", "c"], ["a", "b"])
    assert (verdict.coverage, verdict.precision, verdict.recall) == (0, 0, 0)
    assert verdict.n == 3
    assert av.score_annotation([], []).n == 0


def test_abstaining_costs_coverage_not_precision():
    """This is the whole reason the two are reported separately: a method
    that labels every cell unreliably is not equal to one that abstains."""
    everything = av.score_annotation(["a"] * 10, ["a"] * 5 + ["b"] * 5)
    careful = av.score_annotation(["a"] * 10,
                                  ["a"] * 5 + ["Non_annotated"] * 5)
    assert everything.recall == careful.recall == 0.5
    assert careful.precision == 1.0 and everything.precision == 0.5
    assert careful.coverage == 0.5 and everything.coverage == 1.0


def test_a_guide_nobody_called_has_zero_precision_and_zero_recall():
    verdict = av.score_annotation(["a", "a"], ["a", "a"],
                                  guides=["a", "unused"])
    assert verdict.per_guide["a"] == (1.0, 1.0)
    assert verdict.per_guide["unused"] == (0.0, 0.0)


def test_the_confusion_counts_every_cell_exactly_once():
    verdict = av.score_annotation(["a", "a", "b"], ["a", "b", "b"])
    assert verdict.confusion == {("a", "a"): 1, ("a", "b"): 1, ("b", "b"): 1}
    assert sum(verdict.confusion.values()) == verdict.n


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def test_calibration_bins_only_the_cells_that_were_called():
    out = av.calibration(["a", "a", "b", "b"],
                         ["a", "a", "Non_annotated", "b"],
                         [0.95, 0.9, 0.99, 0.1], bins=2)
    assert sum(count for _, _, count in out) == 3
    high = [row for row in out if row[0] > 0.5][0]
    assert high == (pytest.approx(0.925), 1.0, 2)


def test_a_confidence_of_exactly_one_lands_in_the_top_bin_not_nowhere():
    out = av.calibration(["a"], ["a"], [1.0], bins=4)
    assert out and out[0][2] == 1


def test_no_called_cell_with_a_finite_confidence_is_no_calibration():
    assert av.calibration(["a"], ["Non_annotated"], [0.9]) == []
    assert av.calibration(["a"], ["a"], [float("nan")]) == []
    assert av.calibration(["a", "b"], ["a", "b"], []) == []


# --------------------------------------------------------------------------
# the null
# --------------------------------------------------------------------------

def test_a_permutation_keeps_the_cells_and_breaks_the_link(small_screen):
    null = av.permuted(small_screen, seed=1)
    assert null.features is small_screen.features
    assert null.truth == small_screen.truth
    assert sorted(null.fractions) == sorted(small_screen.fractions)
    assert null.meta["permuted"] is True
    assert sorted(null.true_fractions) == sorted(small_screen.true_fractions)


# --------------------------------------------------------------------------
# order sensitivity
# --------------------------------------------------------------------------

def test_a_method_that_ignores_the_ranking_order_is_reported_stable():
    ranking = [(f"g{i}", 1.0 - i / 10) for i in range(6)]
    out = av.order_sensitivity(lambda order: ["a", "b", "c"], ranking,
                               repeats=3, seed=0)
    assert out["changed"] == 0.0
    assert out["worst"] == 0.0
    assert out["repeats"] == 3


def test_a_method_that_hands_out_the_ranking_order_is_reported_sensitive():
    """Sequential assignment down a ranking is exactly the shape this check
    exists for: swap two near-ties and different cells get different guides."""
    ranking = [(f"g{i}", 0.5) for i in range(8)]
    out = av.order_sensitivity(lambda order: [name for name, _ in order],
                               ranking, repeats=5, seed=1)
    assert out["changed"] > 0.0
    assert out["worst"] >= out["changed"]
    assert out["repeats"] == 5


def test_a_method_that_annotates_nothing_has_no_order_to_be_sensitive_to():
    out = av.order_sensitivity(lambda order: [], [("g0", 1.0)], repeats=3)
    assert out == {"changed": 0.0, "repeats": 0}


def test_a_repeat_that_returns_a_different_number_of_cells_is_discarded():
    """Comparing calls cell by cell needs the same cells; a shorter answer
    would silently score against the wrong ones."""
    calls = iter([["a", "b"], ["a"], ["a", "b"], ["a"]])

    out = av.order_sensitivity(lambda order: next(calls),
                               [("g0", 1.0), ("g1", 0.5)], repeats=3, seed=0)
    assert out["repeats"] == 1


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

def test_the_majority_baseline_gives_a_well_its_biggest_guide(small_screen):
    called = av.baseline_majority(small_screen)
    assert len(called) == len(small_screen)
    for well in set(small_screen.wells):
        here = {called[i] for i, w in enumerate(small_screen.wells)
                if w == well}
        assert here == {max(small_screen.fractions[well],
                            key=small_screen.fractions[well].get)}


def test_a_well_sequencing_reported_nothing_for_is_left_unannotated(
        small_screen):
    """Inventing a guide for a well with no counts would put cells behind a
    label no sequencing supports."""
    blank = av.Screen(features=small_screen.features,
                      scores=small_screen.scores, wells=small_screen.wells,
                      truth=small_screen.truth,
                      fractions={w: {} for w in small_screen.fractions},
                      true_fractions=small_screen.true_fractions,
                      guides=small_screen.guides)
    assert set(av.baseline_majority(blank)) == {"Non_annotated"}
    assert set(av.baseline_chance(blank)) == {"Non_annotated"}


def test_fractions_that_are_all_zero_leave_the_well_unannotated(small_screen):
    """There is nothing to sample from, and normalising zero counts would
    divide by zero to invent a uniform draw."""
    zeroed = av.Screen(features=small_screen.features,
                       scores=small_screen.scores, wells=small_screen.wells,
                       truth=small_screen.truth,
                       fractions={w: {g: 0.0 for g in f}
                                  for w, f in small_screen.fractions.items()},
                       true_fractions=small_screen.true_fractions,
                       guides=small_screen.guides)
    assert set(av.baseline_chance(zeroed)) == {"Non_annotated"}


def test_the_chance_baseline_only_ever_names_a_guide_that_well_reported(
        small_screen):
    called = av.baseline_chance(small_screen, seed=5)
    for index, well in enumerate(small_screen.wells):
        assert called[index] in small_screen.fractions[well]


# --------------------------------------------------------------------------
# the benchmark
# --------------------------------------------------------------------------

def test_the_baselines_are_included_whether_or_not_a_caller_asked():
    """They are what separates 'the method works' from 'the fractions work',
    and a caller who forgot them would read the second as the first."""
    scenes = {"tiny": av.synthesise(wells=4, guides_per_well=2,
                                    cells_per_well=8, guides=3, seed=8)}
    out = av.benchmark({}, scenes)
    assert set(out["tiny"]) == set(av.BASELINES)


def test_a_benchmark_reports_the_gain_over_the_permuted_null():
    scenes = {"clean": av.synthesise(wells=6, guides_per_well=2,
                                     cells_per_well=10, guides=4, seed=9)}

    def always_first(screen):
        return [sorted(screen.fractions[w])[0] for w in screen.wells]

    out = av.benchmark({"first": always_first}, scenes)
    entry = out["clean"]["first"]
    assert entry["gain"] == pytest.approx(
        entry["real"].precision - entry["null"].precision)


def test_a_strategy_that_raises_is_recorded_not_allowed_to_end_the_run():
    """One broken method must not cost the comparison every other method."""
    scenes = {"tiny": av.synthesise(wells=4, guides_per_well=2,
                                    cells_per_well=8, guides=3, seed=10)}

    def broken(screen):
        raise RuntimeError("no model loaded")

    out = av.benchmark({"broken": broken}, scenes)
    assert out["tiny"]["broken"] == {
        "error": "RuntimeError: no model loaded"}
    assert "real" in out["tiny"]["baseline:majority"]


def test_a_strategy_that_only_fails_on_the_null_keeps_its_real_result():
    """The real number is still worth reporting, and the missing null is
    stated rather than replaced by a zero that would inflate the gain."""
    scenes = {"tiny": av.synthesise(wells=4, guides_per_well=2,
                                    cells_per_well=8, guides=3, seed=11)}

    def only_on_real(screen):
        if screen.meta.get("permuted"):
            raise ValueError("cannot run on a permuted screen")
        return list(screen.truth)

    out = av.benchmark({"picky": only_on_real}, scenes)
    entry = out["tiny"]["picky"]
    assert entry["real"].precision == 1.0
    assert entry["null_error"].startswith("ValueError")
    assert "gain" not in entry


def test_the_default_scenarios_isolate_the_confounds_they_are_named_for():
    scenes = av.default_scenarios(seed=100)
    assert set(scenes) == {"clean", "no_effect", "penetrance_0.5",
                           "inflated_fractions", "classifier_0.94",
                           "crowded", "realistic"}
    assert scenes["no_effect"].meta["effect"] == 0.0
    assert scenes["penetrance_0.5"].meta["penetrance"] == 0.5
    assert scenes["inflated_fractions"].meta["fraction_bias"] == 1.8
    assert scenes["classifier_0.94"].meta["classifier_accuracy"] == 0.94
    assert len(scenes["crowded"].guides) == 16


# --------------------------------------------------------------------------
# mixed-ratio control wells
# --------------------------------------------------------------------------

def _mixture(rng, proportion, n=40, offset=4.0):
    positive = rng.normal(loc=offset, size=(int(n * proportion), 3))
    negative = rng.normal(loc=0.0, size=(n - int(n * proportion), 3))
    return np.vstack([positive, negative])


def test_a_mixed_well_recovers_the_share_of_control_cells_in_it():
    rng = np.random.default_rng(0)
    pure_pc = rng.normal(loc=4.0, size=(200, 3))
    pure_nc = rng.normal(loc=0.0, size=(200, 3))
    seen = av.mixture_proportion(_mixture(rng, 0.5, n=200), pure_pc, pure_nc)
    assert seen == pytest.approx(0.5, abs=0.1)


def test_a_mixture_with_nothing_to_measure_is_nan_not_a_number():
    empty = np.empty((0, 3))
    ones = np.ones((5, 3))
    assert np.isnan(av.mixture_proportion(empty, ones, ones))
    assert np.isnan(av.mixture_proportion(ones, empty, ones))
    assert np.isnan(av.mixture_proportion(ones, ones, empty))


def test_two_indistinguishable_controls_give_no_proportion_at_all():
    """There is no line to project onto, and any number would be invented."""
    same = np.ones((10, 3))
    assert np.isnan(av.mixture_proportion(np.zeros((10, 3)), same, same))


def _ratio_series(rng, shares, n=60):
    features, wells, reported = [], [], {}
    for index, share in enumerate(shares):
        label = f"w{index}"
        features.append(_mixture(rng, share, n=n))
        wells.extend([label] * n)
        reported[label] = share
    return np.vstack(features), wells, reported


def test_a_ratio_series_whose_sequencing_agrees_has_a_slope_of_about_one():
    rng = np.random.default_rng(1)
    features, wells, reported = _ratio_series(
        rng, [0.0, 0.25, 0.5, 0.75, 1.0])
    out = av.mixed_ratio_calibration(features, wells, reported,
                                     pure_pc_wells=["w4"],
                                     pure_nc_wells=["w0"])
    assert out["slope"] == pytest.approx(1.0, abs=0.15)
    assert out["reading"] == "sequencing's fractions match the cellular " \
                             "fractions"
    assert out["reference_wells_from_design"] is True
    assert "warning" not in out


def test_picking_the_pure_wells_by_the_number_under_test_is_flagged():
    """Choosing them by the REPORTED fraction is circular: that fraction is
    precisely the biased quantity being measured."""
    rng = np.random.default_rng(2)
    features, wells, reported = _ratio_series(
        rng, [0.0, 0.25, 0.5, 0.75, 1.0])
    out = av.mixed_ratio_calibration(features, wells, reported)
    assert out["reference_wells_from_design"] is False
    assert "pure_pc_wells / pure_nc_wells" in out["warning"]


def test_inflated_sequencing_is_read_as_overstating_the_cellular_fraction():
    """The direction filter-renormalisation produces, named in the reading
    rather than left as a number below one."""
    rng = np.random.default_rng(3)
    truth = [0.0, 0.2, 0.4, 0.6, 0.9]
    features, wells, _ = _ratio_series(rng, truth)
    reported = {f"w{i}": min(1.0, share * 1.8)
                for i, share in enumerate(truth)}
    out = av.mixed_ratio_calibration(features, wells, reported,
                                     pure_pc_wells=["w4"],
                                     pure_nc_wells=["w0"])
    assert out["slope"] < 0.85
    assert "overstate" in out["reading"]


def test_a_plate_with_no_pure_well_at_one_end_cannot_be_calibrated():
    rng = np.random.default_rng(4)
    features, wells, reported = _ratio_series(rng, [0.4, 0.5, 0.6])
    out = av.mixed_ratio_calibration(features, wells, reported)
    assert out["pure_pc"] == out["pure_nc"] == 0
    assert "no pure wells" in out["error"]


def test_wells_with_too_few_cells_to_average_are_left_out_of_the_fit():
    rng = np.random.default_rng(5)
    features, wells, reported = _ratio_series(rng, [0.0, 1.0], n=30)
    features = np.vstack([features, rng.normal(size=(2, 3))])
    wells = list(wells) + ["thin", "thin"]
    reported["thin"] = 0.5
    out = av.mixed_ratio_calibration(features, wells, reported,
                                     pure_pc_wells=["w1"],
                                     pure_nc_wells=["w0"])
    assert "thin" not in out.get("per_well", {})
    assert out["error"].startswith("only 2 usable well(s)")


def test_a_well_whose_proportion_cannot_be_estimated_is_left_out(monkeypatch):
    rng = np.random.default_rng(6)
    features, wells, reported = _ratio_series(rng, [0.0, 0.5, 1.0], n=20)
    monkeypatch.setattr(av, "mixture_proportion",
                        lambda *a, **k: float("nan"))
    out = av.mixed_ratio_calibration(features, wells, reported,
                                     pure_pc_wells=["w2"],
                                     pure_nc_wells=["w0"])
    assert out["per_well"] == {}
    assert "usable well" in out["error"]


def test_a_series_at_one_reported_value_yields_no_slope_at_all():
    """Every pair has the same x, so there is no line; NaN says so instead of
    a slope produced by dividing by zero."""
    slope, intercept = av._theil_sen(np.array([0.5, 0.5, 0.5]),
                                     np.array([0.1, 0.2, 0.3]))
    assert np.isnan(slope) and np.isnan(intercept)


def test_a_slope_that_could_not_be_fitted_says_so():
    assert av._read_slope(float("nan")) == "no slope could be fitted"


def test_imaging_finding_more_control_cells_is_read_as_understated():
    assert "understate" in av._read_slope(1.5)


# --------------------------------------------------------------------------
# count agreement
# --------------------------------------------------------------------------

def test_count_agreement_compares_called_shares_with_reported_ones():
    called = ["g1", "g1", "g2", "g1"]
    wells = ["w1", "w1", "w1", "w2"]
    out = av.count_agreement(called, wells, {"w1": 0.5, "w2": 1.0}, "g1")
    assert out["per_well"] == {"w1": (0.5, pytest.approx(2 / 3)),
                               "w2": (1.0, 1.0)}
    assert out["worst_absolute_error"] == pytest.approx(1 / 6)
    assert out["wells"] == 2


def test_a_well_with_no_imaged_cells_is_skipped_rather_than_scored_zero():
    """No cells is not the same as no calls, and scoring it 0% would report
    a disagreement that was never observed."""
    out = av.count_agreement(["g1"], ["w1"], {"w1": 1.0, "unimaged": 0.5},
                             "g1")
    assert list(out["per_well"]) == ["w1"]
    assert out["wells"] == 1


def test_no_usable_well_leaves_the_errors_undefined_rather_than_zero():
    out = av.count_agreement([], [], {"w1": 0.5}, "g1")
    assert np.isnan(out["median_absolute_error"])
    assert np.isnan(out["worst_absolute_error"])
    assert out["wells"] == 0
