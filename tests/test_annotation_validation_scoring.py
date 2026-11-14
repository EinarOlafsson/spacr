"""Table-driven tests of the simulation, scoring and calibration checks in
``spacr.annotation_validation``.

The simulated screens are the only place where cell-level guide truth
exists, so they are driven end to end and their invariants are asserted
exactly: what a perfect classifier leaves untouched, what the fraction
threshold and bias change and what they must not.  The scoring helpers
(coverage / precision / recall, confidence calibration, order sensitivity,
the benchmark with its baselines and permuted null, mixed-ratio control
wells and count agreement) are checked with small hand-built inputs whose
answers are known exactly.  One general test per unit, each walking a case
table with pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spacr import annotation_validation as av

NA = "Non_annotated"


def test_synthesise_keeps_truth_and_reported_fractions_apart(subtests):
    """The simulation's knobs change only what they are documented to change."""
    with subtests.test(msg="classifier_error_only_flips_score_signs"):
        kw = dict(wells=4, guides_per_well=2, cells_per_well=20, effect=3.0,
                  seed=1)
        honest = av.synthesise(classifier_accuracy=1.0, **kw)
        sloppy = av.synthesise(classifier_accuracy=0.5, **kw)
        assert int((honest.scores < 0).sum()) == 0
        assert int((sloppy.scores < 0).sum()) == 35
        np.testing.assert_array_equal(np.abs(honest.scores),
                                      np.abs(sloppy.scores))
        np.testing.assert_array_equal(honest.features, sloppy.features)
        assert honest.truth == sloppy.truth
        assert honest.meta == {"effect": 3.0, "penetrance": 1.0,
                               "fraction_bias": 1.0,
                               "classifier_accuracy": 1.0, "seed": 1}

    with subtests.test(msg="threshold_dropping_everything_keeps_the_well"):
        screen = av.synthesise(wells=3, guides_per_well=2, cells_per_well=10,
                               fraction_threshold=1.1, seed=2)
        assert sorted(screen.fractions) == ["w000", "w001", "w002"]
        for well, reported in screen.fractions.items():
            honest = screen.true_fractions[well]
            assert set(reported) == set(honest)
            for guide, share in reported.items():
                assert share == pytest.approx(honest[guide], abs=1e-12)

    with subtests.test(msg="threshold_drops_small_guides_and_renormalises"):
        screen = av.synthesise(wells=6, guides_per_well=3, cells_per_well=10,
                               fraction_threshold=0.2, seed=5)
        for well, reported in screen.fractions.items():
            honest = screen.true_fractions[well]
            kept = {g: p for g, p in honest.items() if p >= 0.2} or honest
            assert set(reported) == set(kept)
            assert sum(reported.values()) == pytest.approx(1.0)

    with subtests.test(msg="bias_inflates_reported_but_not_true_fractions"):
        screen = av.synthesise(wells=4, guides_per_well=2, cells_per_well=10,
                               fraction_bias=1.8, seed=4)
        for well, honest in screen.true_fractions.items():
            assert sum(honest.values()) == pytest.approx(1.0, abs=1e-6)
            assert screen.fractions[well] == {
                g: pytest.approx(min(1.0, p * 1.8)) for g, p in honest.items()}
        assert screen.fractions["w000"] == {
            "g005": 1.0, "g003": pytest.approx(0.6316560344507149)}

    with subtests.test(msg="cells_are_shuffled_within_a_well"):
        screen = av.synthesise(wells=2, guides_per_well=3, cells_per_well=40,
                               guides=5, seed=6)
        truth = list(screen.truth[:40])
        runs = 1 + sum(1 for a, b in zip(truth, truth[1:]) if a != b)
        assert runs == 25
        assert len(set(truth)) == 3
        assert set(screen.wells[:40]) == {"w000"}


def test_score_annotation_reports_coverage_precision_and_recall(subtests):
    """Exact verdicts, including abstention and misaligned call lists."""
    # (case, truth, called, guides, coverage, precision, recall, per_guide,
    #  confusion, n)
    cases = [
        ("mixed_calls", ["a", "a", "b", "b"], ["a", NA, "b", "a"], None,
         0.75, 2 / 3, 0.5, {"a": (0.5, 0.5), "b": (1.0, 0.5)},
         {("a", "a"): 1, ("a", NA): 1, ("b", "b"): 1, ("b", "a"): 1}, 4),
        ("labelling_everything_unreliably", ["a"] * 10, ["a"] * 5 + ["b"] * 5,
         None, 1.0, 0.5, 0.5, {"a": (1.0, 0.5)},
         {("a", "a"): 5, ("a", "b"): 5}, 10),
        ("abstaining_costs_coverage_not_precision", ["a"] * 10,
         ["a"] * 5 + [NA] * 5, None, 0.5, 1.0, 0.5, {"a": (1.0, 0.5)},
         {("a", "a"): 5, ("a", NA): 5}, 10),
        ("guide_nobody_called_or_carried", ["a", "a"], ["a", "a"],
         ["a", "unused"], 1.0, 1.0, 1.0,
         {"a": (1.0, 1.0), "unused": (0.0, 0.0)}, {("a", "a"): 2}, 2),
        ("everything_abstained", ["a", "b"], [NA, NA], None, 0.0, 0.0, 0.0,
         {"a": (0.0, 0.0), "b": (0.0, 0.0)}, {("a", NA): 1, ("b", NA): 1}, 2),
        ("one_cell_per_confusion_cell", ["a", "a", "b"], ["a", "b", "b"],
         None, 1.0, 2 / 3, 2 / 3, {"a": (1.0, 0.5), "b": (0.5, 1.0)},
         {("a", "a"): 1, ("a", "b"): 1, ("b", "b"): 1}, 3),
        ("wrong_length_scores_nothing", ["a", "b", "c"], ["a", "b"], None,
         0.0, 0.0, 0.0, {}, {}, 3),
        ("nothing_to_score", [], [], None, 0.0, 0.0, 0.0, {}, {}, 0),
    ]
    for (case_id, truth, called, guides, coverage, precision, recall,
         per_guide, confusion, n) in cases:
        with subtests.test(msg=case_id):
            verdict = av.score_annotation(truth, called, guides=guides)
            assert verdict.coverage == pytest.approx(coverage)
            assert verdict.precision == pytest.approx(precision)
            assert verdict.recall == pytest.approx(recall)
            assert verdict.per_guide == {
                k: pytest.approx(v) for k, v in per_guide.items()}
            assert verdict.confusion == confusion
            assert verdict.n == n
            if n and len(called) == n:
                assert sum(verdict.confusion.values()) == n

    with subtests.test(msg="summary_line"):
        verdict = av.score_annotation(["a", "a", "b", "b"],
                                      ["a", NA, "b", "a"])
        assert verdict.summary() == ("75% annotated, 67% of those correct "
                                     "(50% of all cells)")

    with subtests.test(msg="custom_abstain_label"):
        verdict = av.score_annotation(["a", "a"], ["a", "?"], abstain="?")
        assert (verdict.coverage, verdict.precision, verdict.recall) == \
            (0.5, 1.0, 0.5)


def test_calibration_bins_only_called_cells_with_finite_confidence(subtests):
    """Populated bins report (mean confidence, accuracy, count) exactly."""
    cases = [
        ("abstentions_are_not_binned", ["a", "a", "b", "b"],
         ["a", "a", NA, "b"], [0.95, 0.9, 0.99, 0.1], 2,
         [(0.1, 1.0, 1), (0.925, 1.0, 2)]),
        ("confidence_one_lands_in_the_top_bin", ["a"], ["a"], [1.0], 4,
         [(1.0, 1.0, 1)]),
        ("both_ends_and_a_wrong_call", ["a", "b"], ["a", "a"], [1.0, 0.0], 4,
         [(0.0, 0.0, 1), (1.0, 1.0, 1)]),
        ("default_five_bins", ["a", "a", "b"], ["a", "b", "b"],
         [0.5, 0.55, 0.3], 5, [(0.3, 1.0, 1), (0.525, 0.5, 2)]),
        ("only_abstentions", ["a"], [NA], [0.9], 5, []),
        ("only_nan_confidence", ["a"], ["a"], [float("nan")], 5, []),
        ("no_confidences", ["a", "b"], ["a", "b"], [], 5, []),
    ]
    for case_id, truth, called, confidence, bins, expected in cases:
        with subtests.test(msg=case_id):
            out = av.calibration(truth, called, confidence, bins=bins)
            assert out == [(pytest.approx(m), pytest.approx(acc), count)
                           for m, acc, count in expected]


def test_order_sensitivity_measures_how_calls_move_with_ranking_order(
        subtests):
    """Stable, sensitive, empty and length-changing methods, exactly."""
    def calls_from(sequence):
        answers = iter(sequence)
        return lambda order: next(answers)

    cases = [
        ("ignores_the_order",
         lambda order: ["a", "b", "c"],
         [(f"g{i}", 1.0 - i / 10) for i in range(6)], 3, 0,
         {"changed": 0.0, "worst": 0.0, "repeats": 3}),
        ("hands_out_the_ranking_order",
         lambda order: [name for name, _ in order],
         [(f"g{i}", 0.5) for i in range(8)], 5, 1,
         {"changed": 0.675, "worst": 0.875, "repeats": 5}),
        ("annotates_nothing", lambda order: [], [("g0", 1.0)], 3, 0,
         {"changed": 0.0, "repeats": 0}),
        ("shorter_repeats_are_discarded",
         calls_from([["a", "b"], ["a"], ["a", "b"], ["a"]]),
         [("g0", 1.0), ("g1", 0.5)], 3, 0,
         {"changed": 0.0, "worst": 0.0, "repeats": 1}),
        ("every_repeat_discarded",
         calls_from([["a", "b"], ["a"], ["a"]]),
         [("g0", 1.0), ("g1", 0.5)], 2, 0,
         {"changed": 0.0, "worst": 0.0, "repeats": 0}),
        ("one_cell_always_changes",
         calls_from([["a", "b"], ["a", "c"], ["a", "c"]]),
         [("g0", 1.0)], 2, 0,
         {"changed": 0.5, "worst": 0.5, "repeats": 2}),
    ]
    for case_id, run, ranking, repeats, seed, expected in cases:
        with subtests.test(msg=case_id):
            out = av.order_sensitivity(run, ranking, repeats=repeats,
                                       seed=seed)
            assert out == {k: pytest.approx(v) for k, v in expected.items()}


def test_benchmark_scores_every_strategy_with_baselines_and_a_null(subtests):
    """Baselines are always added; failures are recorded, never fatal."""
    scenes = {"clean": av.synthesise(wells=6, guides_per_well=2,
                                     cells_per_well=10, guides=4, seed=9)}

    def always_first(screen):
        return [sorted(screen.fractions[w])[0] for w in screen.wells]

    def broken(screen):
        raise RuntimeError("no model loaded")

    def only_on_real(screen):
        if screen.meta.get("permuted"):
            raise ValueError("cannot run on a permuted screen")
        return list(screen.truth)

    out = av.benchmark({"first": always_first, "broken": broken,
                        "picky": only_on_real}, scenes)
    entries = out["clean"]

    with subtests.test(msg="every_strategy_and_both_baselines"):
        assert list(out) == ["clean"]
        assert list(entries) == ["baseline:majority", "baseline:chance",
                                 "first", "broken", "picky"]
        assert set(av.BASELINES) == {"baseline:majority", "baseline:chance"}

    precisions = {"baseline:majority": (23 / 30, 2 / 30),
                  "baseline:chance": (22 / 30, 3 / 30),
                  "first": (14 / 30, 11 / 30)}
    for name, (real, null) in precisions.items():
        with subtests.test(msg=f"gain_over_null:{name}"):
            entry = entries[name]
            assert set(entry) == {"real", "null", "gain"}
            assert entry["real"].precision == pytest.approx(real)
            assert entry["null"].precision == pytest.approx(null)
            assert entry["real"].n == entry["null"].n == 60
            assert entry["gain"] == pytest.approx(real - null)

    with subtests.test(msg="a_raising_strategy_is_recorded"):
        assert entries["broken"] == {"error": "RuntimeError: no model loaded"}

    with subtests.test(msg="failing_only_on_the_null_keeps_the_real_result"):
        entry = entries["picky"]
        assert set(entry) == {"real", "null_error"}
        assert entry["real"].precision == 1.0
        assert entry["null_error"] == \
            "ValueError: cannot run on a permuted screen"

    with subtests.test(msg="no_strategies_still_runs_the_baselines"):
        tiny = {"tiny": av.synthesise(wells=4, guides_per_well=2,
                                      cells_per_well=8, guides=3, seed=8)}
        assert list(av.benchmark({}, tiny)["tiny"]) == list(av.BASELINES)


def test_mixture_proportion_projects_a_well_onto_the_control_line(subtests):
    """The PC share of a well's mean, clipped to [0, 1]; NaN when undefined."""
    pc = np.full((2, 3), 4.0)
    nc = np.zeros((2, 3))
    empty = np.empty((0, 3))
    ones = np.ones((5, 3))
    cases = [
        ("quarter_positive", np.vstack([np.full((1, 3), 4.0),
                                        np.zeros((3, 3))]), pc, nc, 0.25),
        ("half_positive", np.vstack([pc, nc]), pc, nc, 0.5),
        ("pure_positive", pc, pc, nc, 1.0),
        ("beyond_positive_clips_to_one", np.full((3, 3), 9.0), pc, nc, 1.0),
        ("below_negative_clips_to_zero", np.full((3, 3), -2.0), pc, nc, 0.0),
        ("empty_well", empty, ones, ones, math.nan),
        ("empty_positive_reference", ones, empty, ones, math.nan),
        ("empty_negative_reference", ones, ones, empty, math.nan),
        ("indistinguishable_controls", np.zeros((10, 3)), ones, ones,
         math.nan),
    ]
    for case_id, well, positive, negative, expected in cases:
        with subtests.test(msg=case_id):
            seen = av.mixture_proportion(well, positive, negative)
            if math.isnan(expected):
                assert math.isnan(seen)
            else:
                assert seen == pytest.approx(expected)


def _ratio_series(shares, n=20):
    """Wells whose cells are exactly ``share`` PC (all 4.0) and NC (all 0.0)."""
    features, wells, reported = [], [], {}
    for index, share in enumerate(shares):
        label = f"w{index}"
        k = int(round(share * n))
        features.append(np.vstack([np.full((k, 3), 4.0),
                                   np.zeros((n - k, 3))]))
        wells.extend([label] * n)
        reported[label] = share
    return np.vstack(features), wells, reported


MATCH = "sequencing's fractions match the cellular fractions"
OVERSTATE = ("imaging finds FEWER control cells than sequencing reported: "
             "the reported fractions overstate the cellular ones, which is "
             "the direction filter-renormalisation produces")
UNDERSTATE = ("imaging finds MORE control cells than sequencing reported: "
              "the reported fractions understate the cellular ones")
CIRCULAR = ("the pure wells were identified by their REPORTED fraction, "
            "which is the quantity under test -- name them from the plate "
            "design with pure_pc_wells / pure_nc_wells")
EVEN = [0.0, 0.25, 0.5, 0.75, 1.0]


def test_mixed_ratio_calibration_reads_the_imaging_vs_sequencing_slope(
        subtests, monkeypatch):
    """Slope, reading and refusals for exact mixed-ratio control plates."""
    design = {"pure_pc_wells": ["w4"], "pure_nc_wells": ["w0"]}
    features, wells, reported = _ratio_series(EVEN)
    biased = [0.0, 0.2, 0.4, 0.6, 1.0]
    b_features, b_wells, _ = _ratio_series(biased)
    # (case, features, wells, reported, kwargs, slope, intercept, reading,
    #  per_well, from_design)
    fits = [
        ("agreeing_sequencing_from_design", features, wells, reported,
         design, 1.0, 0.0, MATCH, {f"w{i}": (s, s) for i, s in
                                   enumerate(EVEN)}, True),
        ("agreeing_sequencing_reference_by_reported_fraction", features,
         wells, reported, {}, 1.0, 0.0, MATCH,
         {f"w{i}": (s, s) for i, s in enumerate(EVEN)}, False),
        ("inflated_sequencing_overstates", b_features, b_wells,
         {f"w{i}": min(1.0, s * 1.8) for i, s in enumerate(biased)}, design,
         0.625, -0.025, OVERSTATE,
         {"w0": (0.0, 0.0), "w1": (0.36, 0.2), "w2": (0.72, 0.4),
          "w3": (1.0, 0.6), "w4": (1.0, 1.0)}, True),
        ("deflated_sequencing_understates", features, wells,
         {f"w{i}": s / 2 for i, s in enumerate(EVEN)}, design, 2.0, 0.0,
         UNDERSTATE, {f"w{i}": (s / 2, s) for i, s in enumerate(EVEN)},
         True),
    ]
    for (case_id, f, w, r, kwargs, slope, intercept, reading, per_well,
         from_design) in fits:
        with subtests.test(msg=case_id):
            out = av.mixed_ratio_calibration(f, w, r, **kwargs)
            assert out["slope"] == pytest.approx(slope)
            assert out["intercept"] == pytest.approx(intercept, abs=1e-12)
            assert out["reading"] == reading
            assert out["per_well"] == {k: pytest.approx(v)
                                       for k, v in per_well.items()}
            assert out["wells"] == len(per_well)
            assert (out["pure_pc_wells"], out["pure_nc_wells"]) == (1, 1)
            assert out["reference_wells_from_design"] is from_design
            assert out.get("warning") == (None if from_design else CIRCULAR)

    with subtests.test(msg="no_pure_well_at_one_end"):
        f, w, r = _ratio_series([0.4, 0.5, 0.6])
        assert av.mixed_ratio_calibration(f, w, r) == {
            "error": "no pure wells at either end; the references that "
                     "define the mixture line cannot be estimated",
            "pure_pc": 0, "pure_nc": 0}

    with subtests.test(msg="wells_with_under_three_cells_are_left_out"):
        f, w, r = _ratio_series([0.0, 1.0])
        f = np.vstack([f, np.ones((2, 3))])
        r["thin"] = 0.5
        out = av.mixed_ratio_calibration(f, w + ["thin", "thin"], r,
                                         pure_pc_wells=["w1"],
                                         pure_nc_wells=["w0"])
        assert out == {"error": "only 2 usable well(s); a slope needs more",
                       "per_well": {"w0": (0.0, 0.0), "w1": (1.0, 1.0)}}

    with subtests.test(msg="unestimable_wells_are_left_out"):
        f, w, r = _ratio_series([0.0, 0.5, 1.0])
        monkeypatch.setattr(av, "mixture_proportion",
                            lambda *a, **k: float("nan"))
        out = av.mixed_ratio_calibration(f, w, r, pure_pc_wells=["w2"],
                                         pure_nc_wells=["w0"])
        monkeypatch.undo()
        assert out == {"error": "only 0 usable well(s); a slope needs more",
                       "per_well": {}}


def test_count_agreement_compares_called_shares_with_reported_ones(subtests):
    """Per-well (reported, called) shares and the median / worst error."""
    cases = [
        ("two_wells", ["g1", "g1", "g2", "g1"], ["w1", "w1", "w1", "w2"],
         {"w1": 0.5, "w2": 1.0}, {"w1": (0.5, 2 / 3), "w2": (1.0, 1.0)},
         1 / 12, 1 / 6),
        ("unimaged_well_is_skipped", ["g1"], ["w1"],
         {"w1": 1.0, "unimaged": 0.5}, {"w1": (1.0, 1.0)}, 0.0, 0.0),
        ("no_usable_well", [], [], {"w1": 0.5}, {}, math.nan, math.nan),
    ]
    for case_id, called, wells, reported, per_well, median, worst in cases:
        with subtests.test(msg=case_id):
            out = av.count_agreement(called, wells, reported, "g1")
            assert out["guide"] == "g1"
            assert out["per_well"] == {k: pytest.approx(v)
                                       for k, v in per_well.items()}
            assert list(out["per_well"]) == list(per_well)
            assert out["wells"] == len(per_well)
            for key, expected in (("median_absolute_error", median),
                                  ("worst_absolute_error", worst)):
                if math.isnan(expected):
                    assert math.isnan(out[key])
                else:
                    assert out[key] == pytest.approx(expected)
