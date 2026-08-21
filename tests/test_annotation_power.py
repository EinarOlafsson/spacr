"""How many cells CAN be annotated, and how big a screen would have to be.

Asked 2026-08-21: "i dont need all cells annotated ... with no method can we
annotate every cell, for that we would need a much largesr screen. that is
something that can be printed in the textbox under the graph".

COVERAGE IS NOT A GOAL. A method that annotates every cell has not done
well, it has declined to abstain -- so the report ranks by cells got RIGHT,
which is the only ordering that rewards neither annotating everything badly
nor annotating nothing safely.

EVERY NUMBER COMES FROM THE CALLER'S SCREEN. Nothing here carries a measured
coefficient, on the same instruction that emptied `classifier_quality`.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import annotation_power as module


class TestTheArithmeticOfARareGuide:

    def test_a_dominant_guide_is_nearly_certain(self):
        assert module.posterior_for_prior(0.5, 0.96, 0.98) > 0.95

    def test_a_rare_guide_is_a_coin_flip_or_worse(self):
        """False positives are a share of the NEGATIVES: when a guide is
        rare, almost every cell in the well is a negative."""
        assert module.posterior_for_prior(0.01, 0.96, 0.98) < 0.5

    def test_the_floor_is_where_those_two_meet(self):
        floor = module.required_fraction(0.96, 0.98, decision=0.55)
        assert module.posterior_for_prior(floor, 0.96, 0.98) == \
            pytest.approx(0.55, abs=1e-6)

    def test_a_better_classifier_lowers_the_floor(self):
        loose = module.required_fraction(0.96, 0.95)
        tight = module.required_fraction(0.96, 0.999)
        assert tight < loose / 5

    def test_specificity_moves_it_far_more_than_sensitivity(self):
        """Which is the practical lesson: the false-positive rate is what
        multiplies against the many, and the sensitivity only against the
        few."""
        base = module.required_fraction(0.96, 0.98)
        better_se = module.required_fraction(0.99, 0.98)
        better_sp = module.required_fraction(0.96, 0.995)
        assert abs(better_se - base) < abs(better_sp - base) / 3


class TestWhatAScreenCanSupport:

    @staticmethod
    def _crowded(wells=10, guides=200, seed=0):
        """A screen shaped like a real pooled one: many guides per well."""
        rng = np.random.default_rng(seed)
        out = {}
        for w in range(wells):
            shares = rng.dirichlet(np.ones(guides) * 0.4)
            out[f"w{w}"] = {f"g{i}": float(s) for i, s in enumerate(shares)}
        return out

    def test_most_of_a_crowded_screen_is_out_of_reach(self):
        power = module.annotatable(self._crowded(), sensitivity=0.96,
                                   specificity=0.98)
        assert power["pairs_clearing_share"] < 0.10
        assert power["guides_unreachable"] > 0

    def test_a_sparse_screen_is_reachable(self):
        fractions = {f"w{i}": {"a": 0.6, "b": 0.4} for i in range(8)}
        power = module.annotatable(fractions, sensitivity=0.96,
                                   specificity=0.98)
        assert power["guides_reachable_share"] == 1.0

    def test_it_counts_the_guides_no_method_can_ever_reach(self):
        """The headline: a guide that never clears the floor in ANY well
        cannot be annotated anywhere, so no method development will produce
        a single cell for it. That is the experiment, not the algorithm."""
        fractions = {"w0": {"big": 0.9, "trace": 0.001},
                     "w1": {"big": 0.8, "trace": 0.002}}
        power = module.annotatable(fractions, sensitivity=0.96,
                                   specificity=0.98)
        assert power["guides_unreachable"] == 1
        assert power["guides_reachable"] == 1


class TestHowMuchBigger:

    def test_it_asks_for_fewer_guides_per_well_not_more_cells(self):
        fractions = {f"w{i}": {f"g{j}": 1 / 300 for j in range(300)}
                     for i in range(20)}
        size = module.screen_size_for(fractions, sensitivity=0.96,
                                      specificity=0.98)
        assert size["guides_per_well_needed"] < size["guides_per_well_now"]
        assert size["wells_multiplier"] > 1.0

    def test_a_screen_already_sparse_enough_needs_no_growth(self):
        fractions = {f"w{i}": {"a": 0.5, "b": 0.5} for i in range(30)}
        size = module.screen_size_for(fractions, sensitivity=0.96,
                                      specificity=0.999)
        assert size["wells_multiplier"] <= 1.5

    def test_the_classifier_is_offered_as_the_other_lever(self):
        fractions = {f"w{i}": {f"g{j}": 1 / 300 for j in range(300)}
                     for i in range(20)}
        size = module.screen_size_for(fractions, sensitivity=0.96,
                                      specificity=0.98)
        needed = size["specificity_needed_at_current_shape"]
        assert 0.98 < needed < 1.0, needed


class TestTheTextbox:

    class _Verdict:
        def __init__(self, coverage, precision, recall, n=100):
            self.coverage, self.precision = coverage, precision
            self.recall, self.n = recall, n

    def test_it_ranks_by_cells_got_right_not_by_coverage(self):
        text = module.quality_report({
            "annotates_everything": self._Verdict(1.0, 0.20, 0.20),
            "careful": self._Verdict(0.30, 0.90, 0.27),
        })
        lines = [l for l in text.splitlines()
                 if l.startswith(("annotates_everything", "careful"))]
        assert lines[0].startswith("careful"), text

    def test_it_says_abstaining_is_correct(self):
        text = module.quality_report({"x": self._Verdict(1.0, 0.2, 0.2)})
        assert "declined to abstain" in text

    def test_it_reports_the_unreachable_guides(self):
        fractions = {"w0": {"big": 0.9, "trace": 0.001}}
        text = module.quality_report(
            {"x": self._Verdict(0.1, 0.9, 0.09)},
            power=module.annotatable(fractions, sensitivity=0.96,
                                     specificity=0.98))
        assert "never reach it in ANY well" in text

    def test_it_says_more_cells_do_not_help(self):
        """The counterintuitive part, and the one worth printing before
        somebody images more fields."""
        fractions = {f"w{i}": {f"g{j}": 1 / 300 for j in range(300)}
                     for i in range(20)}
        text = module.quality_report(
            {"x": self._Verdict(0.1, 0.9, 0.09)},
            size=module.screen_size_for(fractions, sensitivity=0.96,
                                        specificity=0.98))
        assert "MORE CELLS PER WELL DO NOT MOVE ANY OF THIS" in text

    def test_it_works_with_nothing_but_the_verdicts(self):
        text = module.quality_report({"x": self._Verdict(0.5, 0.8, 0.4)})
        assert "ANNOTATION QUALITY" in text


class TestNothingIsBakedIn:

    def test_the_module_holds_no_measured_coefficients(self):
        for name in dir(module):
            if name.startswith("_"):
                continue
            value = getattr(module, name)
            assert not isinstance(value, (dict, list, tuple)) or not any(
                isinstance(v, dict) and "sensitivity" in v
                for v in (value.values() if isinstance(value, dict)
                          else value)), name

    def test_the_classifier_numbers_are_required_arguments(self):
        import inspect

        for name in ("annotatable", "screen_size_for"):
            signature = inspect.signature(getattr(module, name))
            for parameter in ("sensitivity", "specificity"):
                assert (signature.parameters[parameter].default
                        is inspect.Parameter.empty), f"{name}.{parameter}"

    def test_required_fraction_takes_them_positionally(self):
        import inspect

        signature = inspect.signature(module.required_fraction)
        names = list(signature.parameters)
        assert names[:2] == ["sensitivity", "specificity"]
