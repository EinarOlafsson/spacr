"""An impossible design is refused before the run does any work.

Hit live on 2026-08-21: `analysis_unit='cell'` with the permutation test
reached "permuting the guides" THIRTY-ONE SECONDS IN -- after the column
filters, the cell-count threshold plot, three jitter/bar figures and two
saved CSVs -- and only then raised.

THE INCOMPATIBILITY IS KNOWABLE FROM THE SETTINGS ALONE. Nothing about it
needed the data, so nothing about it should have waited for the data. A
check that fires after the work is a check that has already cost the user
the thing it was for.
"""
from __future__ import annotations

import pytest

from spacr.settings_advisor import refusals, requirements_for_unit


def _said(settings) -> str:
    return " | ".join(refusals(settings))


class TestTheCombinationThatWasHit:

    @pytest.mark.parametrize("fragments", [
        ("guide_permutation", "one row per well"),
        ("analysis_unit='well'", "analysis_mode='regression'"),   # both ways out
    ], ids=["refused", "names-both-ways-out"])
    def test_cell_with_the_permutation_test_is_refused_and_names_both_ways_out(
            self, fragments):
        said = _said({"analysis_unit": "cell",
                      "analysis_mode": "guide_permutation"})
        for fragment in fragments:
            assert fragment in said

    def test_it_is_caught_under_the_name_the_user_typed(self):
        """`inference='nonparametric'` SELECTS guide_permutation. The user
        never types the mode, so a refusal that only knows the mode fires
        for a setting nobody set."""
        said = _said({"analysis_unit": "cell",
                      "inference": "nonparametric"})
        assert said
        assert "nonparametric" in said or "guide_permutation" in said

    def test_it_is_not_said_twice_when_both_names_are_present(self):
        """The real settings dict carries both, because `inference` resolves
        into `analysis_mode` before the run."""
        said = refusals({"analysis_unit": "cell",
                         "inference": "nonparametric",
                         "analysis_mode": "guide_permutation"})
        about_permutation = [m for m in said if "permutation" in m]
        assert len(about_permutation) == 1, about_permutation

class TestTheSettingThatWouldBeIgnored:
    """A control the user changed and the run ignored is how somebody
    concludes the setting does nothing."""

    def test_an_aggregation_beside_cell_is_refused(self):
        said = _said({"analysis_unit": "cell", "agg_type": "mean"})
        assert "never read" in said

    def test_no_aggregation_beside_cell_is_fine(self):
        assert refusals({"analysis_unit": "cell",
                         "analysis_mode": "regression",
                         "agg_type": None}) == ()


class TestWellIsUnconstrained:

    def test_the_design_that_ran_is_not_refused(self):
        """The same run with analysis_unit='well' completed in 48 s."""
        assert refusals({"analysis_unit": "well",
                         "analysis_mode": "guide_permutation",
                         "inference": "nonparametric",
                         "agg_type": "mean"}) == ()


class TestWhatAPanelShouldLock:
    """"if cell is chosen then the settings it needs should be chosen and
    displayed and the setting grayed out"."""

    def test_cell_names_what_it_forces(self):
        needed = requirements_for_unit("cell")
        assert needed["analysis_mode"] == "regression"
        assert needed["inference"] == "parametric"

    def test_it_says_which_setting_must_be_empty(self):
        """`None` is a requirement like any other, not an absence of one."""
        needed = requirements_for_unit("cell")
        assert "agg_type" in needed
        assert needed["agg_type"] is None

    @pytest.mark.parametrize("unit", [
        "well",
        "galaxy",   # an unknown unit constrains nothing rather than raising
    ])
    def test_a_unit_that_constrains_nothing_gives_no_requirements(self, unit):
        assert requirements_for_unit(unit) == {}

    def test_applying_the_requirements_clears_the_refusals(self):
        """The end of the contract: what the panel would set must actually
        be runnable."""
        settings = {"analysis_unit": "cell", "inference": "nonparametric",
                    "analysis_mode": "guide_permutation", "agg_type": "mean"}
        settings.update(requirements_for_unit("cell"))
        assert refusals(settings) == ()


class TestTheOtherRefusalsStillFire:
    """The new checks must not have displaced the ones that were there."""

    @pytest.mark.parametrize("correction, fragment", [
        ("combat", "covariate"),
        ("control_center", "batch_control_column"),
    ])
    def test_a_correction_without_what_it_needs_is_still_refused(
            self, correction, fragment):
        said = _said({"batch_correction": correction})
        assert fragment in said

class TestTheUnitChoosesWhatItNeeds:
    """"if cell is chosen then the settings it needs should be chosen and
    displayed and the setting grayed out."

    CHOSEN WHEN IT WAS NOT ASKED FOR, SAID WHEN IT WAS. Left at 'auto' the
    inference is not a conflict -- 'auto' means "pick what the design
    supports", so picking it is the whole job. Set to 'nonparametric'
    explicitly it IS a conflict: two deliberate choices that cannot both
    hold, and resolving that quietly would run something other than what was
    asked for.
    """

    @staticmethod
    def _resolve(settings):
        from spacr.settings import _resolve_regression_analysis_choices

        _resolve_regression_analysis_choices(settings)
        return settings

    def test_auto_becomes_parametric_under_cell(self):
        out = self._resolve({"analysis_unit": "cell", "inference": "auto"})
        assert out["inference"] == "parametric"
        assert out["analysis_mode"] == "regression"

    def test_the_aggregation_is_cleared_with_it(self):
        out = self._resolve({"analysis_unit": "cell", "inference": "auto",
                             "agg_type": "mean"})
        assert out["agg_type"] is None

    def test_a_missing_inference_is_treated_as_auto(self):
        out = self._resolve({"analysis_unit": "cell"})
        assert out["inference"] == "parametric"

    @pytest.mark.parametrize("typed", ["nonparametric", "permutation"])
    def test_an_explicit_conflict_is_refused_in_words(self, typed):
        from spacr.settings import _resolve_regression_analysis_choices

        with pytest.raises(ValueError) as caught:
            _resolve_regression_analysis_choices(
                {"analysis_unit": "cell", "inference": typed})
        said = str(caught.value)
        assert "analysis_unit='well'" in said
        assert "inference='parametric'" in said

    def test_well_is_left_alone(self):
        out = self._resolve({"analysis_unit": "well",
                             "inference": "nonparametric"})
        assert out["inference"] == "nonparametric"
        assert out["analysis_mode"] == "guide_permutation"
        assert out["agg_type"] == "mean"


class TestThePanelGreysWhatTheUnitDecides:

    @staticmethod
    def _rules():
        from spacr.settings import get_setting_dependencies

        return get_setting_dependencies()

    @pytest.mark.parametrize("key", ["inference", "agg_type"])
    def test_it_is_greyed_under_cell_and_live_under_well(self, key):
        rule = self._rules()[key]
        assert rule["predicate"]({"analysis_unit": "well"}, {}) is True
        assert rule["predicate"]({"analysis_unit": "cell"}, {}) is False

    def test_the_reason_names_the_unit_and_the_way_out(self):
        rule = self._rules()["inference"]
        said = rule["reason"]({"analysis_unit": "cell"}, {})
        assert "analysis_unit='cell'" in said
        assert "well" in said

    def test_the_rule_listens_to_the_unit(self):
        """It has to re-evaluate when the unit moves, not only at build."""
        assert "analysis_unit" in self._rules()["inference"]["sources"]
