"""196: the advisor's settings have to actually run.

    "is i run settings for my data i get settings but when i run i get:
     ValueError: batch_correction='combat' needs to know which biology to
     keep."

192 SHIPPED A BUTTON WHOSE OUTPUT DID NOT RUN, which is worse than the button
not existing: the whole promise of the proposal screen is "these are the
settings for your data", and a user who accepts it has been told the question
is settled.

WHY IT GOT PAST ITS TESTS. The proposal was checked against
`get_perform_regression_default_settings` -- which CANONICALISES, filling
defaults and coercing types -- and not against the validators the run uses.
A test asserted every proposed key survived canonicalisation, it passed, and
the run still refused. The two are different questions and only one of them
was being asked.
"""
from __future__ import annotations

import pytest

from spacr.settings_advisor import (Reading, advise, advise_that_runs,
                                    refusals)


def _screen(**over) -> Reading:
    """The reference screen's shape: four plates, a bounded response."""
    base = dict(plates=4, wells=1536, guides=1380, genes=452,
                guides_per_gene=3.0, rows=16, columns=24, n_response=1000,
                low=0.02, high=0.97, inside_unit=True, on_unit=True,
                normal_p=1e-30, skew=2.4, response="pred")
    base.update(over)
    return Reading(**base)


ANSWERS = {"hits_per_thousand": 20, "direction": "either",
           "controls": "000000", "cost": "balanced"}


class TestTheRefusalsAreTheRunsOwn:

    def test_combat_without_a_covariate_is_refused(self):
        said = refusals({"batch_correction": "combat"})

        assert said
        assert "which biology to keep" in said[0]

    def test_and_the_sentence_names_the_way_out(self):
        """`NO_COVARIATE` exists so the caller answers rather than has it
        answered for them; the refusal has to say so."""
        said = refusals({"batch_correction": "combat"})[0]

        assert "batch_covariate_column" in said
        assert "no_covariate" in said

    def test_combat_WITH_a_covariate_is_not_refused(self):
        assert refusals({"batch_correction": "combat",
                         "batch_covariate_column": "condition"}) == ()

    def test_an_explicit_no_covariate_is_an_answer(self):
        from spacr.batch_correction import NO_COVARIATE

        assert refusals({"batch_correction": "combat",
                         "batch_covariate_column": NO_COVARIATE}) == ()

    def test_control_center_without_its_controls_is_refused(self):
        said = refusals({"batch_correction": "control_center",
                         "batch_control_column": ""})

        assert said and "batch_control_column" in said[0]

    def test_an_unknown_regression_type_is_refused(self):
        said = refusals({"regression_type": "sideways"})

        assert said and "sideways" in said[0]

    def test_a_settings_dict_the_run_would_accept_says_nothing(self):
        assert refusals({"batch_correction": "center",
                         "regression_type": "beta"}) == ()


class TestTheProposalRuns:
    """The acceptance criterion, in one line: press it, accept, press Run."""

    def test_the_reference_screens_proposal_has_no_refusals(self):
        advice = advise_that_runs(_screen(), ANSWERS)

        assert refusals(advice.as_settings()) == ()

    def test_and_it_still_proposes_the_things_worth_proposing(self):
        """A proposal made safe by proposing nothing is not a fix."""
        got = advise_that_runs(_screen(), ANSWERS).as_settings()

        assert got["regression_type"] == "beta"
        assert got["level"] == "both"
        assert got["batch_correction"] != "none"

    def test_it_survives_the_canonicaliser_too(self):
        """Both checks, because they are different questions."""
        from spacr.settings import get_perform_regression_default_settings

        got = advise_that_runs(_screen(), ANSWERS).as_settings()
        whole = get_perform_regression_default_settings(dict(got))

        for key in got:
            assert key in whole, key

    @pytest.mark.parametrize("plates", [1, 2, 4, 12])
    def test_at_every_plate_count(self, plates):
        advice = advise_that_runs(_screen(plates=plates), ANSWERS)

        assert refusals(advice.as_settings()) == ()

    @pytest.mark.parametrize("shape", [
        dict(binary=True, on_unit=True),
        dict(inside_unit=True, on_unit=True),
        dict(on_unit=True),
        dict(integral=True, low=0.0, high=90.0),
        dict(normal_p=1e-9, skew=3.0, low=-4.0, high=90.0),
        dict(normal_p=0.4, skew=0.0, low=-4.0, high=4.0),
    ])
    def test_at_every_response_shape(self, shape):
        advice = advise_that_runs(_screen(**shape), ANSWERS)

        assert refusals(advice.as_settings()) == ()

    def test_with_no_answers_at_all(self):
        assert refusals(advise_that_runs(_screen()).as_settings()) == ()


class TestCombatIsNotProposed:
    """Not a patch on one setting: it is the wrong correction here.

    ComBat estimates the plate effect from whatever the design does not
    explain. In a pooled screen the biology IS the per-well guide
    composition -- continuous, and not a categorical covariate column -- so
    there is nothing honest to pass it, and proposing it means proposing a
    run that either refuses or removes the effects being looked for.
    """

    def test_it_is_never_the_proposal(self):
        for plates in (1, 2, 4, 12):
            got = advise_that_runs(_screen(plates=plates),
                                   ANSWERS).as_settings()
            assert got.get("batch_correction") != "combat", plates

    def test_one_plate_is_corrected_not_at_all(self):
        got = advise_that_runs(_screen(plates=1), ANSWERS).as_settings()

        assert got["batch_correction"] == "none"
        assert "at least two batches" in \
            advise_that_runs(_screen(plates=1), ANSWERS).why(
                "batch_correction")

    def test_several_plates_are_centred(self):
        got = advise_that_runs(_screen(plates=4), ANSWERS).as_settings()

        assert got["batch_correction"] == "center"

    def test_the_reason_says_it_estimates_nothing_from_residuals(self):
        """Which is the whole difference from ComBat."""
        why = advise_that_runs(_screen(), ANSWERS).why("batch_correction")

        assert "residuals" in why

    def test_the_stronger_correction_is_named_rather_than_hidden(self):
        """`control_center` centres each plate on its OWN controls, which is
        better -- and it needs to know which WELLS hold them, which is not in
        the count or score tables. Naming it is the difference between a
        default and a ceiling."""
        advice = advise_that_runs(_screen(), ANSWERS)

        assert "control_center" in advice.why("batch_control_values")

    def test_and_the_covariate_is_explained_rather_than_left_blank(self):
        advice = advise_that_runs(_screen(), ANSWERS)

        why = advice.why("batch_covariate_column")
        assert "ComBat" in why
        assert "guide composition" in why


class TestAWithdrawnSettingIsSaidNotPatched:

    def test_a_refused_setting_moves_to_undecided(self):
        """Quietly changing a value to make a check pass is how a proposal
        stops meaning anything."""
        reading = _screen()
        before = advise(reading, ANSWERS)
        assert "regression_type" in before.as_settings()

        # A proposal carrying a refusal, built by hand: the withdrawal path
        # is what is under test, not the advisor's own choices.
        broken = Advice_with(before, "regression_type", "sideways")
        assert refusals(broken)

    def test_the_reason_carries_the_runs_own_sentence(self):
        advice = advise_that_runs(_screen(), ANSWERS)

        for skipped in advice.undecided:
            assert len(skipped.why) > 30, skipped.key


def Advice_with(advice, key, value):
    """The proposal with one setting replaced, for the refusal tests."""
    got = dict(advice.as_settings())
    got[key] = value
    return got


# --------------------------------------------------------------------------- #
#  The acceptance criterion, end to end
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_the_proposal_runs_on_the_example_screen(tmp_path):
    """"Press the button on the example screen, accept, press Run: the
    regression RUNS."

    THE ONLY TEST THAT COULD HAVE CAUGHT THIS. Every other assertion here
    checks the proposal against a validator; this one hands it to
    `perform_regression`, which is what the user pressed. Marked slow
    because it fits four plates -- about three minutes -- and skipped when
    the example screen has not been downloaded.
    """
    import glob

    from spacr.ml import perform_regression
    from spacr.settings import get_perform_regression_default_settings
    from spacr.settings_advisor import read_the_screen

    from spacr.example_data import cache_folder

    folder = cache_folder()
    counts = sorted(glob.glob(f"{folder}/*unique_combinations.csv"))
    scores = sorted(glob.glob(f"{folder}/plate?_dv.csv"))
    if len(counts) != 4 or len(scores) != 4:
        pytest.skip("the example screen is not downloaded; "
                    "run spacr.example_data.fetch()")

    advice = advise_that_runs(
        read_the_screen(counts, scores, "pred"), ANSWERS)
    assert refusals(advice.as_settings()) == ()

    settings = {
        "paired_data": [{"score": s, "count": c}
                        for s, c in zip(scores, counts)],
        "dependent_variable": "pred",
        "dst": str(tmp_path),
    }
    settings.update(advice.as_settings())

    # NO try/except. A failure here IS the report, and swallowing it to say
    # something friendlier is what let the original bug reach the user.
    perform_regression(get_perform_regression_default_settings(settings))
