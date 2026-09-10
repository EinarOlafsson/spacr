"""192: a button that picks the settings for your data.

    "if clicked picks the most correct and best settings for your data fills
    in those settings in teh settings fields. its fine if the button triggers
    a popup that asks the user questions about their data that cant be
    determined by reading the data ... a question might be out of 1000
    perterbations genes how many are expected to be hitts?"

THE DANGER THIS TESTS AGAINST IS AN AUTHORITATIVE GUESS. A button called
"best settings" that silently fills fourteen fields is trusted far past what
it can support, so the assertions here are as much about what it REFUSES to
decide, and about the reason travelling with every value, as about the values
themselves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.settings_advisor import (QUESTIONS, Advice, Reading, advise,
                                    advise_the_screen, questions_for,
                                    read_the_counts, read_the_response,
                                    read_the_screen)


# --------------------------------------------------------------------------- #
#  A screen on disk, shaped like a real one
# --------------------------------------------------------------------------- #

def _counts(path, plates=4, genes=40, guides_per_gene=3, rows=4, columns=6):
    """One row per well and guide, the way `unique_combinations.csv` is."""
    rng = np.random.default_rng(0)
    out = []
    for plate in range(1, plates + 1):
        for row in range(1, rows + 1):
            for column in range(1, columns + 1):
                for gene in range(genes):
                    for guide in range(guides_per_gene):
                        out.append({
                            "plate": f"plate{plate}",
                            "row_name": f"r{row}",
                            "column_name": f"c{column}",
                            "grna_name": f"TGGT1_{gene:06d}_{guide + 1}",
                            "count": int(rng.integers(1, 500)),
                        })
    frame = pd.DataFrame(out)
    frame.to_csv(path, index=False)
    return str(path)


def _scores(path, response, plates=4, rows=4, columns=6, per_well=20):
    rng = np.random.default_rng(1)
    out = []
    for plate in range(1, plates + 1):
        for row in range(1, rows + 1):
            for column in range(1, columns + 1):
                for _ in range(per_well):
                    out.append({
                        "prc": f"plate{plate}_r{row}_c{column}",
                        "pred": float(response(rng)),
                    })
    pd.DataFrame(out).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def screen(tmp_path):
    """A proportion screen: four plates, 40 genes, three guides each."""
    return {
        "counts": [_counts(tmp_path / "counts.csv")],
        "scores": [_scores(tmp_path / "scores.csv",
                           lambda r: float(np.clip(r.beta(2, 5), 1e-4,
                                                   1 - 1e-4)))],
    }


# --------------------------------------------------------------------------- #
#  A. What can be read from the data, and must be
# --------------------------------------------------------------------------- #

class TestItReadsTheDesignExactly:

    def test_the_plates_are_counted(self, screen):
        assert read_the_screen(**screen).plates == 4

    def test_the_wells_guides_and_genes_are_counted(self, screen):
        reading = read_the_screen(**screen)

        assert reading.wells == 4 * 4 * 6
        assert reading.guides == 40 * 3
        assert reading.genes == 40

    def test_the_two_ratios_the_level_turns_on(self, screen):
        reading = read_the_screen(**screen)

        assert reading.guides_per_gene == 3
        assert reading.wells_per_guide == 4 * 4 * 6

    def test_the_gene_is_read_without_assuming_an_organism(self, tmp_path):
        """A `TGGT1_`-shaped rule pools a Plasmodium library into one gene."""
        path = tmp_path / "pf.csv"
        pd.DataFrame({
            "plate": ["plate1"] * 4, "row_name": ["r1"] * 4,
            "column_name": ["c1"] * 4,
            "grna_name": ["PF3D7_0100100_1", "PF3D7_0100100_2",
                          "PF3D7_0100200_1", "PF3D7_0100200_2"],
            "count": [10, 20, 30, 40],
        }).to_csv(path, index=False)

        assert read_the_counts([str(path)])["genes"] == 2

    def test_no_count_table_is_a_sentence(self):
        got = read_the_counts([])

        assert got["trouble"]
        assert "no count table" in got["trouble"][0]


class TestItReadsTheResponseAtTheUnitTheFitUses:

    def test_the_range_is_measured(self, screen):
        reading = read_the_screen(**screen)

        assert 0.0 < reading.low < reading.high < 1.0
        assert reading.inside_unit

    def test_the_response_is_aggregated_to_WELLS(self, tmp_path):
        """A per-object response is aggregated to wells before the model
        touches it, so the family question is about the WELL means -- and the
        object-level spread is much wider than the one being modelled."""
        path = _scores(tmp_path / "s.csv",
                       lambda r: float(r.uniform(0.0, 1.0)), plates=1,
                       rows=2, columns=2, per_well=400)

        reading = read_the_response([path], "pred")

        # 400 uniform draws per well average to near 0.5; the objects
        # themselves span the whole interval.
        assert 0.4 < reading["low"] < 0.6
        assert 0.4 < reading["high"] < 0.6

    def test_every_file_is_read_by_its_OWN_header(self, tmp_path):
        """Taking the columns off the first file and asking every other file
        for them dropped plates 2-4 of the reference screen: plate 1 carries
        `col` and the others do not."""
        first = tmp_path / "one.csv"
        pd.DataFrame({"prc": ["p1_r1_c1"] * 4, "col": ["c1"] * 4,
                      "pred": [0.1, 0.2, 0.3, 0.4]}).to_csv(first, index=False)
        second = tmp_path / "two.csv"
        pd.DataFrame({"prc": ["p2_r1_c1"] * 4,
                      "pred": [0.5, 0.6, 0.7, 0.8]}).to_csv(second, index=False)

        got = read_the_response([str(first), str(second)], "pred")

        assert got["score_files_read"] == 2
        assert not got["trouble"]

    def test_a_missing_dependent_variable_is_named(self, screen):
        got = read_the_response(screen["scores"], "not_a_column")

        assert "not_a_column" in got["trouble"][0]
        assert "response" not in got

    def test_the_sample_cap_is_declared(self, screen):
        reading = read_the_screen(**screen, row_cap=50)

        assert reading.capped
        assert "50" in reading.sample_note()
        assert "sample" in reading.sample_note()

    def test_an_uncapped_read_claims_nothing(self, screen):
        assert read_the_screen(**screen).sample_note() == ""


# --------------------------------------------------------------------------- #
#  B. What cannot be read, and so is asked
# --------------------------------------------------------------------------- #

class TestTheQuestionsAreOnlyTheOnesTheDataCannotAnswer:

    def test_the_maintainers_own_question_is_the_first(self):
        assert QUESTIONS[0].key == "hits_per_thousand"
        assert "1,000" in QUESTIONS[0].prompt

    def test_four_is_a_dialog(self):
        """Twelve is a form nobody finishes."""
        assert len(QUESTIONS) <= 4

    def test_every_question_says_what_it_buys(self):
        for question in QUESTIONS:
            assert len(question.why_it_matters) > 60, question.key

    def test_a_question_the_data_answers_is_not_asked(self):
        """"an increase" is the only direction a binary response has."""
        binary = Reading(binary=True, n_response=100)

        keys = [q.key for q in questions_for(binary)]
        assert "direction" not in keys

    def test_it_is_asked_when_the_data_does_not_answer_it(self, screen):
        keys = [q.key for q in questions_for(read_the_screen(**screen))]

        assert "direction" in keys


# --------------------------------------------------------------------------- #
#  C. It explains itself, and it does not lie
# --------------------------------------------------------------------------- #

class TestEveryFieldCarriesTheReasonItWasSetFor:

    @pytest.fixture
    def advice(self, screen):
        return advise_the_screen(**screen, answers={
            "hits_per_thousand": 20, "direction": "either",
            "controls": "000000", "cost": "balanced"})

    def test_nothing_is_set_without_a_reason(self, advice):
        for choice in advice.chosen:
            assert len(choice.why) > 30, choice.key

    def test_the_reason_names_a_number_from_the_data(self, advice):
        """"regression_type=glm because the response is a proportion strictly
        inside (0,1)" -- a reason a user cannot check is a slogan."""
        why = advice.why("regression_type")

        assert "0, 1" in why or "(0, 1)" in why

    def test_the_level_reason_names_the_guides_per_gene(self, advice):
        assert "3 guides" in advice.why("level")

    def test_the_batch_reason_names_the_plate_count(self, advice):
        assert "4 plates" in advice.why("batch_correction")

    def test_a_setting_it_cannot_decide_is_named_as_undecided(self, screen):
        """Not filled with a default wearing the same authority as the rest."""
        advice = advise_the_screen(**screen, answers={})

        skipped = {u.key for u in advice.undecided}
        assert "fdr_alpha" in skipped
        assert "fdr_alpha" not in advice.as_settings()

    def test_an_undecided_setting_says_why(self, screen):
        advice = advise_the_screen(**screen, answers={})

        assert "prior" in advice.why("fdr_alpha")

    def test_with_no_response_the_model_is_undecided_not_guessed(self):
        advice = advise(Reading(plates=2, wells=96, guides=10))

        assert "regression_type" not in advice.as_settings()
        assert "regression_type" in {u.key for u in advice.undecided}


class TestTheChoicesAreOnesAStatisticianWouldDefend:

    def _advise(self, **reading):
        return advise(Reading(n_response=1000, wells=384, **reading))

    def test_a_proportion_strictly_inside_the_unit_gets_beta(self):
        advice = self._advise(low=0.02, high=0.97, inside_unit=True,
                              on_unit=True)

        assert advice.as_settings()["regression_type"] == "beta"

    def test_a_proportion_touching_a_boundary_does_not(self):
        """Beta's density is undefined at 0 and 1."""
        advice = self._advise(low=0.0, high=1.0, on_unit=True)

        assert advice.as_settings()["regression_type"] == "quasi_binomial"

    def test_binary_gets_logit(self):
        advice = self._advise(low=0.0, high=1.0, on_unit=True, binary=True)

        assert advice.as_settings()["regression_type"] == "logit"

    def test_counts_get_a_glm(self):
        advice = self._advise(low=0.0, high=90.0, integral=True)

        assert advice.as_settings()["regression_type"] == "glm"

    def test_an_unbounded_skewed_response_gets_a_robust_fit(self):
        advice = self._advise(low=-4.0, high=90.0, normal_p=1e-9, skew=3.1)

        assert advice.as_settings()["regression_type"] == "rlm"

    def test_an_unbounded_normal_response_gets_ols(self):
        advice = self._advise(low=-4.0, high=4.0, normal_p=0.4, skew=0.05)

        assert advice.as_settings()["regression_type"] == "ols"

    def test_one_plate_cannot_be_batch_corrected(self):
        """135 greys it for the same reason: two batches or nothing."""
        advice = advise(Reading(plates=1, wells=96, guides=10,
                                n_response=100))

        assert advice.as_settings()["batch_correction"] == "none"

    def test_a_single_row_screen_gets_no_position_term(self):
        advice = advise(Reading(plates=2, wells=24, guides=10, rows=1,
                                columns=24, n_response=100))

        assert advice.as_settings()["model_plate_position"] is False

    def test_one_guide_per_gene_is_not_a_gene_level(self):
        advice = advise(Reading(plates=2, wells=96, guides=40, genes=40,
                                guides_per_gene=1.0, n_response=100))

        assert advice.as_settings()["level"] == "grna"


class TestTheFamilyAndTheTransformAreOneDecision:
    """182: applying both fits logit(log(y))."""

    def _advise(self, **reading):
        return advise(Reading(n_response=1000, wells=384, **reading))

    def test_a_link_carrying_family_takes_no_transform(self):
        advice = self._advise(low=0.02, high=0.97, inside_unit=True,
                              on_unit=True)

        assert advice.as_settings()["transform"] is None

    def test_and_it_says_which_of_the_two_is_transforming(self):
        advice = self._advise(low=0.02, high=0.97, inside_unit=True,
                              on_unit=True)

        assert "carries its own link" in advice.why("transform")

    def test_the_conflict_setting_is_decided_with_it(self):
        advice = self._advise(low=0.02, high=0.97, inside_unit=True,
                              on_unit=True)

        assert advice.as_settings()["glm_transform_conflict"] == "untransformed"

    def test_a_linkless_family_on_skewed_positive_data_takes_the_log(self):
        advice = self._advise(low=0.5, high=900.0, normal_p=1e-9, skew=3.4)

        assert advice.as_settings()["transform"] == "log"


class TestThePriorMovesTheCorrection:

    def _alpha(self, **answers):
        base = {"hits_per_thousand": 20, "direction": "either",
                "controls": "", "cost": "balanced"}
        base.update(answers)
        return advise(Reading(n_response=1000, wells=384, low=0.02,
                              high=0.97, inside_unit=True, on_unit=True),
                      base).as_settings()

    def test_an_expensive_false_positive_tightens_it(self):
        assert self._alpha(cost="precision")["fdr_alpha"] <= 0.01

    def test_an_expensive_false_negative_loosens_it(self):
        assert self._alpha(cost="recall")["fdr_alpha"] >= 0.10

    def test_precision_takes_the_conservative_correction(self):
        assert self._alpha(cost="precision")["multiple_testing_method"] \
            == "fdr_by"

    def test_the_default_posture_is_benjamini_hochberg(self):
        assert self._alpha()["multiple_testing_method"] == "fdr_bh"

    def test_the_reason_quotes_the_prior_back(self):
        advice = advise(Reading(n_response=1000, wells=384, low=0.02,
                                high=0.97, inside_unit=True, on_unit=True),
                        {"hits_per_thousand": 5, "cost": "balanced"})

        assert "5 hit(s) in 1,000" in advice.why("fdr_alpha")


class TestEverythingItProposesIsARealSetting:

    def test_the_settings_round_trip_through_spacrs_own_validator(self,
                                                                  screen):
        """A proposal spaCR would refuse is worse than no proposal."""
        from spacr.settings import get_perform_regression_default_settings

        advice = advise_the_screen(**screen, answers={
            "hits_per_thousand": 20, "direction": "either",
            "controls": "000000", "cost": "balanced"})

        got = get_perform_regression_default_settings(dict(advice.as_settings()))
        for key, value in advice.as_settings().items():
            assert key in got, key

    def test_every_regression_type_it_can_propose_is_supported(self):
        from spacr.ml import REGRESSION_TYPES

        for reading in (
                dict(binary=True, on_unit=True),
                dict(inside_unit=True, on_unit=True),
                dict(on_unit=True),
                dict(integral=True),
                dict(normal_p=1e-9, skew=3.0),
                dict(normal_p=0.4, skew=0.0)):
            advice = advise(Reading(n_response=100, wells=96, low=0.0,
                                    high=1.0, **reading))
            assert advice.as_settings()["regression_type"] in REGRESSION_TYPES
