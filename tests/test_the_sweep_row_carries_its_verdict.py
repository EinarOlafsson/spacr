"""115's last owed item: the sweep ROW's verdict column.

WHY A ROW NEEDS ONE. A sweep is a table of trials ranked by hit count, and
hit count is exactly the number a broken fit inflates -- a rank-deficient
design will happily report more significant guides than an identifiable one.
Without a verdict beside it, the trial that WINS a sweep is sometimes the
trial that is most wrong, and nothing on the row says so.

THE SCORERS STAY THE ONE JUDGEMENT. `score_design` and `score_inference`
read plain dicts, and the row already carries every statistic they read under
spaCR's own column names, so only the names are translated. Re-deriving the
verdicts with fresh rules would be a second opinion about the same numbers,
which is what a sweep table can least afford.
"""
from __future__ import annotations

import pytest

from spacr.trial_metrics import METRIC_COLUMNS, qc_verdicts

HEALTHY = {
    "n_wells": 600, "n_parameters": 40, "design_rank": 40,
    "wells_per_parameter": 15.0, "condition_number": 12.0,
    "non_identifiable_directions": 0,
    "genomic_inflation": 1.02, "n_results": 40,
}

SATURATED = {**HEALTHY, "n_parameters": 800, "design_rank": 600,
             "wells_per_parameter": 0.75, "condition_number": 1e9,
             "non_identifiable_directions": 200, "n_results": 800}


class TestTheVerdictIsOnTheRow:

    def test_a_healthy_design_passes(self):
        assert qc_verdicts(HEALTHY)["qc_design"] == "pass"

    def test_a_rank_deficient_design_fails(self):
        """The case the whole column exists for: this trial can out-hit an
        identifiable one and its coefficients mean nothing."""
        assert qc_verdicts(SATURATED)["qc_design"] == "fail"

    def test_the_worst_of_the_panels_is_what_a_reader_sorts_by(self):
        """Not a pass rate. `worst_verdict`'s own reason holds here: 19
        panels passing and one saying the design is rank deficient is a run
        whose "95% passed" hides exactly the panel the suite was run for."""
        verdicts = qc_verdicts(SATURATED)

        assert verdicts["qc_inference"] == "pass"
        assert verdicts["qc_verdict"] == "fail"

    def test_inflation_is_judged_too(self):
        shifted = {**HEALTHY, "genomic_inflation": 1.4}

        assert qc_verdicts(shifted)["qc_inference"] != "pass"


class TestItNeverSinksARow:
    """"One missing statistic is a NaN in one column, never a lost row.\""""

    def test_an_empty_row_scores_nothing_rather_than_raising(self):
        assert qc_verdicts({}) == {}

    def test_a_row_with_no_p_values_still_scores_its_design(self):
        without = {key: value for key, value in HEALTHY.items()
                   if key not in ("genomic_inflation", "n_results")}

        verdicts = qc_verdicts(without)

        assert verdicts["qc_design"] == "pass"
        assert "qc_inference" not in verdicts

    def test_a_row_with_no_design_still_scores_its_inference(self):
        without = {"genomic_inflation": 1.02, "n_results": 40}

        assert qc_verdicts(without).get("qc_inference") == "pass"

    def test_nonsense_values_do_not_raise(self):
        assert isinstance(qc_verdicts({"n_wells": "many",
                                       "n_parameters": None}), dict)


class TestTheColumnsAreRegistered:
    """ON THE LIST OR THEY BECOME SETTINGS.

    `settings_for_trial`'s rule is "anything not bookkeeping was a setting",
    so a metric column missing from METRIC_COLUMNS is fed back into
    perform_regression as though the user had typed `qc_verdict='fail'`.
    """

    @pytest.mark.parametrize("column", ["qc_design", "qc_inference",
                                        "qc_verdict"])
    def test_every_column_it_emits_is_declared(self, column):
        assert column in METRIC_COLUMNS

    def test_nothing_it_emits_is_undeclared(self):
        emitted = set(qc_verdicts(HEALTHY))

        assert emitted <= METRIC_COLUMNS, sorted(emitted - METRIC_COLUMNS)


class TestItRunsAfterTheStatistics:
    """The verdicts are a judgement ON the row, not another measurement --
    so they must be computed once the blocks above have written it."""

    def test_summarise_trial_scores_what_it_measured(self):
        from spacr.trial_metrics import summarise_trial

        row = summarise_trial({}, {})

        # Nothing to measure, so nothing to judge -- and no exception.
        assert isinstance(row, dict)
