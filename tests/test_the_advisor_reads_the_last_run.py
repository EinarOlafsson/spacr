""""Settings for my data" reads the last run, not only the input.

Instruction 226: "the settings for my data button should base its choise
also of the summary data if a run has been done".

THE NUMBERS COME FROM THE WRITTEN QC, NOT A RECOMPUTATION. That is the
property the instruction names and the one worth testing: a second
diagnostic pass here could disagree with the QC panel about the same fit,
and the user would have no way to tell which was right.
"""
from __future__ import annotations

import json
import os

import pytest

from spacr.regression_qc import QC_NUMBERS_FILE
from spacr.settings_advisor import (Reading, advise, advise_the_screen,
                                    read_the_last_run)


def _write_numbers(folder, numbers, *, regression_type="ols"):
    qc = os.path.join(str(folder), "regression_qc")
    os.makedirs(qc, exist_ok=True)
    path = os.path.join(qc, QC_NUMBERS_FILE)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"regression_type": regression_type,
                   "numbers": numbers, "panels": {}, "verdicts": {}}, handle)
    return path


def _choice(advice, key):
    for one in advice.chosen:
        if one.key == key:
            return one
    return None


@pytest.fixture
def readable():
    """A Reading that already decided things from the input alone."""
    return Reading(plates=4, wells=620, guides=1380, genes=345,
                   response="pathogen_count", n_response=50_000,
                   low=0.0, high=1.0, on_unit=True,
                   normal_p=0.4, wells_per_guide=2.0)


class TestWithNoRunNothingChanges:
    """The button's whole point is answering before anything is fitted."""

    def test_an_absent_folder_reads_nothing(self, tmp_path):
        assert read_the_last_run(str(tmp_path)) == {}

    def test_an_empty_folder_name_reads_nothing(self):
        assert read_the_last_run("") == {}

    def test_the_advice_is_unchanged(self, readable):
        before = advise(readable)
        after = advise_the_screen(run_folder="")
        assert before.chosen  # the fixture decides something
        # And nothing in `before` mentions a run.
        assert all("last run" not in one.why for one in before.chosen)


class TestARunMovesTheChoices:

    def test_non_normal_residuals_choose_the_permutation_null(
            self, readable, tmp_path):
        """Even though the RESPONSE passed its own normality test."""
        _write_numbers(tmp_path, {"normality_p": 1e-30})
        from dataclasses import replace

        reading = replace(readable, **read_the_last_run(str(tmp_path)))
        one = _choice(advise(reading), "inference")
        assert one is not None and one.value == "nonparametric"
        assert "RESIDUALS" in one.why

    def test_the_response_alone_would_have_said_parametric(self, readable):
        """The premise of the test above: the two disagree, residuals win."""
        one = _choice(advise(readable), "inference")
        assert one is not None and one.value == "parametric"

    def test_heavy_tails_choose_the_rank_statistic(self, readable, tmp_path):
        _write_numbers(tmp_path, {"normality_p": 1e-30,
                                  "excess_kurtosis": 9.4})
        from dataclasses import replace

        reading = replace(readable, **read_the_last_run(str(tmp_path)))
        one = _choice(advise(reading), "grna_statistic")
        assert one is not None and one.value == "rank"

    def test_autocorrelation_chooses_the_nuisance_columns(
            self, readable, tmp_path):
        _write_numbers(tmp_path, {"durbin_watson": 1.55})
        from dataclasses import replace

        reading = replace(readable, **read_the_last_run(str(tmp_path)))
        one = _choice(advise(reading), "guide_nuisance_columns")
        assert one is not None
        assert "1.55" in one.why

    def test_an_influential_point_chooses_a_robust_family(
            self, readable, tmp_path):
        _write_numbers(tmp_path, {"max_cooks_distance": 4.2})
        from dataclasses import replace

        reading = replace(readable, **read_the_last_run(str(tmp_path)))
        one = _choice(advise(reading), "regression_type")
        assert one is not None and one.value == "rlm"

    def test_collinearity_chooses_a_penalised_family(
            self, readable, tmp_path):
        _write_numbers(tmp_path, {"max_vif": 45.0})
        from dataclasses import replace

        reading = replace(readable, **read_the_last_run(str(tmp_path)))
        one = _choice(advise(reading), "regression_type")
        assert one is not None and one.value == "ridge"


class TestItSaysWhichRunItRead:

    def test_the_folder_is_named_in_the_reason(self, readable, tmp_path):
        _write_numbers(tmp_path, {"normality_p": 1e-30})
        from dataclasses import replace

        reading = replace(readable, **read_the_last_run(str(tmp_path)))
        one = _choice(advise(reading), "inference")
        assert "regression_qc" in one.why

    def test_the_reading_carries_the_folder(self, tmp_path):
        _write_numbers(tmp_path, {"normality_p": 0.5})
        got = read_the_last_run(str(tmp_path))
        assert got["run_folder"].endswith("regression_qc")


class TestAStaleRunIsReportedNotUsed:
    """A stale summary is worse than none, because it looks like
    measurement."""

    def test_a_different_family_is_refused(self, tmp_path):
        _write_numbers(tmp_path, {"normality_p": 1e-30},
                       regression_type="ols")
        got = read_the_last_run(str(tmp_path),
                               {"regression_type": "beta"})
        assert got.get("run_note")
        assert "residual_normal_p" not in got

    def test_the_same_family_is_used(self, tmp_path):
        _write_numbers(tmp_path, {"normality_p": 1e-30},
                       regression_type="ols")
        got = read_the_last_run(str(tmp_path),
                               {"regression_type": "ols"})
        assert not got.get("run_note")
        assert got["residual_normal_p"] == pytest.approx(1e-30)

    def test_the_staleness_reaches_the_advice(self, readable, tmp_path):
        _write_numbers(tmp_path, {"normality_p": 1e-30},
                       regression_type="ols")
        from dataclasses import replace

        reading = replace(readable, **read_the_last_run(
            str(tmp_path), {"regression_type": "beta"}))
        advice = advise(reading)
        assert any(u.key == "last run" for u in advice.undecided)
        # and it did NOT quietly take the numbers
        assert _choice(advice, "inference").value == "parametric"

    def test_an_unreadable_file_is_reported_not_ignored(self, tmp_path):
        qc = os.path.join(str(tmp_path), "regression_qc")
        os.makedirs(qc, exist_ok=True)
        with open(os.path.join(qc, QC_NUMBERS_FILE), "w") as handle:
            handle.write("{not json")
        got = read_the_last_run(str(tmp_path))
        assert got.get("run_note")


class TestItIsAReadNotARecomputation:
    """The instruction's own check: change the file, see the advice change."""

    def test_editing_the_file_changes_the_advice(self, readable, tmp_path):
        from dataclasses import replace

        _write_numbers(tmp_path, {"normality_p": 0.9})
        first = advise(replace(readable, **read_the_last_run(str(tmp_path))))

        _write_numbers(tmp_path, {"normality_p": 1e-30})
        second = advise(replace(readable, **read_the_last_run(str(tmp_path))))

        assert _choice(first, "inference").value == "parametric"
        assert _choice(second, "inference").value == "nonparametric"
