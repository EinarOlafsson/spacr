"""Regression banner, validation, and persisted run-summary output."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def screen(qtbot):
    """Return a regression screen with a live console."""
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    made = AppScreen("regression")
    qtbot.addWidget(made)
    return made


@pytest.mark.parametrize("settings, expected", [
    ({"inference": "nonparametric", "regression_type": "mixed"}, True),
    ({"analysis_mode": "guide_permutation"}, True),
    ({"inference": "parametric", "regression_type": "mixed"}, False),
    ({"inference": "auto", "regression_type": "mixed"}, False),
])
def test_only_explicit_permutation_inference_changes_the_banner(
        screen, settings, expected):
    assert screen._it_will_permute(settings) is expected


def test_permutation_banner_reports_the_executed_analysis(screen):
    message = screen._say_what_the_permutation_will_do({
        "inference": "nonparametric",
        "regression_type": "mixed",
        "level": "both",
        "guide_permutations": 200_000,
        "guide_permutation_block": "plateID",
        "grna_statistic": "pearson",
    })

    assert "200,000" in message
    assert "within-plateID" in message
    assert "pearson statistic" in message
    assert "reported level: both" in message
    assert "regression_type is not used" in message
    assert "1/(permutations+1)" in message
    assert "5e-06" in message


def test_permutation_p_value_floor_changes_with_the_count(screen):
    one_thousand = screen._say_what_the_permutation_will_do({
        "guide_permutations": 1_000,
    })
    two_hundred_thousand = screen._say_what_the_permutation_will_do({
        "guide_permutations": 200_000,
    })

    assert "0.000999" in one_thousand
    assert "5e-06" in two_hundred_thousand


@pytest.mark.parametrize("value", [None, "", "many", 0, -5])
def test_invalid_permutation_count_is_reported_without_a_false_floor(
        screen, value):
    message = screen._say_what_the_permutation_will_do({
        "guide_permutations": value,
    })

    assert "configuration is invalid" in message
    assert "integer of at least 1" in message
    assert "Minimum attainable" not in message


def test_permutation_editor_cannot_select_a_negative_count(screen):
    widget = screen._settings_model._widgets["guide_permutations"]

    assert widget.minimum() == 1
    widget.setValue(-5)
    assert widget.value() == 1


def test_preflight_rejects_a_nonpositive_permutation_count():
    from spacr.validate import _check_numeric_sanity

    problems = _check_numeric_sanity({"guide_permutations": -5})

    assert len(problems) == 1
    assert problems[0].setting == "guide_permutations"
    assert problems[0].is_error
    assert "must be at least 1" in problems[0].message


def _summary_field(settings, key):
    """Return a public summary field for a minimal completed regression."""
    from spacr.regression_summary import build_run_summary

    summary = build_run_summary(
        settings=settings,
        coef_df=pd.DataFrame({
            "coefficient": [0.1],
            "p_value": [0.5],
            "adjusted_p_value": [0.5],
        }),
        regression_type="ols",
    )
    return summary.field(key)


def test_severe_fraction_filtering_is_visible_in_the_summary():
    field = _summary_field({
        "fraction_threshold": 0.02,
        "_regression_exclusions": {
            "fraction_threshold": 580_214,
            "fraction_threshold_of": 586_038,
        },
    }, "fraction_threshold")

    assert field is not None
    assert "5,824 (1.0% retained)" in field.value
    assert "fewer than 5%" in field.value


def test_routine_fraction_filtering_has_no_low_retention_warning():
    field = _summary_field({
        "fraction_threshold": 0.02,
        "_regression_exclusions": {
            "fraction_threshold": 10_000,
            "fraction_threshold_of": 100_000,
        },
    }, "fraction_threshold")

    assert "90,000 (90.0% retained)" in field.value
    assert "warning" not in field.value.lower()


def test_complete_score_side_pairing_is_reported_as_one_hundred_percent():
    field = _summary_field({
        "_regression_exclusions": {
            "wells_paired": 463,
            "wells_unpaired_counts": 881,
            "wells_unpaired_scores": 0,
        },
    }, "missing_metadata")

    assert field is not None
    assert "463 wells were matched" in field.value
    assert "881 count-table wells and 0 score-table wells" in field.value
    assert "100% of wells in the smaller input paired" in field.value
    assert "warning" not in field.value.lower()


def test_incomplete_pairing_is_flagged_against_the_smaller_input():
    field = _summary_field({
        "_regression_exclusions": {
            "wells_paired": 4,
            "wells_unpaired_counts": 6,
            "wells_unpaired_scores": 10,
        },
    }, "missing_metadata")

    assert "40% of wells in the smaller input paired" in field.value
    assert "fewer than half" in field.value


def test_pairing_checker_records_and_reports_unmatched_wells(capsys):
    from spacr.ml import _check_score_count_pairing

    count_wells = [f"plate1_r1_c{index}" for index in range(1, 1_345)]
    score_wells = count_wells[:463]
    counts = pd.DataFrame({"prc": count_wells})
    scores = pd.DataFrame({"prc": score_wells})
    merged = pd.DataFrame({"prc": score_wells})
    record = {}

    _check_score_count_pairing(counts, scores, merged, record=record)

    assert record == {
        "wells_paired": 463,
        "wells_unpaired_counts": 881,
        "wells_unpaired_scores": 0,
    }
    console = capsys.readouterr().out
    assert "Paired 463 wells" in console
    assert "881 count-table wells" in console
    assert "0 score-table wells" in console


def test_regression_passes_the_pairing_recorder_to_the_checker():
    """The persisted counts must be connected to the full regression path."""
    import inspect

    from spacr import ml

    source = "".join(inspect.getsource(ml._perform_regression).split())
    assert "record=settings.get('_regression_exclusions')" in source
