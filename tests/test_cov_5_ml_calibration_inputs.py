"""Reading a screen's count and score tables, and refusing to guess a cut-off.

The fraction-threshold calibration answers "where does the imaging call a cell
positive, according to the wells whose answer we already know?". Every way it
cannot answer has to come back as a printed reason and the threshold the
settings gave — never as a number, and never as a stack trace out of a run
that already had a usable threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import ml as ML


def _counts(tmp_path, name="counts.csv"):
    path = tmp_path / name
    pd.DataFrame({"prc": ["plate1_r1_c1", "plate1_r1_c2"],
                  "grna": ["pc_1", "nc_1"],
                  "count": [100, 100]}).to_csv(path, index=False)
    return str(path)


def _scores(tmp_path, name="scores.csv", column="pred"):
    path = tmp_path / name
    frame = pd.DataFrame({"prc": ["plate1_r1_c1"] * 4 + ["plate1_r1_c2"] * 4,
                          column: [0.9, 0.8, 0.85, 0.95,
                                   0.1, 0.2, 0.15, 0.05]})
    frame.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Concatenating a screen's per-plate files
# ---------------------------------------------------------------------------

def case_one_file_is_read_as_readily_as_a_list_of_them(tmp_path):
    """A single-plate screen names one path; a four-plate screen names four."""
    _seen = []
    one = _counts(tmp_path, "p1.csv")
    two = _counts(tmp_path, "p2.csv")

    _seen.append(keep(len(ML._concat_named_csvs(one))))
    _seen.append(keep(len(ML._concat_named_csvs([one, two]))))
    _seen.append(keep(list(ML._concat_named_csvs([one, two]).index)))
    return _seen


def case_a_file_that_cannot_be_read_is_named(tmp_path):
    """"could not be read" with no file name sends the user through all four."""
    _seen = []
    good = _counts(tmp_path, "p1.csv")
    missing = str(tmp_path / "never_written.csv")

    with pytest.raises(ValueError, match="never_written.csv could not be read") as _exc1:
        ML._concat_named_csvs([good, missing])
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_naming_no_table_at_all_is_refused(tmp_path):
    _seen = []
    with pytest.raises(ValueError, match="no table was given to read") as _exc1:
        ML._concat_named_csvs(None)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    with pytest.raises(ValueError, match="no table was given to read") as _exc2:
        ML._concat_named_csvs([])
    _seen.append(('ValueError' if isinstance(_exc2.value, ValueError) else type(_exc2.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# The inputs the sweep needs
# ---------------------------------------------------------------------------

def _settings(tmp_path, **changes):
    base = {
        "count_data": _counts(tmp_path),
        "score_data": _scores(tmp_path),
        "positive_control_wells": ["c1"],
        "negative_control_wells": ["c2"],
        "positive_control_id": "pc_1",
        "dependent_variable": "pred",
        "count_well_column": "prc",
    }
    base.update(changes)
    return base


def case_the_inputs_carry_the_wells_the_design_calls_controls(tmp_path):
    _seen = []
    inputs = ML._calibration_inputs(_settings(tmp_path))

    _seen.append(keep(inputs["positive_guide"]))
    _seen.append(keep(inputs["pure_pc_wells"]))
    _seen.append(keep(inputs["pure_nc_wells"]))
    _seen.append(keep(inputs["features"].shape))
    _seen.append(keep(inputs["wells"][0]))
    return _seen


def case_a_score_table_with_no_score_column_cannot_be_calibrated(tmp_path):
    _seen = []
    settings = _settings(tmp_path, dependent_variable="probability")

    with pytest.raises(ValueError, match="'probability' column") as _exc1:
        ML._calibration_inputs(settings)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_score_table_that_does_not_say_which_well_cannot_be_calibrated(
        tmp_path):
    """A cell with no well cannot be compared with a well's sequencing."""
    _seen = []
    path = tmp_path / "no_well.csv"
    pd.DataFrame({"pred": [0.5, 0.6]}).to_csv(path, index=False)
    settings = _settings(tmp_path, score_data=str(path))

    with pytest.raises(ValueError, match="cannot be placed in a well") as _exc1:
        ML._calibration_inputs(settings)
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_design_that_names_no_control_block_cannot_be_calibrated(tmp_path):
    _seen = []
    with pytest.raises(ValueError, match="nothing to calibrate against") as _exc1:
        ML._calibration_inputs(_settings(tmp_path,
                                         positive_control_wells=[]))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_a_design_with_no_positive_guide_cannot_be_calibrated(tmp_path):
    _seen = []
    with pytest.raises(ValueError, match="no gRNA") as _exc1:
        ML._calibration_inputs(_settings(tmp_path, positive_control_id=""))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


def case_control_blocks_that_match_no_well_are_refused(tmp_path):
    _seen = []
    with pytest.raises(ValueError, match="no pure control to anchor the fit") as _exc1:
        ML._calibration_inputs(_settings(tmp_path,
                                         positive_control_wells=["c9"],
                                         negative_control_wells=["c8"]))
    _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
    return _seen


# ---------------------------------------------------------------------------
# What the caller does with each refusal
# ---------------------------------------------------------------------------

def case_a_refused_calibration_prints_its_reason_and_keeps_the_threshold(
        tmp_path, capsys):
    """A user who ticked the box is owed the reason it did nothing."""
    _seen = []
    settings = _settings(tmp_path, positive_control_wells=[])

    _seen.append(bool(
        ML._calibrated_fraction_threshold(settings) is None))
    _seen.append(('calibration did not run' if 'calibration did not run' in (capsys.readouterr().out) else ("missing", 'calibration did not run')))
    return _seen


def case_a_sweep_that_preferred_no_cut_off_says_so(tmp_path, monkeypatch,
                                                    capsys):
    """`chosen` is the key the sweep writes; nothing else is a threshold."""
    _seen = []
    from spacr import fraction_calibration

    monkeypatch.setattr(fraction_calibration, "sweep_fraction_threshold",
                        lambda **_kwargs: {"candidates": [], "chosen": None})

    _seen.append(bool(
        ML._calibrated_fraction_threshold(_settings(tmp_path)) is None))
    _seen.append(('found no cut-off it preferred' if 'found no cut-off it preferred' in (capsys.readouterr().out) else ("missing", 'found no cut-off it preferred')))
    return _seen


def case_a_sweep_that_chose_a_cut_off_returns_it(tmp_path, monkeypatch,
                                                  capsys):
    _seen = []
    from spacr import fraction_calibration

    monkeypatch.setattr(fraction_calibration, "sweep_fraction_threshold",
                        lambda **_kwargs: {"chosen": 0.42, "candidates": []})
    monkeypatch.setattr(fraction_calibration, "describe",
                        lambda _result: "the sweep chose 0.42")

    _seen.append(keep(ML._calibrated_fraction_threshold(_settings(tmp_path))))
    _seen.append(('the sweep chose 0.42' if 'the sweep chose 0.42' in (capsys.readouterr().out) else ("missing", 'the sweep chose 0.42')))
    return _seen


def case_a_sweep_whose_description_fails_still_reports_the_number(
        tmp_path, monkeypatch, capsys):
    _seen = []
    from spacr import fraction_calibration

    def refuse(_result):
        raise KeyError("candidates")

    monkeypatch.setattr(fraction_calibration, "sweep_fraction_threshold",
                        lambda **_kwargs: {"chosen": 0.31})
    monkeypatch.setattr(fraction_calibration, "describe", refuse)

    _seen.append(keep(ML._calibrated_fraction_threshold(_settings(tmp_path))))
    _seen.append(('calibrated to 0.31' if 'calibrated to 0.31' in (capsys.readouterr().out) else ("missing", 'calibrated to 0.31')))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_the_inputs_carry_the_wells_the_design_calls_controls, ['pc_1', ['plate1_r1_c1'], ['plate1_r1_c2'], (8, 1), 'plate1_r1_c1'], id="the_inputs_carry_the_wells_the_design_calls_controls"),
    pytest.param(case_a_score_table_with_no_score_column_cannot_be_calibrated, ['ValueError'], id="a_score_table_with_no_score_column_cannot_be_calibrated"),
    pytest.param(case_a_score_table_that_does_not_say_which_well_cannot_be_calibrated, ['ValueError'], id="a_score_table_that_does_not_say_which_well_cannot_be_calibrated"),
    pytest.param(case_a_design_that_names_no_control_block_cannot_be_calibrated, ['ValueError'], id="a_design_that_names_no_control_block_cannot_be_calibrated"),
    pytest.param(case_a_design_with_no_positive_guide_cannot_be_calibrated, ['ValueError'], id="a_design_with_no_positive_guide_cannot_be_calibrated"),
    pytest.param(case_control_blocks_that_match_no_well_are_refused, ['ValueError'], id="control_blocks_that_match_no_well_are_refused"),
    pytest.param(case_one_file_is_read_as_readily_as_a_list_of_them, [2, 4, [0, 1, 2, 3]], id="one_file_is_read_as_readily_as_a_list_of_them"),
    pytest.param(case_a_file_that_cannot_be_read_is_named, ['ValueError'], id="a_file_that_cannot_be_read_is_named"),
    pytest.param(case_naming_no_table_at_all_is_refused, ['ValueError', 'ValueError'], id="naming_no_table_at_all_is_refused"),
    pytest.param(case_a_refused_calibration_prints_its_reason_and_keeps_the_threshold, [True, 'calibration did not run'], id="a_refused_calibration_prints_its_reason_and_keeps_the_threshold"),
    pytest.param(case_a_sweep_that_preferred_no_cut_off_says_so, [True, 'found no cut-off it preferred'], id="a_sweep_that_preferred_no_cut_off_says_so"),
    pytest.param(case_a_sweep_that_chose_a_cut_off_returns_it, [0.42, 'the sweep chose 0.42'], id="a_sweep_that_chose_a_cut_off_returns_it"),
    pytest.param(case_a_sweep_whose_description_fails_still_reports_the_number, [0.31, 'calibrated to 0.31'], id="a_sweep_whose_description_fails_still_reports_the_number"),
])
def test_5_ml_calibration_inputs_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected

