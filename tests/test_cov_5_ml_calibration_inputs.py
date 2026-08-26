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

def test_one_file_is_read_as_readily_as_a_list_of_them(tmp_path):
    """A single-plate screen names one path; a four-plate screen names four."""
    one = _counts(tmp_path, "p1.csv")
    two = _counts(tmp_path, "p2.csv")

    assert len(ML._concat_named_csvs(one)) == 2
    assert len(ML._concat_named_csvs([one, two])) == 4
    assert list(ML._concat_named_csvs([one, two]).index) == [0, 1, 2, 3]


def test_a_file_that_cannot_be_read_is_named(tmp_path):
    """"could not be read" with no file name sends the user through all four."""
    good = _counts(tmp_path, "p1.csv")
    missing = str(tmp_path / "never_written.csv")

    with pytest.raises(ValueError, match="never_written.csv could not be read"):
        ML._concat_named_csvs([good, missing])


def test_naming_no_table_at_all_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no table was given to read"):
        ML._concat_named_csvs(None)
    with pytest.raises(ValueError, match="no table was given to read"):
        ML._concat_named_csvs([])


# ---------------------------------------------------------------------------
# The inputs the sweep needs
# ---------------------------------------------------------------------------

def _settings(tmp_path, **changes):
    base = {
        "count_data": _counts(tmp_path),
        "score_data": _scores(tmp_path),
        "positive_control_wells": ["c1"],
        "negative_control_wells": ["c2"],
        "positive_control": "pc_1",
        "dependent_variable": "pred",
        "count_well_column": "prc",
    }
    base.update(changes)
    return base


def test_the_inputs_carry_the_wells_the_design_calls_controls(tmp_path):
    inputs = ML._calibration_inputs(_settings(tmp_path))

    assert inputs["positive_guide"] == "pc_1"
    assert inputs["pure_pc_wells"] == ["plate1_r1_c1"]
    assert inputs["pure_nc_wells"] == ["plate1_r1_c2"]
    assert inputs["features"].shape == (8, 1)
    assert inputs["wells"][0] == "plate1_r1_c1"


def test_a_score_table_with_no_score_column_cannot_be_calibrated(tmp_path):
    settings = _settings(tmp_path, dependent_variable="probability")

    with pytest.raises(ValueError, match="'probability' column"):
        ML._calibration_inputs(settings)


def test_a_score_table_that_does_not_say_which_well_cannot_be_calibrated(
        tmp_path):
    """A cell with no well cannot be compared with a well's sequencing."""
    path = tmp_path / "no_well.csv"
    pd.DataFrame({"pred": [0.5, 0.6]}).to_csv(path, index=False)
    settings = _settings(tmp_path, score_data=str(path))

    with pytest.raises(ValueError, match="cannot be placed in a well"):
        ML._calibration_inputs(settings)


def test_a_design_that_names_no_control_block_cannot_be_calibrated(tmp_path):
    with pytest.raises(ValueError, match="nothing to calibrate against"):
        ML._calibration_inputs(_settings(tmp_path,
                                         positive_control_wells=[]))


def test_a_design_with_no_positive_guide_cannot_be_calibrated(tmp_path):
    with pytest.raises(ValueError, match="no gRNA"):
        ML._calibration_inputs(_settings(tmp_path, positive_control=""))


def test_control_blocks_that_match_no_well_are_refused(tmp_path):
    with pytest.raises(ValueError, match="no pure control to anchor the fit"):
        ML._calibration_inputs(_settings(tmp_path,
                                         positive_control_wells=["c9"],
                                         negative_control_wells=["c8"]))


# ---------------------------------------------------------------------------
# What the caller does with each refusal
# ---------------------------------------------------------------------------

def test_a_refused_calibration_prints_its_reason_and_keeps_the_threshold(
        tmp_path, capsys):
    """A user who ticked the box is owed the reason it did nothing."""
    settings = _settings(tmp_path, positive_control_wells=[])

    assert ML._calibrated_fraction_threshold(settings) is None
    assert "calibration did not run" in capsys.readouterr().out


def test_a_sweep_that_preferred_no_cut_off_says_so(tmp_path, monkeypatch,
                                                    capsys):
    """`chosen` is the key the sweep writes; nothing else is a threshold."""
    from spacr import fraction_calibration

    monkeypatch.setattr(fraction_calibration, "sweep_fraction_threshold",
                        lambda **_kwargs: {"candidates": [], "chosen": None})

    assert ML._calibrated_fraction_threshold(_settings(tmp_path)) is None
    assert "found no cut-off it preferred" in capsys.readouterr().out


def test_a_sweep_that_chose_a_cut_off_returns_it(tmp_path, monkeypatch,
                                                  capsys):
    from spacr import fraction_calibration

    monkeypatch.setattr(fraction_calibration, "sweep_fraction_threshold",
                        lambda **_kwargs: {"chosen": 0.42, "candidates": []})
    monkeypatch.setattr(fraction_calibration, "describe",
                        lambda _result: "the sweep chose 0.42")

    assert ML._calibrated_fraction_threshold(_settings(tmp_path)) == 0.42
    assert "the sweep chose 0.42" in capsys.readouterr().out


def test_a_sweep_whose_description_fails_still_reports_the_number(
        tmp_path, monkeypatch, capsys):
    from spacr import fraction_calibration

    def refuse(_result):
        raise KeyError("candidates")

    monkeypatch.setattr(fraction_calibration, "sweep_fraction_threshold",
                        lambda **_kwargs: {"chosen": 0.31})
    monkeypatch.setattr(fraction_calibration, "describe", refuse)

    assert ML._calibrated_fraction_threshold(_settings(tmp_path)) == 0.31
    assert "calibrated to 0.31" in capsys.readouterr().out
