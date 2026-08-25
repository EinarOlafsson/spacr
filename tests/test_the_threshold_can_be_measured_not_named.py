"""The gRNA fraction cut-off can be measured from the control wells.

`fraction_threshold` is a number the user picks and there is no obvious
right one: too low and bleed-through gRNAs survive, too high and real ones
are stripped. The control wells can answer it -- recompute each well's
fractions at a range of cut-offs, refit imaging on sequencing over the
wells the plate design names as pure control, and keep the cut-off where
the two agree best.

It is OFFERED, not defaulted. Turning it on changes which gRNAs survive in
every well of a screen, so it is the user's decision rather than a version
bump's, and a sweep that cannot run says why and leaves the given number
alone rather than taking the run down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import ml
from spacr.settings import (categories, expected_types,
                            get_perform_regression_default_settings, tooltips)


def _settings(**over):
    base = get_perform_regression_default_settings(
        {"src": "/tmp", "count_data": "/tmp/c.csv", "score_data": "/tmp/s.csv"})
    base.update(over)
    return base


def test_it_is_off_until_the_user_asks():
    """It changes which gRNAs survive, so nothing should turn it on quietly."""
    assert _settings()["calibrate_fraction_threshold"] is False
    assert expected_types["calibrate_fraction_threshold"] is bool


def test_the_control_says_what_it_needs():
    text = tooltips["calibrate_fraction_threshold"]
    assert "control" in text
    assert "circular" in text, "the reason the wells must be NAMED is the point"
    assert any("calibrate_fraction_threshold" in keys
               for keys in categories.values())


def test_a_screen_with_no_named_controls_cannot_calibrate():
    with pytest.raises(ValueError, match="no positive and negative control"):
        ml._calibration_inputs(_settings(positive_control="",
                                         negative_control=""))


def test_the_imaging_side_is_the_score_column(tmp_path, monkeypatch):
    """One feature per cell -- the classifier's own answer -- and its well."""
    scores = pd.DataFrame({
        "prc": ["p1_A01", "p1_A01", "p1_B02", "p1_B02"],
        "pred": [0.9, 0.8, 0.1, 0.2],
    })
    counts = pd.DataFrame({
        "prc": ["p1_A01", "p1_B02"],
        "grna": ["pc_guide", "nc_guide"],
        "count": [100, 100],
    })
    monkeypatch.setattr(ml, "_concat_named_csvs",
                        lambda paths: scores if "s" in str(paths) else counts)
    got = ml._calibration_inputs(_settings(positive_control="A01",
                                           negative_control="B02",
                                           score_data="s", count_data="c"))
    assert got["features"].shape == (4, 1)
    assert got["features"].dtype == np.dtype(float)
    assert got["wells"] == ["p1_A01", "p1_A01", "p1_B02", "p1_B02"]
    # The pure wells come from the plate design, never from the fractions.
    assert got["pure_pc_wells"] == ["p1_A01"]
    assert got["pure_nc_wells"] == ["p1_B02"]


def test_a_missing_score_column_is_named_not_guessed(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "_concat_named_csvs",
                        lambda paths: pd.DataFrame({"prc": ["p1_A01"]}))
    with pytest.raises(ValueError, match="no 'pred' column"):
        ml._calibration_inputs(_settings(positive_control="A01",
                                         negative_control="B02"))


def test_a_sweep_that_cannot_run_leaves_the_number_alone(capsys):
    """The calibration improves a value that already exists; it is not a
    prerequisite for having one, so it must never stop the run."""
    got = ml._calibrated_fraction_threshold(_settings(positive_control="",
                                                      negative_control=""))
    assert got is None
    # And it says so -- a user who ticked the box would otherwise believe
    # the number they are looking at was measured.
    assert "did not run" in capsys.readouterr().out


def test_the_measured_number_is_returned_when_the_sweep_answers(monkeypatch,
                                                                capsys):
    monkeypatch.setattr(ml, "_calibration_inputs", lambda s: {})
    monkeypatch.setattr(
        "spacr.fraction_calibration.sweep_fraction_threshold",
        lambda **kw: {"threshold": 0.037, "candidates": []})
    assert ml._calibrated_fraction_threshold(_settings()) == pytest.approx(0.037)


def test_a_sweep_that_prefers_nothing_says_so(monkeypatch, capsys):
    monkeypatch.setattr(ml, "_calibration_inputs", lambda s: {})
    monkeypatch.setattr(
        "spacr.fraction_calibration.sweep_fraction_threshold",
        lambda **kw: {"threshold": None, "candidates": []})
    assert ml._calibrated_fraction_threshold(_settings()) is None
    assert "no cut-off it preferred" in capsys.readouterr().out


def test_several_plates_are_read_as_one_screen(tmp_path):
    """The question is asked of the screen, so the plates are read together."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame({"prc": ["p1_A01"], "count": [5]}).to_csv(a, index=False)
    pd.DataFrame({"prc": ["p2_A01"], "count": [7]}).to_csv(b, index=False)
    got = ml._concat_named_csvs([a, b])
    assert list(got["count"]) == [5, 7]


def test_nothing_to_read_is_an_error_with_a_reason():
    with pytest.raises(ValueError, match="no table"):
        ml._concat_named_csvs([])
