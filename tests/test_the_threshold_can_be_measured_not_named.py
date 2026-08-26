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


def test_a_screen_with_no_named_control_wells_cannot_calibrate():
    """The plate design has to say which wells are pure control.

    This asserted `positive_control` / `negative_control`, which in a
    regression are gene or gRNA ID SUBSTRINGS -- the default is '239740' --
    and never appear in a well label. The wells are the three control-block
    settings, which exist to name them from the plate design.
    """
    with pytest.raises(ValueError, match="no positive_control_wells"):
        ml._calibration_inputs(_settings(positive_control_wells=None,
                                         negative_control_wells=None))


def test_a_screen_with_no_positive_guide_cannot_calibrate():
    """The guide is the other half: its sequenced share is the x axis."""
    with pytest.raises(ValueError, match="no gRNA"):
        ml._calibration_inputs(_settings(positive_control="",
                                         positive_control_wells=["c2"],
                                         negative_control_wells=["c1"]))


def test_the_imaging_side_is_the_score_column(tmp_path, monkeypatch):
    """One feature per cell -- the classifier's own answer -- and its well."""
    scores = pd.DataFrame({
        "prc": ["p1_r1_c2", "p1_r1_c2", "p1_r1_c1", "p1_r1_c1"],
        "pred": [0.9, 0.8, 0.1, 0.2],
    })
    counts = pd.DataFrame({
        "prc": ["p1_r1_c2", "p1_r1_c1"],
        "grna": ["pc_guide", "nc_guide"],
        "count": [100, 100],
    })
    monkeypatch.setattr(ml, "_concat_named_csvs",
                        lambda paths: scores if "s" in str(paths) else counts)
    got = ml._calibration_inputs(_settings(positive_control="pc_guide",
                                           positive_control_wells=["c2"],
                                           negative_control_wells=["c1"],
                                           score_data="s", count_data="c"))
    assert got["features"].shape == (4, 1)
    assert got["features"].dtype == np.dtype(float)
    assert got["wells"] == ["p1_r1_c2", "p1_r1_c2", "p1_r1_c1", "p1_r1_c1"]
    # The pure wells come from the plate design, never from the fractions.
    assert got["pure_pc_wells"] == ["p1_r1_c2"]
    assert got["pure_nc_wells"] == ["p1_r1_c1"]
    # And the guide is the guide, not the well block.
    assert got["positive_guide"] == "pc_guide"


def test_a_column_token_does_not_swallow_a_wider_column(monkeypatch):
    """'c2' in 'p1_r1_c20' is true and says nothing.

    Substring matching would fold column 20 into column 2's reference and
    shift the endpoint the whole calibration is anchored on, on any plate
    with more than nine columns.
    """
    scores = pd.DataFrame({
        "prc": ["p1_r1_c2", "p1_r1_c20", "p1_r1_c1", "p1_r1_c12"],
        "pred": [0.9, 0.5, 0.1, 0.4],
    })
    monkeypatch.setattr(ml, "_concat_named_csvs", lambda paths: scores)
    got = ml._calibration_inputs(_settings(positive_control="pc_guide",
                                           positive_control_wells=["c2"],
                                           negative_control_wells=["c1"]))
    assert got["pure_pc_wells"] == ["p1_r1_c2"]
    assert got["pure_nc_wells"] == ["p1_r1_c1"]


def test_a_missing_score_column_is_named_not_guessed(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "_concat_named_csvs",
                        lambda paths: pd.DataFrame({"prc": ["p1_r1_c2"]}))
    with pytest.raises(ValueError, match="no 'pred' column"):
        ml._calibration_inputs(_settings(positive_control="pc_guide",
                                         positive_control_wells=["c2"],
                                         negative_control_wells=["c1"]))


def test_a_sweep_that_cannot_run_leaves_the_number_alone(capsys):
    """The calibration improves a value that already exists; it is not a
    prerequisite for having one, so it must never stop the run."""
    got = ml._calibrated_fraction_threshold(_settings(
        positive_control_wells=None, negative_control_wells=None))
    assert got is None
    # And it says so -- a user who ticked the box would otherwise believe
    # the number they are looking at was measured.
    assert "did not run" in capsys.readouterr().out


def test_the_measured_number_is_returned_when_the_sweep_answers(monkeypatch,
                                                                capsys):
    """The stand-in returns what the real sweep returns.

    It used to answer ``{"threshold": ...}``, a key
    `sweep_fraction_threshold` has never written, so this test passed over a
    reader that could not read the real result. `chosen` is the key;
    `threshold` names a candidate INSIDE the candidates list.
    """
    monkeypatch.setattr(ml, "_calibration_inputs", lambda s: {})
    monkeypatch.setattr(
        "spacr.fraction_calibration.sweep_fraction_threshold",
        lambda **kw: {"chosen": 0.037, "reason": "measured", "candidates": []})
    assert ml._calibrated_fraction_threshold(_settings()) == pytest.approx(0.037)


def test_a_sweep_that_prefers_nothing_says_so(monkeypatch, capsys):
    """The same correction: `chosen` is None when nothing was preferred."""
    monkeypatch.setattr(ml, "_calibration_inputs", lambda s: {})
    monkeypatch.setattr(
        "spacr.fraction_calibration.sweep_fraction_threshold",
        lambda **kw: {"chosen": None, "candidates": []})
    assert ml._calibrated_fraction_threshold(_settings()) is None
    assert "no cut-off it preferred" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# End to end, through the REAL sweep -- the claim no stand-in can make
# ---------------------------------------------------------------------------

PC_GUIDE = "pc_guide"
#: One spurious barcode per well, every one above 0.015 and below 0.02, so
#: the smallest candidate that removes them all is 0.02.
JUNK = (0.016, 0.017, 0.018)


def _control_plate():
    """Counts and per-cell scores for a plate whose PC share is known.

    Column 1 is pure negative control, column 2 pure positive control and
    column 3 a ratio series, which is the design `mixed_ratio_calibration`
    needs to separate penetrance from fraction bias.
    """
    rows, cells = [], []
    plan = []
    for index in range(6):
        plan.append((f"plate1_r{index + 1}_c1", 0.0))
        plan.append((f"plate1_r{index + 1}_c2", 1.0))
        plan.append((f"plate1_r{index + 1}_c3", 0.1 + 0.15 * index))
    for index, (well, share) in enumerate(plan):
        junk = JUNK[index % len(JUNK)]
        rows.append({"prc": well, "grna": PC_GUIDE,
                     "count": share * (1 - junk) * 100000})
        rows.append({"prc": well, "grna": "nc_guide",
                     "count": (1 - share) * (1 - junk) * 100000})
        rows.append({"prc": well, "grna": f"junk{index}",
                     "count": junk * 100000})
        for _ in range(5):
            cells.append({"prc": well, "pred": share * 5.0})
    return pd.DataFrame(rows), pd.DataFrame(cells)


def test_the_real_sweep_reaches_the_run(monkeypatch, capsys):
    """Drives `_calibrated_fraction_threshold` onto the REAL sweep.

    THIS IS THE CLAIM THE STAND-INS COULD NOT MAKE. Both tests above supplied
    a result dict of their own shape, so the reader and the sweep were free
    to disagree about the key holding the answer -- and they did. Every
    screen that ticked `calibrate_fraction_threshold` was told the sweep
    preferred no cut-off, whatever it had measured, and quietly went on using
    the threshold from the settings.

    0.02 is not assumed here: the largest planted contaminant is 0.018 and
    the sweep has to find the smallest candidate above it from the data.
    """
    counts, scores = _control_plate()
    monkeypatch.setattr(
        ml, "_concat_named_csvs",
        lambda paths: scores if "s.csv" in str(paths) else counts)

    got = ml._calibrated_fraction_threshold(_settings(
        positive_control=PC_GUIDE,
        positive_control_wells=["c2"], negative_control_wells=["c1"],
        dependent_variable="pred", count_data="/tmp/c.csv",
        score_data="/tmp/s.csv"))

    assert got == pytest.approx(0.02)
    # And the run says which wells said so, rather than only the number.
    assert "most consistent" in capsys.readouterr().out


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


def test_the_screen_csvs_arrive_with_canonical_key_names(tmp_path):
    """The counts and scores go through the one reader, so the keys are canonical."""
    one = tmp_path / "plate.csv"
    pd.DataFrame({"Plate": ["p1"], "column_name": ["1"], "count": [5]}
                 ).to_csv(one, index=False)
    got = ml._concat_named_csvs([one])
    assert "plateID" in got.columns and "columnID" in got.columns
    assert "column_name" not in got.columns
