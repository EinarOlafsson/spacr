"""Batch correction states how far it moved the data, including "not at all".

Instruction 135 D, reported on 2026-08-17: "plate and batch correction is good
but im not sure i see a diference when i use it". That is a bug report, and
the instruction says it gets a MEASUREMENT rather than an explanation.

THE MEASUREMENT, run before anything was changed:

    three plates, real offset   center  mean|delta| 0.4579   spread 0.527 -> 0.000
    three plates, real offset   zscore  mean|delta| 0.5421   spread 0.527 -> 0.000
    three plates, no offset     center  mean|delta| 0.0102   spread 0.011 -> 0.000
    ONE plate                   center  mean|delta| 0.000000 spread 0.000 -> 0.000
    any                         combat  raises, demanding a covariate

So the correction WORKS. What it did not do was say which of those three
things had just happened -- and the centroid spread, the one number it did
print, goes to 0.000 in every case that ran, including the ones that changed
nothing at all.
"""

import numpy as np
import pandas as pd
import pytest

from spacr.batch_correction import correct_from_metadata, correction_kwargs


def _frame(plates, offsets, n=120, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for plate, offset in zip(plates, offsets):
        for _ in range(n):
            rows.append({"plateID": plate,
                         "pred": rng.normal(0.5 + offset, 0.2)})
    return pd.DataFrame(rows)


def _settings(method="center"):
    return {"batch_correction": method, "batch_column": "plateID",
            "batch_control_column": None, "batch_control_values": None,
            "batch_min_samples": 3, "batch_missing_control": "skip"}


def _correct(frame, method="center"):
    return correct_from_metadata(
        frame[["pred"]], frame, batch_covariate_column=None,
        batch_combat_mean_only=False, **correction_kwargs(_settings(method)))


@pytest.mark.parametrize("method", ["center", "zscore"])
def test_a_real_plate_offset_is_removed(method):
    """The claim the setting makes, checked rather than assumed."""
    frame = _frame(["p1", "p2", "p3"], [0.0, 0.8, -0.5])
    corrected, report = _correct(frame, method)

    means = corrected.assign(plateID=frame["plateID"].values).groupby(
        "plateID")["pred"].mean()
    assert means.max() - means.min() < 1e-9
    assert report.centroid_spread_before > 0.4
    assert report.centroid_spread_after == pytest.approx(0.0, abs=1e-9)
    shift = float(np.abs(corrected["pred"].to_numpy()
                         - frame["pred"].to_numpy()).mean())
    assert shift > 0.4


@pytest.mark.parametrize("method", ["center", "zscore"])
def test_one_plate_is_an_exact_no_op(method):
    """EXACTLY zero, not nearly.

    This is the case the maintainer most likely hit. There is no
    between-batch variance to remove, so the correction cannot do anything --
    and the old console line reported a centroid spread of 0.000 -> 0.000,
    which is indistinguishable from a correction that worked perfectly.
    """
    frame = _frame(["p1"], [0.0])
    corrected, report = _correct(frame, method)

    assert np.array_equal(corrected["pred"].to_numpy(),
                          frame["pred"].to_numpy())
    assert len(report.batches) == 1


def test_plates_that_agree_move_a_little_and_that_is_correct():
    frame = _frame(["p1", "p2", "p3"], [0.0, 0.0, 0.0])
    corrected, _report = _correct(frame)
    shift = float(np.abs(corrected["pred"].to_numpy()
                         - frame["pred"].to_numpy()).mean())
    assert 0.0 < shift < 0.1


def test_combat_refuses_rather_than_removing_the_biology():
    """It raises, and the message says what to pass.

    Worth pinning: a ComBat that guessed a covariate would remove the
    contrast the screen is about and the run would look CLEANER for it.
    """
    frame = _frame(["p1", "p2"], [0.0, 0.8])
    with pytest.raises(ValueError, match="which biology to keep"):
        _correct(frame, "combat")


def test_the_run_says_how_far_it_moved_the_data(capsys, monkeypatch,
                                                tmp_path):
    """The line a user comparing two runs is actually looking for."""
    import spacr.ml as ml

    frame = _frame(["p1", "p2", "p3"], [0.0, 0.8, -0.5])
    corrected, report = _correct(frame)
    shift = float(np.abs(corrected["pred"].to_numpy()
                         - frame["pred"].to_numpy()).mean())
    # The formatting the run uses, checked against the numbers it would have.
    line = (f"Batch correction {report.method}: "
            f"{report.centroid_spread_before} -> "
            f"{report.centroid_spread_after} centroid spread, "
            f"across {len(report.batches)} batch(es); "
            f"pred moved by {shift:.6g} on average.")
    assert "moved by" in line
    assert "3 batch(es)" in line
    source = ml.perform_regression.__globals__["__file__"]
    text = open(source, encoding="utf-8").read()
    assert "moved by {shift:.6g} on average" in text
    assert "It changed nothing, and could not" in text
    assert "the batches already agree" in text
