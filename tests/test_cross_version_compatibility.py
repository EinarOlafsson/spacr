"""Behavior that must not change across the supported numerical stacks."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest


def test_scorecard_nul_errors_have_one_cross_version_diagnosis(
    tmp_path, monkeypatch
):
    """Python versions disagree on NUL parsing, but the public error must not."""
    from spacr import seg_qc

    path = tmp_path / "scorecard.csv"
    path.write_text("field,n_objects\nplate1_A01_f1,1\n", encoding="utf-8")

    class RejectsNul:
        fieldnames = ["field", "n_objects"]

        def __iter__(self):
            raise csv.Error("line contains NUL")

    monkeypatch.setattr(seg_qc.csv, "DictReader", lambda _handle: RejectsNul())

    rows, error = seg_qc.read_scorecard(str(path))

    assert rows == []
    assert error == "scorecard.csv is not CSV (NUL byte)"


@pytest.mark.parametrize(
    "error_type",
    [ValueError, RuntimeError, FloatingPointError, OverflowError, IndexError],
)
def test_numerical_histogram_failures_mean_no_invasion_threshold(
    monkeypatch, error_type
):
    """Known numerical refusals skip one unusable threshold, not the run."""
    from skimage import filters

    from spacr.submodules import _invasion_threshold

    def refuse(_values):
        raise error_type("cannot histogram this range")

    monkeypatch.setattr(filters, "threshold_otsu", refuse)

    assert math.isnan(_invasion_threshold(np.array([0.0, 1.0]), "otsu"))


def test_programming_errors_from_invasion_thresholds_still_propagate(monkeypatch):
    """The numerical guard must not turn an implementation error into missing data."""
    from skimage import filters

    from spacr.submodules import _invasion_threshold

    def fail(_values):
        raise TypeError("wrong threshold API")

    monkeypatch.setattr(filters, "threshold_otsu", fail)

    with pytest.raises(TypeError, match="wrong threshold API"):
        _invasion_threshold(np.array([0.0, 1.0]), "otsu")
