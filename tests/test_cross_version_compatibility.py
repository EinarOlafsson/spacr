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


def test_equal_distance_keypoints_have_a_stable_cap_and_descriptor_order(
    tmp_path, monkeypatch
):
    """A NumPy sort implementation must not decide which descriptor survives."""
    from spacr.spacrops import spacrStitcher

    stitcher = spacrStitcher(
        outdir=str(tmp_path / "out"),
        max_keypoints=3,
        downsample=1.0,
        feature_cache_mode="ram",
        save_qc=False,
        save_stitched_default=False,
    )
    points = np.array(
        [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]],
        dtype=np.float32,
    )
    descriptors = np.arange(5 * 32, dtype=np.uint8).reshape(5, 32)
    monkeypatch.setattr(
        stitcher,
        "_read_plane",
        lambda _path, ch: np.zeros((8, 8), dtype=np.uint16),
    )
    monkeypatch.setattr(
        stitcher,
        "_detect_and_describe",
        lambda _image: (points, descriptors),
    )

    features = stitcher._compute_features_one("unused.tif", channel_index=0)

    assert np.array_equal(features["pts"], points[[4, 0, 3]])
    assert np.array_equal(features["desc"], descriptors[[4, 0, 3]])


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
