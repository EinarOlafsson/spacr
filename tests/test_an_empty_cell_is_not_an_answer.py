"""An empty cell in a settings CSV refused an ordinary regression run.

Driving the real four-plate tsg101 screen on its own settings/regression.csv
stopped with:

    ValueError: regression_type='ols' does not read hinge_threshold=nan: it
    is the cut that turns a continuous response into the two classes a hinge
    loss separates ... Leave hinge_threshold at its default (None), or
    choose a regression type that uses it (hinge).

Nobody had set ``hinge_threshold``. The line in the file is
``hinge_threshold,`` -- which is what a SAVED settings file looks like for
every box the user did not fill -- and ``pandas.read_csv`` turns an empty
cell into ``float('nan')``.

``_left_blank`` knew three spellings of empty: None, ``''`` and whitespace.
NaN is the fourth and the one that actually arrives from a file, and being
neither None nor a str it read as ANSWERED. So the guard that exists to
catch a setting the chosen backend cannot read fired on a value nobody
typed, and refused the run.

NaN is never a threshold, a covariance type or a quantile. There is no
reading of it that means "answered".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.ml import _left_blank


@pytest.mark.parametrize("value, expected", [
    pytest.param(None, True, id="none-is-blank"),
    pytest.param("", True, id="empty-text-is-blank"),
    pytest.param("   ", True, id="spaces-are-blank"),
    pytest.param("\t", True, id="tab-is-blank"),
    pytest.param("\t\n", True, id="tab-newline-is-blank"),
    pytest.param(float("nan"), True, id="float-nan-is-blank"),
    pytest.param(np.nan, True, id="np-nan-is-blank"),
    pytest.param(np.float64("nan"), True, id="np-float64-nan-is-blank"),
    pytest.param(0, False, id="zero-is-an-answer"),
    pytest.param(0.0, False, id="zero-float-is-an-answer"),
    pytest.param(0.5, False, id="half-is-an-answer"),
    pytest.param(1.345, False, id="a-tuning-constant-is-an-answer"),
    pytest.param(-1, False, id="minus-one-is-an-answer"),
    pytest.param(False, False, id="false-is-an-answer"),
    pytest.param("HC3", False, id="a-covariance-type-is-an-answer"),
    pytest.param("auto", False, id="auto-is-an-answer"),
])
def test_only_nan_and_blank_text_count_as_empty(value, expected):
    """Zero and False are answers. Only None, NaN and blank text are not."""
    assert _left_blank(value) is expected


def test_the_nan_comes_from_reading_a_real_settings_file(tmp_path):
    """The path that produced it, rather than a hand-made float."""
    path = tmp_path / "regression.csv"
    path.write_text("Key,Value\nregression_type,ols\nhinge_threshold,\n"
                    "cov_type,\n", encoding="utf-8")

    frame = pd.read_csv(path)
    settings = dict(zip(frame["Key"], frame["Value"]))

    assert _left_blank(settings["hinge_threshold"]), (
        "an unfilled box in a saved settings file read as an answer")
    assert _left_blank(settings["cov_type"])


def test_an_ols_run_is_not_refused_over_an_unfilled_box():
    """The refusal this was found through."""
    from spacr.ml import _reject_unused_settings

    # Nothing raised: the two blanks are not requests. The return value is
    # asserted rather than the absence of an exception, so the test fails if
    # the function is ever changed to REPORT instead of raise.
    assert _reject_unused_settings("ols", {
        "hinge_threshold": (float("nan"), None),
        "cov_type": ("", None),
    }) is None


def test_a_setting_that_really_was_asked_for_is_still_refused():
    """Tolerating blanks may not become tolerating mistakes."""
    from spacr.ml import _reject_unused_settings

    with pytest.raises(ValueError, match="hinge_threshold"):
        _reject_unused_settings("ols", {"hinge_threshold": (0.15, None)})
