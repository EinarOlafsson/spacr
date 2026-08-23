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


@pytest.mark.parametrize("value", [None, "", "   ", "\t\n",
                                   float("nan"), np.nan, np.float64("nan")])
def test_every_spelling_of_empty_is_empty(value):
    assert _left_blank(value) is True


@pytest.mark.parametrize("value", [0, 0.0, 0.5, False, "HC3", "auto", -1])
def test_an_answer_is_not_empty(value):
    """Zero and False are answers. Only NaN and blank text are not."""
    assert _left_blank(value) is False


def test_the_nan_comes_from_reading_a_real_settings_file(tmp_path):
    """The path that produced it, rather than a hand-made float."""
    path = tmp_path / "regression.csv"
    path.write_text("Key,Value\nregression_type,ols\nhinge_threshold,\n"
                    "cov_type,\n")

    frame = pd.read_csv(path)
    settings = dict(zip(frame["Key"], frame["Value"]))

    assert _left_blank(settings["hinge_threshold"]), (
        "an unfilled box in a saved settings file read as an answer")
    assert _left_blank(settings["cov_type"])


def test_an_ols_run_is_not_refused_over_an_unfilled_box():
    """The refusal this was found through."""
    from spacr.ml import _reject_unused_settings

    # Nothing raised: the two blanks are not requests.
    _reject_unused_settings("ols", {
        "hinge_threshold": (float("nan"), None),
        "cov_type": ("", None),
    })


def test_a_setting_that_really_was_asked_for_is_still_refused():
    """Tolerating blanks may not become tolerating mistakes."""
    from spacr.ml import _reject_unused_settings

    with pytest.raises(ValueError, match="hinge_threshold"):
        _reject_unused_settings("ols", {"hinge_threshold": (0.15, None)})
