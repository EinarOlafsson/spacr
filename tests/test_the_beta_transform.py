"""Instruction 174: a proportion may be fitted on the logit scale.

The user asked for "beta transform as an option for the regression as well",
alongside the beta option for picking cells. A classification score and its
well aggregate are both PROPORTIONS, and a proportion is the one shape `log`
handles badly: it is bounded at both ends, so the skew `log` is meant to pull
out is not the skew a proportion has.

The whole risk of the feature is the endpoint. A screen produces wells at
exactly 0 and exactly 1, and logit(0) is -inf. Every test here is about what
happens at the ends.
"""
import numpy as np
import pytest

from spacr.ml import BETA_SQUEEZE_NOTE, apply_transformation, beta_logit


def test_the_middle_is_the_plain_logit():
    """No squeeze when no endpoint is present: the user's number, untouched."""
    values = np.array([0.25, 0.5, 0.75])
    got = beta_logit(values)
    want = np.log(values / (1.0 - values))
    assert np.allclose(got, want)


def test_a_half_is_zero():
    assert beta_logit([0.5]) == pytest.approx(0.0)


def test_it_is_monotone():
    """A transform that reordered the wells would reorder the hits."""
    values = np.linspace(0.0, 1.0, 51)
    out = beta_logit(values)
    assert np.all(np.diff(out) > 0)


def test_the_endpoints_stay_finite():
    """THE POINT OF THE SQUEEZE. logit(0) is -inf and -inf ends the fit."""
    out = beta_logit([0.0, 0.5, 1.0])
    assert np.all(np.isfinite(out))


def test_the_endpoints_stay_symmetric():
    out = beta_logit([0.0, 1.0])
    assert out[0] == pytest.approx(-out[1])


def test_a_wider_screen_squeezes_less():
    """The squeeze is (y*(n-1)+0.5)/n: it shrinks as the screen grows.

    A rule that squeezed by a fixed epsilon would make the extreme wells of
    a 1,536-well plate say the same thing as those of a 96-well plate.
    """
    narrow = beta_logit(np.r_[0.0, np.full(10, 0.5), 1.0])
    wide = beta_logit(np.r_[0.0, np.full(2000, 0.5), 1.0])
    assert abs(wide[-1]) > abs(narrow[-1])


def test_a_nan_stays_a_nan():
    """Missing stays missing -- it must not become an extreme well."""
    out = beta_logit([0.2, np.nan, 0.8])
    assert np.isnan(out[1])
    assert np.all(np.isfinite(out[[0, 2]]))


def test_a_nan_does_not_change_the_squeeze():
    """`n` counts the wells that are THERE, not the length of the column."""
    without = beta_logit([0.0, 0.5, 1.0])
    with_gap = beta_logit([0.0, 0.5, 1.0, np.nan])
    assert np.allclose(without, with_gap[:3], equal_nan=True)


def test_an_empty_column_is_not_an_error():
    out = beta_logit([])
    assert out.shape == (0,)


def test_an_all_nan_column_is_returned_not_raised():
    out = beta_logit([np.nan, np.nan])
    assert np.all(np.isnan(out))


def test_it_does_not_write_through_the_caller_s_array():
    """A transform that edited the response in place would corrupt the run."""
    values = np.array([0.0, 0.5, 1.0])
    beta_logit(values)
    assert np.allclose(values, [0.0, 0.5, 1.0])


def test_the_pipeline_offers_it_by_name():
    """`transform='beta'` has to reach the fit, not just exist in ml.py."""
    transformer = apply_transformation(None, "beta")
    assert transformer is not None
    got = transformer.fit_transform(np.array([[0.25], [0.75]]))
    assert np.allclose(got.ravel(), beta_logit([0.25, 0.75]))


def test_an_unknown_transform_is_still_none():
    assert apply_transformation(None, "logit") is None


def test_the_squeeze_is_reported_not_silent():
    """A transform that moved a user's 0 without saying so changed their data."""
    assert "0" in BETA_SQUEEZE_NOTE and "1" in BETA_SQUEEZE_NOTE
    assert "squeeze" in BETA_SQUEEZE_NOTE.lower()


@pytest.mark.parametrize("source", [
    "spacr/qt/screens/settings_model.py",
    "spacr/settings_spec.py",
])
def test_the_settings_panel_offers_beta(source):
    """Both option lists, or the user cannot choose it in the GUI."""
    import pathlib
    text = pathlib.Path(source).read_text()
    line = next(l for l in text.splitlines()
                if "'transform'" in l or '"transform"' in l)
    assert "beta" in line, line



def _scores(values):
    """The shape `process_scores` reads: one row per object, with a well."""
    import pandas as pd
    n = len(values)
    return pd.DataFrame({
        "pred": values,
        "plateID": ["plate1"] * n,
        "rowID": ["r1"] * n,
        "columnID": [f"c{i % 3 + 1}" for i in range(n)],
        "prc": [f"plate1_r1_c{i % 3 + 1}" for i in range(n)],
    })


def test_a_response_outside_zero_to_one_is_refused():
    """THE failure mode: a logit of an intensity is still a plausible number.

    Nothing downstream can tell that the coefficients are meaningless -- they
    have the right sign, the right magnitude and a p-value -- so the run has
    to stop here or not at all.
    """
    from spacr.ml import process_scores

    frame = _scores([120.0, 4300.0, 900.0] * 30)
    with pytest.raises(ValueError) as excinfo:
        process_scores(frame, "pred", plate="plate1", min_cell_count=1,
                       transform="beta")
    message = str(excinfo.value)
    assert "proportion" in message
    assert "beta" in message
    assert "pred" in message


def test_a_proportion_response_is_accepted():
    """The same call, on the column the transform is actually for."""
    from spacr.ml import process_scores

    rng = np.random.default_rng(0)
    frame = _scores(rng.uniform(0.0, 1.0, 90))
    out, name = process_scores(frame, "pred", plate="plate1",
                               min_cell_count=1, transform="beta")
    assert name == "beta_pred"
    assert np.all(np.isfinite(out[name]))


def test_the_endpoints_survive_the_whole_call():
    """A well where every cell was called positive aggregates to exactly 1."""
    from spacr.ml import process_scores

    frame = _scores([1.0] * 45 + [0.0] * 45)
    out, name = process_scores(frame, "pred", plate="plate1",
                               min_cell_count=1, transform="beta")
    assert np.all(np.isfinite(out[name])), "an all-positive well became -inf"


def test_the_summary_reports_the_squeeze_where_the_transform_is_named():
    """168's rule: the run says what it did to the user's data.

    Pinned on the SOURCE because building a whole fitted run to read one
    field would test the fixture, not the report.
    """
    import pathlib
    text = pathlib.Path("spacr/regression_summary.py").read_text()
    assert "BETA_SQUEEZE_NOTE" in text, (
        "the summary names the transform but not what it did to the scale")
