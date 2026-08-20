"""0.05 minus one scroll of the wheel is -0.95, and that reached a run.

Reported 2026-08-19 on the maintainer's own screen: a regression ran for
forty seconds -- writing its figures, its regression_data.csv and three QC
plots -- and then died inside the permutation with

    ValueError: alpha must be strictly between 0 and 1; got -0.95

A QDoubleSpinBox's default `singleStep` is 1.0. The settings panel built
every float field with `setRange(-1e12, 1e12)` and never set a step, so ONE
wheel tick over `fdr_alpha` wrote 0.05 - 1. The same tick turns
`fraction_threshold` 0.02 into -0.98 and `l1_ratio` 0.5 into -0.5.
"""
import pytest

from spacr.ml import _reject_impossible_probabilities
from spacr.qt.screens.settings_model import _float_domain


@pytest.mark.parametrize("key, default", [
    ("fdr_alpha", 0.05), ("fraction_threshold", 0.02),
    ("l1_ratio", 0.5), ("quantile", 0.5),
])
def test_one_step_never_crosses_zero(key, default):
    low, _high, step = _float_domain(key, default)

    assert max(low, default - step) > 0, (
        f"one down-click on {key} lands at {default - step}")


def test_a_setting_that_states_its_domain_is_held_to_it():
    low, high, _step = _float_domain("fdr_alpha", 0.05)

    assert (low, high) == (1e-6, 1.0)
    # NOT 0.0: the code that reads it refuses 0, so clamping a bad saved
    # value to zero would only move the failure.
    assert low > 0


def test_a_setting_with_no_stated_domain_keeps_its_range_but_gains_a_step():
    low, high, step = _float_domain("huber_t", 1.345)

    assert (low, high) == (-1e12, 1e12), "do not invent a domain"
    assert step == 0.1


def test_the_run_refuses_before_it_writes_anything():
    with pytest.raises(ValueError) as raised:
        _reject_impossible_probabilities({"fdr_alpha": -0.95})

    message = str(raised.value)
    assert "fdr_alpha" in message
    assert "0.05 may be the number you set" in message, (
        "name the value the user probably meant, not just the rule")


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.95, "x"])
def test_every_impossible_value_is_refused(bad):
    with pytest.raises(ValueError):
        _reject_impossible_probabilities({"fdr_alpha": bad})


@pytest.mark.parametrize("good", [0.05, 0.01, 0.5, 0.999])
def test_a_usable_alpha_passes(good):
    settings = {"fdr_alpha": good}
    _reject_impossible_probabilities(settings)
    assert settings == {"fdr_alpha": good}


def test_the_penalty_alpha_is_NOT_treated_as_a_probability():
    # `alpha` is the ridge/lasso PENALTY. 1.0 is its default and a perfectly
    # ordinary value; refusing it would break every penalised fit.
    settings = {"alpha": 1.0}
    _reject_impossible_probabilities(settings)
    assert settings == {"alpha": 1.0}

    settings = {"alpha": 25.0}
    _reject_impossible_probabilities(settings)
    assert settings == {"alpha": 25.0}


def test_fdr_bh_is_the_default_correction():
    """Requested 2026-08-19. The tooltip already said it was."""
    from spacr.settings import get_perform_regression_default_settings

    settings = get_perform_regression_default_settings(
        {"score_data": ["a.csv"], "count_data": ["b.csv"]})

    assert settings["multiple_testing_method"] == "fdr_bh"
    assert settings["fdr_alpha"] == 0.05
