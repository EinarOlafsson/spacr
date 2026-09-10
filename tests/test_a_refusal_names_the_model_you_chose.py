"""A refusal names the model the USER asked for, not the one underneath it.

`horseshoe` is a sparse Poisson GLM, so it reaches
`_validate_poisson_response` and was refused with "Poisson regression
requires integer count data; use a continuous response model for fractional
values" -- an error naming a model the user did not choose, followed by
advice for a choice they never made. Found by fitting all nineteen
regression types against the real TSG101 screen.
"""
import numpy as np
import pytest

from spacr.ml import _validate_poisson_response


FRACTIONS = np.array([0.5, 1.5, 2.5] * 4)


def test_the_default_wording_is_unchanged_for_poisson():
    with pytest.raises(ValueError, match="Poisson regression requires integer"):
        _validate_poisson_response(FRACTIONS)


def test_horseshoe_is_refused_in_its_own_name():
    with pytest.raises(ValueError) as raised:
        _validate_poisson_response(FRACTIONS, model="horseshoe")

    assert "horseshoe requires integer count data" in str(raised.value)
    assert "Poisson regression requires" not in str(raised.value)


@pytest.mark.parametrize("bad, fragment", [
    (np.array([np.nan] * 12), "finite count data"),
    (np.array([-1.0] * 12), "non-negative count data"),
    (np.zeros(12), "at least one positive count"),
    (np.array([1.0, 2.0]), "too few observations"),
])
def test_every_refusal_carries_the_name_through(bad, fragment):
    with pytest.raises(ValueError) as raised:
        _validate_poisson_response(bad, model="horseshoe")

    message = str(raised.value)
    assert fragment in message
    assert message.startswith("horseshoe")


def test_the_real_horseshoe_path_names_itself():
    # The call site, not just the validator: a fractional response through
    # `regression_type='horseshoe'` must say horseshoe.
    from spacr.ml import _fit_horseshoe_poisson

    X = np.ones((12, 2))
    with pytest.raises(ValueError) as raised:
        _fit_horseshoe_poisson(X, FRACTIONS, None)

    assert "horseshoe" in str(raised.value)
