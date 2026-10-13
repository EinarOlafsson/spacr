"""Validation and dispatch coverage for Poisson GLMs."""

import numpy as np
import pandas as pd
import pytest


def _design(n, columns=2):
    rng = np.random.default_rng(810)
    data = {"const": np.ones(n)}
    for index in range(1, columns):
        data[f"x{index}"] = rng.normal(size=n)
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ([0, 1, 2, -1, 3, 4, 2, 1], "non-negative count data"),
        ([0, 1, 2, 1.5, 3, 4, 2, 1], "integer count data"),
        ([0, 1, 2, np.nan, 3, 4, 2, 1], "finite count data"),
        ([0, 1, 2, np.inf, 3, 4, 2, 1], "finite count data"),
        ([0, 0, 0, 0, 0, 0, 0, 0], "at least one positive count"),
    ],
)
def test_explicit_poisson_rejects_invalid_counts_before_fit(counts, message):
    from spacr.ml import regression_model

    with pytest.raises(ValueError, match=message):
        regression_model(
            _design(len(counts)),
            pd.Series(counts),
            regression_type="poisson",
        )


def test_poisson_requires_the_absolute_minimum_sample_size():
    from spacr.ml import MIN_POISSON_SAMPLES, regression_model

    n = MIN_POISSON_SAMPLES - 1
    with pytest.raises(ValueError, match=rf"at least {MIN_POISSON_SAMPLES}"):
        regression_model(
            _design(n),
            pd.Series([0, 1, 2, 1, 3, 2, 1]),
            regression_type="poisson",
        )


def test_poisson_requires_residual_degrees_of_freedom():
    from spacr.ml import regression_model

    n = 8
    with pytest.raises(ValueError, match=r"at least 9.*8 model parameters"):
        regression_model(
            _design(n, columns=n),
            pd.Series([0, 1, 2, 1, 3, 2, 1, 4]),
            regression_type="poisson",
        )


def test_explicit_poisson_fits_unscaled_integer_counts():
    from statsmodels.genmod.families import Poisson

    from spacr.ml import regression_model

    rng = np.random.default_rng(811)
    n = 100
    x = rng.normal(size=n)
    X = pd.DataFrame({"const": 1.0, "x": x})
    y = pd.Series(rng.poisson(np.exp(0.8 + 0.35 * x)))

    model = regression_model(X, y, regression_type="poisson")

    assert isinstance(model.model.family, Poisson)
    assert model.params["x"] == pytest.approx(0.35, abs=0.2)


def test_auto_glm_validates_small_count_samples_before_fit():
    from spacr.ml import regression_model

    with pytest.raises(ValueError, match="too few observations"):
        regression_model(
            _design(7),
            pd.Series([0, 1, 2, 1, 3, 2, 1]),
            regression_type="glm",
        )
