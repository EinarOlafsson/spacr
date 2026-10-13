"""A pyfixest fit with no fixed effects falls back instead of dying.

REPORTED FROM A LIVE RUN, 2026-08-20, twenty seconds into a four-plate
regression:

    ValueError: regression_backend='pyfixest' absorbs the rowID and columnID
    fixed effects, and this design has neither -- either
    model_plate_position=False took them out of the model or the screen sits
    on one row and one column.

`model_plate_position=False` removes the row and column terms, and the
absorbing backend then has nothing to project out.

ITS OWN REFUSAL SAYS WHY REFUSING IS WRONG: "the fit would be the statsmodels
fit with an extra projection in front of it". With nothing to absorb the two
backends compute the SAME numbers, so falling back is not substituting a
different method -- it is the identical fit by the only route left.

THAT IS WHAT MAKES THIS FALLBACK SAFE AND THE MONTAGE'S MULTIVARIATE ONE NOT
(186 A). There, the alternative answered a different question and so had to
be asked about; here the alternative IS the answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.ml import regression_model


@pytest.fixture
def design():
    """A design with no rowID/columnID terms -- what the report describes."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"Intercept": 1.0, "fraction": rng.uniform(0.0, 1.0, n)})
    y = pd.Series(0.2 + 0.5 * X["fraction"] + rng.normal(0.0, 0.05, n),
                  name="pred")
    return X, y


@pytest.fixture(autouse=True)
def pyfixest_backend_is_selectable(monkeypatch):
    """Exercise the absorbing route without requiring the optional package."""
    import sys
    from types import ModuleType

    import spacr.ml as ml

    require_backend = ml._require_backend

    def _available(regression_type, backend):
        if ml.resolve_backend_name(backend) == "pyfixest":
            return "pyfixest"
        return require_backend(regression_type, backend)

    monkeypatch.setattr(ml, "_require_backend", _available)
    pyfixest = ModuleType("pyfixest")
    core = ModuleType("pyfixest.core")
    demean_module = ModuleType("pyfixest.core.demean")

    def _unexpected_demean(*_args, **_kwargs):
        raise AssertionError("a design with no fixed effects must not demean")

    demean_module.demean = _unexpected_demean
    monkeypatch.setitem(sys.modules, "pyfixest", pyfixest)
    monkeypatch.setitem(sys.modules, "pyfixest.core", core)
    monkeypatch.setitem(sys.modules, "pyfixest.core.demean", demean_module)


class TestItFitsInsteadOfFailing:

    def test_the_run_no_longer_dies(self, design):
        X, y = design

        model = regression_model(X, y, "ols", regression_backend="pyfixest")

        assert model is not None
        assert "fraction" in model.params

    def test_the_numbers_are_the_statsmodels_numbers(self, design):
        """Not close -- the same. If they differed, the fallback would be a
        different model wearing the requested one's name."""
        X, y = design

        fell_back = regression_model(X, y, "ols",
                                     regression_backend="pyfixest")
        direct = regression_model(X, y, "ols",
                                  regression_backend="statsmodels")

        assert np.allclose(fell_back.params.to_numpy(),
                           direct.params.to_numpy())
        assert np.allclose(fell_back.bse.to_numpy(), direct.bse.to_numpy())

    def test_it_says_it_fell_back(self, design, capsys):
        """A backend swapped in silence is a run whose summary names a
        backend it did not use."""
        X, y = design

        regression_model(X, y, "ols", regression_backend="pyfixest")
        said = capsys.readouterr().out

        assert "nothing to absorb" in said
        assert "statsmodels" in said
        assert "same numbers" in said, "say why the swap is not a compromise"
        assert "model_plate_position" in said, "and what put it here"

    def test_wls_falls_back_too(self, design):
        X, y = design
        weights = np.linspace(1.0, 4.0, len(y))

        fell_back = regression_model(X, y, "wls", weights=weights,
                                     regression_backend="pyfixest")
        direct = regression_model(X, y, "wls", weights=weights,
                                  regression_backend="statsmodels")

        assert np.allclose(fell_back.params.to_numpy(),
                           direct.params.to_numpy())


class TestItStillRefusesWhatItShould:
    """The fallback must not swallow the backend's other refusals."""

    def test_only_the_nothing_to_absorb_refusal_is_caught(self, design,
                                                          monkeypatch):
        """The fallback is keyed on THAT refusal, not on ValueError.

        Catching the class would turn every genuine problem in the absorbing
        backend -- a singular design, a bad input -- into a quiet swap to
        another backend, and the run would report a fit that the failure had
        nothing to do with.
        """
        import spacr.ml as ml

        X, y = design

        def _explode(*_args, **_kwargs):
            raise ValueError("the normal equations are singular")

        monkeypatch.setattr(ml, "_fit_absorbed_least_squares", _explode)

        with pytest.raises(ValueError, match="singular"):
            regression_model(X, y, "ols", regression_backend="pyfixest")

    def test_wls_without_weights_is_still_refused(self, design):
        """WLS with unit weights IS OLS, and a run labelled 'wls' that fitted
        OLS is exactly the silent mislabelling this refusal prevents."""
        X, y = design

        with pytest.raises(ValueError, match="needs per-well weights"):
            regression_model(X, y, "wls", regression_backend="pyfixest")
