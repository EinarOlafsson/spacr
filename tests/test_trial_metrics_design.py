"""Design diagnostics read off a fitted model (``spacr.trial_metrics.design_diagnostics``).

A sweep row must survive a model whose design is not a numeric matrix of rows
(nothing is reported) and a correlation matrix that will not fit in memory
(only the pair statistics are lost, never the rank or identifiability).
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.trial_metrics import design_diagnostics


class _Model:
    """A stand-in fit whose attributes are set per case."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)


_EXOG = np.column_stack([np.ones(30),
                         np.random.default_rng(0).normal(size=(30, 3))])

_CORE = {"n_parameters": 4, "design_rank": 4, "non_identifiable_directions": 0,
         "residual_degrees_of_freedom": 26, "design_identifiable": True,
         "wells_per_parameter": 7.5}


def test_design_diagnostics_report_what_the_fit_supports(subtests, monkeypatch):
    """Each design gives its exact statistics, or an empty dict, never a crash."""
    fitted = _Model(model=_Model(exog=_EXOG, rank=4, k_constant=1), df_resid=26)
    cases = [
        ("no model", None, {}),
        ("no design matrix", _Model(model=_Model(exog=None)), {}),
        ("a design of text", _Model(model=_Model(
            exog=np.array([["a", "b"], ["c", "d"]]))), {}),
        ("one dimension", _Model(model=_Model(exog=np.array([1.0, 2.0, 3.0]))),
         {}),
        ("no rows", _Model(model=_Model(exog=np.zeros((0, 3)))), {}),
        ("a healthy design", fitted,
         dict(_CORE, n_collinear_pairs=0,
              max_abs_predictor_correlation=pytest.approx(0.1714002334231973))),
    ]
    for case_id, model, expected in cases:
        with subtests.test(msg=case_id):
            assert design_diagnostics(model) == expected

    def refuse(*args, **kwargs):
        raise MemoryError("the correlation matrix does not fit")

    monkeypatch.setattr(np, "corrcoef", refuse)
    with subtests.test(msg="a correlation matrix too big costs only the pairs"):
        assert design_diagnostics(fitted) == _CORE
