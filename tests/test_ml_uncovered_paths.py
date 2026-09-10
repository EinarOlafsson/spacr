"""Why singular absorbed designs are refused before backend arithmetic.

:func:`spacr.ml._fit_absorbed_least_squares` builds the demeaned cross-product
matrix ``xtx`` once.  An explicit rank check must reject a deficient design
before ``np.linalg.solve`` because LAPACK backends are not required to raise
for every numerically singular matrix.  The test below pins that refusal to
the library's invariant instead of one backend's exception behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyfixest")

from spacr.ml import _fit_absorbed_least_squares  # noqa: E402


def test_a_rank_deficient_absorbed_design_fails_before_the_solve(monkeypatch):
    """The explicit rank guard refuses the design without entering LAPACK.

    A backend-dependent ``LinAlgError`` is deliberately not the cause: some
    supported LAPACK builds return an arbitrary solution for this matrix.
    """
    n = 12
    x = np.arange(n, dtype=float)
    X = pd.DataFrame({
        'Intercept': np.ones(n),
        'rowID[T.B]': np.tile([0., 1., 0.], 4),
        'rowID[T.C]': np.tile([0., 0., 1.], 4),
        'x': x,
        'x_dup': x,
    })
    y = 2.0 * x + 1.0

    def solve_was_reached(*_args, **_kwargs):
        pytest.fail("rank-deficient normal equations reached np.linalg.solve")

    monkeypatch.setattr(np.linalg, "solve", solve_was_reached)

    with pytest.raises(ValueError) as excinfo:
        _fit_absorbed_least_squares(X, y)

    message = str(excinfo.value)
    assert "normal equations are singular" in message
    assert "its 2 coefficients are not identified" in message, message
    assert "cross-product matrix is singular" not in message, (
        "the covariance guard reported this, which means the solve let a "
        "singular design through")
    assert excinfo.value.__cause__ is None
