"""Why the covariance guard in the absorbing backend cannot fire.

:func:`spacr.ml._fit_absorbed_least_squares` builds the demeaned cross-product
matrix ``xtx`` once and then touches it twice: ``np.linalg.solve`` for the
coefficients, and ``np.linalg.inv`` for the covariance. Only the first of
those two guards is reachable, because both calls decide singularity from the
same LU factorisation of the same unmodified array and the right-hand side
plays no part in that decision. These tests pin both halves of the argument:
the solve is what refuses a rank-deficient design, and numpy really does
answer ``inv`` for every matrix ``solve`` has just accepted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyfixest")

from spacr.ml import _fit_absorbed_least_squares  # noqa: E402


def test_a_rank_deficient_absorbed_design_fails_at_the_solve_not_the_inverse():
    """The refusal chains from the LinAlgError the coefficient solve raised.

    ``xtx`` is never rebuilt between the two calls, so a design that is
    singular reaches the covariance inverse only if the solve let it through.
    Asserting the chained cause -- not just the message -- is what shows which
    of the two guards actually fired.
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

    with pytest.raises(ValueError) as excinfo:
        _fit_absorbed_least_squares(X, y)

    message = str(excinfo.value)
    assert "normal equations are singular" in message
    assert "its 2 coefficients are not identified" in message, message
    assert "cross-product matrix is singular" not in message, (
        "the covariance guard reported this, which means the solve let a "
        "singular design through")
    assert isinstance(excinfo.value.__cause__, np.linalg.LinAlgError)
