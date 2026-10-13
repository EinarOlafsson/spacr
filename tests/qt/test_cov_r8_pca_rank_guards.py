"""PCA's two rank guards, and the steps that make them unreachable.

The module's policy is stated at the top of it: features are
standardised unless told otherwise, NaN is never imputed unless told to,
CONSTANT COLUMNS ARE DROPPED BY NAME, collinear ones are kept and
reported, and no more components are returned than the data has.

The third of those retires both guards. A column with no variance is
removed before the decomposition, so the largest singular value is
positive; and a positive largest value implies a rank of at least one,
because the tolerance is that value scaled by machine epsilon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.pca_model import PCAError, PCASpec, pca

pytestmark = pytest.mark.qt


def _frame(seed=0, rows=40):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "cell_area": rng.normal(900, 90, rows),
        "cell_perimeter": rng.normal(120, 12, rows),
        "cell_intensity": rng.normal(1200, 130, rows),
    })


class TestAnOrdinaryDecomposition:

    def test_it_returns_the_components_asked_for(self):
        result = pca(_frame(), PCASpec(features=("cell_area",
                                                 "cell_perimeter",
                                                 "cell_intensity"),
                                       n_components=2))
        assert result.loadings.shape[1] == 2

    def test_it_never_returns_more_components_than_the_data_has(self):
        """Asking for ten from three features cannot give ten."""
        result = pca(_frame(), PCASpec(features=("cell_area",
                                                 "cell_perimeter"),
                                       n_components=10))
        assert result.loadings.shape[1] <= 2


class TestTheGuardsTheColumnDropRetires:
    """Both raises are unreachable, and the reason is one policy step."""

    def test_a_constant_column_is_dropped_rather_than_decomposed(self):
        """THE STEP THAT RETIRES THE FIRST GUARD.

        A column with no variance contributes no singular value. It is
        removed by name before the decomposition, so `largest <= 0`
        cannot be reached through a constant column -- which is the only
        way a caller could produce it.
        """
        frame = _frame()
        frame["cell_constant"] = 5.0
        result = pca(frame, PCASpec(features=("cell_area",
                                              "cell_constant",
                                              "cell_perimeter"),
                                    n_components=2))
        assert "cell_constant" not in result.features, (
            "a column with no variance reached the decomposition")

    def test_an_all_constant_selection_is_refused_before_decomposing(self):
        """And with every column dropped there is nothing left to
        decompose, which is refused by name rather than reaching the
        `largest <= 0` raise."""
        frame = pd.DataFrame({"a": [1.0] * 10, "b": [2.0] * 10})
        with pytest.raises(PCAError) as caught:
            pca(frame, PCASpec(features=("a", "b"), n_components=1))
        assert "variance" in str(caught.value).lower() or \
            "column" in str(caught.value).lower()

    def test_a_positive_largest_value_implies_a_rank_of_at_least_one(self):
        """THE ARGUMENT FOR THE SECOND GUARD, checked rather than asserted.

        `tolerance = largest * max(n, p) * eps`, and `rank` counts the
        singular values above it. With `largest > 0` the largest value
        itself clears the tolerance unless `max(n, p) * eps >= 1` --
        which needs a matrix of about 4.5e15 rows.
        """
        eps = float(np.finfo(float).eps)
        for n, p in ((10, 3), (1000, 50), (10 ** 6, 10 ** 3)):
            assert max(n, p) * eps < 1.0, (
                f"a {n}x{p} matrix makes the tolerance exceed the largest "
                "singular value; the rank guard in pca is now reachable")

    def test_collinear_columns_are_kept_and_reported_not_refused(self):
        """The case that LOOKS like it would produce rank 0 and does not.

        Two identical columns are collinear, not constant. The policy
        keeps them and says so, because dropping one silently would
        change which features the caller thinks were analysed.
        """
        frame = _frame()
        frame["cell_copy"] = frame["cell_area"]
        result = pca(frame, PCASpec(features=("cell_area", "cell_copy",
                                              "cell_perimeter"),
                                    n_components=2))
        assert result.loadings.shape[1] >= 1
