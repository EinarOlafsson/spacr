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
import pytest

pytestmark = pytest.mark.qt


class TestTheGuardsTheColumnDropRetires:
    """Both raises are unreachable, and the reason is one policy step."""

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
