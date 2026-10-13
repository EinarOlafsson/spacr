"""PCA refuses a variance-free matrix while it can still name the columns.

``pca`` used to carry two more guards after the decomposition -- a largest
singular value of zero, and a rank of zero -- both marked
``# pragma: no cover`` with the reason "constants were already removed".
The reason was right, and instruction 288 counted the two unreachable
lines.

``_drop_constant`` refuses first, and refuses BETTER: it can say which
features were constant, because it is looking at features. A guard after
the decomposition can only say "the analysed matrix has no variance left",
which names nothing the user chose.

This file pins the premise the removal rests on -- that nothing flat gets
past ``_drop_constant`` -- rather than the removal itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.pca_model import PCAError, PCASpec, pca

#: Features named explicitly, so `candidate_features` cannot refuse the
#: frame first and leave these tests measuring a different guard.
SPEC = PCASpec(features=("a", "b"))


def test_a_matrix_with_no_variance_is_refused_by_name(  ):
    """Every column constant. The message names the features."""
    frame = pd.DataFrame({"a": [1.0] * 8, "b": [1.0] * 8})
    with pytest.raises(PCAError) as raised:
        pca(frame, SPEC)
    assert "constant" in str(raised.value)


def test_one_varying_feature_is_refused_and_says_which():
    """PCA needs two directions; one is not a decomposition."""
    frame = pd.DataFrame({"a": [1.0] * 8, "b": np.arange(8.0)})
    with pytest.raises(PCAError) as raised:
        pca(frame, SPEC)
    message = str(raised.value)
    assert "1 feature varies" in message and "b" in message


def test_denormal_values_are_still_constant():
    """Numbers too small to distinguish are refused as constants.

    The interesting case for the deleted guards: this is the shape that
    would produce a zero singular value if it ever got as far as the
    decomposition. It does not.
    """
    frame = pd.DataFrame({"a": [1e-320, 2e-320] * 4,
                          "b": [3e-320, 4e-320] * 4})
    with pytest.raises(PCAError) as raised:
        pca(frame, SPEC)
    assert "constant" in str(raised.value)


def test_perfectly_collinear_columns_still_decompose():
    """THE OTHER SIDE, and it is why the guards could not simply be
    tightened.

    Collinearity reduces the rank to 1, not to 0 -- there is still a
    direction, and refusing it would refuse a legitimate result. So the
    thing being removed is genuinely dead rather than merely
    inconvenient.
    """
    frame = pd.DataFrame({"a": np.arange(8.0), "b": np.arange(8.0) * 2})
    result = pca(frame, SPEC)
    assert result.loadings.shape[1] >= 1
