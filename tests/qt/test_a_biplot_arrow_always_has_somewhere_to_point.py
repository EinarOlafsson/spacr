"""``pca()`` never hands the biplot a non-finite correlation.

Instruction 288. ``_draw_arrows`` skips any arrow whose coordinates are
not finite, and that guard carried ``# pragma: no cover - correlations
are clipped``.

The pragma was wrong twice over. The line was already covered, by
``test_a_feature_whose_correlation_is_not_a_number_gets_no_arrow`` -- and
the guard is NOT unreachable, because ``set_result`` is public, takes any
``PCAResult``, and the dataclass validates nothing.

What IS true is the narrower claim the pragma was reaching for: a result
built by ``pca()`` cannot carry a non-finite correlation, because that
function ends its correlation block with::

    correlations = np.clip(np.nan_to_num(correlations), -1.0, 1.0)

These tests pin that narrower claim, which is what lets the guard be read
as protecting the public API rather than the normal path. If the line
changed -- to drop the ``nan_to_num``, say, keeping only the clip, which
does NOT remove NaN -- arrows would start disappearing from ordinary
plots, and that is worth a failing test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.pca_model import PCAError, PCASpec, pca


def _frame(array):
    return pd.DataFrame(array,
                        columns=[f"f{i}" for i in range(array.shape[1])])


def _result(array):
    frame = _frame(array)
    return pca(frame, PCASpec(features=tuple(frame.columns)))


@pytest.mark.parametrize("name,array", [
    ("plain", np.random.default_rng(0).normal(0, 1, (20, 4))),
    ("zero variance column",
     np.column_stack([np.full(20, 5.0),
                      np.random.default_rng(1).normal(0, 1, (20, 3))])),
    ("perfectly collinear",
     np.column_stack([np.arange(20.0), np.arange(20.0) * 2.0,
                      np.random.default_rng(2).normal(0, 1, 20)])),
    ("huge", np.random.default_rng(3).normal(0, 1, (20, 3)) * 1e12),
    ("tiny", np.random.default_rng(4).normal(0, 1, (20, 3)) * 1e-12),
])
def test_correlations_come_back_finite(name, array):
    """The inputs most likely to produce a NaN correlation."""
    try:
        result = _result(array)
    except PCAError:
        pytest.skip(f"{name} is refused before a result exists")
    assert np.isfinite(result.correlations).all(), (
        f"{name} produced a non-finite correlation")
    assert np.all(np.abs(result.correlations) <= 1.0), (
        "a correlation outside [-1, 1] means the clip stopped working")


def test_a_column_of_nan_does_not_leak_into_the_correlations():
    """NaN in, finite out -- the specific thing nan_to_num is there for."""
    rng = np.random.default_rng(5)
    array = rng.normal(0, 1, (20, 3))
    array[0, 0] = np.nan
    try:
        result = _result(array)
    except PCAError:
        pytest.skip("a NaN column is refused outright")
    assert np.isfinite(result.correlations).all()


def test_the_clip_alone_would_not_be_enough():
    """WHY the premise names nan_to_num and not just the clip.

    `np.clip` passes NaN straight through. If somebody simplified that
    line to a bare clip, correlations could be NaN again and the deleted
    guard would be needed. Pinning numpy's behaviour makes the reason
    explicit rather than folklore.
    """
    assert np.isnan(np.clip(np.nan, -1.0, 1.0))
    assert not np.isnan(np.nan_to_num(np.nan))


def test_the_model_still_clips_and_denans():
    """Read the source line the premise rests on.

    A behavioural test cannot distinguish "clipped" from "the data
    happened to be in range", so the line itself is asserted too.
    """
    import inspect

    from spacr.qt.widgets import pca_model

    source = inspect.getsource(pca_model)
    assert "np.clip(np.nan_to_num(correlations)" in source, (
        "the correlation clip changed shape; the biplot's finite check "
        "was deleted on the strength of this line")
