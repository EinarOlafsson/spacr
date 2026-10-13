"""A null shuffle that measured nothing is dropped, not scored as zero.

Instruction 310, entry A57. ``_null_threshold`` initialised ``top = 0.0``
before the feature loop and only ever raised it with a finite score, so a
permutation in which every ``_separation`` returned NaN -- each candidate
class empty once the finite mask was applied -- appended a literal 0.0. The
null distribution then contained "chance reached zero" for a shuffle that had
in fact measured nothing.

WHY IT MATTERS: the threshold is the 95th percentile of that distribution.
Spurious zeros drag it down, and ``ExplorerResult.above_null()`` then lists
features that never beat chance. On a sparsely measured table -- which is the
ordinary case for per-object measurements, where many columns are populated
for only some objects -- a large share of the null can be those zeros.

The case below is the extreme end of it, chosen because it is deterministic:
with a single finite value in the column, every shuffle leaves one candidate
class empty, so EVERY shuffle is unmeasurable. Before the fix the null was all
zeros and the threshold came back 0.0, which passes everything; after it, no
shuffle is scored, the null is empty and the honest answer is None.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt.widgets.feature_rank import ExplorerSpec, _null_threshold


def _one_finite_value_column(rows: int = 12):
    """A column measured for exactly one object, and two balanced classes."""
    values = np.full(rows, np.nan)
    values[0] = 1.0
    keys = np.array(["a"] * (rows // 2) + ["b"] * (rows - rows // 2))
    return {"sparse": values}, keys, ["a", "b"]


def test_a_null_that_measured_nothing_is_none_not_zero():
    columns, keys, levels = _one_finite_value_column()
    notices: list[str] = []
    spec = ExplorerSpec(label="cls", n_permutations=32, seed=0)

    threshold = _null_threshold(columns, keys, levels, spec, notices)

    assert threshold is None, (
        f"every shuffle was unmeasurable, so the null is empty; a threshold "
        f"of {threshold!r} is the A57 defect -- a floor of zero passes every "
        "feature through above_null()"
    )


def test_the_dropped_shuffles_are_reported():
    """Say what the number cannot say: the null is thinner than requested."""
    columns, keys, levels = _one_finite_value_column()
    notices: list[str] = []
    spec = ExplorerSpec(label="cls", n_permutations=32, seed=0)

    _null_threshold(columns, keys, levels, spec, notices)

    assert any("measured no feature" in note for note in notices), notices
    assert any("32" in note for note in notices), notices


def test_a_measurable_column_still_produces_a_threshold():
    """The ordinary path is unchanged: real data still calibrates a null."""
    rng = np.random.default_rng(0)
    rows = 60
    columns = {"real": rng.normal(size=rows)}
    keys = np.array(["a"] * (rows // 2) + ["b"] * (rows // 2))
    notices: list[str] = []
    spec = ExplorerSpec(label="cls", n_permutations=32, seed=0)

    threshold = _null_threshold(columns, keys, ["a", "b"], spec, notices)

    assert threshold is not None and np.isfinite(threshold)
    assert threshold > 0.0
    assert not notices, f"nothing should have been dropped: {notices}"


@pytest.mark.parametrize("permutations", [0])
def test_no_permutations_still_means_no_null(permutations):
    columns, keys, levels = _one_finite_value_column()
    spec = ExplorerSpec(label="cls", n_permutations=permutations, seed=0)
    assert _null_threshold(columns, keys, levels, spec, []) is None
