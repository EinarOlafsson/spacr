"""``derive_replot_recipe`` cannot be handed a nameless group.

Instruction 310, A15. The function ended with
``if frame[DERIVED_GROUP].nunique() < 1: return None`` -- a guard that
could not fire, and that told a reader nameless groups were a real case
to handle.

The group column comes only from ``_named``, which returns a tick label
or ``f"{float(x):g}"``. Both are strings; a NaN x becomes the string
``"nan"``, not a missing value. With at least two rows guaranteed by the
check above it, ``nunique()`` is one or more by construction.

This pins that premise rather than the deletion, because the premise is
what would have to change for the removal to be wrong.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("PySide6")

import pandas

from spacr.qt.widgets.figure_settings import DERIVED_GROUP, _named


@pytest.mark.parametrize("labels,x", [
    ({}, 1.0),
    ({}, 0.0),
    ({}, -3.5),
    ({}, float("nan")),
    ({}, float("inf")),
    ({1.0: "treated"}, 1.0),
    ({1.0: ""}, 1.0),
    ({1.0: "   "}, 1.0),
    ({1.0: "near"}, 1.2),
])
def test_named_always_returns_a_string(labels, x):
    """THE PREMISE. Never None, never NaN -- a string every time.

    The NaN case is the one that matters: it comes back as the string
    "nan", which pandas counts as a value. That is what makes a column
    built from this function impossible to count as empty.
    """
    result = _named(labels, x)
    assert isinstance(result, str)
    assert result is not None


def test_a_column_of_named_values_is_never_empty():
    """The premise, carried to the thing the guard tested.

    Two rows of anything `_named` can produce give a nunique of at least
    one, so the deleted guard had no input that could reach it.
    """
    values = [_named({}, float("nan")), _named({}, 1.0)]
    assert pandas.DataFrame({DERIVED_GROUP: values})[
        DERIVED_GROUP].nunique() >= 1


def test_only_real_missing_values_could_have_tripped_it():
    """What the guard was watching for, and why it cannot arrive.

    A column of actual None does count as empty -- so the guard was not
    nonsense, it was unreachable. Shown here so the deletion is
    understood as "no input produces this" rather than "this can never
    happen to a pandas column".
    """
    assert pandas.DataFrame({DERIVED_GROUP: [None, None]})[
        DERIVED_GROUP].nunique() == 0
    assert not any(
        _named(labels, x) is None
        for labels in ({}, {1.0: "a"})
        for x in (0.0, 1.0, float("nan"), float("inf")))
