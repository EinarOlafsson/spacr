"""A baseline request on a table with no effect column still states a baseline.

Every figure has to carry a sentence saying what its effects are measured
from. A coefficient table whose effect column is spelled something else --
`estimate` where the caller asked for `coefficient` -- must not silently
produce a figure with no baseline sentence, and must not raise either: it
falls back to zero and carries the reason the request could not be honoured.
"""
from __future__ import annotations

import pandas as pd

from spacr import baseline


def test_a_missing_effect_column_falls_back_to_zero_and_says_why():
    """The reason has to name the column that was looked for, because a
    caller reading only 'measured from zero' cannot tell a deliberate zero
    baseline from one it was silently demoted to."""
    frame = pd.DataFrame({"feature": ["g1", "g2"],
                          "estimate": [0.4, -0.2],
                          "condition": ["nc", "test"]})

    result = baseline.resolve(frame, baseline.CONTROLS, column="coefficient")

    assert result.kind == baseline.ZERO
    assert result.shift == 0.0
    assert result.n == 0
    assert result.moves is False
    assert result.reason == "the table has no 'coefficient' column"
    assert "zero" in result.sentence.lower()


def test_a_named_baseline_on_a_columnless_object_reports_the_same_reason():
    """`getattr(frame, "columns", ())` is the guard for a caller that hands
    in something that is not a frame at all; it must reach the same stated
    fallback rather than an AttributeError."""
    result = baseline.resolve(object(), baseline.NAMED, name="GRA14")

    assert result.kind == baseline.ZERO
    assert result.reason == "the table has no 'coefficient' column"
    assert result.sentence
