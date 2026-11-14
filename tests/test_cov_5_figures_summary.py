"""The regression summary paragraph refuses to guess and refuses to crash.

The paragraph is what a reader takes away from a run, so it must never be a
sentence about numbers that are not there. These pin the three ways the
summary can be handed something incomplete: a table with no fitted effect at
all, a caller-supplied effect-size cut, and a value that will not turn into
a float.
"""
from __future__ import annotations


from spacr.figures.summary import _fmt


def test_a_value_that_is_not_a_number_prints_as_itself():
    """A non-numeric quantity is written out, not raised over.

    Every number in the paragraph goes through this formatter. A missing or
    textual value has to degrade to its own text: the alternative is that one
    absent quantity takes the whole summary down and the reader is told
    nothing about a run that finished.
    """
    assert _fmt(None) == "None"
    assert _fmt("not measured") == "not measured"
    assert _fmt(0.000123456) == "0.000123"
