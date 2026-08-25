"""The regression summary paragraph refuses to guess and refuses to crash.

The paragraph is what a reader takes away from a run, so it must never be a
sentence about numbers that are not there. These pin the three ways the
summary can be handed something incomplete: a table with no fitted effect at
all, a caller-supplied effect-size cut, and a value that will not turn into
a float.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spacr.figures.summary import _fmt, summarise


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


def test_a_table_with_no_fitted_effect_gets_no_paragraph():
    """A coefficient table missing its effect column summarises to nothing.

    The caller says so itself rather than being handed a paragraph about a
    quantity the fit never produced -- "0 coefficients were tested" would
    read as a finished run that found nothing.
    """
    frame = pd.DataFrame({"feature": ["fraction:gene[abc]"],
                          "p_value": [0.01]})
    assert summarise(frame) == ""


def test_the_effect_size_cut_you_set_is_the_one_that_is_applied():
    """A numeric ``effect_threshold`` overrides the control-derived cut.

    The default measures the cut from the non-targeting guides; a caller who
    names a number is asking for that number, and the paragraph has to both
    apply it and say where it came from, or the hit count cannot be
    reproduced from what is written.
    """
    frame = pd.DataFrame({
        "feature": [f"fraction:gene[g{i}]" for i in range(6)],
        "gene": [f"g{i}" for i in range(6)],
        "coefficient": [3.0, -2.5, 0.2, 0.1, -0.05, 1.9],
        "p_value": [1e-6, 1e-6, 1e-6, 0.4, 0.6, 0.7],
    })
    text = summarise(frame, effect_threshold=2.0)

    assert "the value you set" in text
    assert "effect-size cut of 2" in text
    # 3.0 and -2.5 are significant AND clear the cut; 0.2 is significant but
    # does not; 1.9 clears nothing because its p is 0.7.
    assert "3 pass" in text
    assert "2 also clear" in text
    assert "g0" in text and "g1" in text
    assert "g2" not in text


def test_an_effect_threshold_of_none_leaves_every_significant_hit_listed():
    """Switching the cut off reports the significance rule on its own.

    ``None`` is how a caller asks for no effect-size filter at all, and the
    paragraph must not then quote a cut it did not apply.
    """
    frame = pd.DataFrame({
        "feature": [f"fraction:gene[g{i}]" for i in range(4)],
        "gene": [f"g{i}" for i in range(4)],
        "coefficient": [3.0, 0.2, 0.1, -0.05],
        "p_value": [1e-6, 1e-6, 0.4, 0.6],
    })
    text = summarise(frame, effect_threshold=None)

    assert "effect-size cut" not in text
    assert "2 pass" in text
