"""Three last branches in label normalisation and the annotation-power report.

Two of the three are cases where a report must NOT print a line: no
unreachable guides, no finite specificity target. A report that prints "0
guides never reach it" is a report people learn to skim.
"""
from __future__ import annotations

import datetime as dt
import decimal

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# agreement._scalar_label — arc 257 -> 261, a value pandas cannot call NaN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [
    decimal.Decimal("2"),                       # a type none of the isinstance
    dt.date(2026, 8, 29),                       # branches above claim
    ("a", "tuple"),
])
def test_a_label_of_an_unexpected_type_is_passed_through_unchanged(value):
    """The ``if pd.isna(value):`` branch not taken, reaching the final return.

    The function's job is to make "1" and 1 compare equal, not to police
    types. A hand-edited database can hold anything, and a value this
    normaliser does not recognise must survive as ITSELF -- two annotators who
    both wrote the same odd value still agree, and turning it into None would
    silently record both as abstentions.
    """
    from spacr.agreement import _scalar_label

    assert _scalar_label(value) == value


def test_the_three_spellings_of_a_missing_label_all_become_none():
    """The branches above, which the pass-through must not swallow."""
    from spacr.agreement import _scalar_label

    assert _scalar_label(None) is None
    assert _scalar_label(float("nan")) is None
    assert _scalar_label(np.nan) is None


def test_a_string_and_an_integer_label_compare_equal():
    """The whole point of the function, stated as a test."""
    from spacr.agreement import _scalar_label

    assert _scalar_label("1") == _scalar_label(1)
    assert _scalar_label(True) == _scalar_label(1)
    assert _scalar_label(2.0) == _scalar_label(2)


# ---------------------------------------------------------------------------
# annotation_power.quality_report — arcs 299 -> 307 and 328 -> 332
# ---------------------------------------------------------------------------

def test_a_screen_where_every_guide_is_reachable_says_nothing_about_the_rest():
    """The ``if unreachable:`` branch not taken.

    "0 guides never reach it in any well" is a sentence that makes a healthy
    screen look like it has a finding. The paragraph is printed only when
    there is something to report, and a well-designed library is exactly the
    case that had never been passed through the report.
    """
    from spacr.annotation_power import quality_report

    text = quality_report({}, power={"guides": 1000,
                                     "guides_reachable": 1000,
                                     "guides_reachable_share": 1.0,
                                     "guides_unreachable": 0})

    assert "guides reachable anywhere" in text
    assert "never reach it in any well" not in text


def test_unreachable_guides_are_named_when_there_are_any():
    """The taken side, so the silence above is visibly a decision."""
    from spacr.annotation_power import quality_report

    text = quality_report({}, power={"guides": 1000,
                                     "guides_reachable": 940,
                                     "guides_reachable_share": 0.94,
                                     "guides_unreachable": 60})

    assert "60 guides never reach it in any well" in text


def test_no_specificity_target_is_offered_when_none_is_attainable():
    """The ``if np.isfinite(needed_sp):`` branch not taken.

    The value is NaN when no specificity at the current shape would do -- the
    screen cannot be fixed that way. Printing "raise specificity to nan" is
    worse than printing nothing: it reads as an instruction, and it is not one.
    """
    from spacr.annotation_power import quality_report

    text = quality_report({}, size={"library_if_wells_fixed": 500.0,
                                    "specificity_needed_at_current_shape":
                                        float("nan")})

    assert "cut the library to" in text
    assert "raise specificity" not in text
    assert "nan" not in text.lower().replace("annotate", "")


def test_an_attainable_specificity_target_is_offered():
    """The taken side, printed to five places as the report promises."""
    from spacr.annotation_power import quality_report

    text = quality_report({}, size={"library_if_wells_fixed": 500.0,
                                    "specificity_needed_at_current_shape":
                                        0.99875})

    assert "raise specificity to 0.99875" in text
