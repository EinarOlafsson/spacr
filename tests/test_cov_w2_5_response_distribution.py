"""The response-shape caption and pen: no empty parentheses, a 2 px outline.

The rest of :mod:`spacr.response_distribution` is covered by
tests/test_response_distribution_before_after.py.
"""
from __future__ import annotations

import pytest

from spacr import response_distribution as rd


# ---------------------------------------------------------------------------
# compare and caption
# ---------------------------------------------------------------------------

def test_a_family_without_measurable_statistics_shows_only_its_label():
    """A finite family with NaN statistics prints no empty parentheses."""
    part = {"family": "ols", "name": "normal",
            "normality_p": float("nan"), "skew": float("nan")}
    result = {"changed": True, "transform": "none",
              "before": part, "after": part}

    text = rd.caption(result)

    assert "normal" in text
    assert "(" not in text


# ---------------------------------------------------------------------------
# fast_panel
# ---------------------------------------------------------------------------

def test_the_pen_is_two_pixels_wide(qapp):
    """The outline is drawn heavy enough to read over the other one."""
    pen = rd._pen("#4C72B0", "before")

    assert pen.widthF() == pytest.approx(2.0)
