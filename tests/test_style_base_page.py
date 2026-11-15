"""Shared page settings that change what a figure means (``spacr.style_base.apply_page``).

An axis limit, an inverted axis and a painted background cannot be recovered
from the picture, so each one the style names has to reach the axes, and the
default ``"none"`` background has to leave the page unpainted.
"""
from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from spacr.style_base import FigureStyle, apply_page  # noqa: E402


def test_apply_page_puts_the_styled_limits_direction_and_background_on_the_axes(
        subtests):
    """Each style field lands exactly on the figure; untouched ones stay put."""
    cases = [
        ("y limit", FigureStyle(y_lim=(-3.0, 7.0)),
         lambda f, a, before: a.get_ylim() == (-3.0, 7.0)),
        ("x limit", FigureStyle(x_lim=(-1.0, 5.0)),
         lambda f, a, before: a.get_xlim() == (-1.0, 5.0)),
        ("inverted x reverses the limits", FigureStyle(invert_x=True),
         lambda f, a, before: a.get_xlim() == before["xlim"][::-1]),
        ("inverted y reverses the limits", FigureStyle(invert_y=True),
         lambda f, a, before: a.get_ylim() == before["ylim"][::-1]),
        ("log y scale", FigureStyle(y_scale="log"),
         lambda f, a, before: a.get_yscale() == "log"),
        ("title and labels", FigureStyle(title="T", x_label="X", y_label="Y"),
         lambda f, a, before: (a.get_title(), a.get_xlabel(), a.get_ylabel())
         == ("T", "X", "Y")),
        ("named background paints page and axes",
         FigureStyle(background_color="#101820"),
         lambda f, a, before: f.patch.get_facecolor() == to_rgba("#101820")
         and a.get_facecolor() == to_rgba("#101820")),
        ("default background leaves the page alone", FigureStyle(),
         lambda f, a, before: f.patch.get_facecolor() == before["page"]
         and a.get_facecolor() == before["axes"]
         and a.get_xlim() == before["xlim"]),
    ]
    for case_id, style, check in cases:
        with subtests.test(msg=case_id):
            figure, axes = plt.subplots()
            try:
                axes.plot([1, 2, 3], [1, 2, 4])
                before = {"xlim": axes.get_xlim(), "ylim": axes.get_ylim(),
                          "page": figure.patch.get_facecolor(),
                          "axes": axes.get_facecolor()}
                apply_page(figure, axes, style)
                assert check(figure, axes, before)
            finally:
                plt.close(figure)
