"""Every volcano style option has to reach the picture, or it is a lie.

The style object is what the figure menu writes into, so a field the
renderer quietly ignores looks to the user like a control that does nothing.
These drive the options that were never exercised -- axis limits and scales,
inversion, the grid switch, the plot background, the title, the effect-size
lines and the point labels -- and the two refusals that keep a malformed
frame from producing a blank figure instead of a message.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from spacr.volcano_style import (VolcanoStyle, _resolve_effect_threshold,
                                 render_volcano)


def _results(n=40, seed=0):
    """A small coefficient table with a handful of real hits."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(0.0, 0.4, n)
    p = rng.uniform(0.2, 1.0, n)
    effect[:4] = [2.4, -2.1, 1.9, -1.7]
    p[:4] = [1e-6, 1e-5, 1e-4, 1e-4]
    return pd.DataFrame({
        "standardized_marginal_effect": effect,
        "adjusted_p_value": p,
        "guide": [f"g{i:03d}" for i in range(n)],
        "plate": ["p1" if i % 2 else "p2" for i in range(n)],
    })


def test_a_column_of_nothing_numeric_sets_no_effect_cut():
    """An effect column with no finite value yields no threshold at all.

    A spread estimated from nothing is either a crash or a zero cut, and a
    zero cut marks every guide in the screen significant. Returning None
    draws the volcano with no effect line, which is the honest picture.
    """
    style = VolcanoStyle(threshold_method="mad")
    values = np.array([np.nan, np.nan, np.inf * 0])

    assert _resolve_effect_threshold(values, style) is None


def test_a_frame_without_the_y_column_says_which_columns_it_has():
    """A missing y column raises and names the columns that are there.

    The y column is a style field a user can type into. Silently plotting
    NaN would give an empty panel with axes, which reads as "no hits" rather
    than "you named a column that does not exist".
    """
    frame = _results().drop(columns=["adjusted_p_value"])
    style = VolcanoStyle()

    with pytest.raises(ValueError) as excinfo:
        render_volcano(frame, style)

    message = str(excinfo.value)
    assert "adjusted_p_value" in message
    assert "standardized_marginal_effect" in message


def test_a_volcano_with_no_hits_still_draws_the_rest():
    """When nothing is significant the significant group is skipped.

    An empty scatter group is not an error and must not become a legend
    entry for a series with no points -- a reader would take the entry as
    evidence there were hits.
    """
    frame = _results()
    frame["adjusted_p_value"] = 0.9
    style = VolcanoStyle()

    figure, panels = render_volcano(frame, style)
    try:
        labels = [t.get_label() for t in panels[0].collections]
        assert "significant" not in labels
        assert "not significant" in labels
    finally:
        matplotlib.pyplot.close(figure)


def test_every_axis_option_reaches_the_panel():
    """Scale, limits, inversion, grid, background and title all take effect.

    Each of these is a menu entry on the figure. Reading them back off the
    drawn axes is the only way to know the menu is wired to the renderer and
    not just to the saved style file.
    """
    style = VolcanoStyle(
        x_scale="symlog", x_lim=(-3.0, 3.0), y_lim=(0.0, 8.0),
        invert_y=True, grid=False, background_color="#F0F0F0",
        title="tsg101 screen", effect_threshold=1.0,
        threshold_multiplier=1.0, show_effect_lines=True,
    )

    figure, panels = render_volcano(_results(), style)
    try:
        axis = panels[0]
        assert axis.get_xscale() == "symlog"
        assert axis.get_xlim() == (-3.0, 3.0)
        # invert_y flips the limits it was given.
        assert axis.get_ylim() == (8.0, 0.0)
        assert not axis.xaxis._major_tick_kw.get("gridOn", False)
        assert axis.get_facecolor() == matplotlib.colors.to_rgba("#F0F0F0")
        assert axis.get_title() == "tsg101 screen"

        # The effect-size cut is drawn on BOTH sides: an effect of -1.5 is as
        # big as +1.5 and the reader has to see the symmetry.
        verticals = sorted(round(line.get_xdata()[0], 6)
                           for line in axis.get_lines()
                           if len(set(line.get_xdata())) == 1)
        assert -1.0 in verticals and 1.0 in verticals
    finally:
        matplotlib.pyplot.close(figure)


def test_labels_are_skipped_when_the_label_column_is_not_there():
    """A label column that does not exist annotates nothing, quietly.

    The label column defaults to ``guide``; a gene-level table has no such
    column, and asking for labels on one must not take the whole figure down.
    """
    frame = _results().drop(columns=["guide"])
    style = VolcanoStyle(annotate_significant=True)

    figure, panels = render_volcano(frame, style)
    try:
        assert len(panels[0].texts) == 0
    finally:
        matplotlib.pyplot.close(figure)


def test_asking_for_the_hits_to_be_named_names_them():
    """``annotate_significant`` prints every significant point's label.

    A volcano nobody can read the hits off is a picture of a result rather
    than the result, and this is the switch that turns the names on.
    """
    frame = _results()
    style = VolcanoStyle(annotate_significant=True,
                         effect_threshold=1.0,
                         threshold_multiplier=1.0)

    figure, panels = render_volcano(frame, style)
    try:
        printed = {t.get_text() for t in panels[0].texts}
        assert {"g000", "g001", "g002", "g003"} <= printed
        assert "g010" not in printed
    finally:
        matplotlib.pyplot.close(figure)


def test_a_colour_category_that_matches_no_row_is_skipped(monkeypatch):
    """A colour category with no points draws nothing and no legend entry.

    Colour-by categories come from the column's own values, but a column
    carrying a null contributes a category that equals none of its own
    rows. Drawing it would add an empty series to the legend, which reads as
    a group that was measured and found to have no members.
    """
    from spacr import volcano_style as vs

    values = np.array(["p1", "p2", np.nan, "p1"] * 10, dtype=object)
    monkeypatch.setattr(vs, "_colour_values", lambda frame, style: (values, True))

    style = VolcanoStyle(color_by="plate")
    figure, panels = render_volcano(_results(), style)
    try:
        labels = [t.get_label() for t in panels[0].collections]
        assert "p1" in labels and "p2" in labels
        assert "nan" not in labels
    finally:
        matplotlib.pyplot.close(figure)
