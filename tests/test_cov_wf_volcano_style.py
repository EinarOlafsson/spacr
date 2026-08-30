"""The volcano's ink, its rules and its labels, on the paths nothing drove.

Every setting in :class:`spacr.volcano_style.VolcanoStyle` is a control the
figure menu writes into, so a branch the renderer never takes is a switch
that silently does nothing to the picture. The cases here are the ones the
suite had never exercised: a compartment ticked on a table that cannot name a
gene, the reference lines switched OFF, a label whose point sits in the gap of
a broken axis, a screen render with no ground named, and -- the whole reason
``axis_color`` exists -- an EMPTY ink that has to leave the ambient matplotlib
settings alone instead of painting everything black.

Each test drives the positive case alongside the negative one, because "no
line was drawn" and "nothing was drawn at all" look identical from a single
render.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

from spacr.localisation import table
from spacr.volcano_style import VolcanoStyle, page_ground, point_localizations, render_volcano


def _results(n=24, seed=3):
    """A small coefficient table with three real hits and a numeric covariate."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(0.0, 0.5, n)
    p = rng.uniform(0.2, 1.0, n)
    effect[:3] = [2.2, -2.0, 1.8]
    p[:3] = [1e-7, 1e-6, 1e-5]
    return pd.DataFrame({
        "standardized_marginal_effect": effect,
        "adjusted_p_value": p,
        "guide": [f"g{i:03d}" for i in range(n)],
        "score": np.linspace(0.0, 1.0, n),
    })


def _series_labels(axis):
    """The legend label of every scatter series drawn on ``axis``."""
    return [collection.get_label() for collection in axis.collections]


def test_a_ticked_compartment_on_a_nameless_table_still_draws_the_volcano():
    """A compartment ticked on a table with no gene column must not blank the plot.

    Ticking a compartment hands the colour channel to the localisation
    lookup. When the frame the user opened carries no column the reference
    table has ever heard of -- a term/coefficient/p-value export, say -- the
    lookup can name nothing, and the renderer has to fall back to the ordinary
    significant / not-significant colouring. If it instead drew the
    compartment path with an empty answer the user would get an all-grey
    figure, or an exception, from ticking a menu entry.
    """
    nameless = _results().drop(columns=["guide"])
    style = VolcanoStyle(localizations=("micronemes",))

    assert point_localizations(nameless, style) is None
    figure, panels = render_volcano(nameless, style)
    try:
        fallback = _series_labels(panels[0])
    finally:
        plt.close(figure)

    # The two-tone scheme, not the compartment scheme.
    assert "significant" in fallback
    assert "not significant" in fallback
    assert "elsewhere" not in fallback

    # The control: the same ticked compartment on a table that CAN name a
    # gene really does colour by compartment, so the fallback above is the
    # missing column and not a broken lookup.
    genes = [gene for gene, place in table().items() if place == "micronemes"][:6]
    named = pd.DataFrame({
        "gene": genes,
        "standardized_marginal_effect": np.linspace(-1.0, 1.0, len(genes)),
        "adjusted_p_value": [0.5] * len(genes),
    })
    figure, panels = render_volcano(named, style)
    try:
        assert "micronemes" in _series_labels(panels[0])
    finally:
        plt.close(figure)


def test_the_alpha_and_zero_reference_lines_can_each_be_switched_off():
    """Turning a reference line off has to remove it from the picture.

    The alpha horizontal and the zero vertical are what a reader measures a
    point against, and both are checkboxes in the figure menu. A checkbox that
    clears while the rule stays on the axes is worse than no checkbox: the
    exported figure then disagrees with the settings saved beside it.
    """
    frame = _results()

    drawn, panels = render_volcano(frame, VolcanoStyle())
    try:
        rules = list(panels[0].lines)
        horizontals = [line.get_ydata()[0] for line in rules
                       if line.get_ydata()[0] == line.get_ydata()[1]]
    finally:
        plt.close(drawn)

    assert len(rules) == 2, "alpha rule and zero rule, and nothing else"
    assert horizontals == [pytest.approx(-np.log10(0.05), abs=1e-9)]

    bare, panels = render_volcano(
        frame, VolcanoStyle(show_alpha_line=False, show_zero_line=False))
    try:
        assert list(panels[0].lines) == []
        # and the points are still there, so "no lines" is not "no figure"
        assert len(panels[0].collections) >= 1
    finally:
        plt.close(bare)


def test_a_label_lands_on_the_panel_whose_range_can_show_its_point():
    """On a broken axis, an annotation must follow its point to the right panel.

    A split volcano is two stacked panels with a gap between them. Printing
    every label on the top panel would place a guide's name far from the dot
    it names -- and a point that falls INTO the gap has no panel at all, so
    the renderer has to pick one rather than dropping the label or raising.
    """
    frame = pd.DataFrame({
        "standardized_marginal_effect": [-1.0, 0.0, 1.0],
        "y": [8.0, 3.5, 0.5],
        "guide": ["hit", "gap", "low"],
    })
    style = VolcanoStyle(
        y_column="y", y_neg_log10=False, split_axis=True,
        split_y_lims=((0.0, 2.0), (5.0, 10.0)),
        annotations={"hit": "HIT", "gap": "GAP", "low": "LOW"})

    figure, panels = render_volcano(frame, style)
    try:
        assert len(panels) == 2
        upper = {text.get_text() for text in panels[0].texts}
        lower = {text.get_text() for text in panels[1].texts}
    finally:
        plt.close(figure)

    # y=8 is inside the upper panel, y=0.5 inside the lower one.
    assert "HIT" in upper and "HIT" not in lower
    assert "LOW" in lower and "LOW" not in upper
    # y=3.5 is in the break, in neither range: it falls back to the first
    # panel rather than vanishing.
    assert "GAP" in upper
    assert upper | lower == {"HIT", "GAP", "LOW"}


def test_a_screen_render_with_no_ground_named_keeps_the_transparent_page():
    """An unset screen background must not invent a white rectangle.

    ``background_color`` ships as "none" so a saved figure drops onto any page
    without carrying a block of colour with it; ``screen_background`` is the
    separate, readable ground the live explorer paints. When BOTH are cleared
    the answer has to be "leave the figure as it is" -- painting a default
    would put that rectangle back into every exported PDF.
    """
    blank = VolcanoStyle(background_color="none", screen_background="none")
    empty = VolcanoStyle(background_color="none", screen_background="  ")
    dark = VolcanoStyle(background_color="none", screen_background="#101010")

    assert page_ground(blank, screen=True) is None
    assert page_ground(empty, screen=True) is None
    assert page_ground(dark, screen=True) == "#101010"
    assert page_ground(dark, screen=False) is None

    frame = _results()
    painted, _ = render_volcano(frame, dark, screen=True)
    try:
        assert painted.patch.get_facecolor() == to_rgba("#101010")
    finally:
        plt.close(painted)

    plain, _ = render_volcano(frame, blank, screen=True)
    try:
        assert plain.patch.get_facecolor() != to_rgba("#101010")
    finally:
        plt.close(plain)


def test_an_empty_axis_colour_leaves_the_ambient_ink_alone():
    """Clearing ``axis_color`` has to mean "do not touch", not "paint black".

    The setting exists so a figure's ink is reproducible rather than inherited
    from whatever the process drew before it -- which means the empty value has
    to be a real option: it restores the pre-``axis_color`` behaviour, leaving
    the spines, the ticks, the labels and the colour bar at the ambient
    matplotlib settings. If the empty string were treated as a colour, every
    figure drawn under a dark theme would come back with black-on-black axes.
    """
    frame = _results()
    inked = VolcanoStyle(color_by="score", show_colorbar=True,
                         axis_color="#FF0000")
    ambient = VolcanoStyle(color_by="score", show_colorbar=True,
                           axis_color="")

    figure, panels = render_volcano(frame, inked)
    try:
        assert len(figure.axes) == 2, "the colour bar is an axis of its own"
        red_spine = panels[0].spines["left"].get_edgecolor()
        red_bar_label = figure.axes[-1].yaxis.label.get_color()
    finally:
        plt.close(figure)

    assert red_spine == to_rgba("#FF0000")
    assert to_rgba(red_bar_label) == to_rgba("#FF0000")

    figure, panels = render_volcano(frame, ambient)
    try:
        assert len(figure.axes) == 2, "the colour bar is drawn either way"
        plain_spine = panels[0].spines["left"].get_edgecolor()
        plain_bar_label = figure.axes[-1].yaxis.label.get_color()
        # The mapping itself survives: an untouched ink is not an undrawn plot.
        assert panels[0].collections and panels[0].collections[0].get_array() is not None
    finally:
        plt.close(figure)

    assert plain_spine != to_rgba("#FF0000")
    assert to_rgba(plain_bar_label) != to_rgba("#FF0000")


def test_a_named_y_label_wins_over_the_derived_one():
    """A y label the user typed must survive the -log10 auto-label.

    With ``y_neg_log10`` on, an unset label is filled in as
    ``-log10(<column>)`` so the axis is never mislabelled by default. But the
    column name is a database identifier, and a user who typed "Significance"
    into the figure menu has to see it -- a menu field the renderer overwrites
    is a field that does nothing.
    """
    frame = _results()

    derived, panels = render_volcano(frame, VolcanoStyle())
    try:
        auto = panels[-1].get_ylabel()
        x_axis = panels[-1].get_xlabel()
    finally:
        plt.close(derived)

    assert auto == "-log10(adjusted_p_value)"
    assert x_axis == "Standardized marginal effect"

    chosen, panels = render_volcano(
        frame, VolcanoStyle(y_label="Significance (-log10 q)"))
    try:
        assert panels[-1].get_ylabel() == "Significance (-log10 q)"
    finally:
        plt.close(chosen)
