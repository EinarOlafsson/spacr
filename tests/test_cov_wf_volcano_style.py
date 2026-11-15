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
from matplotlib.colors import to_rgba

from spacr.volcano_style import VolcanoStyle, page_ground, render_volcano


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


