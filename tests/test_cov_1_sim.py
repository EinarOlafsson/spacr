"""The simulator's colour-bar styling leaves a bar-less figure alone.

``_style_colour_bar`` restyles ``fig.axes[-1]``, which is the colour bar only
because seaborn appends it after the data axes. A figure drawn without a
colour bar has the DATA axes last, so without the length guard the helper
would strip the plot's own spines and recolour its ticks -- the visible
damage this test pins down.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spacr.sim as sim


def test_a_figure_without_a_colour_bar_keeps_its_own_axis():
    """One-axes figure: the data axes must survive untouched."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.tick_params(colors="#123456")
    try:
        assert sim._style_colour_bar(fig) is None

        assert len(fig.axes) == 1, "guard is only meaningful with a lone axes"
        assert all(spine.get_visible() for spine in ax.spines.values()), (
            "the data axes lost its spines to the colour-bar styling")
        colours = {t.get_color() for t in ax.get_xticklabels()}
        assert colours == {"#123456"}, (
            f"the data axes tick colour was overwritten: {colours}")
    finally:
        plt.close(fig)
