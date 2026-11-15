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

from spacr.volcano_style import VolcanoStyle, _resolve_effect_threshold


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


def test_style_to_dict_returns_detached_plain_data():
    """Editing serialized annotations must not mutate the active style."""
    style = VolcanoStyle(annotations={"g001": "hit"})

    values = style.to_dict()
    values["annotations"]["g001"] = "changed"

    assert values["x_column"] == style.x_column
    assert style.annotations == {"g001": "hit"}


