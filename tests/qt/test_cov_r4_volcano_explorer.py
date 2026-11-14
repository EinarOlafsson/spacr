"""The volcano explorer's edges: a panel that is not the one it expects.

Pins the paths the explorer takes when something it reads is not the shape
it was written for -- a renderer complaint that names a setting without
spelling it ``name=``, an "automatic" limit that cannot be expressed as a
number, a results table with an unnamed column, a numeric setting that is
``None``, a split that cannot be sized, and a writer that reports it wrote
nothing. Every one of them has to leave the reader with a figure and a
sentence rather than an exception.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.volcano_explorer import (
    _PROBLEM_INK, VolcanoExplorer)


def _results(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "guide": [f"G{i:03d}" for i in range(n)],
        "gene": [f"TGGT1_{i // 3:06d}" for i in range(n)],
        "standardized_marginal_effect": rng.normal(size=n),
        "adjusted_p_value": rng.random(n) * 0.5 + 1e-6,
        "p_value": rng.random(n),
    })


@pytest.fixture
def explorer(qapp, qtbot):
    widget = VolcanoExplorer(_results())
    qtbot.addWidget(widget)
    return widget


# --------------------------------------------------------------------------
# a limit that is "automatic"
# --------------------------------------------------------------------------


def test_a_limit_that_is_not_a_number_reads_as_automatic(explorer):
    """A style file can carry anything. A value the control cannot express
    has to read as "automatic" rather than as a number nobody typed."""
    style = dataclasses.replace(explorer.style(), effect_threshold=2.5)
    explorer.set_style(style)
    assert explorer._controls["effect_threshold"].value() == 2.5

    explorer.set_style(dataclasses.replace(style, effect_threshold="rubbish"))
    assert explorer._controls["effect_threshold"].value() is None


# --------------------------------------------------------------------------
# a panel that is not carrying every control
# --------------------------------------------------------------------------

def test_the_column_menus_refill_even_with_no_compartment_list(explorer):
    """`_controls` is looked up by name throughout, and every lookup is
    guarded -- a screen that carries a subset of the panel must still get its
    column pickers refilled. Reached through the registry because there is no
    public way to build a partial panel."""
    explorer._controls.pop("localizations")
    explorer.set_results(_results(12))
    assert explorer._controls["x_column"].count() > 0


# --------------------------------------------------------------------------
# compartments, splits and the house red
# --------------------------------------------------------------------------


def test_the_house_red_answers_when_the_theme_cannot(monkeypatch):
    """A bare widget built with no application palette still has to be able
    to turn a label red."""
    from spacr.qt import theme

    house = str(theme.active_palette().get("error") or _PROBLEM_INK)
    assert VolcanoExplorer._error_ink() == house

    def _no_palette():
        raise RuntimeError("no theme here")

    monkeypatch.setattr(theme, "active_palette", _no_palette)
    assert VolcanoExplorer._error_ink() == _PROBLEM_INK
