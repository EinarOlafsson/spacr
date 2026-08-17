"""Beside every label on a graph is how many things it names.

Asked for on 2026-08-17: "beside the label on the graph should be the count
of each label".

IT IS NOT DECORATION. On the real screen `pc` is 3 points and `nc` is 24,
among 1,213. A label that names them without saying so lets a three-point
group be read as a group -- which is the same failure the mark advice exists
to prevent, one step earlier.

The interactive plot and the saved figure are both covered here, because the
two must not disagree about what they show.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

pytestmark = pytest.mark.qt

N = 1213


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(N)],
        "coefficient": rng.normal(0, .5, N),
        "p_value": rng.uniform(size=N),
        "condition": list(rng.choice(["nc", "pc", "other"], N,
                                     p=[.02, .0025, .9775])),
    })


def _panel(qtbot, frame):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(frame)
    return panel


def _tick_labels(plot):
    return [text for _pos, text in plot.plot.getAxis("bottom")._tickLevels[0]]


# --------------------------------------------------------------------------- #
#  The colour-by legend
# --------------------------------------------------------------------------- #

def test_every_legend_entry_carries_its_count(qtbot):
    frame = _frame()
    panel = _panel(qtbot, frame)

    counts = frame["condition"].value_counts().to_dict()
    legend = set(panel.volcano._legend_colours)
    for name, count in counts.items():
        assert f"{name} ({count})" in legend, (name, count, legend)


def test_the_counts_are_the_real_ones(qtbot):
    """A count computed a second way is a count that can drift from the one
    the plot drew."""
    frame = _frame()
    panel = _panel(qtbot, frame)

    total = 0
    for entry in panel.volcano._legend_colours:
        total += int(entry.rsplit("(", 1)[1].rstrip(")"))
    assert total == len(frame)


# --------------------------------------------------------------------------- #
#  The grouped plot's axis
# --------------------------------------------------------------------------- #

def test_the_group_labels_carry_their_n(qtbot):
    """It was in the note BELOW the panel and the axis said only "pc", so a
    reader had to carry three numbers from one line to another."""
    panel = _panel(qtbot, _frame())

    labels = _tick_labels(panel.controls)
    assert labels, "the control panel drew no groups"
    for label in labels:
        assert "n=" in label, labels


def test_the_axis_agrees_with_the_note_beneath_it(qtbot):
    """Both are taken from the same `group_sizes`, so they cannot disagree --
    and a test that only checked the axis would not notice if they did."""
    panel = _panel(qtbot, _frame())

    sizes = panel.controls.group_sizes()
    labels = _tick_labels(panel.controls)
    for label, size in zip(labels, sizes):
        assert f"n={size}" in label, (label, size)


def test_a_three_point_group_says_three(qtbot):
    """The specific case that motivated it."""
    frame = _frame()
    panel = _panel(qtbot, frame)

    smallest = min(panel.controls.group_sizes())
    assert smallest < 10, "fixture no longer has a tiny group"
    assert any(f"n={smallest}" in label
               for label in _tick_labels(panel.controls))


# --------------------------------------------------------------------------- #
#  The saved figure says the same thing
# --------------------------------------------------------------------------- #

def test_the_saved_panel_carries_the_counts_too():
    import matplotlib.pyplot as plt

    from spacr.figures import build_panel

    figure, _panel = build_panel("controls", _frame())
    try:
        labels = [t.get_text() for t in figure.axes[0].get_xticklabels()]
        assert labels
        for label in labels:
            assert "n=" in label, labels
    finally:
        plt.close(figure)


def test_the_saved_panel_keeps_the_median_in_its_annotation():
    """The axis cannot carry the median, and the median is the number the
    panel is actually comparing -- so moving n to the axis must not lose
    it."""
    import matplotlib.pyplot as plt

    from spacr.figures import build_panel

    figure, _panel = build_panel("controls", _frame())
    try:
        texts = " ".join(t.get_text() for t in figure.axes[0].texts)
        assert "median" in texts, texts
    finally:
        plt.close(figure)
