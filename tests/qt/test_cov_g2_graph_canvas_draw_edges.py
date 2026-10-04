"""Graph canvas drawing helpers with nothing to draw or unusual artists."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from matplotlib.figure import Figure  # noqa: E402

from spacr.qt.widgets import graph_builder as gb  # noqa: E402

PALETTE = {"fg": "#000000", "fg_dim": "#333333", "fg_muted": "#666666"}


def _canvas(spec, scales, figure=None):
    return SimpleNamespace(_spec=spec, _scales=scales, _figure=figure,
                           _series_colour=lambda index: "#123456")


def test_jitter_without_levels_draws_nothing():
    ax = Figure().add_subplot()
    canvas = _canvas(SimpleNamespace(x="c", y="v", seed=0),
                     SimpleNamespace(x_levels=(), y_levels=()))
    gb.GraphCanvas._draw_jitter(canvas, ax, pd.DataFrame({"c": [1.0], "v": ["a"]}),
                                PALETTE, over_bars=False)
    assert not ax.collections


def test_a_distribution_without_values_draws_nothing():
    ax = Figure().add_subplot()
    canvas = _canvas(SimpleNamespace(x="g", y="v"),
                     SimpleNamespace(x_levels=("a",), y_levels=()))
    rows = pd.DataFrame({"g": ["a"], "v": [np.nan]})
    gb.GraphCanvas._draw_distribution(canvas, ax, rows, gb.VIOLIN, PALETTE)
    assert not ax.collections


def test_a_violin_without_median_lines_still_colours_its_bodies(monkeypatch):
    ax = Figure().add_subplot()
    body = SimpleNamespace(colour=None, set_facecolor=lambda c: setattr(body, "colour", c),
                           set_alpha=lambda a: None)
    monkeypatch.setattr(ax, "violinplot", lambda *a, **k: {"bodies": [body]})
    canvas = _canvas(SimpleNamespace(x="g", y="v"),
                     SimpleNamespace(x_levels=("a",), y_levels=()))
    rows = pd.DataFrame({"g": ["a", "a"], "v": [1.0, 2.0]})
    gb.GraphCanvas._draw_distribution(canvas, ax, rows, gb.VIOLIN, PALETTE)
    assert body.colour == "#123456"


def test_a_mean_bar_without_a_column_has_no_axis_label():
    ax = Figure().add_subplot()
    canvas = _canvas(SimpleNamespace(spread=""), None)
    text = pd.Series(["a", "a"])
    gb.GraphCanvas._draw_mean_bar(canvas, ax, text, pd.Series([1.0, 3.0]),
                                  ["a"], PALETTE, "")
    assert ax.get_ylabel() == "" and len(ax.patches) == 1


def test_a_legend_without_a_title_still_colours_its_labels():
    labels = []
    legend = SimpleNamespace(get_texts=lambda: [SimpleNamespace(
        set_color=labels.append)], get_title=lambda: None)
    figure = SimpleNamespace(legend=lambda **k: legend)
    canvas = _canvas(SimpleNamespace(colour="well"),
                     SimpleNamespace(colour_levels=["a", "b"]), figure)
    gb.GraphCanvas._draw_legend(canvas, "scatter", PALETTE)
    assert labels == ["#333333"]
