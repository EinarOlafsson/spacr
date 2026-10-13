"""Figure error bars follow the selected summary and confidence level."""

from __future__ import annotations

import math

import pandas as pd
from scipy.stats import t as student_t


def test_grouped_error_bar_choices_have_distinct_spans(monkeypatch):
    from spacr.plot import spacrGraph

    frame = pd.DataFrame({"group": ["a"] * 3 + ["b"] * 3,
                          "value": [1, 2, 3, 2, 3, 4]})
    graph = spacrGraph(frame, "group", "value", save=False)
    row = {"std": 4.0, "sem": 2.0, "count": 9}
    chosen = {}
    monkeypatch.setattr(graph, "_user_style", lambda: chosen)

    assert graph._error_half_width(row) == 4.0
    chosen["error_bars"] = "none"
    assert graph._error_half_width(row) is None
    chosen["error_bars"] = "sd"
    assert graph._error_half_width(row) == 4.0
    chosen["error_bars"] = "sem"
    assert graph._error_half_width(row) == 2.0

    chosen["error_bars"] = "ci95"
    assert math.isclose(graph._error_half_width(row),
                        2.0 * student_t.ppf(0.975, 8))
    chosen.update(error_bars="ci", ci_level=80)
    assert math.isclose(graph._error_half_width(row),
                        2.0 * student_t.ppf(0.9, 8))
    assert graph._error_half_width({**row, "count": 1}) is None


def test_an_invalid_preference_dpi_keeps_the_valid_legacy_value(monkeypatch):
    from spacr import plot
    from spacr.figures import style
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_figure_format", lambda: "png")
    monkeypatch.setattr(preferences, "get_figure_png_dpi", lambda: 240)
    monkeypatch.setattr(style, "_preference_deltas",
                        lambda *_args, **_kwargs: {"dpi": "not-a-number"})

    assert plot.figure_output_preferences() == ("png", 240)


def test_point_overlay_can_be_disabled_without_losing_group_means(monkeypatch):
    from matplotlib.figure import Figure

    from spacr.plot import spacrGraph

    frame = pd.DataFrame({"group": ["a"] * 3 + ["b"] * 3,
                          "value": [1, 2, 3, 2, 3, 4]})
    graph = spacrGraph(frame, "group", "value", graph_type="jitter_bar",
                       save=False)
    chosen = {"point_overlay": False, "marker_size": 36, "point_alpha": 0.4}
    monkeypatch.setattr(graph, "_user_style", lambda: chosen)
    axes = Figure().add_subplot()
    graph.create_plot(axes)

    assert graph._point_look(16) == (0.4, 6.0)
    assert len(axes.collections) == 0
    assert [bar.get_height() for bar in axes.patches if bar.get_width() > 0] \
        == [2.0, 3.0]
    assert graph.summary_df["mean"].tolist() == [2.0, 3.0]


def test_unavailable_palette_preserves_the_existing_colour_fallback(monkeypatch):
    from spacr import figure_style
    from spacr.plot import spacrGraph

    frame = pd.DataFrame({"group": ["a", "b"], "value": [1, 2]})
    graph = spacrGraph(frame, "group", "value", save=False)
    chosen = {}
    monkeypatch.setattr(graph, "_user_style", lambda: chosen)
    before = graph._plot_palette(2)
    chosen["palette"] = "unavailable"
    monkeypatch.setattr(figure_style, "palette_colours", lambda _name: [])
    assert graph._plot_palette(2) == before
