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
