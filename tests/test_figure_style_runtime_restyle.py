"""Changed figure preferences reach the artists of an already drawn plot."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from spacr.figure_style import palette_colours
from spacr.figures import style


def test_restyle_updates_existing_text_lines_marks_and_axes(monkeypatch):
    """An older module's completed figure receives every changed preference."""
    changed = {
        "font_family": "DejaVu Sans", "title_size": 19, "label_size": 15,
        "tick_size": 13, "legend_size": 11, "background": "#eeeeee",
        "grid": True, "spines": "none", "spine_width": 2.5,
        "despine_offset": 7, "line_width": 4, "marker_size": 49,
        "colormap": "cividis", "palette": "bright",
    }
    monkeypatch.setattr(style, "_preference_deltas", lambda kind: changed)
    figure = Figure()
    ax = figure.subplots()
    figure.suptitle("Overview")
    ax.set_title("A panel")
    ax.set_xlabel("Time")
    ax.set_ylabel("Intensity")
    line, = ax.plot([0, 1], [1, 2], marker="o", label="cells",
                    color="#1f77b4")
    marks = ax.scatter([0, 1], [2, 3], s=[12, 12], c=[0, 1],
                       cmap="viridis")
    image = ax.imshow(np.arange(4).reshape(2, 2), cmap="viridis")
    legend = ax.legend()

    assert style._apply_user_style(figure) == changed
    assert figure._suptitle.get_fontsize() == 19
    assert ax.title.get_fontsize() == 19
    assert ax.xaxis.label.get_fontsize() == 15
    assert {t.get_fontsize() for t in ax.get_xticklabels()} == {13}
    assert legend.get_texts()[0].get_fontsize() == 11
    assert to_hex(figure.patch.get_facecolor()) == "#eeeeee"
    assert to_hex(ax.patch.get_facecolor()) == "#eeeeee"
    assert not any(spine.get_visible() for spine in ax.spines.values())
    assert all(spine.get_linewidth() == 2.5 for spine in ax.spines.values())
    assert all(spine.get_position() == ("outward", 7) for spine in ax.spines.values())
    assert line.get_linewidth() == 4
    assert line.get_markersize() == pytest.approx(7)
    assert to_hex(line.get_color()) == palette_colours("bright")[0]
    assert marks.get_sizes().tolist() == [49]
    assert image.get_cmap().name == "cividis"
    assert marks.get_cmap().name == "cividis"
    assert any(grid.get_visible() for grid in ax.xaxis.get_gridlines())


def test_a_broken_preference_record_leaves_an_existing_figure_drawable(
        monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_figure_style", lambda: object())
    monkeypatch.setattr(preferences, "get_figure_style_per_graph", lambda: {})
    figure = Figure()
    line, = figure.subplots().plot([0, 1], [1, 2])
    width = line.get_linewidth()

    assert style._apply_user_style(figure) == {}
    assert line.get_linewidth() == width
    figure.canvas.draw()


def test_an_invalid_background_preference_cannot_break_the_figure(monkeypatch):
    monkeypatch.setattr(style, "_preference_deltas",
                        lambda kind: {"background": "not-a-colour"})
    figure = Figure()
    figure.subplots().plot([0, 1], [1, 2])
    original = figure.patch.get_facecolor()

    assert style._apply_user_style(figure) == {}
    assert figure.patch.get_facecolor() == original
    figure.canvas.draw()


def test_one_malformed_artist_colour_does_not_block_the_others(monkeypatch):
    figure = Figure()
    ax = figure.subplots()
    good, = ax.plot([0, 1], [1, 2], color="#1f77b4")
    bad, = ax.plot([0, 1], [2, 3], color="#1f77b4")
    marks = ax.scatter([0, 1], [2, 3], c="blue")
    monkeypatch.setattr(bad, "get_color", lambda: object())
    monkeypatch.setattr(marks, "get_facecolor", lambda: "invalid RGBA")
    monkeypatch.setattr(style, "_preference_deltas",
                        lambda kind: {"palette": "bright"})

    assert style._apply_user_style(figure) == {"palette": "bright"}
    assert to_hex(good.get_color()) == palette_colours("bright")[0]
