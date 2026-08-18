"""The matplotlib figures get instruction 152's TWO colour controls.

The pyqtgraph half shipped first: one "Line colour" reaching every line
including the axis spines and the tick MARKS, one "Font colour" reaching
every piece of text including the tick LABELS. The matplotlib figures were
still on one ink -- ``_style_figure_colors`` called
``ax.tick_params(colors=fg)``, which sets the mark and the label together, so
the two could not be told apart at all, and the Figure settings dialog's one
"Text colour" row drove the spines as well.

    "dosnt look like there is an option to change the exis color for the
     colcano plot"

That report is about a figure. This file asserts the same division on the
matplotlib side, at every place a colour reaches a figure:

1. the shared recolour pass (``_style_figure_colors``), which every render
   goes through;
2. the preference that feeds it, stored as a TOKEN like the other two;
3. the per-figure controls -- ``apply_line_colour`` / ``apply_font_colour``
   and the right-click menu that offers them.

TICK MARKS ARE LINES AND TICK LABELS ARE TEXT. That is the one place the two
controls meet and the one a reader could take either way, so it has a test of
its own rather than a comment.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from spacr.qt.widgets.figure_queue import _style_figure_colors


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def figure():
    """A figure with the artists the two controls divide between them."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4], label="data")
    ax.axhline(1.0, linestyle="--", color="#ff0000")
    ax.set_title("a title")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.text(1.0, 1.0, "GRA14")
    ax.legend(title="condition")
    fig.suptitle("a run")
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def _hex(colour) -> str:
    from matplotlib.colors import to_hex
    return to_hex(colour)


def _tick_mark_colour(axis) -> str:
    return _hex(axis.xaxis.get_major_ticks()[0].tick1line.get_color())


def _tick_label_colour(axis) -> str:
    return _hex(axis.xaxis.get_major_ticks()[0].label1.get_color())


def _spine_colour(axis) -> str:
    return _hex(list(axis.spines.values())[0].get_edgecolor())


# ---------------------------------------------------------------------------
# 1. the shared recolour pass
# ---------------------------------------------------------------------------

def test_the_tick_mark_is_a_line_and_the_tick_label_is_text(figure):
    """The one place the two controls meet, asserted rather than argued.

    THE REGRESSION. ``tick_params(colors=...)`` sets both, so before the
    split these two came back equal whatever was asked for.
    """
    _style_figure_colors(figure, "none", "#00ff00", 0, "#0000ff")
    axis = figure.axes[0]
    assert _tick_mark_colour(axis) == "#0000ff"
    assert _tick_label_colour(axis) == "#00ff00"


def test_the_spines_follow_the_line_colour(figure):
    _style_figure_colors(figure, "none", "#00ff00", 0, "#0000ff")
    assert _spine_colour(figure.axes[0]) == "#0000ff"


def test_the_title_and_the_labels_follow_the_font_colour(figure):
    _style_figure_colors(figure, "none", "#00ff00", 0, "#0000ff")
    axis = figure.axes[0]
    assert _hex(axis.title.get_color()) == "#00ff00"
    assert _hex(axis.xaxis.label.get_color()) == "#00ff00"
    assert _hex(axis.yaxis.label.get_color()) == "#00ff00"


def test_the_legend_title_is_text_too(figure):
    """A legend's title is not among its ``get_texts()`` -- the miss that
    issue #108 was three quarters of."""
    _style_figure_colors(figure, "none", "#00ff00", 0, "#0000ff")
    legend = figure.axes[0].get_legend()
    assert _hex(legend.get_title().get_color()) == "#00ff00"


def test_no_line_colour_means_the_font_colour(figure):
    """A store that has never been touched renders exactly as it did before
    there were two controls."""
    _style_figure_colors(figure, "none", "#00ff00", 0)
    axis = figure.axes[0]
    assert _spine_colour(axis) == "#00ff00"
    assert _tick_mark_colour(axis) == "#00ff00"


def test_the_data_lines_are_not_repainted_by_the_theme(figure):
    """The pass runs on EVERY render. A theme that repainted every series in
    one ink would flatten every multi-series figure in the package."""
    _style_figure_colors(figure, "none", "#00ff00", 0, "#0000ff")
    assert _hex(figure.axes[0].lines[1].get_color()) == "#ff0000"


def test_the_grid_is_not_repainted(figure):
    """A grid in the ink is a cage over the data -- ``figure_style``'s own
    rule for the save path, agreed with rather than re-argued."""
    figure.axes[0].grid(True, color="#dddddd")
    _style_figure_colors(figure, "none", "#00ff00", 0, "#0000ff")
    gridline = figure.axes[0].xaxis.get_gridlines()[0]
    assert _hex(gridline.get_color()) == "#dddddd"


def test_the_queue_helper_and_the_module_helper_are_one_implementation(figure):
    """``FigureQueue._style_figure`` was a character-for-character copy and
    the two had begun to drift."""
    from spacr.qt.widgets.figure_queue import FigureQueue

    FigureQueue._style_figure(figure, "none", "#00ff00", 0, "#0000ff")
    axis = figure.axes[0]
    assert _tick_mark_colour(axis) == "#0000ff"
    assert _tick_label_colour(axis) == "#00ff00"


# ---------------------------------------------------------------------------
# 2. the preference behind it
# ---------------------------------------------------------------------------

@pytest.fixture
def prefs(monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences as preferences_module

    store = tmp_path / "prefs.ini"
    monkeypatch.setattr(
        preferences_module, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))
    return preferences_module


def test_the_line_colour_is_stored_as_a_token(prefs, monkeypatch):
    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "dark")
    assert prefs.get_figure_line_token() == "auto"
    # Automatic means "the same ink as the text", so the split costs nobody
    # a changed figure until they choose one.
    assert prefs.get_figure_line_colour() == prefs.get_figure_colors()[1]


def test_a_chosen_line_colour_is_honoured_and_the_font_is_not(prefs,
                                                              monkeypatch):
    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "light")
    prefs.set_figure_line_colour("#0000ff")
    assert prefs.get_figure_line_colour() == "#0000ff"
    assert prefs.get_figure_colors()[1] == "#000000"


def test_follow_the_theme_un_sets_the_line_colour_too(prefs, monkeypatch):
    """A way back that left one of the three frozen would be the trap it
    exists to be the way out of."""
    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "dark")
    prefs.set_figure_line_colour("#0000ff")
    prefs.set_figure_colors("#123456", "#654321")
    prefs.set_figure_colors_auto()
    assert prefs.get_figure_line_token() == "auto"
    assert prefs.get_figure_color_tokens() == ("auto", "auto")


def test_a_frozen_line_colour_is_migrated_back_to_auto(prefs, monkeypatch,
                                                       caplog):
    """Section A's migration covers the new key on the same pass: it did not
    exist before the migration shipped, so a store already marked cannot hold
    a frozen one."""
    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "light")
    store = prefs._settings()
    store.setValue("prefs/figure_line", "#ffffff")
    store.sync()
    with caplog.at_level("INFO"):
        assert prefs.get_figure_line_token() == "auto"
    assert "line colour" in caplog.text


def test_the_renderer_reads_the_line_preference(prefs, monkeypatch, figure,
                                                tmp_path):
    """Not "the getter returns blue" -- the render pass actually asks for it.

    This is the wiring that a green model-layer test would miss.
    """
    from spacr.qt.widgets import figure_queue as queue_module

    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "light")
    prefs.set_figure_line_colour("#0000ff")
    monkeypatch.setattr(queue_module, "_export_vector_pdf",
                        lambda *a, **k: True)
    assert queue_module.render_figure_to_png(figure,
                                             str(tmp_path / "f.png"))
    axis = figure.axes[0]
    assert _spine_colour(axis) == "#0000ff"
    assert _tick_label_colour(axis) == "#000000"


# ---------------------------------------------------------------------------
# 3. the per-figure controls
# ---------------------------------------------------------------------------

def test_apply_line_colour_reaches_the_data_lines_and_the_axes(figure):
    """The per-figure control is a user asking about ONE figure, so it does
    reach the data's lines -- the division the theme pass declines to make."""
    from spacr.qt.widgets.figure_settings import apply_line_colour

    assert apply_line_colour(figure, "#0000ff") > 0
    axis = figure.axes[0]
    assert _hex(axis.lines[0].get_color()) == "#0000ff"
    assert _hex(axis.lines[1].get_color()) == "#0000ff"
    assert _spine_colour(axis) == "#0000ff"
    assert _tick_mark_colour(axis) == "#0000ff"


def test_the_dashes_survive_a_recolour(figure):
    """A threshold line stays dashed: the dash is what tells a reader which
    line is a threshold and which is the data's own."""
    from spacr.qt.widgets.figure_settings import apply_line_colour

    apply_line_colour(figure, "#0000ff")
    assert figure.axes[0].lines[1].get_linestyle() == "--"
    assert figure.axes[0].lines[0].get_linestyle() == "-"


def test_apply_line_colour_leaves_every_piece_of_text_alone(figure):
    from spacr.qt.widgets.figure_settings import apply_line_colour

    apply_line_colour(figure, "#0000ff")
    axis = figure.axes[0]
    assert _hex(axis.title.get_color()) != "#0000ff"
    assert _tick_label_colour(axis) != "#0000ff"


def test_apply_font_colour_reaches_the_three_that_are_easily_missed(figure):
    """An annotation, the suptitle and the legend's title -- issue #108."""
    from spacr.qt.widgets.figure_settings import apply_font_colour

    apply_font_colour(figure, "#00ff00")
    axis = figure.axes[0]
    assert _hex(axis.texts[0].get_color()) == "#00ff00"
    assert _hex(figure._suptitle.get_color()) == "#00ff00"
    assert _hex(axis.get_legend().get_title().get_color()) == "#00ff00"
    assert _tick_label_colour(axis) == "#00ff00"


def test_apply_font_colour_leaves_the_lines_alone(figure):
    from spacr.qt.widgets.figure_settings import apply_font_colour

    apply_font_colour(figure, "#00ff00")
    axis = figure.axes[0]
    assert _spine_colour(axis) != "#00ff00"
    assert _tick_mark_colour(axis) != "#00ff00"
    assert _hex(axis.lines[1].get_color()) == "#ff0000"


def test_the_tick_label_colour_survives_a_redraw(figure):
    """matplotlib rebuilds its tick labels on every draw, so a colour set on
    today's label objects is gone at the next autoscale. Set on the TICK."""
    from spacr.qt.widgets.figure_settings import apply_font_colour

    apply_font_colour(figure, "#00ff00")
    figure.axes[0].set_xlim(-5, 5)
    figure.canvas.draw()
    assert _tick_label_colour(figure.axes[0]) == "#00ff00"


def test_follow_the_theme_puts_a_figure_back(figure, prefs, monkeypatch):
    from spacr.qt.widgets.figure_settings import (apply_font_colour,
                                                  apply_line_colour,
                                                  figure_follows_the_theme)

    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "light")
    apply_line_colour(figure, "#0000ff")
    apply_font_colour(figure, "#00ff00")
    figure_follows_the_theme(figure)
    axis = figure.axes[0]
    assert _spine_colour(axis) == "#000000"
    assert _hex(axis.title.get_color()) == "#000000"


def test_the_right_click_menu_offers_both_colours(qtbot, figure):
    """A control nobody can find is a control that does not exist, and the
    report that opened 152 was a user who could not find one."""
    from PySide6.QtWidgets import QWidget
    from spacr.qt.widgets.figure_settings import build_figure_context_menu

    parent = QWidget()
    qtbot.addWidget(parent)
    menu = build_figure_context_menu(parent, figure)
    appearance = [a.menu() for a in menu.actions()
                  if a.menu() is not None and a.text() == "Appearance"]
    assert appearance, [a.text() for a in menu.actions()]
    entries = [a.text() for a in appearance[0].actions()]
    assert entries == ["Line colour…", "Font colour…",
                       "Follow the theme (colours)"]


def test_the_menu_entry_actually_recolours_the_figure(qtbot, figure,
                                                      monkeypatch):
    """Driven through the ACTION, not through the helper it calls: a menu
    entry connected to nothing is the failure this repo has already shipped
    once."""
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QWidget
    from spacr.qt.widgets import figure_settings as module

    monkeypatch.setattr(module, "pick_colour",
                        lambda *a, **k: QColor("#0000ff"))
    parent = QWidget()
    qtbot.addWidget(parent)
    menu = module.build_figure_context_menu(parent, figure)
    appearance = [a.menu() for a in menu.actions()
                  if a.menu() is not None and a.text() == "Appearance"][0]
    appearance.actions()[0].trigger()
    assert _spine_colour(figure.axes[0]) == "#0000ff"


def test_the_settings_dialog_has_a_line_row_and_a_font_row(qtbot, figure,
                                                           prefs, monkeypatch):
    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "light")
    from PySide6.QtWidgets import QLabel

    dialog = FigureSettingsDialog(figure)
    qtbot.addWidget(dialog)
    texts = {w.text() for w in dialog.findChildren(QLabel)}
    assert "Line colour" in texts
    assert "Font colour" in texts
    assert "All text colour" not in texts
