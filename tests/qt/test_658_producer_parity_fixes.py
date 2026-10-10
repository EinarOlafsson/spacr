"""N658 parity audit fixes: the producers that missed parity now have it.

The 2026-10-10 audit (``tools/audit_figure_producers.py``) found thirteen
producers without full parity:

* eight embedded screens on the shared menu canvas (control chart x3, dose
  response, map barcodes, Feature Explorer, PCA scree, volcano) whose
  figures never took the Preferences figure settings: the canvas now applies
  them to every new drawing before it draws;
* the hyperparameter search's trial thumbnails and its summary panel (four
  producers), shown as rendered pictures with no figure menu: a right-click
  now opens the shared menu on the live figure and the picture follows
  every edit;
* ``response_distribution.panel``, whose own figure was never styled.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=False)
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QContextMenuEvent, QPixmap  # noqa: E402

from spacr.figures import bundle, style  # noqa: E402

PLOT_ENTRIES = {"Edit figure…", "Change graph type", "Statistics…",
                "Save figure (zip)…"}


@pytest.fixture
def tick_17(monkeypatch):
    """The user changed one figure setting: tick labels at 17 pt."""
    monkeypatch.setattr(style, "_preference_deltas",
                        lambda kind=None: {"tick_size": 17})


@pytest.fixture
def opened(monkeypatch):
    """Menus that would open, captured instead of shown modally."""
    from spacr.qt.widgets import figure_settings

    seen = []
    monkeypatch.setattr(figure_settings, "_exec_menu",
                        lambda menu, position: seen.append(menu))
    return seen


def _visible(menu):
    return {a.text(): a for a in menu.actions()
            if a.isVisible() and a.text()}


def _tick_size(ax):
    return ax.xaxis.get_major_ticks()[0].label1.get_fontsize()


def _trial(index=0, score=0.8):
    from spacr.hyperparam import Trial

    rng = np.random.default_rng(index)
    return Trial(params={"n_neighbors": 5 + index}, score=score,
                 extra_metrics={"embedding": rng.normal(size=(12, 2))},
                 duration=0.01, index=index)


def test_the_shared_canvas_styles_each_new_drawing(qapp, tick_17):
    """New content takes the settings; an edit survives a plain repaint."""
    from spacr.qt.widgets.graph_builder import _canvas_class

    figure = Figure()
    canvas = _canvas_class()(figure)
    ax = figure.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    canvas.draw()
    assert _tick_size(ax) == 17

    ax.tick_params(labelsize=9)
    canvas.draw()
    assert _tick_size(ax) == 9, "a repaint undid the user's edit"

    ax.plot([0, 1], [1, 1], gid="spacr-stats")
    canvas.draw()
    assert _tick_size(ax) == 9, "a statistics bracket counted as new data"

    ax.plot([1, 0], [0, 1])
    canvas.draw()
    assert _tick_size(ax) == 17, "a new drawing missed the settings"
    canvas.deleteLater()


def test_style_new_content_edges(tick_17):
    """No figure, an empty figure and a broken figure are all harmless."""
    assert style._style_new_content(None) == {}
    assert style._style_new_content(Figure()) == {}

    class Broken:
        axes = property(lambda self: 1 / 0)

    assert style._style_new_content(Broken()) == {}
    figure = Figure()
    figure.add_subplot(111).plot([0, 1])
    assert style._style_new_content(figure) == {"tick_size": 17}
    assert style._style_new_content(figure) == {}


def test_a_trial_figure_registers_and_is_styled(tick_17, tmp_path):
    """The trial figure carries its embedding and the user's settings."""
    from spacr.qt.screens.hyperparam import _trial_figure as trial_figure
    from spacr.qt.screens.hyperparam import render_trial_figure

    figure = trial_figure(_trial(), "trustworthiness")
    frame, spec = bundle._figure_record(figure)
    assert len(frame) == 12 and spec["kind"] == "scatter"
    assert getattr(figure, style._STYLED_FLAG)

    from spacr.hyperparam import Trial
    empty = Trial(params={}, score=None, extra_metrics={}, index=1)
    assert trial_figure(empty, "x") is None
    assert render_trial_figure(empty, "x", str(tmp_path / "e.png")) is False
    assert render_trial_figure(_trial(), "x", str(tmp_path / "t.png"))
    assert (tmp_path / "t.png").exists()


def test_a_grid_cell_right_click_asks_for_the_menu(qtbot, tmp_path):
    """Right-click requests the menu; only a left click opens the file."""
    from spacr.qt.screens.hyperparam import render_trial_figure
    from spacr.qt.widgets.figure_grid import SearchFigureGrid

    png = tmp_path / "t.png"
    render_trial_figure(_trial(), "x", str(png))
    grid = SearchFigureGrid(["n_neighbors"])
    qtbot.addWidget(grid)
    grid.resize(400, 300)
    index = grid.add_figure(str(png), {"n_neighbors": 5})
    from PySide6.QtWidgets import QLabel
    label = grid.findChild(QLabel, f"SearchFigureCell_{index}")
    assert label is not None

    clicked, asked = [], []
    grid.cell_clicked.connect(clicked.append)
    grid.cell_menu_requested.connect(lambda i, p: asked.append((i, p)))

    from PySide6.QtTest import QTest
    QTest.mouseClick(label, Qt.RightButton)
    assert clicked == []
    QTest.mouseClick(label, Qt.LeftButton)
    assert clicked == [index]
    label.contextMenuEvent(QContextMenuEvent(
        QContextMenuEvent.Mouse, QPoint(1, 1), QPoint(7, 9)))
    assert asked == [(index, QPoint(7, 9))]

    before = grid._cells[index].pixmap.cacheKey()
    grid._set_cell_pixmap(index, QPixmap())
    grid._set_cell_pixmap(index + 5, QPixmap(4, 4))
    assert grid._cells[index].pixmap.cacheKey() == before
    fresh = QPixmap(4, 4)
    grid._set_cell_pixmap(index, fresh)
    assert grid._cells[index].pixmap.cacheKey() == fresh.cacheKey()


@pytest.fixture
def ml_panel(qtbot, qt_theme_applied):
    """A search panel with the trial grid and the summary label."""
    from spacr.qt.screens.hyperparam import HyperparamPanel

    panel = HyperparamPanel("ml_analyze")
    qtbot.addWidget(panel)
    assert panel._figure_grid is not None
    return panel


def test_a_trial_thumbnail_opens_the_figure_menu(ml_panel, opened, tmp_path):
    """The menu edits the live trial figure and the cell shows the edit."""
    from spacr.qt.screens.hyperparam import render_trial_figure

    trial = _trial()
    png = tmp_path / "t.png"
    render_trial_figure(trial, "score", str(png))
    ml_panel._on_trial_ready(trial, 1, 1, str(png))
    assert ml_panel._grid_trials == {0: trial}

    ml_panel._trial_figure_menu(5, QPoint(0, 0))
    assert opened == [], "a cell with no trial opened a menu"

    ml_panel._trial_figure_menu(0, QPoint(0, 0))
    entries = _visible(opened[-1])
    assert PLOT_ENTRIES <= set(entries)
    figure = ml_panel._trial_figures[0]
    frame, _spec = bundle._figure_record(figure)
    assert len(frame) == 12

    before = ml_panel._figure_grid._cells[0].pixmap.cacheKey()
    kinds = [a for a in entries["Change graph type"].menu().actions()
             if a.isEnabled() and not a.isChecked()]
    kinds[0].trigger()
    assert ml_panel._figure_grid._cells[0].pixmap.cacheKey() != before

    ml_panel._trial_figure_menu(0, QPoint(0, 0))
    assert ml_panel._trial_figures[0] is figure, "edits were thrown away"


def test_a_trial_without_an_embedding_opens_nothing(ml_panel, opened):
    from spacr.hyperparam import Trial

    ml_panel._grid_trials[0] = Trial(params={}, score=None,
                                     extra_metrics={}, index=0)
    ml_panel._trial_figure_menu(0, QPoint(0, 0))
    assert opened == [] and ml_panel._trial_figures == {}


def test_the_summary_panel_opens_the_figure_menu(ml_panel, opened):
    """The summary picture keeps its figure and follows every edit."""
    from spacr.hyperparam import SearchResult, SearchSpace

    ml_panel._preview_figure_menu(QPoint(1, 1))
    assert opened == [], "a menu opened with no summary drawn"

    trials = [_trial(i, 0.9 - 0.1 * i) for i in range(3)]
    for trial in trials:
        trial.extra_metrics.pop("embedding")
    result = SearchResult(trials=trials, best=trials[0],
                          space=SearchSpace({"n_neighbors": [5, 6, 7]}),
                          metric="score")
    ml_panel._draw_preview(result)
    figure = ml_panel._preview_figure
    assert figure is not None
    frame, _spec = bundle._figure_record(figure)
    assert len(frame) == 3

    ml_panel._preview_figure_menu(QPoint(1, 1))
    entries = _visible(opened[-1])
    assert PLOT_ENTRIES <= set(entries)
    before = ml_panel._preview.pixmap().cacheKey()
    kinds = [a for a in entries["Change graph type"].menu().actions()
             if a.isEnabled() and not a.isChecked()]
    kinds[0].trigger()
    assert ml_panel._preview.pixmap().cacheKey() != before


def test_the_rendered_picture_menu_routes_every_change(qapp, monkeypatch):
    """Toggles repaint; a house replacement is drawn onto the same figure;
    Edit opens the editor on the live figure."""
    from spacr.qt.widgets import figure_settings

    captured = {}

    def fake_menu(widget, figure, *, on_change=None, open_settings=None):
        captured.update(on_change=on_change, open_settings=open_settings)
        return figure_settings.QMenu()

    retyped, edited, painted = [], [], []
    monkeypatch.setattr(figure_settings, "build_figure_context_menu",
                        fake_menu)
    monkeypatch.setattr(figure_settings, "_exec_menu", lambda m, p: None)
    monkeypatch.setattr(figure_settings, "_retype",
                        lambda fig, kind, _x: retyped.append((fig, kind)))
    monkeypatch.setattr(figure_settings, "_open_editor",
                        lambda fig, parent, change: edited.append(fig)
                        or change())

    figure = Figure()
    figure_settings._show_rendered_figure_menu(None, figure, QPoint(0, 0),
                                               painted.append)
    captured["on_change"](preview=True)
    assert painted == [figure] and retyped == []

    replacement = Figure()
    replacement._spacr_replot = {"graph_type": "bar"}
    captured["on_change"](replacement)
    assert retyped == [(figure, figure_settings._house_kind("bar"))]
    assert figure._spacr_replot == {"graph_type": "bar"}
    assert figure._spacr_data is None

    captured["open_settings"]()
    assert edited == [figure] and painted[-1] is figure


def test_response_distribution_styles_only_its_own_figure(tick_17):
    """A figure the panel makes takes the settings; a caller's does not."""
    from spacr.response_distribution import panel

    values = np.random.default_rng(0).lognormal(size=60)
    result = panel(values, "log", dependent_variable="y")
    figure = result["axes"].figure
    assert getattr(figure, style._STYLED_FLAG)
    assert bundle._figure_record(figure)[0] is not None

    own = Figure()
    ax = own.add_subplot(111)
    panel(values, "log", ax=ax)
    assert not getattr(own, style._STYLED_FLAG, False)
