"""The pyqtgraph figures refuse cleanly when there is nowhere to draw.

Every one of these helpers runs after a model has been fitted and every object
scored, so none of them may raise: with no Qt application they print the reason
and return None, and when the plot itself declines to draw they return None
without leaving a widget behind. Both conditions are arranged here, because a
machine with a display never produces either.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from spacr import ml


REFUSAL = 'there is no Qt application to draw under'


@pytest.fixture
def no_qt(monkeypatch):
    """Make ``figures.headless.application`` report that it cannot draw."""
    import spacr.figures.headless as headless

    monkeypatch.setattr(headless, 'application', lambda: (None, REFUSAL))
    return REFUSAL


class DecliningPlot:
    """A FastPlot stand-in whose every draw call declines."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.deleted = False
        DecliningPlot.instances.append(self)

    def resize(self, *args):
        return None

    def add_beeswarm(self, *args, **kwargs):
        return False

    def add_curve(self, *args, **kwargs):
        return False

    def add_radar(self, *args, **kwargs):
        return False

    def add_ranked_bars(self, *args, **kwargs):
        return False

    def deleteLater(self):
        self.deleted = True


@pytest.fixture
def declining_plot(monkeypatch):
    """A working application whose plots all decline to draw."""
    import spacr.figures.headless as headless
    import spacr.qt.widgets.fast_plots as fast_plots

    application = types.SimpleNamespace(processEvents=lambda: None)
    monkeypatch.setattr(headless, 'application', lambda: (application, ''))
    DecliningPlot.instances = []
    monkeypatch.setattr(fast_plots, 'FastPlot', DecliningPlot)
    return DecliningPlot


# ---------------------------------------------------------------------------
# nowhere to draw
# ---------------------------------------------------------------------------

def test_the_response_panel_says_why_it_could_not_be_drawn(no_qt, capsys):
    """No Qt prints the refusal and returns None rather than raising.

    This is drawn from inside a regression run; an exception would lose the fit
    to a distribution plot.
    """
    assert ml._draw_response_panel_in_pyqtgraph(
        [0.1, 0.2, 0.3], 'log', 'pathogen_rate', None) is None
    assert REFUSAL in capsys.readouterr().out


def test_the_shap_summary_says_why_it_could_not_be_drawn(no_qt, capsys):
    """The explanation bundle is skipped, with the reason on the console.

    The CSV of importances has already been written by then, so the picture is
    the only thing lost.
    """
    frame = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    assert ml._draw_shap_summary_in_pyqtgraph(
        np.zeros((2, 2)), frame, None, 'shap', 2) is None
    assert REFUSAL in capsys.readouterr().out


def test_the_cell_count_sweep_says_why_it_could_not_be_drawn(no_qt, capsys,
                                                             tmp_path):
    """The sample-size sweep returns None instead of blocking on a show.

    Blocking here is what this figure was moved off matplotlib to stop; the
    headless answer has to be a return, not a wait.
    """
    summary = pd.DataFrame({'sample_size': [10, 20],
                            'smoothed_mean_abs_diff': [0.5, 0.2],
                            'std_abs_diff': [0.1, 0.05]})
    assert ml._draw_the_cell_count_sweep(
        summary, 15, str(tmp_path / 'sweep.pdf')) is None
    assert REFUSAL in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_the_radar_says_why_it_could_not_be_drawn(no_qt, capsys):
    """The radar bundle is skipped and the refusal explains it.

    None here is not a failure -- the caller has the table already.
    """
    assert ml._draw_radar_in_pyqtgraph(
        ['a', 'b', 'c'], [0.1, 0.5, 0.9], 'radar', None, 'radar') is None
    assert REFUSAL in capsys.readouterr().out


def test_the_importance_chart_says_why_it_could_not_be_drawn(no_qt, capsys):
    """The ranked bars are skipped, with the refusal said out loud.

    Silence here left a run with no figure and nothing saying why.
    """
    frame = pd.DataFrame({'feature': ['a', 'b'], 'importance': [0.2, 0.1]})
    assert ml._draw_importance_in_pyqtgraph(
        frame, 'importance', None, 'importance', 2) is None
    assert REFUSAL in capsys.readouterr().out


def test_a_shap_beeswarm_with_no_scene_returns_nothing(no_qt, monkeypatch,
                                                       capsys):
    """``shap_analysis`` returns None when there is no application to draw under.

    It returns a live plot, so the absence of Qt is the one case where there is
    no plot to return, and it is said rather than logged.
    """
    values = types.SimpleNamespace(shape=(4, 2),
                                   values=np.zeros((4, 2)))
    monkeypatch.setattr(ml, '_shap_values', lambda *a, **k: (values, ''))
    frame = pd.DataFrame({'a': [1.0] * 4, 'b': [2.0] * 4})

    assert ml.shap_analysis(object(), frame, frame) is None
    assert REFUSAL in capsys.readouterr().out


# ---------------------------------------------------------------------------
# a plot that declines to draw
# ---------------------------------------------------------------------------

def test_a_response_panel_with_no_finite_values_is_reported(monkeypatch,
                                                            capsys):
    """A response holding nothing finite draws no panel, and says so.

    An empty axes would read as "the response is flat" rather than "there was
    nothing to plot".
    """
    import spacr.figures.headless as headless
    import spacr.response_distribution as response_distribution

    application = types.SimpleNamespace(processEvents=lambda: None)
    monkeypatch.setattr(headless, 'application', lambda: (application, ''))
    monkeypatch.setattr(response_distribution, 'fast_panel',
                        lambda *a, **k: None)

    assert ml._draw_response_panel_in_pyqtgraph(
        [float('nan')], 'none', 'pathogen_rate', None) is None
    assert 'holds no finite values' in capsys.readouterr().out


def test_a_shap_matrix_that_is_not_two_dimensional_draws_nothing(
        declining_plot):
    """A 1-D or empty SHAP matrix has no beeswarm, so nothing is written.

    Ranking features off a matrix with no feature axis would index the wrong
    column names onto the values.
    """
    frame = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    assert ml._draw_shap_summary_in_pyqtgraph(
        np.zeros(4), frame, None, 'shap', 2) is None
    assert declining_plot.instances == []


def test_a_beeswarm_that_declines_leaves_no_widget_behind(declining_plot):
    """When ``add_beeswarm`` returns False the plot is deleted and None returned.

    The widget is created before the draw is attempted; leaking it would hold a
    Qt object alive for the life of the process on every failed figure.
    """
    frame = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    assert ml._draw_shap_summary_in_pyqtgraph(
        np.zeros((2, 2)), frame, None, 'shap', 2) is None
    assert [p.deleted for p in declining_plot.instances] == [True]


def test_a_sweep_curve_that_declines_writes_no_file(declining_plot, tmp_path):
    """A curve that will not draw returns None rather than writing an empty file.

    ``write_plot`` is never reached, so the run folder keeps no stub PDF that
    a reader would open expecting a sweep.
    """
    summary = pd.DataFrame({'sample_size': [10, 20],
                            'smoothed_mean_abs_diff': [0.5, 0.2],
                            'std_abs_diff': [0.1, 0.05]})
    assert ml._draw_the_cell_count_sweep(
        summary, 15, str(tmp_path / 'sweep.pdf')) is None
    assert not list(tmp_path.iterdir())


def test_a_radar_that_declines_leaves_no_widget_behind(declining_plot):
    """``add_radar`` returning False deletes the plot and reports nothing drawn.

    Same contract as the beeswarm: the bundle is not written and the widget
    does not survive.
    """
    assert ml._draw_radar_in_pyqtgraph(
        ['a', 'b', 'c'], [0.1, 0.5, 0.9], 'radar', None, 'radar') is None
    assert [p.deleted for p in declining_plot.instances] == [True]


def test_ranked_bars_that_decline_leave_no_widget_behind(declining_plot):
    """``add_ranked_bars`` returning False deletes the plot and returns None.

    The importance CSV is already written; the picture failing must cost only
    the picture.
    """
    frame = pd.DataFrame({'feature': ['a', 'b'], 'importance': [0.2, 0.1]})
    assert ml._draw_importance_in_pyqtgraph(
        frame, 'importance', None, 'importance', 2) is None
    assert [p.deleted for p in declining_plot.instances] == [True]


def test_a_shap_summary_whose_values_are_not_a_matrix_returns_nothing(
        declining_plot, monkeypatch):
    """``shap_analysis`` refuses a non-2-D explanation rather than reshaping it.

    A 1-D array here would be plotted against feature names it does not line up
    with, which is a picture of the wrong thing.
    """
    values = types.SimpleNamespace(shape=(4,), values=np.zeros(4))
    monkeypatch.setattr(ml, '_shap_values', lambda *a, **k: (values, ''))
    frame = pd.DataFrame({'a': [1.0] * 4, 'b': [2.0] * 4})

    assert ml.shap_analysis(object(), frame, frame) is None


def test_a_shap_beeswarm_that_declines_is_deleted_and_not_returned(
        declining_plot, monkeypatch):
    """A plot that could not draw the beeswarm is deleted, and None comes back.

    The caller writes whatever it is given; returning a blank plot would put an
    empty SHAP panel in the report.
    """
    values = types.SimpleNamespace(shape=(4, 2), values=np.zeros((4, 2)))
    monkeypatch.setattr(ml, '_shap_values', lambda *a, **k: (values, ''))
    frame = pd.DataFrame({'a': [1.0] * 4, 'b': [2.0] * 4})

    assert ml.shap_analysis(object(), frame, frame) is None
    assert [p.deleted for p in declining_plot.instances] == [True]


# ---------------------------------------------------------------------------
# announcing what was written
# ---------------------------------------------------------------------------

def test_no_bundle_folder_means_no_gallery_tile(tmp_path):
    """A missing or empty folder is announced as nothing, not as a tile.

    A tile pointing at a file that was never written is a dead link in the
    gallery.
    """
    assert ml._announce_the_bundle(None, 'a figure') is None
    assert ml._announce_the_bundle(str(tmp_path / 'gone'), 'a figure') is None


def test_a_bundle_holding_no_figure_announces_nothing(monkeypatch, tmp_path):
    """Neither the preferred format nor a PDF present means nothing to show.

    The fallback to PDF exists because a bundle always writes one; a folder
    with neither is not a bundle, and inventing a tile for it would be worse
    than staying quiet.
    """
    import spacr.plot as plot_module

    monkeypatch.setattr(plot_module, 'figure_output_preferences',
                        lambda: ('svg', 300))
    (tmp_path / 'data.csv').write_text('a,b\n1,2\n')

    assert ml._announce_the_bundle(str(tmp_path), 'a figure') is None


def test_nothing_is_written_for_a_plot_that_was_never_built(tmp_path):
    """``write_plot(None, ...)`` writes nothing, announces nothing, returns None.

    A plot that could not be built must not take the run down after the model
    has been fitted and every object scored.
    """
    assert ml.write_plot(None, str(tmp_path / 'x.pdf'), 'title') is None
    assert not list(tmp_path.iterdir())
