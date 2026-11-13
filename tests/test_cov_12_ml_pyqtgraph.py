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

def case_the_response_panel_says_why_it_could_not_be_drawn(no_qt, capsys):
    """No Qt prints the refusal and returns None rather than raising.

    This is drawn from inside a regression run; an exception would lose the fit
    to a distribution plot.
    """
    _seen = []
    _seen.append(bool(
        ml._draw_response_panel_in_pyqtgraph(
        [0.1, 0.2, 0.3], 'log', 'pathogen_rate', None) is None))
    _seen.append(bool(
        REFUSAL in capsys.readouterr().out))
    return _seen


def case_the_shap_summary_says_why_it_could_not_be_drawn(no_qt, capsys):
    """The explanation bundle is skipped, with the reason on the console.

    The CSV of importances has already been written by then, so the picture is
    the only thing lost.
    """
    _seen = []
    frame = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    _seen.append(bool(
        ml._draw_shap_summary_in_pyqtgraph(
        np.zeros((2, 2)), frame, None, 'shap', 2) is None))
    _seen.append(bool(
        REFUSAL in capsys.readouterr().out))
    return _seen


def case_the_cell_count_sweep_says_why_it_could_not_be_drawn(no_qt, capsys,
                                                             tmp_path):
    """The sample-size sweep returns None instead of blocking on a show.

    Blocking here is what this figure was moved off matplotlib to stop; the
    headless answer has to be a return, not a wait.
    """
    _seen = []
    summary = pd.DataFrame({'sample_size': [10, 20],
                            'smoothed_mean_abs_diff': [0.5, 0.2],
                            'std_abs_diff': [0.1, 0.05]})
    _seen.append(bool(
        ml._draw_the_cell_count_sweep(
        summary, 15, str(tmp_path / 'sweep.pdf')) is None))
    _seen.append(bool(
        REFUSAL in capsys.readouterr().out))
    _seen.append(bool(
        not list(tmp_path.iterdir())))
    return _seen


def case_the_radar_says_why_it_could_not_be_drawn(no_qt, capsys):
    """The radar bundle is skipped and the refusal explains it.

    None here is not a failure -- the caller has the table already.
    """
    _seen = []
    _seen.append(bool(
        ml._draw_radar_in_pyqtgraph(
        ['a', 'b', 'c'], [0.1, 0.5, 0.9], 'radar', None, 'radar') is None))
    _seen.append(bool(
        REFUSAL in capsys.readouterr().out))
    return _seen


def case_the_importance_chart_says_why_it_could_not_be_drawn(no_qt, capsys):
    """The ranked bars are skipped, with the refusal said out loud.

    Silence here left a run with no figure and nothing saying why.
    """
    _seen = []
    frame = pd.DataFrame({'feature': ['a', 'b'], 'importance': [0.2, 0.1]})
    _seen.append(bool(
        ml._draw_importance_in_pyqtgraph(
        frame, 'importance', None, 'importance', 2) is None))
    _seen.append(bool(
        REFUSAL in capsys.readouterr().out))
    return _seen


def case_a_shap_beeswarm_with_no_scene_returns_nothing(no_qt, monkeypatch,
                                                       capsys):
    """``shap_analysis`` returns None when there is no application to draw under.

    It returns a live plot, so the absence of Qt is the one case where there is
    no plot to return, and it is said rather than logged.
    """
    _seen = []
    values = types.SimpleNamespace(shape=(4, 2),
                                   values=np.zeros((4, 2)))
    monkeypatch.setattr(ml, '_shap_values', lambda *a, **k: (values, ''))
    frame = pd.DataFrame({'a': [1.0] * 4, 'b': [2.0] * 4})

    _seen.append(bool(
        ml.shap_analysis(object(), frame, frame) is None))
    _seen.append(bool(
        REFUSAL in capsys.readouterr().out))
    return _seen


# ---------------------------------------------------------------------------
# a plot that declines to draw
# ---------------------------------------------------------------------------

def case_a_response_panel_with_no_finite_values_is_reported(monkeypatch,
                                                            capsys):
    """A response holding nothing finite draws no panel, and says so.

    An empty axes would read as "the response is flat" rather than "there was
    nothing to plot".
    """
    _seen = []
    import spacr.figures.headless as headless
    import spacr.response_distribution as response_distribution

    application = types.SimpleNamespace(processEvents=lambda: None)
    monkeypatch.setattr(headless, 'application', lambda: (application, ''))
    monkeypatch.setattr(response_distribution, 'fast_panel',
                        lambda *a, **k: None)

    _seen.append(bool(
        ml._draw_response_panel_in_pyqtgraph(
        [float('nan')], 'none', 'pathogen_rate', None) is None))
    _seen.append(('holds no finite values' if 'holds no finite values' in (capsys.readouterr().out) else ("missing", 'holds no finite values')))
    return _seen


def case_a_shap_matrix_that_is_not_two_dimensional_draws_nothing(
        declining_plot):
    """A 1-D or empty SHAP matrix has no beeswarm, so nothing is written.

    Ranking features off a matrix with no feature axis would index the wrong
    column names onto the values.
    """
    _seen = []
    frame = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    _seen.append(bool(
        ml._draw_shap_summary_in_pyqtgraph(
        np.zeros(4), frame, None, 'shap', 2) is None))
    _seen.append(keep(declining_plot.instances))
    return _seen


def case_a_beeswarm_that_declines_leaves_no_widget_behind(declining_plot):
    """When ``add_beeswarm`` returns False the plot is deleted and None returned.

    The widget is created before the draw is attempted; leaking it would hold a
    Qt object alive for the life of the process on every failed figure.
    """
    _seen = []
    frame = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    _seen.append(bool(
        ml._draw_shap_summary_in_pyqtgraph(
        np.zeros((2, 2)), frame, None, 'shap', 2) is None))
    _seen.append(keep([p.deleted for p in declining_plot.instances]))
    return _seen


def case_a_sweep_curve_that_declines_writes_no_file(declining_plot, tmp_path):
    """A curve that will not draw returns None rather than writing an empty file.

    ``write_plot`` is never reached, so the run folder keeps no stub PDF that
    a reader would open expecting a sweep.
    """
    _seen = []
    summary = pd.DataFrame({'sample_size': [10, 20],
                            'smoothed_mean_abs_diff': [0.5, 0.2],
                            'std_abs_diff': [0.1, 0.05]})
    _seen.append(bool(
        ml._draw_the_cell_count_sweep(
        summary, 15, str(tmp_path / 'sweep.pdf')) is None))
    _seen.append(bool(
        not list(tmp_path.iterdir())))
    return _seen


def case_a_radar_that_declines_leaves_no_widget_behind(declining_plot):
    """``add_radar`` returning False deletes the plot and reports nothing drawn.

    Same contract as the beeswarm: the bundle is not written and the widget
    does not survive.
    """
    _seen = []
    _seen.append(bool(
        ml._draw_radar_in_pyqtgraph(
        ['a', 'b', 'c'], [0.1, 0.5, 0.9], 'radar', None, 'radar') is None))
    _seen.append(keep([p.deleted for p in declining_plot.instances]))
    return _seen


def case_ranked_bars_that_decline_leave_no_widget_behind(declining_plot):
    """``add_ranked_bars`` returning False deletes the plot and returns None.

    The importance CSV is already written; the picture failing must cost only
    the picture.
    """
    _seen = []
    frame = pd.DataFrame({'feature': ['a', 'b'], 'importance': [0.2, 0.1]})
    _seen.append(bool(
        ml._draw_importance_in_pyqtgraph(
        frame, 'importance', None, 'importance', 2) is None))
    _seen.append(keep([p.deleted for p in declining_plot.instances]))
    return _seen


def case_a_shap_summary_whose_values_are_not_a_matrix_returns_nothing(
        declining_plot, monkeypatch):
    """``shap_analysis`` refuses a non-2-D explanation rather than reshaping it.

    A 1-D array here would be plotted against feature names it does not line up
    with, which is a picture of the wrong thing.
    """
    _seen = []
    values = types.SimpleNamespace(shape=(4,), values=np.zeros(4))
    monkeypatch.setattr(ml, '_shap_values', lambda *a, **k: (values, ''))
    frame = pd.DataFrame({'a': [1.0] * 4, 'b': [2.0] * 4})

    _seen.append(bool(
        ml.shap_analysis(object(), frame, frame) is None))
    return _seen


def case_a_shap_beeswarm_that_declines_is_deleted_and_not_returned(
        declining_plot, monkeypatch):
    """A plot that could not draw the beeswarm is deleted, and None comes back.

    The caller writes whatever it is given; returning a blank plot would put an
    empty SHAP panel in the report.
    """
    _seen = []
    values = types.SimpleNamespace(shape=(4, 2), values=np.zeros((4, 2)))
    monkeypatch.setattr(ml, '_shap_values', lambda *a, **k: (values, ''))
    frame = pd.DataFrame({'a': [1.0] * 4, 'b': [2.0] * 4})

    _seen.append(bool(
        ml.shap_analysis(object(), frame, frame) is None))
    _seen.append(keep([p.deleted for p in declining_plot.instances]))
    return _seen


# ---------------------------------------------------------------------------
# announcing what was written
# ---------------------------------------------------------------------------

def case_no_bundle_folder_means_no_gallery_tile(tmp_path):
    """A missing or empty folder is announced as nothing, not as a tile.

    A tile pointing at a file that was never written is a dead link in the
    gallery.
    """
    _seen = []
    _seen.append(bool(
        ml._announce_the_bundle(None, 'a figure') is None))
    _seen.append(bool(
        ml._announce_the_bundle(str(tmp_path / 'gone'), 'a figure') is None))
    return _seen


def case_a_bundle_holding_no_figure_announces_nothing(monkeypatch, tmp_path):
    """Neither the preferred format nor a PDF present means nothing to show.

    The fallback to PDF exists because a bundle always writes one; a folder
    with neither is not a bundle, and inventing a tile for it would be worse
    than staying quiet.
    """
    _seen = []
    import spacr.plot as plot_module

    monkeypatch.setattr(plot_module, 'figure_output_preferences',
                        lambda: ('svg', 300))
    (tmp_path / 'data.csv').write_text('a,b\n1,2\n', encoding="utf-8")

    _seen.append(bool(
        ml._announce_the_bundle(str(tmp_path), 'a figure') is None))
    return _seen


def case_nothing_is_written_for_a_plot_that_was_never_built(tmp_path):
    """``write_plot(None, ...)`` writes nothing, announces nothing, returns None.

    A plot that could not be built must not take the run down after the model
    has been fitted and every object scored.
    """
    _seen = []
    _seen.append(bool(
        ml.write_plot(None, str(tmp_path / 'x.pdf'), 'title') is None))
    _seen.append(bool(
        not list(tmp_path.iterdir())))
    return _seen


def case_a_bare_plot_path_uses_its_absolute_parent(monkeypatch, tmp_path):
    """A filename without a directory still exports in the working folder.

    ``dirname(abspath(target))`` is always non-empty, including for this bare
    path.  The writer therefore creates that resolved parent unconditionally
    rather than carrying an impossible branch for a case that cannot occur.
    """
    _seen = []
    import spacr.figure_sink as figure_sink
    import spacr.plot as plot_module

    class Plot:
        deleted = False

        def export(self, target):
            with open(target, 'w', encoding='utf-8') as handle:
                handle.write('plot')
            return target

        def deleteLater(self):
            self.deleted = True

    announced = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plot_module, 'figure_output_preferences',
                        lambda: ('svg', 300))
    monkeypatch.setattr(figure_sink, 'publish_file',
                        lambda path, *, title=None: announced.append(
                            (path, title)))
    plot = Plot()

    written = ml.write_plot(plot, 'result.pdf', 'response')

    _seen.append(keep(written))
    _seen.append(keep((tmp_path / 'result.svg').read_text(encoding='utf-8')))
    _seen.append(keep(announced))
    _seen.append(bool(
        plot.deleted is True))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_the_shap_summary_says_why_it_could_not_be_drawn, [True, True], id="the_shap_summary_says_why_it_could_not_be_drawn"),
    pytest.param(case_the_cell_count_sweep_says_why_it_could_not_be_drawn, [True, True, True], id="the_cell_count_sweep_says_why_it_could_not_be_drawn"),
    pytest.param(case_a_shap_beeswarm_with_no_scene_returns_nothing, [True, True], id="a_shap_beeswarm_with_no_scene_returns_nothing"),
    pytest.param(case_a_shap_matrix_that_is_not_two_dimensional_draws_nothing, [True, []], id="a_shap_matrix_that_is_not_two_dimensional_draws_nothing"),
    pytest.param(case_a_beeswarm_that_declines_leaves_no_widget_behind, [True, [True]], id="a_beeswarm_that_declines_leaves_no_widget_behind"),
    pytest.param(case_a_sweep_curve_that_declines_writes_no_file, [True, True], id="a_sweep_curve_that_declines_writes_no_file"),
    pytest.param(case_a_shap_summary_whose_values_are_not_a_matrix_returns_nothing, [True], id="a_shap_summary_whose_values_are_not_a_matrix_returns_nothing"),
    pytest.param(case_a_shap_beeswarm_that_declines_is_deleted_and_not_returned, [True, [True]], id="a_shap_beeswarm_that_declines_is_deleted_and_not_returned"),
    pytest.param(case_nothing_is_written_for_a_plot_that_was_never_built, [True, True], id="nothing_is_written_for_a_plot_that_was_never_built"),
    pytest.param(case_a_bare_plot_path_uses_its_absolute_parent, ['result.svg', 'plot', [('result.svg', 'response')], True], id="a_bare_plot_path_uses_its_absolute_parent"),
    pytest.param(case_the_response_panel_says_why_it_could_not_be_drawn, [True, True], id="the_response_panel_says_why_it_could_not_be_drawn"),
    pytest.param(case_the_radar_says_why_it_could_not_be_drawn, [True, True], id="the_radar_says_why_it_could_not_be_drawn"),
    pytest.param(case_the_importance_chart_says_why_it_could_not_be_drawn, [True, True], id="the_importance_chart_says_why_it_could_not_be_drawn"),
    pytest.param(case_a_response_panel_with_no_finite_values_is_reported, [True, 'holds no finite values'], id="a_response_panel_with_no_finite_values_is_reported"),
    pytest.param(case_a_radar_that_declines_leaves_no_widget_behind, [True, [True]], id="a_radar_that_declines_leaves_no_widget_behind"),
    pytest.param(case_ranked_bars_that_decline_leave_no_widget_behind, [True, [True]], id="ranked_bars_that_decline_leave_no_widget_behind"),
    pytest.param(case_no_bundle_folder_means_no_gallery_tile, [True, True], id="no_bundle_folder_means_no_gallery_tile"),
    pytest.param(case_a_bundle_holding_no_figure_announces_nothing, [True], id="a_bundle_holding_no_figure_announces_nothing"),
])
def test_12_ml_pyqtgraph_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected
