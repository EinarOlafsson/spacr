"""A run must survive every way its figures can fail to be drawn.

The fit is the expensive half and the panels are the cheap half, so every
drawing helper in :mod:`spacr.ml` is written to swallow its own failures. That
is only true if it is checked: these block the imports, make the panel builders
raise, make the save fail, and assert that the helper reports the loss and
returns rather than taking the run down with it.
"""
from __future__ import annotations

import builtins
import types

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spacr import ml


def block_figures(monkeypatch):
    """Make every ``from .figures...`` import inside spacr.ml raise."""
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'figures' or name.startswith('figures.') \
                or name.startswith('spacr.figures'):
            raise ImportError('figures package is unavailable')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', blocked)


def frame_with_fractions():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'prc': [f'plate1_r{i // 4 + 1}_c{i % 4 + 1}' for i in range(16)],
        'grna': [f'g{i % 4}' for i in range(16)],
        'fraction': rng.uniform(0.05, 0.9, 16),
        'plateID': ['plate1'] * 16,
        'rowID': [f'r{i // 4 + 1}' for i in range(16)],
        'columnID': [f'c{i % 4 + 1}' for i in range(16)],
        'pathogen_rate': rng.uniform(0.0, 1.0, 16),
    })


class Panel:
    """A built panel that reports whether anything was actually drawn."""

    def __init__(self, drawn, title='a panel'):
        self.drawn = drawn
        self.title = title


# ---------------------------------------------------------------------------
# the distribution panels
# ---------------------------------------------------------------------------

def case_distribution_panels_that_cannot_be_imported_send_the_caller_back(
        monkeypatch, capsys):
    """A missing figures package returns False so the old histogram is used.

    The helper's contract is that False means "draw it the old way"; raising
    instead would lose a completed fit over a picture.
    """
    _seen = []
    block_figures(monkeypatch)
    _seen.append(bool(
        ml._show_well_distributions(frame_with_fractions(),
                                       'pathogen_rate', None) is False))
    _seen.append(('Could not load the distribution panels' if 'Could not load the distribution panels' in (capsys.readouterr().out) else ("missing", 'Could not load the distribution panels')))
    return _seen


def case_a_distribution_panel_that_raises_is_named_and_the_rest_still_draw(
        monkeypatch, capsys):
    """One panel blowing up costs that panel, not the whole set.

    The two panels are independent; a response column the builder cannot handle
    must not also cost the guide-fraction histogram.
    """
    _seen = []
    from spacr.figures import distributions

    real_build = distributions.build_panel

    def selective(key, frame, **kwargs):
        if key == 'guide_fraction':
            raise ValueError('no fraction column')
        return real_build(key, frame, **kwargs)

    monkeypatch.setattr(distributions, 'build_panel', selective)
    plt.close('all')

    _seen.append(bool(
        ml._show_well_distributions(frame_with_fractions(),
                                       'pathogen_rate', None, plot=False) \
        is True))
    _seen.append(('Distribution panel guide_fraction did not draw' if 'Distribution panel guide_fraction did not draw' in (capsys.readouterr().out) else ("missing", 'Distribution panel guide_fraction did not draw')))
    plt.close('all')
    return _seen


def case_panels_that_drew_nothing_are_closed_rather_than_shown(monkeypatch):
    """A panel reporting ``drawn=False`` is closed and does not count.

    An empty axes shown to the user reads as "the distribution is empty" when
    it means the builder had nothing to plot.
    """
    _seen = []
    from spacr.figures import distributions

    def empty(key, frame, **kwargs):
        return plt.figure(), Panel(drawn=False)

    monkeypatch.setattr(distributions, 'build_panel', empty)
    plt.close('all')

    _seen.append(bool(
        ml._show_well_distributions(frame_with_fractions(),
                                       'pathogen_rate', None,
                                       plot=False) is False))
    _seen.append(keep(plt.get_fignums()))
    return _seen


def case_a_distribution_panel_that_cannot_be_saved_is_still_counted_as_drawn(
        monkeypatch, tmp_path):
    """A failed save costs the file, not the panel or the run.

    The panel was drawn and the user can see it; refusing to report that
    because the disk was full would hide the figure that does exist.
    """
    _seen = []
    import spacr.plot as plot_module

    def boom(*args, **kwargs):
        raise OSError('no space left on device')

    monkeypatch.setattr(plot_module, 'save_figure', boom)
    plt.close('all')

    _seen.append(bool(
        ml._show_well_distributions(frame_with_fractions(),
                                       'pathogen_rate', str(tmp_path),
                                       plot=False) is True))
    _seen.append(bool(
        not list(tmp_path.iterdir())))
    plt.close('all')
    return _seen


# ---------------------------------------------------------------------------
# the plate panel
# ---------------------------------------------------------------------------

def case_a_plate_panel_that_cannot_be_imported_returns_false(monkeypatch,
                                                             capsys):
    """The plate heatmap is optional; its import failing is reported and skipped.

    Same trade as the distributions: the run has already produced the numbers.
    """
    _seen = []
    block_figures(monkeypatch)
    _seen.append(bool(
        ml._show_plates(frame_with_fractions(), 'fraction', None) is False))
    _seen.append(('Could not load the plate panel' if 'Could not load the plate panel' in (capsys.readouterr().out) else ("missing", 'Could not load the plate panel')))
    return _seen


def case_a_plate_panel_that_raises_is_reported_by_name(monkeypatch, capsys):
    """A builder exception is printed and False returned, not propagated.

    A screen whose plate layout the builder cannot read still has a valid fit.
    """
    _seen = []
    import spacr.figures.plates as plates_module

    def boom(*args, **kwargs):
        raise ValueError('no plate layout in this frame')

    monkeypatch.setattr(plates_module, 'build_plates', boom)
    _seen.append(bool(
        ml._show_plates(frame_with_fractions(), 'fraction', None) is False))
    _seen.append(('The plate panel did not draw' if 'The plate panel did not draw' in (capsys.readouterr().out) else ("missing", 'The plate panel did not draw')))
    return _seen


def case_a_plate_panel_with_nothing_on_it_is_closed(monkeypatch):
    """``drawn=False`` closes the figure and reports that nothing was shown.

    Leaving the blank figure open puts an empty plate map in the gallery.
    """
    _seen = []
    import spacr.figures.plates as plates_module

    monkeypatch.setattr(plates_module, 'build_plates',
                        lambda *a, **k: (plt.figure(), Panel(drawn=False)))
    plt.close('all')

    _seen.append(bool(
        ml._show_plates(frame_with_fractions(), 'fraction', None) is False))
    _seen.append(keep(plt.get_fignums()))
    return _seen


def case_a_plate_panel_that_cannot_be_saved_is_still_shown(monkeypatch,
                                                           tmp_path):
    """A save failure leaves the panel drawn and the helper reporting True.

    The figure reached the screen, which is what the return value promises.
    """
    _seen = []
    import spacr.plot as plot_module
    import spacr.figures.plates as plates_module

    monkeypatch.setattr(plates_module, 'build_plates',
                        lambda *a, **k: (plt.figure(), Panel(drawn=True)))
    monkeypatch.setattr(plot_module, 'save_figure',
                        lambda *a, **k: (_ for _ in ()).throw(
                            OSError('read-only file system')))
    plt.close('all')

    _seen.append(bool(
        ml._show_plates(frame_with_fractions(), 'fraction',
                           str(tmp_path)) is True))
    _seen.append(bool(
        not list(tmp_path.iterdir())))
    plt.close('all')
    return _seen


def case_drawn_helpers_release_their_figures_and_preserve_the_callers(
        monkeypatch):
    """Boolean/count helpers own figures they never return.

    ``plt.show`` is the bridge hand-off: the figure must still be registered
    during that call, then its exact pyplot manager must be released without
    disturbing a figure the caller already had open.
    """
    _seen = []
    caller = plt.figure()
    before = tuple(plt.get_fignums())
    shown = []

    def capture_show(*_args, **_kwargs):
        shown.extend(
            plt.figure(number) for number in plt.get_fignums()
            if number not in before
        )

    monkeypatch.setattr(plt, 'show', capture_show)

    try:
        assert ml._show_well_distributions(
            frame_with_fractions(), 'pathogen_rate', None,
            plot=False) is True
        assert tuple(plt.get_fignums()) == before

        assert ml._show_plates(
            frame_with_fractions(), 'fraction', None) is True
        assert len(shown) == 1
        assert len(shown[0].axes) == 2, (
            'the display bridge did not retain the drawn plate figure')
        shown[0].canvas.draw()
        assert tuple(plt.get_fignums()) == before
    finally:
        plt.close(caller)
    return _seen


def case_house_style_panels_that_cannot_be_imported_draw_none(monkeypatch,
                                                              capsys):
    """A blocked figures import returns a count of zero, not an exception.

    The count is what the caller prints; zero says plainly that no panel was
    drawn.
    """
    _seen = []
    block_figures(monkeypatch)
    coef = pd.DataFrame({'coefficient': [0.1], 'p_value': [0.01]})
    _seen.append(keep(ml._show_house_style_panels(coef)))
    _seen.append(('Could not load the figure style' if 'Could not load the figure style' in (capsys.readouterr().out) else ("missing", 'Could not load the figure style')))
    return _seen


# ---------------------------------------------------------------------------
# the publication sheet
# ---------------------------------------------------------------------------

def case_an_empty_coefficient_table_writes_no_sheet(tmp_path):
    """With no coefficients there is nothing to draw, and None says so.

    A fit that produced no rows must not leave an empty PDF that reads as a
    result.
    """
    _seen = []
    _seen.append(bool(
        ml._write_regression_sheet(None, tmp_path) is None))
    _seen.append(bool(
        ml._write_regression_sheet(pd.DataFrame(), tmp_path) is None))
    _seen.append(bool(
        not list(tmp_path.iterdir())))
    return _seen


def case_a_sheet_that_cannot_be_built_is_reported_and_costs_nothing_else(
        monkeypatch, tmp_path, capsys):
    """A builder exception prints the reason and returns None.

    This is the publication figure; losing the whole run because it could not
    be composed would be the worst possible trade.
    """
    _seen = []
    import spacr.figures as figures_module

    def boom(*args, **kwargs):
        raise RuntimeError('a panel had no data')

    monkeypatch.setattr(figures_module, 'build_sheet', boom)
    coef = pd.DataFrame({'coefficient': [0.1], 'p_value': [0.01]})

    _seen.append(bool(
        ml._write_regression_sheet(coef, tmp_path) is None))
    _seen.append(('Could not draw the regression figure' if 'Could not draw the regression figure' in (capsys.readouterr().out) else ("missing", 'Could not draw the regression figure')))
    return _seen


def case_a_sheet_whose_figure_cannot_be_closed_is_still_written(monkeypatch,
                                                               tmp_path,
                                                               capsys):
    """Closing the figure is best-effort; the path is returned regardless.

    The PDF and its legend are already on disk by then, so a close that raises
    must not turn a written figure into a reported failure.
    """
    _seen = []
    import spacr.figures as figures_module
    import spacr.figure_sink as sink_module

    class Sheet:
        figure = object()          # not a real figure, so plt.close raises
        panels = ('volcano', 'qq')
        skipped = ('controls',)

        def legend(self):
            return 'Panel a, the volcano.'

    monkeypatch.setattr(figures_module, 'build_sheet',
                        lambda *a, **k: Sheet())
    monkeypatch.setattr(sink_module, 'publish', lambda fig, path, **k: path)
    coef = pd.DataFrame({'coefficient': [0.1], 'p_value': [0.01]})

    path = ml._write_regression_sheet(coef, tmp_path)

    _seen.append(bool(
        path == str(tmp_path / 'regression_figure.pdf')))
    legend = (tmp_path / 'regression_figure_legend.txt').read_text(encoding="utf-8")
    _seen.append(keep(legend))
    _seen.append(('2 panels, 1 not applicable' if '2 panels, 1 not applicable' in (capsys.readouterr().out) else ("missing", '2 panels, 1 not applicable')))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_distribution_panels_that_cannot_be_imported_send_the_caller_back, [True, 'Could not load the distribution panels'], id="distribution_panels_that_cannot_be_imported_send_the_caller_back"),
    pytest.param(case_a_distribution_panel_that_raises_is_named_and_the_rest_still_draw, [True, 'Distribution panel guide_fraction did not draw'], id="a_distribution_panel_that_raises_is_named_and_the_rest_still_draw"),
    pytest.param(case_panels_that_drew_nothing_are_closed_rather_than_shown, [True, []], id="panels_that_drew_nothing_are_closed_rather_than_shown"),
    pytest.param(case_a_distribution_panel_that_cannot_be_saved_is_still_counted_as_drawn, [True, True], id="a_distribution_panel_that_cannot_be_saved_is_still_counted_as_drawn"),
    pytest.param(case_drawn_helpers_release_their_figures_and_preserve_the_callers, [], id="drawn_helpers_release_their_figures_and_preserve_the_callers"),
    pytest.param(case_a_plate_panel_that_cannot_be_imported_returns_false, [True, 'Could not load the plate panel'], id="a_plate_panel_that_cannot_be_imported_returns_false"),
    pytest.param(case_a_plate_panel_that_raises_is_reported_by_name, [True, 'The plate panel did not draw'], id="a_plate_panel_that_raises_is_reported_by_name"),
    pytest.param(case_a_plate_panel_with_nothing_on_it_is_closed, [True, []], id="a_plate_panel_with_nothing_on_it_is_closed"),
    pytest.param(case_a_plate_panel_that_cannot_be_saved_is_still_shown, [True, True], id="a_plate_panel_that_cannot_be_saved_is_still_shown"),
    pytest.param(case_an_empty_coefficient_table_writes_no_sheet, [True, True, True], id="an_empty_coefficient_table_writes_no_sheet"),
    pytest.param(case_a_sheet_that_cannot_be_built_is_reported_and_costs_nothing_else, [True, 'Could not draw the regression figure'], id="a_sheet_that_cannot_be_built_is_reported_and_costs_nothing_else"),
    pytest.param(case_a_sheet_whose_figure_cannot_be_closed_is_still_written, [True, 'Panel a, the volcano.\n', '2 panels, 1 not applicable'], id="a_sheet_whose_figure_cannot_be_closed_is_still_written"),
    pytest.param(case_house_style_panels_that_cannot_be_imported_draw_none, [0, 'Could not load the figure style'], id="house_style_panels_that_cannot_be_imported_draw_none"),
])
def test_12_ml_figures_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected

