"""Read-only physical panel checks retain manual sizes and all scientific values."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from spacr.figures import style

pytestmark = pytest.mark.qt

MESSAGE = 'Use the shared 40 mm panel dimension; preserve explicit manual sizes with a diagnostic'


@pytest.mark.parametrize('kind', ['scatter', 'bar', 'horizontal', 'split'])
@pytest.mark.parametrize('scale', [1., 1.5])
def test_matplotlib_checks_absolute_panel_size_even_when_shape_and_slots_are_correct(kind, scale):
    """A 60-mm square previously passed the aspect check; compare 40/60-mm fixtures with the existing one-device-pixel margin while retaining exact manual geometry and data."""
    comparison = {'orientation': 'horizontal'} if kind == 'horizontal' else {'split_fraction': .4} if kind == 'split' else {}
    spec = style.resolve_style('print', 'bar' if kind == 'horizontal' else 'scatter' if kind == 'split' else kind,
        conditions=('reference', 'treated') if kind in ('bar', 'horizontal') else None,
        comparison=comparison, legend={'caption': 'Synthetic values.', 'reference': 'reference', 'unit': 'dimensionless'})
    geometry = style.axes_geometry(spec)
    width, height = geometry.width_inches, geometry.height_inches
    if kind == 'bar':
        height *= scale
    elif kind == 'horizontal':
        width *= scale
    else:
        width *= scale
        height *= scale
    figure = Figure(figsize=(8., 8.), dpi=100.)
    ax = figure.add_axes([.1, .1, width / 8., height / 8.])
    source = ax.scatter([0., 1.], [.2, .8], color=style.Palette.ACCENT, alpha=.6)
    ax._spacr_style_spec = spec
    before = source.get_offsets().copy(), ax.get_position().bounds, figure.get_size_inches().copy()
    result = style.finalize_figure(figure, diagnostics=False)
    assert (MESSAGE in result.problems) == (scale != 1.)
    np.testing.assert_array_equal(source.get_offsets(), before[0])
    assert ax.get_position().bounds == before[1]
    np.testing.assert_array_equal(figure.get_size_inches(), before[2])


@pytest.mark.parametrize('scale', [1., 1.5])
def test_native_checks_the_actual_categorical_panel_height_without_changing_manual_geometry(qtbot, scale):
    """The native policy fixture measures 40 mm within half a logical pixel; a manual 60-mm height must be diagnosed with the existing one-pixel margin, without changing its .66-inch slots or coefficient values."""
    from spacr.qt.widgets.fast_plots import EffectRankPlot

    widget = EffectRankPlot()
    qtbot.addWidget(widget)
    widget.resize(720, 480)
    frame = pd.DataFrame({'gene': ['a', 'b', 'c'], 'coefficient': [1., 2., 3.], 'std_err': [.1, .2, .3], 'q_value': [.1, .2, .3]})
    widget.set_results(frame, label_column='gene', drop_untested=False)
    widget.show()
    qtbot.waitExposed(widget)
    source = widget.frame().to_csv(index=False)
    with widget._held_at_policy_geometry():
        plot = widget.plot.plotItem
        view = widget.plot.getViewBox()
        before = view.geometry()
        view.setGeometry(before.adjusted(0., 0., 0., before.height() * (scale - 1.)))
        geometry = view.geometry()
        observed = widget._spacr_style_observation()
        assert observed['width_inches'] == pytest.approx(.66, abs=.5/widget.logicalDpiX())
        assert observed['height_inches'] == pytest.approx(40/25.4*scale, abs=.5/widget.logicalDpiX())
        result = style.finalize_figure(widget, diagnostics=False)
        assert (MESSAGE in result.problems) == (scale != 1.)
        assert view.geometry() == geometry
    assert widget.frame().to_csv(index=False) == source
