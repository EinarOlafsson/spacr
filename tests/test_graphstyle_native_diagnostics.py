"""Native diagnostics use shared presentation without changing numerical coordinates."""
import numpy as np
import pytest

pytestmark=pytest.mark.qt


@pytest.mark.parametrize('name',['p_values','effects','qq','residuals','scale_location','influence'])
def test_native_diagnostics_register_actual_marks_and_preserve_source_arrays(qtbot,name):
    """The diagnostic branches previously bypassed shared registration; require square export, actual .6 marks, black reference lines and unchanged histogram/quantile/residual/influence values."""
    from spacr.figures import style
    from spacr.qt.widgets import fast_plots as native

    values=np.geomspace(.0001,.9,60)
    fitted=np.linspace(.1,2.,60)
    residual=np.random.default_rng(271).normal(0.,.2,60)
    before=[array.tobytes() for array in (values,fitted,residual)]
    if name=='p_values':
        widget=native.PValueHistogram();widget.set_p_values(values,bins=12)
        expected=np.histogram(values,bins=12,range=(0.,1.))
        np.testing.assert_array_equal(widget._counts,expected[0])
        np.testing.assert_array_equal(widget._edges,expected[1])
    elif name=='effects':
        widget=native.EffectDistribution();widget.set_effects(residual,bins=12)
        expected=np.histogram(residual,bins=12)
        np.testing.assert_array_equal(widget._counts,expected[0])
        np.testing.assert_array_equal(widget._edges,expected[1])
    elif name=='qq':
        widget=native.QQPlot();widget.set_p_values(values)
        points=next(item for item in widget.plot.plotItem.items if isinstance(item,native.pg.ScatterPlotItem))
        x,y=points.getData()
        np.testing.assert_array_equal(x,-np.log10((np.arange(1,61)-.5)/60))
        np.testing.assert_array_equal(y,-np.log10(values))
    elif name=='residuals':
        widget=native.ResidualPlot();widget.set_residuals(fitted,residual)
    elif name=='scale_location':
        widget=native.ScaleLocationPlot();widget.set_scale_location(fitted,residual)
    else:
        widget=native.InfluencePlot();widget.set_influence(values,residual,np.linspace(0.,.1,60),n_params=3)
    qtbot.addWidget(widget);widget.resize(760,560);widget.show();qtbot.waitExposed(widget)
    assert widget._spacr_style_spec.kind=='continuous'
    assert dict(widget._spacr_style_spec.legend)['reference']
    assert not widget.plot.plotItem.titleLabel.text
    with widget._held_at_policy_geometry() as applied:
        assert applied
        observed=widget._spacr_style_observation()
        assert observed['width_inches']==pytest.approx(observed['height_inches'],abs=.5/widget.logicalDpiX())
        assert observed['marks'] and all(mark['role']!='unregistered' for mark in observed['marks'])
        assert all(colour[3]*mark['opacity']==pytest.approx(.6,abs=1/65535) for mark in observed['marks'] for colour in mark['colours'])
        references=[mark for mark in observed['marks'] if mark['role']=='reference']
        for mark in references:
            assert all(colour[:3]==pytest.approx([0.]*3,abs=1/65535) for colour in mark['colours'])
    assert [array.tobytes() for array in (values,fitted,residual)]==before


def test_native_diagnostic_redraw_keeps_the_caption_and_an_explicit_manual_title(qtbot):
    """Moving a default title outside the data area must retain its caption across redraws and preserve a subsequent explicit title choice."""
    from spacr.qt.widgets.fast_plots import PValueHistogram

    widget=PValueHistogram();qtbot.addWidget(widget)
    widget.set_p_values([.1,.2,.3],bins=3)
    caption=dict(widget._spacr_style_spec.legend)['caption']
    assert 'p-value distribution' in caption
    widget.set_p_values([.2,.3,.4],bins=3)
    assert 'p-value distribution' in dict(widget._spacr_style_spec.legend)['caption']
    widget.plot.setTitle('Explicit manual title')
    widget.set_p_values([.3,.4,.5],bins=3)
    assert widget.plot.plotItem.titleLabel.text=='Explicit manual title'


def test_manual_native_reference_colour_is_kept_and_diagnosed_without_changing_its_position(qtbot):
    """The real native zero-reference line defaults to shared black; a later explicit magenta pen remains exact and produces the reference-line diagnostic, with unchanged residual inputs and location."""
    import pyqtgraph as pg
    from spacr.figures import style
    from spacr.qt.widgets.fast_plots import ResidualPlot

    widget = ResidualPlot()
    qtbot.addWidget(widget)
    fitted = np.array([1., 2., 3., 4.])
    residual = np.array([-.2, .1, -.1, .2])
    widget.set_residuals(fitted, residual)
    line = next(item for item in widget.plot.plotItem.items if isinstance(item, pg.InfiniteLine))
    position = line.value()
    before = fitted.tobytes(), residual.tobytes()
    pen = pg.mkPen(line.pen)
    pen.setColor(pg.mkColor('#C878A8'))
    line.setPen(pen)
    result = style.finalize_figure(widget, diagnostics=False)
    assert 'Draw reference lines in black; preserve explicit colours with a diagnostic' in result.problems
    assert line.pen.color().name() == '#c878a8' and line.value() == position
    assert before == (fitted.tobytes(), residual.tobytes())
