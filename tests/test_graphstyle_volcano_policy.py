"""Volcano presentation retains exact effect, evidence and explicit opacity encodings."""
import numpy as np
import pandas as pd
import pytest

pytestmark=pytest.mark.qt


@pytest.mark.parametrize('opacity_mode',['default','q','column'])
def test_volcano_uses_shared_defaults_and_preserves_explicit_opacity_encodings(qtbot,opacity_mode):
    """Default points and thresholds use .6 shared marks; explicit evidence/category opacity retains its existing values and remains an honest diagnostic when incompatible with mandatory opacity."""
    from spacr.figures import style
    from spacr.qt.widgets import fast_plots as native

    frame=pd.DataFrame({'feature':[f'gene_{i}' for i in range(36)],
        'coefficient':np.linspace(-1.,1.,36),'p_value':np.geomspace(.00001,.9,36),
        'category':np.resize(['a','b'],36),'opacity_group':np.resize(['u','v','w'],36)})
    before=frame.to_csv(index=False)
    widget=native.VolcanoPlot();qtbot.addWidget(widget)
    if opacity_mode=='q':
        widget._q_mark='opacity'
    kwargs={'category_column':'category','opacity_column':'opacity_group'} if opacity_mode=='column' else {}
    assert widget.set_results(frame,drop_untested=False,**kwargs)==len(frame)
    widget.resize(760,560);widget.show();qtbot.waitExposed(widget)
    points=next(item for item in widget.plot.plotItem.items if isinstance(item,native.pg.ScatterPlotItem))
    x,y=points.getData()
    np.testing.assert_array_equal(x,frame.coefficient)
    np.testing.assert_array_equal(y,-np.log10(frame.p_value))
    assert widget._spacr_style_spec.kind=='continuous'
    assert dict(widget._spacr_style_spec.legend)['reference']
    actual=[point.brush().color().alpha() for point in points.points()]
    if opacity_mode=='default':
        assert set(actual)=={153}
    else:
        brushes=[native.pg.mkBrush(native.HIGHLIGHT)]*len(frame)
        expected=(widget._q_opacity(brushes,widget._q_values,len(frame)) if opacity_mode=='q'
                  else widget._categorical_opacity(brushes,frame.opacity_group,len(frame)))
        assert actual==[brush.color().alpha() for brush in expected]
        assert len(set(actual))>1
    with widget._held_at_policy_geometry() as applied:
        assert applied
        observed=widget._spacr_style_observation()
        assert observed['width_inches']==pytest.approx(observed['height_inches'],abs=.5/widget.logicalDpiX())
        assert all(mark['role']!='unregistered' for mark in observed['marks'])
        report=style.finalize_figure(widget)
        assert any('sixty percent' in problem for problem in report.problems)==(opacity_mode!='default')
    assert frame.to_csv(index=False)==before
