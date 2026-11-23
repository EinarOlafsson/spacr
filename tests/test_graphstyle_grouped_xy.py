"""Native continuous grouped plots retain source coordinates and use shared presentation."""
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.qt


@pytest.mark.parametrize('kind', ['line','scatter'])
@pytest.mark.parametrize('negative', [False,True])
def test_grouped_xy_exports_square_data_area_without_changing_coordinates_or_manual_choices(qtbot,kind,negative):
    """The old numeric-group branch skipped shared registration and exported an opaque line; test exact original sorting, source rows, honest line origins, reversible square geometry and .6 marks."""
    from spacr.figures import style
    from spacr.qt.widgets.grouped_plot import GroupedPlot,PlotSpec

    frame = pd.DataFrame({'coordinate':[3.,1.,4.,2.], 'value':np.array([1.2,1.,1.3,1.1])*(-1. if negative else 1.)})
    frame.attrs['units']={'coordinate':'min','value':'a.u.'}
    frame.attrs['reference']='Synthetic measured coordinate origin'
    before=frame.to_csv(index=False)
    widget=GroupedPlot(PlotSpec(frame,'value',group='coordinate',kind=kind))
    qtbot.addWidget(widget); widget.resize(760,560); widget.show(); qtbot.waitExposed(widget)
    items=widget.plot.listDataItems()
    assert len(items)==1
    x,y=items[0].getData()
    order=np.argsort(frame.coordinate) if kind=='line' else np.arange(len(frame))
    np.testing.assert_array_equal(x,frame.coordinate.to_numpy()[order])
    np.testing.assert_array_equal(y,frame.value.to_numpy()[order])
    assert widget._spacr_style_spec.kind=='continuous'
    assert not style._quantity_unit_problems(widget._spacr_style_spec)
    if kind=='line':
        low,high=widget.plot.getViewBox().viewRange()[1]
        assert high>=0. if negative else low==0.
        assert low<frame.value.min() if negative else high>frame.value.max()
    geometry=widget.plot.plotItem.geometry()
    ranges=widget.plot.getViewBox().viewRange()
    with widget._held_at_policy_geometry() as applied:
        assert applied
        area=widget.plot.getViewBox().geometry()
        assert area.width()==pytest.approx(area.height(),abs=.5)
        observation=widget._spacr_style_observation()
        assert observation['marks']
        assert all(mark['role']!='unregistered' for mark in observation['marks'])
        assert all(colour[3]*mark['opacity']==pytest.approx(.6,abs=1/255) for mark in observation['marks'] for colour in mark['colours'])
    assert widget.plot.plotItem.geometry()==geometry
    np.testing.assert_array_equal(widget.plot.getViewBox().viewRange(),ranges)
    assert widget.spec.frame.to_csv(index=False)==before
    widget.set_export_size(100.,50.)
    with widget._held_at_policy_geometry() as applied:
        assert not applied
        assert widget.export_size()==(100.,50.)


def test_continuous_numeric_ticks_use_normal_painting_after_a_categorical_view(qtbot,monkeypatch):
    """A numeric XY redraw must clear stale category names and send complete numeric text to the ordinary Qt axis painter instead of rotating it as condition labels."""
    import pyqtgraph as pg
    from PySide6.QtGui import QPainter,QPicture
    from spacr.qt.widgets.grouped_plot import GroupedPlot,PlotSpec

    widget=GroupedPlot();qtbot.addWidget(widget)
    axis=widget.plot.getAxis('bottom')
    axis.set_categories(['old_a','old_b'])
    widget.show_spec(PlotSpec(pd.DataFrame({'x':[1.,2.,3.],'y':[2.,3.,4.]}),'y',group='x',kind='scatter'))
    widget.resize(760,560);widget.show();qtbot.waitExposed(widget)
    assert axis._category_labels==()
    calls=[]
    original=pg.AxisItem.drawPicture

    def record(self,painter,axis_spec,tick_specs,text_specs):
        """Observe the actual unrotated text passed to the normal Qt renderer."""
        calls.append([text for _rect,_flags,text in text_specs])
        return original(self,painter,axis_spec,tick_specs,text_specs)

    monkeypatch.setattr(pg.AxisItem,'drawPicture',record)
    picture=QPicture();painter=QPainter(picture)
    try:
        specs=axis.generateDrawSpecs(painter)
        assert specs is not None and specs[2]
        axis.drawPicture(painter,*specs)
    finally:
        painter.end()
    assert calls==[[text for _rect,_flags,text in specs[2]]]
