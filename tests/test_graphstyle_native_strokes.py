"""Native export keeps cosmetic stroke widths physical across file resolutions."""
import numpy as np
import pytest

pytestmark=pytest.mark.qt


@pytest.mark.parametrize('dpi',[150,300,600])
def test_native_exported_frame_ink_keeps_its_point_width_and_restores_the_live_scene(qtbot,tmp_path,monkeypatch,dpi):
    """Pixel integration measured .39/.20/.10 pt for the same .6-pt frame; require exact device-pen scaling and ink within half an output pixel plus .1 pixel for crossing ticks, since Qt deliberately paints crisp axis lines without antialiasing."""
    from PIL import Image
    from PySide6.QtGui import QPen
    from pyqtgraph.exporters import ImageExporter
    from spacr.qt.widgets.fast_plots import PValueHistogram

    widget=PValueHistogram();qtbot.addWidget(widget)
    widget.set_p_values(np.geomspace(.00001,.95,80),bins=12)
    widget.resize(760,560);widget.show();qtbot.waitExposed(widget)
    widget._export_dpi=dpi
    before=[QPen(axis.pen()) for axis in widget.axis_items()]
    measured=[]
    original=ImageExporter.export

    def capture(exporter,*args,**kwargs):
        """Inspect the actual Qt pen inside the raster write, before restoration."""
        measured.append(widget.plot.getAxis('bottom').pen().widthF()*72./dpi)
        return original(exporter,*args,**kwargs)

    monkeypatch.setattr(ImageExporter,'export',capture)
    counts,edges=widget._counts.tobytes(),widget._edges.tobytes()
    path=tmp_path/f'frame_{dpi}.png';widget.export(str(path))
    rgb=np.asarray(Image.open(path).convert('RGB')).astype(float)
    darkness=255.-rgb.max(axis=2)
    black=(rgb.max(axis=2)-rgb.min(axis=2)<2.)&(darkness>20.)
    row=int(black.sum(axis=1).argmax())
    columns=np.flatnonzero(black[row]);left,right=int(columns.min()),int(columns.max())
    left+=int((right-left)*.2);right-=int((right-left)*.2)
    profile=darkness[:,left:right].mean(axis=1)/255.
    physical_width=float(profile[max(0,row-4):row+5].sum())*72./dpi
    assert measured==pytest.approx([.6],abs=1e-12)
    assert physical_width==pytest.approx(.6,abs=.6*72./dpi)
    assert [axis.pen() for axis in widget.axis_items()]==before
    assert (widget._counts.tobytes(),widget._edges.tobytes())==(counts,edges)


def test_native_svg_frame_uses_the_same_physical_point_width(qtbot,tmp_path):
    """A 72-dpi vector device must emit .6-point black frame strokes rather than unscaled .8-logical-pixel widths."""
    import xml.etree.ElementTree as ET
    from spacr.qt.widgets.fast_plots import PValueHistogram

    widget=PValueHistogram();qtbot.addWidget(widget)
    widget.set_p_values([.1,.2,.3],bins=3)
    path=tmp_path/'frame.svg';widget.export(str(path))
    root=ET.parse(path).getroot()
    widths=[float(element.attrib['stroke-width']) for element in root.iter()
            if element.attrib.get('stroke')=='#000000'
            and element.attrib.get('stroke-width') not in (None,'1')]
    assert widths and all(width==pytest.approx(.6,abs=1e-5) for width in widths)


def test_device_pen_scaling_restores_mixed_manual_pens_ranges_and_autorange_after_failure(qtbot):
    """An exporter failure must restore per-point, data-curve and axis pens, preserve noncosmetic manual pens and leave exact view limits and automatic-range choices intact."""
    import pyqtgraph as pg
    from PySide6.QtGui import QPen
    from spacr.figures.style import _native_file_strokes
    from spacr.qt.widgets.fast_plots import QQPlot

    widget=QQPlot();qtbot.addWidget(widget)
    widget.set_p_values([.001,.02,.2,.8])
    widget.resize(760,560);widget.show();qtbot.waitExposed(widget)
    cosmetic=pg.mkPen('#123456',width=1.7)
    fixed=pg.mkPen('#654321',width=.02,cosmetic=False)
    points=pg.ScatterPlotItem(x=[.2,.5],y=[.3,.4],pen=[cosmetic,fixed])
    widget.plot.addItem(points)
    view=widget.plot.getViewBox();view.updateAutoRange()
    ranges=[list(pair) for pair in view.viewRange()];automatic=list(view.state['autoRange'])
    axes=[QPen(axis.pen()) for axis in widget.axis_items()]
    pens=list(points.data['pen'])
    curves=[item for item in widget.plot.plotItem.items if isinstance(item,pg.PlotDataItem)]
    previous=[QPen(item.curve.opts['pen']) for item in curves]
    with pytest.raises(RuntimeError,match='injected paint failure'):
        with _native_file_strokes(widget.plot.plotItem,dpi=300) as applied:
            assert applied
            assert points.data['pen'][0].widthF()==pytest.approx(1.7*300/widget.logicalDpiX())
            assert points.data['pen'][1]==fixed
            raise RuntimeError('injected paint failure')
    assert [axis.pen() for axis in widget.axis_items()]==axes
    assert list(points.data['pen'])==pens
    assert [item.curve.opts['pen'] for item in curves]==previous
    assert view.viewRange()==ranges
    assert list(view.state['autoRange'])==automatic


def test_translated_scene_uses_its_declared_coordinate_resolution(qtbot):
    """A Matplotlib scene built in 300-dpi coordinates must retain those pen units rather than treating them as 96-dpi screen pixels."""
    from PySide6.QtGui import QPen
    from matplotlib.figure import Figure
    from spacr.figures import scene
    from spacr.figures.style import _native_file_strokes

    figure=Figure(figsize=(3.,3.));ax=figure.add_subplot();ax.plot([0.,1.],[0.,1.])
    widget,_report=scene.build_scene(figure,dpi=300)
    qtbot.addWidget(widget)
    items=[item for item in widget.ci.items if hasattr(item,'getAxis')]
    assert items
    axis=items[0].getAxis('bottom');before=QPen(axis.pen())
    with _native_file_strokes(widget.ci,dpi=300) as applied:
        assert applied
        assert axis.pen()==before
    with _native_file_strokes(widget.ci,dpi=150) as applied:
        assert applied
        assert axis.pen().widthF()==pytest.approx(before.widthF()/2.)
    assert axis.pen()==before
