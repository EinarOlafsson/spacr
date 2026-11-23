"""Translated axes retain actual source frame and tick presentation."""
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import to_hex

from spacr.figures import scene

pytestmark=pytest.mark.qt


@pytest.mark.parametrize('width',[.6,1.3,0.])
@pytest.mark.parametrize('mode',['screen','print'])
def test_translated_frame_uses_actual_source_spine_width_colour_and_alpha(qtbot,width,mode):
    """The translator hardcoded .8-point default ink for every frame; preserve each source width and manual colour/alpha through the existing export-colour owner without moving data."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    figure=Figure(figsize=(3.,3.));ax=figure.add_subplot()
    source,=ax.plot([0.,1.],[1.,2.],alpha=.6)
    before=source.get_xydata().copy()
    colours=['#0A77B8','#0A9878','#C878A8','#283868']
    edges=('bottom','left','top','right')
    for name,colour in zip(edges,colours):
        ax.spines[name].set_linewidth(width)
        ax.spines[name].set_edgecolor(colour)
        ax.spines[name].set_alpha(.4)
    FigureCanvasAgg(figure);figure.canvas.draw()
    widget,report=scene.build_scene(figure,mode=mode,dpi=150);qtbot.addWidget(widget)
    assert not report.missing
    plot=next(item for item in widget.ci.items if isinstance(item,pg.PlotItem))
    look=scene._Look(mode,dpi=150)
    for name in edges:
        pen=plot.getAxis(name).pen()
        if width==0.:
            assert pen.style()==Qt.PenStyle.NoPen
        else:
            assert pen.widthF()==pytest.approx(width*150./72.,abs=1e-12)
            expected=look.paint(to_hex(ax.spines[name].get_edgecolor()),'chrome')
            assert pen.color().name()==expected.lower()
            assert pen.color().alphaF()==pytest.approx(.4,abs=1/255)
    np.testing.assert_array_equal(source.get_xydata(),before)


def test_translated_tick_pens_lengths_text_ink_and_visibility_follow_the_source(qtbot):
    """Major ticks may differ from their frame; retain their actual source width, outward length, label ink and shared-axis hiding instead of Qt defaults."""
    import pyqtgraph as pg

    figure=Figure(figsize=(3.,3.));ax=figure.add_subplot();ax.plot([0.,1.],[1.,2.])
    ax.tick_params(axis='both',width=1.1,length=4.,colors='#283868',labelcolor='#0A9878',
        top=True,right=True,labeltop=False,labelright=False,direction='out')
    ax.spines['left'].set_visible(False)
    FigureCanvasAgg(figure);figure.canvas.draw()
    widget,report=scene.build_scene(figure,mode='screen',dpi=150);qtbot.addWidget(widget)
    assert not report.missing
    plot=next(item for item in widget.ci.items if isinstance(item,pg.PlotItem))
    for name in ('bottom','left','top','right'):
        axis=plot.getAxis(name)
        assert axis.tickPen().widthF()==pytest.approx(1.1*150./72.,abs=1e-12)
        assert axis.tickPen().color().name()=='#283868'
        assert axis.textPen().color().name()=='#0a9878'
        assert axis.style['tickLength']==round(4.*150./72.)
    from PySide6.QtCore import Qt

    assert plot.getAxis('left').isVisible()
    assert plot.getAxis('left').pen().style()==Qt.PenStyle.NoPen
    assert not plot.getAxis('top').style['showValues']
    assert not plot.getAxis('right').style['showValues']


@pytest.mark.parametrize('visibility',['visible','transparent','hidden'])
def test_translated_title_and_axis_label_retain_their_actual_source_ink(qtbot,visibility):
    """Actual source text colours and alpha must survive screen-mode translation instead of being replaced by default white ink; explicitly hidden labels stay hidden."""
    import pyqtgraph as pg
    from PySide6.QtGui import QColor,QTextCursor

    figure=Figure(figsize=(3.,3.));ax=figure.add_subplot();ax.plot([0.,1.],[1.,2.])
    ax.set_xlabel('Coordinate',color='#0A77B8',alpha=.4)
    ax.set_ylabel('Measurement',color='#C878A8',alpha=.7)
    ax.set_title('Source title',color='#0A9878',alpha=.6)
    if visibility=='transparent':ax.xaxis.label.set_alpha(0.)
    if visibility=='hidden':ax.title.set_visible(False);ax.yaxis.label.set_visible(False)
    FigureCanvasAgg(figure);figure.canvas.draw()
    widget,report=scene.build_scene(figure,mode='screen',dpi=150);qtbot.addWidget(widget)
    assert not report.missing
    plot=next(item for item in widget.ci.items if isinstance(item,pg.PlotItem))
    sources=[(plot.getAxis('bottom').label,ax.xaxis.label)]
    if visibility=='hidden':
        assert not plot.titleLabel.text
        assert not plot.getAxis('left').labelText
    else:
        sources += [(plot.getAxis('left').label,ax.yaxis.label),(plot.titleLabel.item,ax.title)]
    for item,source in sources:
        cursor=QTextCursor(item.document())
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        cursor.movePosition(QTextCursor.MoveOperation.Right,QTextCursor.MoveMode.KeepAnchor)
        actual=cursor.charFormat().foreground().color()
        expected=QColor(source.get_color())
        assert actual.name()==expected.name()
        assert actual.alphaF()==pytest.approx(source.get_alpha(),abs=1/255)
