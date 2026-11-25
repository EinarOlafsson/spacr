"""Measured native-renderer opacity, boxed-frame and image preservation regressions."""
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from spacr.figures import scene


@pytest.fixture
def translated(qapp):
    """Build real scene items and release each widget after its measured comparison."""
    widgets=[]

    def build(figure):
        """Translate the exact source figure and retain its widget for bounded cleanup."""
        FigureCanvasAgg(figure);figure.canvas.draw()
        widget,report=scene.build_scene(figure,mode='screen',dpi=150)
        widgets.append(widget)
        assert not report.missing
        return widget

    yield build
    for widget in widgets:
        widget.close();widget.deleteLater()
    qapp.processEvents()


def _plots(widget):
    """Return real plot items, excluding labels and layout support objects."""
    import pyqtgraph as pg

    return [item for item in widget.ci.items if isinstance(item,pg.PlotItem)]


def _scatters(widget):
    """Return translated data scatters without treating a curve as a second source row."""
    import pyqtgraph as pg

    return [item for plot in _plots(widget) for item in plot.items if isinstance(item,pg.ScatterPlotItem)]


def test_collection_alpha_is_not_multiplied_twice(translated):
    """The baseline .6/.6 scatter measured92/255; target153/255 allows half an 8-bit alpha channel."""
    figure=Figure();ax=figure.add_subplot(111)
    source=ax.scatter([0.,1.,2.],[1.,2.,1.5],color=(.1,.4,.7,.6),alpha=.6)
    values=source.get_offsets().copy()
    widget=translated(figure)
    brushes=_scatters(widget)[0].data['brush']
    assert [brush.color().alpha() for brush in brushes]==pytest.approx([153]*3,abs=.5)
    np.testing.assert_array_equal(source.get_offsets(),values)


def test_line_markers_keep_source_alpha(translated):
    """The old marker branch omitted alpha entirely; source .6 must produce153/255, within half a channel."""
    figure=Figure();ax=figure.add_subplot(111)
    source,=ax.plot([0.,1.],[1.,2.],marker='o',alpha=.6,color='#0A77B8')
    values=source.get_xydata().copy()
    widget=translated(figure)
    brush=_scatters(widget)[0].opts['brush']
    assert brush.color().alpha()==pytest.approx(153,abs=.5)
    np.testing.assert_array_equal(source.get_xydata(),values)


@pytest.mark.parametrize('boxed',[True,False])
def test_translated_spines_match_the_source_axes(translated,boxed):
    """Baseline hid both top/right spines; preserve each measured source visibility exactly."""
    figure=Figure();ax=figure.add_subplot(111)
    ax.spines['top'].set_visible(boxed);ax.spines['right'].set_visible(boxed)
    ax.plot([0.,1.],[1.,2.],alpha=.6)
    plot=_plots(translated(figure))[0]
    assert plot.getAxis('top').isVisible()==boxed
    assert plot.getAxis('right').isVisible()==boxed
    assert plot.getAxis('left').isVisible() and plot.getAxis('bottom').isVisible()


def test_heatmap_opacity_and_masked_values_survive_translation(translated):
    """The .6 heatmap measured1.0 in Qt; preserve .6 within half a channel and keep missing cells distinct from zero."""
    import pyqtgraph as pg

    figure=Figure();ax=figure.add_subplot(111)
    values=np.ma.array([[0,1],[2,3]],mask=[[False,True],[False,False]])
    source=ax.imshow(values,alpha=.6,cmap='Blues',vmin=0,vmax=3)
    before=source.get_array().copy()
    plot=_plots(translated(figure))[0]
    image=next(item for item in plot.items if isinstance(item,pg.ImageItem))
    assert image.opacity()==pytest.approx(.6,abs=.5/255)
    assert np.isnan(image.image.T[0,1]) and image.image.T[0,0]==0
    np.testing.assert_array_equal(source.get_array().data,before.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(source.get_array()),np.ma.getmaskarray(before))



def test_scene_fonts_use_bundled_family_actual_light_and_bold_weights(qapp):
    """The source adapter used Qt5 values50/75 and no family; require real Open Sans Light300 and Bold700 at the exact 144-dpi pixel size."""
    from PySide6.QtGui import QFont,QFontInfo

    look=scene._Look('screen',dpi=144)
    body=scene._font(8.,look,family='Open Sans',weight='light')
    letter=scene._font(10.,look,family='Open Sans',weight='bold')
    assert QFontInfo(body).family()=='Open Sans' and QFontInfo(body).weight()==QFont.Light
    assert QFontInfo(letter).family()=='Open Sans' and QFontInfo(letter).weight()==QFont.Bold
    assert body.pixelSize()==16 and letter.pixelSize()==20


def test_translated_registered_geometry_is_measured_and_never_inherits_source_certification(qapp):
    """A .44-inch source panel cannot inherit certification on a stretched native grid; explicitly decline changed geometry and preserve all source observations."""
    from spacr.figures import style

    with style.figure_style('print'):
        figure=Figure();ax=figure.add_subplot()
        arrays=[np.array([1.]),np.array([2.])]
        style.condition_plot(ax,arrays,control=0)
        ax.set_xlabel('Condition');ax.set_ylabel('Value (a.u.)')
        spec=style.resolve_style('print','categorical',conditions=('reference','treated'),sample_counts=(1,1),control=0,
            legend={'caption':'Synthetic values.','reference':'reference','unit':'a.u.',
                    'quantities':{'x':'condition','y':'value'}})
        style.apply_axes_style(ax,spec)
        source=[value.tobytes() for value in arrays]
        canvas=figure.canvas
        widget,report=scene.build_scene(figure,mode='screen',dpi=150)
        try:
            assert 'registered physical axes geometry' in report.missing
            assert not report.complete and not widget._spacr_style_diagnostics.compliant
            assert widget._spacr_source_style_diagnostics is not widget._spacr_style_diagnostics
            assert ax._spacr_style_spec is spec and figure.canvas is canvas
            assert [value.tobytes() for value in arrays]==source
        finally:
            widget.close();widget.deleteLater();qapp.processEvents()



@pytest.mark.parametrize('dpi,dots_per_metre',[(None,11811),(150,5906)])
def test_translated_scene_files_keep_source_inches_and_honest_png_resolution(qapp,tmp_path,dpi,dots_per_metre):
    """The old scene PNG kept96dpi and vector pages used160mm; require a two-inch page (144PDFpoints), matching raster pixels, and exact rounded300/150-dpi metadata without changing source arrays."""
    import re
    import struct
    from tests.test_graphstyle_native_geometry import _png_physical_resolution

    figure=Figure(figsize=(2.,2.));ax=figure.add_subplot()
    line,=ax.plot([0.,1.],[.2,.8],color='#0A77B8',alpha=.6)
    before=line.get_xydata().tobytes()
    options={} if dpi is None else {'dpi':dpi}
    widget,report=scene.build_scene(figure,mode='screen',**options)
    try:
        assert report.complete,report.missing
        assert widget._spacr_scene_inches==(2.,2.)
        assert widget._spacr_scene_dpi==(300 if dpi is None else dpi)
        png=tmp_path/'native.png';pdf=tmp_path/'native.pdf'
        assert scene.export_scene(widget,png)==str(png)
        assert scene.export_scene(widget,pdf)==str(pdf)
        assert _png_physical_resolution(png)==(dots_per_metre,dots_per_metre,1)
        expected=(600,600) if dpi is None else (300,300)
        assert struct.unpack('>II',png.read_bytes()[16:24])==expected
        match=re.search(rb'/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]',pdf.read_bytes())
        assert match is not None
        assert tuple(float(value) for value in match.groups())==(144.,144.)
        assert line.get_xydata().tobytes()==before
    finally:
        widget.close();widget.deleteLater();qapp.processEvents()
