"""Native coefficient geometry, font resolution and reversible interactive export state."""
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.qt


@pytest.fixture
def rank(qtbot):
    """Build a real rank widget with identifiable rows and unequal coefficient order."""
    from spacr.qt.widgets.fast_plots import EffectRankPlot

    widget=EffectRankPlot();qtbot.addWidget(widget)
    frame=pd.DataFrame({'gene':['LONG_GENE_ALPHA','LONG_GENE_BETA','LONG_GENE_GAMMA'],
        'coefficient':[-2.,.5,1.25],'std_err':[.2,.1,.3],'q_value':[.001,.8,.02]})
    widget.resize(720,480);widget.set_results(frame,label_column='gene',drop_untested=False)
    widget.show();qtbot.waitExposed(widget)
    return widget


def test_native_categorical_slots_use_the_shared_physical_geometry(rank):
    """The v2 fixture exported 1000 pixels regardless of condition count; require .22-inch slots within half a display pixel."""
    dpi=float(rank.logicalDpiX())
    source=rank._frame.to_csv(index=False).encode()
    with rank._held_at_policy_geometry() as applied:
        assert applied
        area=rank.plot.getViewBox().geometry()
        assert area.width()/dpi==pytest.approx(3*.22,abs=.5/dpi)
        assert area.height()/dpi==pytest.approx(40/25.4,abs=.5/dpi)
        assert rank._export_dpi==300
    assert rank._frame.to_csv(index=False).encode()==source


def test_native_export_restores_interactive_layout_even_after_failure(rank):
    """Captured v2 scene and page settings must remain exactly unchanged when an exporter raises."""
    item=rank.plot.plotItem
    before=(item.geometry(),item.layout.getContentsMargins(),rank.export_size(),rank._export_dpi)
    with pytest.raises(RuntimeError,match='injected exporter failure'):
        with rank._held_at_policy_geometry():
            raise RuntimeError('injected exporter failure')
    assert (item.geometry(),item.layout.getContentsMargins(),rank.export_size(),rank._export_dpi)==before


def test_native_category_labels_use_complete_names_and_bundled_light_font(rank):
    """The v2 preview retained identities; require all complete long names and an actual Open Sans Light face at eight points."""
    from PySide6.QtGui import QFont, QFontInfo

    axis=rank.plot.getAxis('bottom')
    assert axis._category_labels==('LONG_GENE_ALPHA','LONG_GENE_GAMMA','LONG_GENE_BETA')
    font=axis.style['tickFont'];resolved=QFontInfo(font)
    assert resolved.family()=='Open Sans'
    assert resolved.weight()==QFont.Light
    assert font.pointSizeF()>=8.


def test_manual_page_shape_keeps_its_explicit_geometry(rank):
    """An explicit square page is a manual appearance choice; the automatic categorical adapter must leave it intact."""
    rank.set_canvas_shape('square')
    before=rank.plot.plotItem.geometry()
    with rank._held_at_policy_geometry() as applied:
        assert not applied
        assert rank.plot.plotItem.geometry()==before


@pytest.mark.parametrize("count", [1,20,21,1000,1001])
def test_native_automatic_forms_preserve_observations_and_dense_point_rule(qtbot,count):
    """Before used boxes at every n and first auto previews had outline-only violins; require preserved points through 999 and a filled interior with no points from 1000."""
    import pyqtgraph as pg
    from spacr.qt.widgets.grouped_plot import GroupedPlot,PlotSpec

    frame=pd.DataFrame({'group':['reference']*count,'value':np.linspace(1.,2.,count)})
    source=frame.to_csv(index=False).encode()
    widget=GroupedPlot(PlotSpec(frame,'value',group='group',background='reference'))
    qtbot.addWidget(widget)
    assert widget.spec.kind=='auto'
    scatters=[item for item in widget.plot.plotItem.items if isinstance(item,pg.ScatterPlotItem)]
    assert sum(len(item.points()) for item in scatters)==(count if count<1000 else 0)
    for item in scatters:
        assert all(point.brush().color().name()=='#8c8c8c' for point in item.points())
        assert all(point.brush().color().alpha()==153 for point in item.points())
    bars=[item for item in widget.plot.plotItem.items if isinstance(item,pg.BarGraphItem)]
    if count<=20:
        assert len(bars)==1 and np.array_equal(bars[0].opts['height'],[frame.value.mean()])
    elif count<1000:
        assert len(bars)==1 and np.array_equal(bars[0].opts['y0'],[frame.value.quantile(.25)])
    else:
        assert not bars
        from PySide6.QtCore import QPointF

        curves=[item for item in widget.plot.plotItem.items if isinstance(item,pg.PlotCurveItem) and len(item.xData)>10]
        assert curves
        assert any(item.opts['fillLevel']=='enclosed' and
                   item._getFillPath().contains(QPointF(0.,float(frame.value.median()))) and
                   item.opts['brush'].color().alpha()==153 for item in curves)
    assert widget.spec.frame.to_csv(index=False).encode()==source
    assert widget.plot.getAxis('bottom')._category_labels==('reference',)


def test_native_automatic_caption_retains_the_selected_spread(qtbot):
    """Automatic mean bars still carry the existing SD whiskers; the external caption must identify that unchanged quantity."""
    from spacr.qt.widgets.grouped_plot import GroupedPlot,PlotSpec
    from spacr.figures.spread import spread_label

    frame=pd.DataFrame({'group':['reference']*20,'value':np.linspace(1.,2.,20)})
    widget=GroupedPlot(PlotSpec(frame,'value',group='group',background='reference',spread='sd'))
    qtbot.addWidget(widget)
    assert spread_label('sd') in widget._headline


def test_manual_free_page_size_is_preserved_by_the_native_adapter(rank):
    """An explicitly chosen 100-by50-mm page remains exact even with a free canvas; policy geometry must not replace it."""
    rank.set_export_size(100.,50.)
    before=rank.plot.plotItem.geometry()
    with rank._held_at_policy_geometry() as applied:
        assert not applied
        assert rank.export_size()==(100.,50.)
        assert rank.plot.plotItem.geometry()==before


def test_native_colour_change_preserves_explicit_font_choices(rank):
    """The old colour setter recreated the bundled font and discarded a manual family/size; independent choices must remain exact."""
    from PySide6.QtGui import QFont

    font=QFont('DejaVu Sans',11);font.setWeight(QFont.DemiBold)
    axis=rank.plot.getAxis('bottom');axis.setStyle(tickFont=font)
    axis.setLabel('Gene',**{'font-family':'DejaVu Sans','font-size':'11pt'})
    rank.plot.setTitle('Manual title',family='DejaVu Sans',size='13pt',bold=True)
    rank.set_font_colour('#123456')
    assert axis.style['tickFont']==font
    assert axis.labelStyle['font-family']=='DejaVu Sans'
    title=rank.plot.plotItem.titleLabel
    assert title.opts['family']=='DejaVu Sans' and title.opts['size']=='13pt' and title.opts['bold'] is True


def test_explicit_native_file_colours_reach_the_writer_unchanged(rank,monkeypatch,tmp_path):
    """The first dark native preview saved black chrome despite explicit white ink; inspect the actual write boundary and exact restoration."""
    from PySide6.QtGui import QPen

    axis=rank.plot.getAxis('bottom');before=(QPen(axis.pen()),QPen(axis.textPen()))
    observed=[]

    def capture(item,path,width,height,exporters):
        """Read actual pens and ground at the final output boundary without changing data."""
        observed.append((axis.pen().color().name(),axis.textPen().color().name(),rank._export_ground().name()))

    monkeypatch.setattr(rank,'_write_export',capture)
    rank.export_styled(str(tmp_path/'manual.svg'),ink='#FFFFFF',background='#202630')
    assert observed==[('#ffffff','#ffffff','#202630')]
    assert (axis.pen(),axis.textPen())==before
    assert rank._explicit_file_colours is False


def test_temporary_native_file_styling_restores_fonts_and_each_mixed_pen_on_failure(rank):
    """Before stored one line width and a nullable font preference; preserve every actual pen and font when a styled export raises."""
    from PySide6.QtGui import QFont,QPen

    axes=rank.axis_items()
    before_fonts=[QFont(axis.style['tickFont']) if axis.style.get('tickFont') is not None else None for axis in axes]
    lines=rank.line_items();before_pens=[QPen(rank._pen_of(item)) for item in lines]
    title=rank.plot.plotItem.titleLabel;before_title=dict(title.opts)
    with pytest.raises(RuntimeError,match='injected styled export failure'):
        with rank._dressed_for_the_file(font_size=14,line_width=3.,text_colour='#123456',line_colour='#654321'):
            assert rank.plot.getAxis('bottom').style['tickFont'].pointSize()==14
            raise RuntimeError('injected styled export failure')
    assert [axis.style.get('tickFont') for axis in axes]==before_fonts
    assert [rank._pen_of(item) for item in lines]==before_pens
    assert title.opts==before_title


def test_native_shared_finalizer_observes_manual_font_and_opacity_without_restoring_defaults(rank):
    """Before native exports bypassed finalization; detect actual DejaVu Sans and .2 opacity while preserving both choices and exact coefficients."""
    from PySide6.QtGui import QFont
    from spacr.figures.style import finalize_figure
    import pyqtgraph as pg

    before=rank._frame.to_csv(index=False).encode()
    axis=rank.plot.getAxis('bottom')
    font=QFont('DejaVu Sans',11);axis.setStyle(tickFont=font)
    scatter=next(item for item in rank.plot.plotItem.items if isinstance(item,pg.ScatterPlotItem))
    scatter.setOpacity(.2)
    result=finalize_figure(rank)
    assert any('bundled Open Sans' in value for value in result.problems)
    assert 'Draw every data mark at sixty percent opacity' in result.problems
    assert not result.compliant
    assert axis.style['tickFont']==font and scatter.opacity()==.2
    assert rank._frame.to_csv(index=False).encode()==before


def test_native_snapshot_keeps_policy_actual_fonts_and_independent_pen_styles(rank):
    """Native settings previously retained only column names; capture actual font, black CI pen and measured viewport in JSON without touching the scene."""
    import json
    from spacr.figures.style import POLICY_VERSION

    axis=rank.plot.getAxis('bottom')
    before=axis.pen()
    saved=rank.export_settings()
    json.dumps(saved)
    assert saved['presentation']['policy_version']==POLICY_VERSION
    actual=saved['appearance']
    assert actual['backend']=='pyqtgraph'
    assert actual['ranges']==rank.plot.getViewBox().viewRange()
    bottom=next(edge for edge in actual['frame'] if edge['edge']=='bottom')
    assert bottom['tick_font']['family']=='Open Sans'
    assert bottom['tick_font']['size']>=8.
    assert any(pen['colour'][:3]==[0.,0.,0.] for mark in actual['marks'] for pen in mark['pens'])
    assert actual['text_bounds_checked'] is True
    assert 'text bounds' in saved['presentation_diagnostics']['checks']
    assert 'actual native data limits' in saved['presentation_diagnostics']['checks']
    assert any('plotted quantities and units' in value for value in saved['presentation_diagnostics']['problems'])
    assert axis.pen()==before


def test_native_export_runs_shared_finalization_at_actual_physical_write_boundary(rank,monkeypatch,tmp_path):
    """The shared finalizer must inspect the .66-inch native axes used by the writer, rather than reuse a source Matplotlib certificate."""
    observed=[]

    def capture(item,path,width,height,exporters):
        """Read the actual diagnostics and physical axes at the final writer boundary."""
        observed.append((rank._spacr_style_diagnostics,rank._spacr_style_observation()['width_inches']))

    monkeypatch.setattr(rank,'_write_export',capture)
    rank.export(str(tmp_path/'native.svg'))
    assert len(observed)==1 and observed[0][1]==pytest.approx(3*.22,abs=.5/rank.logicalDpiX())
    assert 'Use the fixed physical width for the declared number of conditions' not in observed[0][0].problems



def test_native_conditions_feed_shared_actual_form_count_width_and_control_checks(qtbot):
    """The native adapter previously had no condition roles; require exact1/21/1000 forms, point counts and widths, then detect altered control, hidden point and .3-slot width without changing CSV bytes."""
    import pyqtgraph as pg
    from spacr.figures.style import finalize_figure
    from spacr.qt.widgets.grouped_plot import GroupedPlot,PlotSpec

    counts=(1,21,1000)
    labels=('reference','many','dense')
    frame=pd.DataFrame({'group':np.repeat(labels,counts),
                        'value':np.concatenate([np.linspace(1.,2.,n) for n in counts])})
    before=frame.to_csv(index=False).encode()
    widget=GroupedPlot(PlotSpec(frame,'value',group='group',background='reference'))
    qtbot.addWidget(widget)
    result=finalize_figure(widget)
    assert 'actual condition forms, counts and identities' in result.checks
    assert not any('count-dependent' in value or 'every finite observation' in value or
                   'six-tenths' in value or 'explicit control in grey' in value or
                   'Register native artist roles' in value for value in result.problems)
    bar=next(item for item in widget.plot.plotItem.items if isinstance(item,pg.BarGraphItem) and
             getattr(item,'_spacr_condition_index',None)==0)
    point=next(item for item in widget.plot.plotItem.items if isinstance(item,pg.ScatterPlotItem) and
               getattr(item,'_spacr_condition_index',None)==0)
    ink=pg.mkColor('#0A77B8');ink.setAlphaF(.6)
    bar.setOpts(brush=pg.mkBrush(ink),width=.3)
    point.setVisible(False)
    changed=finalize_figure(widget)
    assert any('every finite observation' in value for value in changed.problems)
    assert any('six-tenths' in value for value in changed.problems)
    assert any('explicit control in grey' in value for value in changed.problems)
    assert not point.isVisible() and bar.opts['width']==.3
    assert widget.spec.frame.to_csv(index=False).encode()==before


def test_native_estimate_rank_registers_real_bar_and_interval_roles(rank):
    """Native rank keeps three exact estimate slots and black error roles rather than claiming unobserved distribution counts."""
    from spacr.figures.style import finalize_figure

    observed=rank._spacr_style_observation()
    assert {mark['role'] for mark in observed['marks']}=={'bar','error','estimate_point','reference'}
    result=finalize_figure(rank)
    assert not any('count-dependent' in value or 'finite observation counts' in value or
                   'Register native artist roles' in value for value in result.problems)
    assert dict(rank._spacr_style_spec.comparison)['representation']=='estimates'



def test_rotated_native_text_uses_actual_polygons_and_stays_in_the_physical_canvas(rank):
    """At96dpi the long-name envelopes overlap by0.0346547813pixels (14.96875px font height minus .22*96/sqrt(2) slot separation); apply the existing half-display-pixel limit to overlap thickness rather than the long bounding diagonal."""
    from spacr.qt.widgets.fast_plots import _native_text_geometry

    with rank._held_at_policy_geometry():
        measured=_native_text_geometry(rank.plot.plotItem)
        assert measured['checked'],measured['findings']
        assert not measured['overlaps'],measured['elements']
        assert not measured['outside'],measured['elements']


def test_native_text_bounds_detect_a_manually_clipped_title_without_repositioning_it(rank):
    """Moving the actual title1000pixels beyond the page must report clipping while retaining that manual position and all coefficient bytes."""
    from PySide6.QtCore import QPointF
    from spacr.qt.widgets.fast_plots import _native_text_geometry

    rank.plot.setTitle('Manual title outside the canvas')
    title=rank.plot.plotItem.titleLabel.item
    title.setPos(QPointF(-1000.,-1000.))
    before=(title.pos(),rank._frame.to_csv(index=False).encode())
    measured=_native_text_geometry(rank.plot.plotItem)
    assert measured['checked'] and measured['outside']
    assert title.pos()==before[0] and rank._frame.to_csv(index=False).encode()==before[1]



def test_long_native_categories_reach_the_actual_rotated_painter(rank):
    """The physical .66-inch fixture dropped every long label in pyqtgraph's unrotated filter; all three explicit visible strings must reach paint specifications."""
    from PySide6.QtGui import QPainter,QPicture

    with rank._held_at_policy_geometry():
        axis=rank.plot.getAxis('bottom')
        picture=QPicture();painter=QPainter(picture)
        try:
            specs=axis.generateDrawSpecs(painter)
        finally:
            painter.end()
        assert tuple(text for rect,flags,text in specs[2])==axis._category_labels



def test_native_text_geometry_still_detects_a_real_title_label_collision(rank):
    """A title placed directly over the x-axis label has a many-pixel intersection and must survive the half-pixel boundary exclusion."""
    from spacr.qt.widgets.fast_plots import _native_text_geometry

    rank.plot.setTitle('Title colliding with axis label')
    title=rank.plot.plotItem.titleLabel.item
    label=rank.plot.getAxis('bottom').label
    target=label.mapToScene(label.boundingRect().topLeft())
    title.setPos(title.parentItem().mapFromScene(target))
    before=title.pos()
    result=_native_text_geometry(rank.plot.plotItem)
    assert result['checked'] and result['overlaps']
    assert title.pos()==before



def test_native_complete_observations_check_units_font_roles_and_manual_zoom_without_repair(qtbot):
    """A fully declared two-singleton fixture must pass actual checks; changing body weight toBold and clipping y to.5must report both while keeping chosen font, limits and exact CSV bytes."""
    from PySide6.QtGui import QFont
    from spacr.figures.style import finalize_figure
    from spacr.qt.widgets.grouped_plot import GroupedPlot,PlotSpec

    frame=pd.DataFrame({'group':['reference','treated'],'value':[1.,2.]})
    frame.attrs['units']={'x':'condition','y':'a.u.'}
    widget=GroupedPlot(PlotSpec(frame,'value',group='group',background='reference',x_label='Condition',y_label='Value (a.u.)'))
    qtbot.addWidget(widget);widget.resize(720,480);widget.show();qtbot.waitExposed(widget)
    source=frame.to_csv(index=False).encode()
    with widget._held_at_policy_geometry():
        result=finalize_figure(widget)
        assert result.compliant,result.problems
        axis=widget.plot.getAxis('bottom')
        font=QFont(axis.style['tickFont']);font.setWeight(QFont.Bold);axis.setStyle(tickFont=font)
        widget.plot.setYRange(0.,.5,padding=0.)
        limits=widget.plot.getViewBox().viewRange()
        result=finalize_figure(widget)
        assert any('Light body and Regular title' in value for value in result.problems)
        assert any('data marks within the viewport' in value for value in result.problems)
        assert axis.style['tickFont']==font and widget.plot.getViewBox().viewRange()==limits
    assert frame.to_csv(index=False).encode()==source



def _png_physical_resolution(path):
    """Read only the PNG physical-resolution header, without opening or transforming image pixels."""
    import struct

    data=path.read_bytes()
    assert data[:8]==b'\x89PNG\r\n\x1a\n'
    offset=8
    while offset<len(data):
        count=struct.unpack('>I',data[offset:offset+4])[0]
        if data[offset+4:offset+8]==b'pHYs':
            return struct.unpack('>IIB',data[offset+8:offset+8+count])
        offset+=12+count
    return None


def test_native_bundle_captures_successful_file_state_and_exact_numerical_tables(rank,tmp_path):
    """The measured PNG had 3780 dots/m despite 300-dpi dimensions; require 11811 dots/m, actual .66-inch file axes, licensed fonts and byte-identical data/statistics while restoring the live geometry."""
    import json
    from pathlib import Path
    from spacr.figures.bundle import statistics_frame

    source=rank.frame().to_csv(index=False).encode()
    statistics=statistics_frame(rank.comparison_groups(),unit=rank.comparison_unit()).to_csv(index=False).encode()
    live=rank.plot.plotItem.geometry()
    folder=Path(rank.export_bundle(str(tmp_path),name='native'))
    assert (folder/'data.csv').read_bytes()==source
    assert (folder/'statistics.csv').read_bytes()==statistics
    assert (folder/'native.pdf').read_bytes().startswith(b'%PDF')
    assert _png_physical_resolution(folder/'native.png')==(11811,11811,1)
    saved=json.loads((folder/'settings.json').read_text())
    exported=saved['export_presentation']['appearance']
    assert exported['width_inches']==pytest.approx(3*.22,abs=.5/rank.logicalDpiX())
    assert exported['export_dpi']==300 and exported['output_format']=='png'
    assert exported['raster_resolution_applied'] is True
    assert exported['output_target']=='file'
    assert all((folder/record['name']).is_file() for record in saved['fonts'])
    assert (folder/'OpenSans-OFL.txt').is_file()
    assert rank.plot.plotItem.geometry()==live
    saved_again=rank.export_settings()
    saved_again['export_presentation']['appearance']['width_inches']=999.
    assert rank.export_settings()['export_presentation']['appearance']['width_inches']!=999.


def test_native_explicit_raster_resolution_is_recorded_and_restored(rank,tmp_path):
    """A manual 150-dpi file must retain 5906 dots/m, while the original unset live resolution is restored exactly."""
    before=rank._export_dpi
    destination=tmp_path/'manual.png'
    rank.export_styled(str(destination),dpi=150)
    assert _png_physical_resolution(destination)==(5906,5906,1)
    assert rank._export_dpi==before
    assert rank.export_settings()['export_presentation']['appearance']['export_dpi']==150


@pytest.mark.parametrize("count", [1000, 1001])
@pytest.mark.parametrize("level", [0., -2., 3.])
def test_constant_dense_native_violins_keep_zero_spread_without_individual_points(qtbot, count, level):
    """The 1000-value constant fixture fell back to 1000 points; require zero points, exact shared y and the existing .6-slot width with only floating subtraction rounding."""
    import pyqtgraph as pg
    from spacr.qt.widgets.grouped_plot import GroupedPlot, PlotSpec
    from spacr.figures.style import _condition_record_problems

    frame = pd.DataFrame({'group': ['reference'] * count, 'value': np.full(count, level)})
    source = frame.to_csv(index=False).encode()
    widget = GroupedPlot(PlotSpec(frame, 'value', group='group', background='reference'))
    qtbot.addWidget(widget)
    assert not any(isinstance(item, pg.ScatterPlotItem) for item in widget.plot.plotItem.items)
    bodies = [item for item in widget.plot.plotItem.items
              if getattr(item, '_spacr_condition_role', '') == 'violin']
    assert len(bodies) == 1
    np.testing.assert_array_equal(bodies[0].yData, np.full(4, level))
    np.testing.assert_array_equal(bodies[0].xData, [.3, .3, -.3, -.3])
    assert bodies[0].opts['fillLevel'] == 'enclosed'
    assert bodies[0].opts['brush'].color().name() == '#8c8c8c'
    assert bodies[0].opts['brush'].color().alpha() == 153
    assert 'zero-spread violin' in widget._headline
    snapshot = widget._spacr_style_observation()
    records = [dict(condition=mark['condition'], role=mark['role'], visible=True,
                    width=mark.get('width'), point_count=mark['point_count'],
                    colours=[colour[:3] for colour in mark['colours']])
               for mark in snapshot['marks'] if mark['condition'] is not None]
    assert not _condition_record_problems(widget._spacr_style_spec, records, colour_tolerance=1./65535)
    assert widget.spec.frame.to_csv(index=False).encode() == source
