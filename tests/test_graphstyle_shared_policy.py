"""Shared graph-policy semantics, physical geometry and numerical preservation."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.collections import PathCollection

from spacr.figures import style


def test_specification_freezes_explicit_semantics_and_uses_reference_colours():
    """The measured baseline uses explicit c0 grey; new metadata preserves that identity without label guessing."""
    legend={'caption':'Synthetic intensity by condition.','reference':'untreated','unit':'a.u.'}
    spec=style.resolve_style('print','bar',conditions=('untreated','treated'),
        sample_counts=(1,21),control='untreated',legend=legend)
    legend['caption']='changed after resolution'
    assert spec.controls==(0,) and spec.colours==(style.Palette.CONTROL,style.Palette.ACCENT)
    assert spec.kinds==('bar','box')
    assert dict(spec.legend)['caption']=='Synthetic intensity by condition.'
    with pytest.raises(FrozenInstanceError):
        spec.alpha=.1


@pytest.mark.parametrize('conditions,width',[(2,.44),(3,.66),(6,1.32)])
def test_condition_geometry_matches_measured_slots_with_half_pixel_margin(conditions,width):
    """Untouched 300-dpi fixtures measured .44/.66/1.32-inch axes; allow half a rendered pixel."""
    spec=style.resolve_style('print','categorical',conditions=conditions)
    geometry=style.axes_geometry(spec)
    assert geometry.width_inches==pytest.approx(width,abs=.5/300)
    assert geometry.element_width==.6 and geometry.gap==.4


def test_continuous_and_matrix_geometry_preserve_square_cells():
    """The audit measured nonsquare 2.754-by1.976-inch panels; the resolved target is square, not a crop."""
    continuous=style.axes_geometry(style.resolve_style('print','histogram'))
    assert continuous.width_inches==continuous.height_inches
    heatmap=style.axes_geometry(style.resolve_style('print','heatmap'),cell_count=(2,3))
    assert heatmap.width_inches/3==pytest.approx(heatmap.height_inches/2)
    image=style.axes_geometry(style.resolve_style('print','image'))
    assert image.aspect=='content' and image.width_inches is None


def test_single_observations_remain_visible_and_dense_groups_drop_individual_points():
    """Baseline n1 had no point and n1001 showed every point; require one point and dense violin with zero points."""
    arrays=[np.arange(n,dtype=float) for n in (1,20,21,1001)]
    before=[value.tobytes() for value in arrays]
    figure=Figure();ax=figure.add_subplot(111)
    assert style.condition_plot(ax,arrays,control=0)==['bar','bar','box','violin']
    scatter=[artist for artist in ax.collections if isinstance(artist,PathCollection)]
    assert sum(len(artist.get_offsets()) for artist in scatter)==1+20+21
    assert all(artist.get_alpha()==.6 for artist in ax.lines)
    assert all(value.tobytes()==prior for value,prior in zip(arrays,before))


def test_unregistered_empty_checker_result_is_not_a_compliance_certificate():
    """The audit checker missed real baseline defects; absence of semantic registration must remain visible."""
    figure=Figure();ax=figure.add_subplot(111)
    line,=ax.plot([0.,1.],[1.,2.],color='purple',alpha=.2)
    values=line.get_xydata().copy()
    result=style.finalize_figure(figure,diagnostics=False)
    assert not result.compliant and 'registered semantics' not in result.checks
    np.testing.assert_array_equal(line.get_xydata(),values)
    assert line.get_color()=='purple' and line.get_alpha()==.2


def test_registered_geometry_and_effective_alpha_are_checked_without_repairing_data():
    """Baseline raw scatter could be opaque and nonsquare; detect both rather than silently restyling it."""
    figure=Figure(figsize=(6,3));ax=figure.add_subplot(111)
    points=ax.scatter([0.,1.],[1.,2.],alpha=.2)
    values=points.get_offsets().copy()
    spec=style.resolve_style('print','scatter',legend={'caption':'Synthetic values.','reference':'identity'})
    result=style.finalize_figure(figure,spec,diagnostics=False)
    assert not result.compliant
    assert any('square plot area' in problem for problem in result.problems)
    assert any('sixty percent' in problem for problem in result.problems)
    np.testing.assert_array_equal(points.get_offsets(),values)
    assert points.get_alpha()==.2


def test_numpy_reference_indexes_and_identity_bindings_are_explicit():
    """Baseline numpy integer controls raised TypeError; explicit reference and role bindings must resolve exactly."""
    spec=style.resolve_style('print','bar',conditions=('reference','treated'),
        control=np.int64(0),identities={'reference':'up','treated':'down'})
    assert spec.controls==(0,)
    assert spec.colours==(style.Palette.CONTROL,style.Palette.MAGENTA)
    assert style.resolve_style('print','unknown renderer').kind=='unspecified'


def test_skipped_render_checks_never_certify_a_registered_figure():
    """The preliminary finalizer could certify without layout checks; the renderer must actually run before certification."""
    figure=Figure(figsize=(3,3));ax=figure.add_subplot(111)
    ax.scatter([0.,1.],[0.,1.],alpha=.6)
    spec=style.resolve_style('print','scatter',legend={'caption':'Synthetic values.','reference':'zero','unit':'dimensionless'})
    result=style.finalize_figure(figure,spec,diagnostics=False)
    assert not result.compliant
    assert 'completed presentation checks' not in result.checks


def test_neutral_grey_errorbars_and_unreadable_dark_text_are_rejected():
    """Baseline accepted neutral grey CI ink and black-on-dark text; require exact black whiskers and at least 4.5 contrast."""
    figure=Figure(figsize=(3,3),facecolor='#202630');ax=figure.add_subplot(111)
    ax.set_facecolor('#202630');ax.set_xlabel('Value',color='black')
    ax.errorbar([0.,1.],[0.,1.],yerr=[.1,.2],ecolor='#808080',alpha=.6)
    spec=style.resolve_style('print','scatter',legend={'caption':'Synthetic values.','reference':'zero','unit':'dimensionless'})
    result=style.finalize_figure(figure,spec,diagnostics=False)
    assert 'Draw every error bar in black' in result.problems
    assert 'Use readable text contrast against the actual plot background' in result.problems


def test_heatmap_cell_geometry_and_opacity_are_checked_without_array_changes():
    """The two-by-three baseline stretched cells and used opacity 1; detect each with one-pixel geometry margin."""
    figure=Figure(figsize=(6,2));ax=figure.add_subplot(111)
    values=np.arange(6,dtype=float).reshape(2,3)
    image=ax.imshow(values,aspect='auto',alpha=1.)
    original=image.get_array().tobytes()
    spec=style.resolve_style('print','heatmap',legend={'caption':'Synthetic values.','reference':'zero','unit':'dimensionless'})
    result=style.finalize_figure(figure,spec,diagnostics=False)
    assert 'Use square cells for heatmap data' in result.problems
    assert 'Draw every data mark at sixty percent opacity' in result.problems
    assert image.get_array().tobytes()==original


def test_actual_single_rank_layout_keeps_intervals_and_measured_slot_width():
    """The baseline 14-gene axes measured 2.754 inches; require 14 times .22 with half a 300-dpi pixel margin."""
    import pandas as pd
    from matplotlib.collections import LineCollection
    from spacr.figures.sheet import build_panel

    frame=pd.DataFrame({'gene':[f'gene_{i:02d}' for i in range(14)],
        'coefficient':np.linspace(-2.,3.,14),'std_err':np.full(14,.2)})
    original=frame.to_csv(index=False).encode()
    figure,panel=build_panel('effect_rank',frame,target='print')
    ax=figure.axes[0];box=ax.get_position()
    assert box.width*figure.get_size_inches()[0]==pytest.approx(14*.22,abs=.5/300)
    intervals=next(item for item in ax.collections if isinstance(item,LineCollection)).get_segments()
    assert len(intervals)==14
    observed=sorted((float(segment[0,1]),float(segment[1,1])) for segment in intervals)
    expected=sorted((float(value-1.96*.2),float(value+1.96*.2)) for value in frame.coefficient)
    np.testing.assert_array_equal(observed,expected)
    assert frame.to_csv(index=False).encode()==original
    assert dict(ax._spacr_style_spec.comparison)['reference']=='Zero regression coefficient'


def test_read_only_finalization_does_not_attach_state_to_source():
    """Unregistered baseline scene translation is read-only; diagnostics must remain on the translated widget."""
    figure=Figure();figure.add_subplot(111)
    before=set(vars(figure))
    style.finalize_figure(figure,record=False)
    assert set(vars(figure))==before


def test_empty_collection_faces_do_not_hide_opaque_line_segments():
    """LineCollection before exposed empty face colours and opaque segment colours; the segment opacity must still be diagnosed."""
    from matplotlib.collections import LineCollection

    figure=Figure();ax=figure.add_subplot(111)
    segments=LineCollection([[(0.,0.),(1.,1.)]],colors='#0A77B8')
    ax.add_collection(segments)
    specification=style.resolve_style('print','scatter',legend={
        'caption':'Synthetic segments.','reference':'identity','unit':'dimensionless'})
    result=style.finalize_figure(figure,specification,diagnostics=False)
    assert 'Draw every data mark at sixty percent opacity' in result.problems


def test_diagnostics_restore_the_original_canvas_and_skip_disabled_tick_labels():
    """A bare Figure had its canvas replaced during diagnostics; keep it and exclude only decorations that are actually disabled."""
    figure=Figure();ax=figure.add_subplot(111);ax.set_axis_off()
    ax.set_xlabel('hidden label',fontsize=3)
    ax.text(.5,.5,'visible overlay',fontsize=3)
    canvas=figure.canvas
    findings=style.figure_problems(figure)
    assert figure.canvas is canvas
    assert any('visible overlay' in finding and 'under 8 pt' in finding for finding in findings)
    assert not any('hidden label' in finding for finding in findings)


def test_packed_mixed_panels_preserve_slot_width_square_area_and_numerical_values():
    """Before sheets used equal grid cells; require .88-inch four-condition slots, square continuous axes, and shared top edges within half a 300-dpi pixel."""
    with style.figure_style('print'):
        figure=Figure(figsize=(7,3));categorical=figure.add_subplot(121);continuous=figure.add_subplot(122)
        groups=[np.array([float(value)]) for value in range(1,5)]
        style.condition_plot(categorical,groups,control=0)
        categorical.set_xticks(range(4),['reference','one','two','three'])
        style.rotate_ticks(categorical);categorical.set_ylabel('Value (a.u.)')
        first=style.resolve_style('print','categorical',conditions=['reference','one','two','three'],
            sample_counts=[1]*4,control=0,legend={'caption':'Synthetic values.','reference':'reference','unit':'a.u.'})
        style.panel_letter(categorical,'a')
        points=continuous.scatter([.2,.8],[.3,.7],alpha=.6)
        continuous.set_xlabel('Coordinate (a.u.)');continuous.set_ylabel('Value (a.u.)')
        second=style.resolve_style('print','scatter',legend={'caption':'Synthetic values.','reference':'origin','unit':'a.u.'})
        style.panel_letter(continuous,'b')
        style.apply_axes_style(categorical,first,layout=False);style.apply_axes_style(continuous,second,layout=False)
        original=points.get_offsets().tobytes();limits=[(ax.get_xlim(),ax.get_ylim()) for ax in (categorical,continuous)]
        style.pack_axes_geometry([categorical,continuous],maximum_width=5.)
        width,height=figure.get_size_inches();left=categorical.get_position();right=continuous.get_position()
        assert left.width*width==pytest.approx(4*.22,abs=.5/300)
        assert right.width*width==pytest.approx(right.height*height,abs=.5/300)
        assert left.y1*height==pytest.approx(right.y1*height,abs=.5/300)
        assert points.get_offsets().tobytes()==original
        for ax,expected in zip((categorical,continuous),limits):
            assert (ax.get_xlim(),ax.get_ylim())==expected
        letters=[text for ax in (categorical,continuous) for text in ax.texts if text.get_gid()=='spacr-panel-letter']
        assert [text.get_text() for text in letters]==['A','B']
        assert all(text.get_fontsize()==10. and text.get_fontweight()=='bold' for text in letters)


def test_finalizer_observes_real_forms_counts_width_and_reference_colour_without_repair():
    """The initial checker accepted a coloured control and hidden n1 point; detect both, then detect a .3-slot bar against the exact .6-slot rule."""
    figure=Figure();ax=figure.add_subplot()
    arrays=[np.array([1.]),np.linspace(1.,2.,21),np.linspace(1.,2.,1000)]
    source=[value.tobytes() for value in arrays]
    style.condition_plot(ax,arrays,control=0)
    spec=style.resolve_style('print','categorical',conditions=('reference','many','dense'),sample_counts=(1,21,1000),control=0,
        legend={'caption':'Synthetic observation distributions.','reference':'reference','unit':'a.u.',
                'quantities':{'x':'condition','y':'signal'}})
    style.apply_axes_style(ax,spec)
    assert style._condition_presentation_problems(ax,spec)==()
    first_point=next(artist for artist in ax.collections if artist.get_gid()=='spacr-condition:0:points')
    first_point.set_visible(False)
    ax.patches[0].set_facecolor(style.Palette.ACCENT)
    ax.patches[0].set_width(.3)
    result=style.finalize_figure(figure,diagnostics=False)
    assert any('every finite observation' in value for value in result.problems)
    assert any('explicit control in grey' in value for value in result.problems)
    assert any('six-tenths slot width' in value for value in result.problems)
    assert not first_point.get_visible()
    assert ax.patches[0].get_width()==.3
    assert [value.tobytes() for value in arrays]==source


def test_finalizer_rejects_obsolete_dense_bars_and_out_of_palette_colours():
    """A manually selected dense bar stays drawn, but cannot be certified as the required violin; brown remains an explicit palette finding."""
    figure=Figure();ax=figure.add_subplot()
    style.condition_plot(ax,[np.linspace(1.,2.,1000)],kind='bar',points=False)
    spec=style.resolve_style('print','categorical',conditions=('measured',),sample_counts=(1000,),
        legend={'caption':'Synthetic values.','reference':'zero','unit':'a.u.','quantities':{'x':'condition','y':'signal'}})
    style.apply_axes_style(ax,spec)
    ax.patches[0].set_facecolor('#6D2B00')
    result=style.finalize_figure(figure,diagnostics=False)
    assert any('count-dependent bar, box or violin' in value for value in result.problems)
    assert any('shared data palette' in value for value in result.problems)
    assert len(ax.patches)==1 and ax.patches[0].get_facecolor()[:3]==pytest.approx((109/255,43/255,0.))


def test_finalizer_matches_reference_legend_to_actual_line_and_rejects_floating_counts():
    """Earlier validation ignored a grey key with a blue reference line and floating n; report both without removing the line or text."""
    figure=Figure();ax=figure.add_subplot()
    line=style.reference_line(ax,y=0.,label='Declared reference')
    line.set_color(style.Palette.ACCENT)
    note=ax.text(.5,.5,'n = 12',transform=ax.transAxes)
    spec=style.resolve_style('print','scatter',legend={'caption':'Synthetic coordinates.','reference':'zero','unit':'dimensionless',
        'quantities':{'x':'coordinate','y':'coordinate'}})
    style.apply_axes_style(ax,spec)
    result=style.finalize_figure(figure,diagnostics=False)
    assert 'Match reference legend colours to their actual plotted lines' in result.problems
    assert any('ordinary in-panel text' in value for value in result.problems)
    assert note in ax.texts and line.get_color()==style.Palette.ACCENT


def test_image_validation_requires_declared_display_metadata_without_changing_pixels():
    """Opaque images are allowed, but missing display range, gamma and processing remain actionable rather than invented."""
    figure=Figure();ax=figure.add_subplot()
    image=ax.imshow(np.arange(6).reshape(2,3),alpha=1.)
    source=image.get_array().tobytes()
    spec=style.resolve_style('print','image',legend={'caption':'Synthetic image.'})
    result=style.finalize_figure(figure,spec,diagnostics=False)
    assert any('display range, gamma and processing' in value for value in result.problems)
    assert not any('sixty percent' in value for value in result.problems)
    assert image.get_alpha()==1. and image.get_array().tobytes()==source



def test_matplotlib_finalizer_checks_declared_font_role_without_changing_manual_weight():
    """The initial checker accepted any weight inOpenSans; bodyLight and titleRegular are explicit immutable policy fields, and a manualBold label staysBold while diagnosed."""
    with style.figure_style('print'):
        figure=Figure();ax=figure.add_subplot()
        ax.scatter([0.,1.],[0.,1.],color=style.Palette.ACCENT,alpha=.6)
        ax.set_xlabel('Coordinate (a.u.)')
        spec=style.resolve_style('print','scatter',legend={'caption':'Synthetic values.','reference':'zero','unit':'a.u.',
            'quantities':{'x':'coordinate','y':'value'}})
        style.apply_axes_style(ax,spec)
        assert dict(spec.font_weights)=={'body':'light','title':'regular','emphasis':'semibold','panel_letter':'bold'}
        ax.xaxis.label.set_fontweight('bold')
        result=style.finalize_figure(figure,diagnostics=False)
        assert 'Use the declared body, title and panel-letter font weights' in result.problems
        assert ax.xaxis.label.get_fontweight()=='bold'



def test_producer_policy_preserves_explicit_stored_choices_and_reports_conflicts(monkeypatch):
    """Producer styling previously replaced a stored7-point DejaVu label and open frame; retain those choices, grid and data bytes, then report the mandatory-rule conflicts."""
    from spacr.qt import preferences

    def manual_preferences():
        """Return explicit stored choices that conflict with the automatic publication defaults."""
        return {'font_family':'DejaVu Sans','tick_size':7.,'spines':'left_bottom','grid':True}

    monkeypatch.setattr(preferences,'get_figure_style',manual_preferences)
    monkeypatch.setattr(preferences,'get_figure_style_per_graph',dict)
    figure=Figure();ax=figure.add_subplot()
    points=ax.scatter([0.,1.],[1.,2.],alpha=.6,color=style.Palette.ACCENT)
    before=points.get_offsets().tobytes()
    spec=style.resolve_style('print','scatter',legend={'caption':'Synthetic values.','reference':'zero','unit':'a.u.',
        'quantities':{'x':'coordinate','y':'value'}})
    style.apply_axes_style(ax,spec)
    assert ax.get_xticklabels()[0].get_fontfamily()==['DejaVu Sans']
    assert ax.get_xticklabels()[0].get_fontsize()==7.
    assert not ax.spines['top'].get_visible() and not ax.spines['right'].get_visible()
    assert any(line.get_visible() for line in ax.get_xgridlines())
    result=style.finalize_figure(figure,diagnostics=False)
    assert 'Use all four frame spines for statistical plots' in result.problems
    assert any('Open Sans' in value or 'eight points' in value for value in result.problems)
    assert points.get_offsets().tobytes()==before


def test_panel_specific_stored_choices_preserve_sibling_appearance(monkeypatch):
    """A volcano-only tick override must leave the other panel's font and the shared page ground byte-for-byte unchanged."""
    from spacr.qt import preferences

    def volcano_preferences():
        """Return one graph-specific override without modifying the live settings store."""
        return {'volcano':{'tick_size':13.}}

    monkeypatch.setattr(preferences,'get_figure_style',dict)
    monkeypatch.setattr(preferences,'get_figure_style_per_graph',volcano_preferences)
    with style.figure_style('print'):
        figure=Figure();left,right=figure.subplots(1,2)
        for ax in (left,right):ax.scatter([0.,1.],[1.,2.],alpha=.6,color=style.Palette.ACCENT)
        before=(right.get_xticklabels()[0].get_fontsize(),figure.get_facecolor())
        spec=style.resolve_style('print','scatter',comparison={'graph_type':'volcano'},
            legend={'caption':'Synthetic values.','reference':'zero','unit':'a.u.'})
        style.apply_axes_style(left,spec,layout=False)
        assert left.get_xticklabels()[0].get_fontsize()==13.
        assert (right.get_xticklabels()[0].get_fontsize(),figure.get_facecolor())==before



def test_direct_style_construction_freezes_caller_owned_semantic_containers():
    """Frozen attributes alone left direct constructor metadata mutable; later caller edits must not change captured condition identities, nested quantities or font roles."""
    from dataclasses import replace

    labels = ['reference', 'treated']
    quantities = {'x': ['condition'], 'y': ['intensity']}
    legend = {'caption': 'Captured quantities.', 'quantities': quantities}
    fonts = [['body', 'light'], ['title', 'regular']]
    record = replace(style.resolve_style('print', 'categorical', conditions=labels),
                     condition_labels=labels, legend=legend, font_weights=fonts)
    labels.append('later')
    quantities['y'].append('later')
    fonts[0][1] = 'bold'
    legend['caption'] = 'later'
    assert record.condition_labels == ('reference', 'treated')
    assert dict(record.legend)['caption'] == 'Captured quantities.'
    assert dict(dict(record.legend)['quantities'])['y'] == ('intensity',)
    assert dict(record.font_weights)['body'] == 'light'



def test_figure_legend_fonts_weights_and_actual_frame_contrast_are_validated():
    """Figure-level keys escaped font and contrast validation; certify the declared outside key, then detect manual seven-point bold DejaVu text and white ink while preserving all coordinates."""
    with style.figure_style('print'):
        figure = Figure(); ax = figure.add_subplot()
        points = ax.scatter([.2, .8], [.3, .7], color=style.Palette.ACCENT, alpha=.6)
        ax.set_xlabel('Coordinate'); ax.set_ylabel('Value')
        spec = style.resolve_style('print', 'scatter', legend={'caption': 'Synthetic coordinates.',
            'reference': 'origin', 'unit': 'a.u.', 'quantities': {'x': 'coordinate', 'y': 'value'}})
        style.apply_axes_style(ax, spec, layout=False)
        legend = figure.legend([points], ['Measured condition'], title='Condition', loc='upper right', frameon=False, fontsize=8)
        legend.get_title().set_fontweight('regular')
        style.pack_axes_geometry([ax], legend_space=True)
    before = points.get_offsets().tobytes()
    assert style.finalize_figure(figure).compliant
    text = legend.get_texts()[0]
    text.set_fontfamily('DejaVu Sans'); text.set_fontsize(7); text.set_fontweight('bold'); text.set_color('white')
    result = style.finalize_figure(figure)
    assert any('Open Sans' in value for value in result.problems)
    assert any('eight points' in value for value in result.problems)
    assert any('font weights' in value for value in result.problems)
    assert any('readable text contrast' in value for value in result.problems)
    text.set_fontfamily('Open Sans'); text.set_fontsize(8); text.set_fontweight('light')
    legend.set_frame_on(True); legend.get_frame().set_facecolor('black'); legend.get_frame().set_alpha(1.)
    legend.get_title().set_color('white')
    assert not any('readable text contrast' in value for value in style.finalize_figure(figure).problems)
    assert points.get_offsets().tobytes() == before


@pytest.mark.parametrize('orientation', ['vertical', 'horizontal'])
@pytest.mark.parametrize('count', [1, 20, 21, 1000])
def test_shared_condition_orientation_keeps_exact_observations_and_physical_slots(orientation, count):
    """Horizontal bindings previously bypassed the shared forms; require unchanged finite values, .6-slot marks and .22-inch category slots within half a 300-dpi pixel in both orientations."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    values = [np.linspace(1., 2., count), np.linspace(2., 3., count)]
    before = [item.tobytes() for item in values]
    with style.figure_style('print'):
        figure = Figure(dpi=300); ax = figure.add_subplot()
        used = style.condition_plot(ax, values, orientation=orientation, control=0, labels=['reference', 'treated'])
        (ax.set_yticks if orientation == 'horizontal' else ax.set_xticks)([0, 1], ['reference', 'treated'])
        spec = style.resolve_style('print', 'categorical', conditions=['reference', 'treated'], sample_counts=[count, count], control=0,
                                   comparison={'orientation': orientation})
        style.apply_axes_style(ax, spec)
    canvas = FigureCanvasAgg(figure); canvas.draw()
    area = ax.get_window_extent(canvas.get_renderer())
    length = area.height if orientation == 'horizontal' else area.width
    assert length / figure.dpi == pytest.approx(.44, abs=.5/300)
    assert used == [style.condition_kind(np.arange(count))] * 2
    assert not style._condition_presentation_problems(ax, spec)
    points = [artist.get_offsets() for artist in ax.collections if isinstance(artist, PathCollection)]
    if count < 1000:
        coordinate = 0 if orientation == 'horizontal' else 1
        np.testing.assert_array_equal(np.concatenate([item[:, coordinate] for item in points]), np.concatenate(values))
    else:
        assert not points
    assert [item.tobytes() for item in values] == before


@pytest.mark.parametrize('many', ['box', 'violin'])
def test_both_approved_many_value_forms_pass_the_shared_condition_checks(many):
    """The authority permits either box or violin above twenty; both must retain all 21 observations without a false representation failure."""
    figure = Figure(); ax = figure.add_subplot()
    values = [np.linspace(1., 2., 21), np.linspace(2., 3., 21)]
    style.condition_plot(ax, values, many=many, control=0)
    spec = style.resolve_style('print', 'categorical', conditions=['reference', 'treated'], sample_counts=[21, 21], control=0)
    assert not style._condition_presentation_problems(ax, spec)


@pytest.mark.parametrize('limits, bounds, expected', [
    ((.9, 2.1), (1., 2.), (0., 2.1)),
    ((-.1, 1.1), (0., 1.), (0., 1.1)),
    ((-.5, 1.3), (-.3, 1.2), (-.5, 1.3)),
    ((-3.1, -.9), (-3., -1.), (-3.1, 0.)),
])
def test_line_origin_keeps_zero_and_every_observed_interval(limits, bounds, expected):
    """Positive lines lacked zero; require the declared origin exactly while retaining negative data and interval padding rather than clipping it."""
    assert style._line_origin_limits(limits, bounds) == expected


@pytest.mark.parametrize('choice', ['manual', 'authorized', 'inverted'])
def test_shared_line_origin_preserves_explicit_views_and_reports_unauthorized_cuts(choice):
    """Appearance policy must leave explicitly selected manual or inverted views intact and require an explanation for a positive cut instead of silently resetting it."""
    figure = Figure(); ax = figure.add_subplot()
    line, = ax.plot([1., 2.], [1., 2.])
    bounds = (2.1, .9) if choice == 'inverted' else (.9, 2.1)
    ax.set_ylim(*bounds)
    comparison = {'graph_type': 'line'}
    if choice == 'manual':
        comparison['manual_limits'] = bounds
    elif choice == 'authorized':
        comparison['cut_axis_reason'] = 'A declared detailed view.'
    spec = style.resolve_style('print', 'line', comparison=comparison)
    source = line.get_xydata().tobytes()
    style.apply_axes_style(ax, spec)
    assert ax.get_ylim() == bounds
    result = style.finalize_figure(figure)
    assert any('Start line and sweep axes' in value for value in result.problems) == (choice != 'authorized')
    assert line.get_xydata().tobytes() == source
