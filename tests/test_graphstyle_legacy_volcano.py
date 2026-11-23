"""The independent Matplotlib volcano follows shared defaults without changing transformed observations."""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from spacr import plot
from spacr.figures import style
from spacr.qt import preferences


def _frame():
    """Return literal signed effects and p values with three selected observations."""
    return pd.DataFrame({'effect':[-2.,-.2,.1,1.,2.],'p':[.001,.5,.8,.04,.002],'name':['a','b','c','d','e']})


def _defaults(monkeypatch):
    """Isolate live preference stores from deterministic producer checks."""
    monkeypatch.setattr(preferences,'get_figure_style',dict);monkeypatch.setattr(preferences,'get_figure_style_per_graph',dict)


def test_legacy_volcano_uses_shared_ground_geometry_alpha_and_explained_colour_keys(monkeypatch):
    """Keep literal effect/-log10(p) coordinates and unchanged input while the default plot is square with four-square-point .6-alpha observations and an actual external key."""
    _defaults(monkeypatch);source=_frame();before=source.to_csv(index=False)
    figure,ax,hits=plot.volcano_plot(source,fold_change_col='effect',p_value_col='p',p_value_threshold=.05,show=False)
    assert source.to_csv(index=False)==before and hits==[]
    np.testing.assert_array_equal(ax.collections[0].get_offsets()[:,0],source['effect'])
    np.testing.assert_array_equal(ax.collections[0].get_offsets()[:,1],-np.log10(source['p']))
    spec=ax._spacr_style_spec
    assert spec.kind=='continuous' and dict(spec.legend)['unit']=='dimensionless'
    assert ax.collections[0].get_alpha()==.6
    np.testing.assert_array_equal(ax.collections[0].get_sizes(),[4.])
    assert ax.get_position().width*figure.get_figwidth()==pytest.approx(40/25.4,abs=.5/figure.dpi)
    assert ax.get_position().height*figure.get_figheight()==pytest.approx(40/25.4,abs=.5/figure.dpi)
    assert all(spine.get_visible() for spine in ax.spines.values())
    assert ax.get_legend() is not None
    figure.canvas.draw()
    for handle in ax.get_legend().legend_handles:
        if isinstance(handle,plot.mpl.collections.PathCollection):
            np.testing.assert_array_equal(handle.get_transform().get_matrix(),np.eye(3))
            centres=handle.get_offset_transform().transform(handle.get_offsets())
            assert all(ax.get_legend().get_window_extent().contains(*point) for point in centres)
    assert all(line.get_alpha()==.6 for line in ax.lines)
    assert all(np.allclose(plot.mpl.colors.to_rgb(line.get_color()),[0.,0.,0.],rtol=0,atol=1e-12) for line in ax.lines)
    assert not style.figure_problems(figure)
    plot.plt.close(figure)


def test_volcano_explicit_view_marks_and_parent_contents_remain_unchanged(monkeypatch):
    """Existing parent data/geometry and explicit alpha/marker size/limits are manual choices and must survive producer registration."""
    _defaults(monkeypatch);figure=Figure(figsize=(4.,3.));ax=figure.subplots()
    line,=ax.plot([-2.,2.],[.2,.8],color='purple',alpha=.2)
    figure._spacr_caption='Existing parent caption.'
    position=ax.get_position().bounds;coords=line.get_xydata().copy()
    returned,axis,hits=plot.volcano_plot(_frame(),fold_change_col='effect',p_value_col='p',ax=ax,
        alpha=.35,point_size=17.,xlim=(-3.,3.),ylim=(0.,4.),show=False)
    assert returned is figure and axis is ax
    assert ax.collections[0].get_alpha()==.35
    np.testing.assert_array_equal(ax.collections[0].get_sizes(),[17.])
    np.testing.assert_array_equal(line.get_xydata(),coords)
    assert line.get_alpha()==.2 and line.get_color()=='purple'
    assert ax.get_position().bounds==position
    assert ax.get_xlim()==(-3.,3.) and ax.get_ylim()==(0.,4.)
    assert ax._spacr_style_spec.kind=='continuous'
    assert figure._spacr_caption.startswith('Existing parent caption.')


def test_cloud_without_thresholds_uses_background_points_and_keeps_every_input_value(monkeypatch):
    """Without a selection threshold there are no highlighted categories, so the original complete effect/p data form a light grey cloud."""
    _defaults(monkeypatch);source=_frame()
    figure,ax,hits=plot.volcano_plot(source,fold_change_col='effect',p_value_col='p',show=False)
    colours=ax.collections[0].get_facecolors()
    expected=plot.mpl.colors.to_rgb(style.Palette.BACKGROUND)
    assert all(np.allclose(value[:3],expected,rtol=0,atol=1e-12) for value in colours)
    assert len(ax.collections[0].get_offsets())==len(source) and hits==[]
    plot.plt.close(figure)


def test_explicit_created_figure_choices_are_retained_and_diagnosed(monkeypatch):
    """A manually selected page, alpha, marker area, title and coloured threshold stay exact after producer defaults rather than being silently reset."""
    _defaults(monkeypatch)
    figure,ax,hits=plot.volcano_plot(_frame(),fold_change_col='effect',p_value_col='p',
        alpha=.35,point_size=17.,figsize=(5.,3.),title='Explicit dataset title',
        p_value_threshold=.05,threshold_line_kwargs={'color':'#C878A8','alpha':.4},show=False)
    np.testing.assert_array_equal(figure.get_size_inches(),[5.,3.])
    assert ax.collections[0].get_alpha()==.35
    np.testing.assert_array_equal(ax.collections[0].get_sizes(),[17.])
    assert ax.get_title()=='Explicit dataset title'
    assert ax.lines[0].get_alpha()==.4 and ax.lines[0].get_color()=='#C878A8'
    report=style.finalize_figure(figure,diagnostics=False)
    assert 'Move the plot title into the figure legend' in report.problems
    assert 'Draw reference lines in black; preserve explicit colours with a diagnostic' in report.problems
    assert ax.collections[0].get_alpha()==.35 and ax.lines[0].get_color()=='#C878A8'
    plot.plt.close(figure)


def test_default_black_reference_is_checked_against_actual_manual_ink():
    """The shared reference defaults to black, while a later explicitly grey line is diagnosed without changing its position or colour."""
    figure=Figure();ax=figure.subplots()
    line=style.reference_line(ax,x=0.,label='Displayed zero')
    assert plot.mpl.colors.to_rgb(line.get_color())==(0.,0.,0.)
    spec=style.resolve_style('print','continuous',legend={'caption':'Synthetic coordinates.','reference':'Displayed zero',
        'quantities':{'x':'x','y':'y'},'unit':'dimensionless'})
    style.apply_axes_style(ax,spec)
    line.set_color(style.Palette.CONTROL);before=line.get_xydata().copy()
    report=style.finalize_figure(figure,diagnostics=False)
    assert 'Draw reference lines in black; preserve explicit colours with a diagnostic' in report.problems
    assert line.get_color()==style.Palette.CONTROL
    np.testing.assert_array_equal(line.get_xydata(),before)


@pytest.mark.parametrize('transforms',[('none','-log10'),('log2','-log10'),('none','none')])
def test_unannotated_legacy_volcano_replays_source_points_keys_and_reference_coordinates(monkeypatch,transforms):
    """Existing reference recipes retain exact thresholds and zero coordinates; standalone replay must reproduce every source point, mask, key and line rather than report missing lines."""
    import json
    from spacr.figures import bundle

    _defaults(monkeypatch);source=_frame()
    if transforms[0]=='log2':source['effect']=[.5,1.,2.,4.,8.]
    figure,ax,hits=plot.volcano_plot(source,fold_change_col='effect',p_value_col='p',
        x_transform=transforms[0],y_transform=transforms[1],fold_change_threshold=1.5,p_value_threshold=.05,
        annotate=False,show=False)
    frame,spec=bundle._figure_record(figure);before=frame.to_csv(index=False)
    assert len(spec['references'])==4
    captured=json.loads(json.dumps(bundle._capture_view(figure,spec)))
    namespace={'__name__':'standalone_legacy_volcano'}
    exec(compile(bundle._recreate_script(captured),'recreate_figure.py','exec',dont_inherit=True),namespace)
    target=Figure();replay=namespace['_draw'](target,frame,captured)
    assert not target._spacr_replay_problems
    assert frame.to_csv(index=False)==before
    assert len(replay.collections)==len(ax.collections) and len(replay.lines)==len(ax.lines)
    np.testing.assert_array_equal(replay.collections[0].get_offsets(),ax.collections[0].get_offsets())
    assert style._artist_appearance(replay.collections[0])==style._artist_appearance(ax.collections[0])
    for original,copied in zip(ax.lines,replay.lines):
        np.testing.assert_array_equal(copied.get_xydata(),original.get_xydata())
        assert style._artist_appearance(copied)==style._artist_appearance(original)
    assert [text.get_text() for text in replay.get_legend().get_texts()]==[text.get_text() for text in ax.get_legend().get_texts()]
    assert replay.get_xlim()==ax.get_xlim() and replay.get_ylim()==ax.get_ylim()
    plot.plt.close(figure);plot.plt.close(target)
