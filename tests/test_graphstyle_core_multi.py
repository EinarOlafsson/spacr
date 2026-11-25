"""Multiple-field condition slots retain original numerical tables and explicit choices."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from spacr import plot
from spacr.figures import style
from spacr.qt import preferences


def _graph(monkeypatch,kind,count,**options):
    """Build actual grouped observations with reversed group/measurement order and a declared reference."""
    monkeypatch.setattr(preferences,'get_figure_style',dict)
    monkeypatch.setattr(preferences,'get_figure_style_per_graph',dict)
    frame=pd.DataFrame({'group':np.repeat(['a','b'],count),
        'late':np.tile(np.arange(count,dtype=float),2)+np.repeat([1.,4.],count),
        'early':np.tile(np.arange(count,dtype=float),2)*2+np.repeat([2.,8.],count)})
    graph=plot.spacrGraph(frame,'group',['late','early'],graph_type=kind,
        order=['b','a'],compare_group='a',**options)
    graph.create_plot()
    assert frame.to_csv(index=False)==graph.raw_df.to_csv(index=False)
    assert frame.to_csv(index=False)==graph.df.to_csv(index=False)
    return graph,graph.fig.axes[0]


@pytest.mark.parametrize('count,form',[(1,'bar'),(20,'bar'),(21,'box'),(1001,'violin')])
def test_multi_field_auto_slots_observe_thresholds_and_keep_original_bookkeeping(monkeypatch,count,form):
    """Literal 1/20/21/1001 samples must each retain four .22-inch slots, explicit control colours and the original melted summary bytes."""
    graph,ax=_graph(monkeypatch,'jitter_box',count)
    assert len(graph.fig.axes)==1
    assert [item.get_text() for item in ax.get_xticklabels()]==['b - late','b - early','a - late','a - early']
    spec=ax._spacr_style_spec
    assert spec.conditions==4 and spec.controls==(2,3)
    assert spec.sample_counts==(count,)*4
    assert ax.get_position().width*graph.fig.get_figwidth()==pytest.approx(4*.22,abs=.5/graph.fig.dpi)
    assert ax.get_ylabel()=='late, early'
    assert graph.summary_df.to_csv(index=False)==graph.df_melted.to_csv(index=False)
    assert ax._spacr_condition_kinds==(form,)*4
    points=[item for item in ax.collections if isinstance(item,PathCollection)]
    assert sum(len(item.get_offsets()) for item in points)==(count*4 if count<1000 else 0)
    assert not style._condition_presentation_problems(ax,spec)
    assert not any(item.get_text() in ('+','-') for item in ax.texts)
    assert not style.figure_problems(graph.fig)
    plot.plt.close(graph.fig)


@pytest.mark.parametrize('kind',['bar','box','violin','jitter'])
def test_multi_field_manual_forms_keep_values_and_use_shared_slots(monkeypatch,kind):
    """An explicit form remains explicit; SEM literals are the original groupby result on the 21-value fixture, kept exact with no tolerance."""
    graph,ax=_graph(monkeypatch,kind,21)
    assert graph.graph_type==kind
    assert ax._spacr_style_spec.controls==(2,3)
    assert ax._spacr_style_spec.sample_counts==(21,)*4
    assert len(graph.fig.axes)==1
    assert ax.get_position().width*graph.fig.get_figwidth()==pytest.approx(.88,abs=.5/graph.fig.dpi)
    if kind=='jitter':
        assert all(np.array_equal(item.get_sizes(),[4.]) for item in ax.collections if isinstance(item,PathCollection) and len(item.get_offsets()))
    if kind=='bar':
        assert sorted(bar.get_width() for bar in ax.patches)==pytest.approx([.6]*4,abs=1e-12)
        assert sorted(bar.get_x()+bar.get_width()/2. for bar in ax.patches)==[0.,1.,2.,3.]
        expected=pd.DataFrame({'Combined Group':['a - early','a - late','b - early','b - late'],
            'mean':[22.,11.,28.,14.], 'std':[np.std(np.arange(21.)*2,ddof=1),np.std(np.arange(21.),ddof=1)]*2,
            'sem':[2.70801280154532,1.35400640077266]*2,'count':[21]*4})
        pd.testing.assert_frame_equal(graph.summary_df,expected,check_exact=True)
    else:
        assert graph.summary_df.to_csv(index=False)==graph.df_melted.to_csv(index=False)
    plot.plt.close(graph.fig)


def test_multi_field_manual_palette_keeps_its_original_identity_mapping(monkeypatch):
    """The original hue encounter order maps purple/red/cyan to a-late/b-late/a-early; reordering tick slots must not reassign those choices."""
    graph,ax=_graph(monkeypatch,'bar',21,colors=['#663399','#D62728','#17BECF'])
    expected={0:to_rgb('#D62728'),1:to_rgb('#663399'),2:to_rgb('#663399'),3:to_rgb('#17BECF')}
    for bar in ax.patches:
        index=int(round(bar.get_x()+bar.get_width()/2.))
        np.testing.assert_allclose(bar.get_facecolor()[:3],expected[index],rtol=0,atol=1e-12)
    assert 'Draw each explicit control in grey and preserve declared identity colours' in style._condition_presentation_problems(ax,ax._spacr_style_spec)
    plot.plt.close(graph.fig)


@pytest.mark.parametrize('count',[1,21,1001])
def test_multi_field_portable_replay_preserves_each_wide_source_column(monkeypatch,count):
    """Replaying four slots must preserve exact measurement values and jitter from the original wide CSV, including no points for 1001 observations."""
    import json
    from matplotlib.figure import Figure
    from spacr.figures import bundle

    graph,ax=_graph(monkeypatch,'jitter_box',count)
    frame,spec=bundle._figure_record(graph.fig)
    before=frame.to_csv(index=False)
    captured=json.loads(json.dumps(bundle._capture_view(graph.fig,spec)))
    namespace={'__name__':'portable_multi_field'}
    exec(compile(bundle._recreate_script(captured),'recreate_figure.py','exec',dont_inherit=True),namespace)
    restored=Figure();replay=namespace['_draw'](restored,frame,captured)
    assert frame.to_csv(index=False)==before
    assert not restored._spacr_replay_problems
    assert replay._spacr_condition_kinds==ax._spacr_condition_kinds
    assert [text.get_text() for text in replay.get_xticklabels()]==[text.get_text() for text in ax.get_xticklabels()]
    assert replay.get_position().bounds==ax.get_position().bounds
    assert replay.get_xlim()==ax.get_xlim() and replay.get_ylim()==ax.get_ylim()
    source_points=[item.get_offsets() for item in ax.collections if isinstance(item,PathCollection)]
    replay_points=[item.get_offsets() for item in replay.collections if isinstance(item,PathCollection)]
    assert len(source_points)==len(replay_points)
    for source,target in zip(source_points,replay_points):
        np.testing.assert_array_equal(target,source)
    assert len(replay.patches)==len(ax.patches)
    for source,target in zip(ax.patches,replay.patches):
        np.testing.assert_array_equal(target.get_path().vertices,source.get_path().vertices)
    plot.plt.close(graph.fig);plot.plt.close(restored)


def test_slot_defaults_preserve_manual_marker_area_and_error_value_coordinates():
    """Original horizontal offset -.4 becomes slot 0 while every error/cap y value and the explicitly selected 17-square-point marker remain exact."""
    from matplotlib.figure import Figure

    figure=Figure();ax=figure.subplots()
    points=ax.scatter([-.2,.1],[2.,4.],s=17.,color='purple')
    ax.bar(-.4,3.,width=.8,color='purple')
    error=ax.errorbar(-.4,3.,yerr=2.,fmt='none',capsize=2.)
    before=[segment[:,1].copy() for collection in error.lines[2] for segment in collection.get_segments()]
    caps=[np.asarray(line.get_ydata()).copy() for line in error.lines[1]]
    style.categorical_slot_defaults(ax,'bar',conditions=1,point_area=None)
    np.testing.assert_array_equal(points.get_sizes(),[17.])
    assert ax.patches[0].get_x()+ax.patches[0].get_width()/2.==0.
    for expected,segment in zip(before,[segment for collection in error.lines[2] for segment in collection.get_segments()]):
        np.testing.assert_array_equal(segment[:,1],expected)
        np.testing.assert_array_equal(segment[:,0],[0.,0.])
    for expected,line in zip(caps,error.lines[1]):
        np.testing.assert_array_equal(line.get_ydata(),expected)
        np.testing.assert_array_equal(line.get_xdata(),[0.])
