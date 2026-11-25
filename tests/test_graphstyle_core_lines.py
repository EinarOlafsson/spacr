"""Core trends name actual axes, pooled summaries and explicit reference identities."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from spacr import plot
from spacr.figures import style,bundle
from spacr.qt import preferences


def _draw(monkeypatch,kind,columns):
    """Use actual line producers on complete literal observations, with declared units and a reordered reference."""
    monkeypatch.setattr(preferences,'get_figure_style',dict)
    monkeypatch.setattr(preferences,'get_figure_style_per_graph',dict)
    frame=pd.DataFrame({'group':np.repeat(['a','b','c'],6),'time':np.tile([1.,1.,2.,2.,3.,3.],3),
        'signal':[1.,3.,2.,4.,3.,5.,2.,4.,4.,6.,6.,8.,3.,5.,6.,8.,9.,11.]})
    frame.attrs['units']={'time':'s','signal':'a.u.'}
    graph=plot.spacrGraph(frame,'group',columns,graph_type=kind,order=['c','a','b'],compare_group='a')
    graph.create_plot();return graph,graph.fig.axes[0]


def test_numeric_line_metadata_names_the_actual_xy_columns_and_reference(monkeypatch):
    """The actual x/time and y/signal bindings must replace the former categorical metadata, while explicit reference a is grey."""
    graph,ax=_draw(monkeypatch,'line',['time','signal'])
    spec=ax._spacr_style_spec
    assert dict(dict(spec.legend)['quantities'])=={'x':'time','y':'signal'}
    legend_colours={line.get_label():to_rgb(line.get_color()) for line in ax.lines if line.get_label() in ['a','b','c']}
    assert legend_colours=={'a':to_rgb(style.Palette.CONTROL),'b':to_rgb(style.Palette.ACCENT),'c':to_rgb(style.Palette.ACCENT)}
    assert ax.get_legend().get_title().get_text()=='group'
    assert all(line.get_markersize()==2. for line in ax.lines if line.get_marker()=='o')
    assert all(line.get_markeredgewidth()==0. for line in ax.lines if line.get_marker()=='o')
    assert all(handle.get_alpha()==.6 for handle in ax.get_legend().legend_handles)
    assert graph.summary_df.to_csv(index=False)==graph.df.to_csv(index=False)
    assert not style.figure_problems(graph.fig)
    plot.plt.close(graph.fig)


def test_pooled_line_caption_does_not_claim_a_separately_drawn_reference(monkeypatch):
    """The existing mean/SD pivot pools all groups; describe exactly that and keep the literal original summary values."""
    graph,ax=_draw(monkeypatch,'line_std',['time','signal'])
    spec=ax._spacr_style_spec;legend=dict(spec.legend);comparison=dict(spec.comparison)
    assert dict(legend['quantities'])=={'x':'time','y':'signal'}
    assert 'pooled across groups' in legend['caption']
    assert 'one standard deviation' in legend['caption']
    assert comparison['statistical_reference']=='a'
    assert comparison['reference'] is None and legend['reference'] is None
    expected=graph.df.pivot_table(index='time',values='signal',aggfunc=['mean','std']).reset_index()
    expected.columns=['time','mean_signal','std_mean_signal']
    pd.testing.assert_frame_equal(graph.summary_df,expected,check_exact=True)
    assert any('pooled across groups' in entry['label'] for entry in ax._spacr_legend_entries)
    plot.plt.close(graph.fig)


def test_group_trend_points_show_the_explicit_reference_and_replay_exactly(monkeypatch):
    """Retain all original group centres and SD whiskers while the actual reference marker is grey and portable replay reproduces every point exactly."""
    import json
    from matplotlib.figure import Figure

    graph,ax=_draw(monkeypatch,'line',['signal'])
    points=[item for item in ax.collections if isinstance(item,PathCollection)]
    assert len(points)==1
    for container in ax.containers:
        if isinstance(container,plot.mpl.container.ErrorbarContainer):
            assert all(to_rgb(cap.get_markeredgecolor())==(0.,0.,0.) for cap in container.lines[1])
    np.testing.assert_array_equal(points[0].get_offsets()[:,1],graph.summary_df['centre'])
    np.testing.assert_array_equal(points[0].get_sizes(),[4.])
    np.testing.assert_allclose(points[0].get_facecolor()[:,:3],[to_rgb(style.Palette.ACCENT),to_rgb(style.Palette.CONTROL),to_rgb(style.Palette.ACCENT)],rtol=0,atol=1e-12)
    frame,spec=bundle._figure_record(graph.fig)
    captured=json.loads(json.dumps(bundle._capture_view(graph.fig,spec)))
    namespace={'__name__':'portable_group_trend'}
    exec(compile(bundle._recreate_script(captured),'recreate_figure.py','exec',dont_inherit=True),namespace)
    target=Figure();replay=namespace['_draw'](target,frame,captured)
    copied=[item for item in replay.collections if isinstance(item,PathCollection)]
    assert len(copied)==1
    np.testing.assert_array_equal(copied[0].get_offsets(),points[0].get_offsets())
    np.testing.assert_array_equal(copied[0].get_facecolor(),points[0].get_facecolor())
    assert not target._spacr_replay_problems
    plot.plt.close(graph.fig);plot.plt.close(target)


def test_actual_error_cap_marker_colour_is_validated_without_repair(monkeypatch):
    """A cap can report a black line colour while its visible underscore remains blue; inspect actual marker-edge ink and retain the explicit edit."""
    graph,ax=_draw(monkeypatch,'line',['signal'])
    container=next(item for item in ax.containers if isinstance(item,plot.mpl.container.ErrorbarContainer))
    cap=container.lines[1][0]
    cap.set_color('black');cap.set_markeredgecolor(style.Palette.ACCENT)
    before=np.asarray(cap.get_ydata()).copy()
    report=style.finalize_figure(graph.fig,diagnostics=False)
    assert 'Draw every error bar in black' in report.problems
    assert to_rgb(cap.get_markeredgecolor())==to_rgb(style.Palette.ACCENT)
    np.testing.assert_array_equal(cap.get_ydata(),before)
    plot.plt.close(graph.fig)
