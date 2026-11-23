"""Manual core distribution forms preserve data while using shared physical body widths."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.patches import PathPatch
from spacr import plot
from spacr.figures import style
from spacr.qt import preferences


@pytest.mark.parametrize('kind',['box','violin'])
def test_core_manual_distribution_bodies_use_equal_slots_and_keep_the_original_tables(monkeypatch,kind):
    """Explicit box/violin choices retain their original observations, bookkeeping and type while every body spans .6 of the shared .22-inch condition slot."""
    monkeypatch.setattr(preferences,'get_figure_style',dict)
    monkeypatch.setattr(preferences,'get_figure_style_per_graph',dict)
    frame=pd.DataFrame({'group':np.repeat(['a','b','c'],21),'value':np.tile(np.arange(21.),3)+np.repeat([0.,1.,2.],21)})
    source=frame.to_csv(index=False)
    graph=plot.spacrGraph(frame,'group','value',graph_type=kind,order=['c','a','b'],compare_group='a')
    graph.create_plot();figure=graph.fig;ax=figure.axes[0]
    bodies=list(ax.patches) if kind=='box' else [mark for mark in ax.collections if isinstance(mark,PolyCollection)]
    bodies=[mark for mark in bodies if isinstance(mark,(PathPatch,PolyCollection))]
    assert len(bodies)==3
    for mark in bodies:
        paths=[mark.get_path()] if isinstance(mark,PathPatch) else mark.get_paths()
        width=max(path.vertices[:,0].max() for path in paths)-min(path.vertices[:,0].min() for path in paths)
        assert width==pytest.approx(.6,abs=1e-12)
        assert mark.get_alpha()==.6
    assert ax.get_position().width*figure.get_size_inches()[0]==pytest.approx(3*.22,abs=.5/figure.dpi)
    assert ax.get_ylabel()=='value'
    assert graph.df.to_csv(index=False)==source
    assert graph.raw_df.to_csv(index=False)==source
    assert graph.summary_df['Value'].tolist()==graph.df_melted['Value'].tolist()
    assert graph.graph_type==kind
    if kind=='violin':
        assert all(plot.mpl.colors.to_rgb(line.get_markeredgecolor())==(0.,0.,0.)
                   for line in ax.lines if line.get_marker() not in (None,'None','none',''))
    if kind=='violin':
        medians=[line for line in ax.lines if len(line.get_xdata())==2
                 and np.ptp(line.get_ydata())==0 and np.isclose(np.ptp(line.get_xdata()),.6)]
        assert len(medians)==3
        np.testing.assert_array_equal([line.get_ydata()[0] for line in medians],[12.,10.,11.])
    if kind=='box':
        expected=[(7.,17.),(5.,15.),(6.,16.)]
        for mark,quartiles in zip(bodies,expected):
            np.testing.assert_array_equal(np.unique(mark.get_path().vertices[:,1]),quartiles)
    assert any(np.allclose(mark.get_facecolor()[:3] if isinstance(mark,PathPatch) else mark.get_facecolor()[0,:3],
                          plot.mpl.colors.to_rgb(style.Palette.CONTROL),rtol=0,atol=1e-12) for mark in bodies)
