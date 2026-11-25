"""Bar error widths stay attached to their own multiple-measurement categories."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from spacr.plot import spacrGraph
from spacr.figures.style import pipeline_figures


def _frame():
    """Return four literal mean/SD pairs:3/2, 21/1, 10/0 and 11/4."""
    return pd.DataFrame({'group':['a']*3+['b']*3,
        'late':[1.,3.,5.,20.,21.,22.], 'early':[10.,10.,10.,7.,11.,15.]})


def _draw(columns,order):
    """Use the real producer without unrelated statistical model or layout work."""
    frame=_frame();before=frame.to_csv(index=False)
    graph=spacrGraph(frame,'group',columns,graph_type='bar',order=order)
    graph.df_melted=pd.melt(graph.df,id_vars=['group'],value_vars=columns,
        var_name='Data Column',value_name='Value')
    graph.hue='Data Column' if len(columns)>1 else 'group'
    graph.jitter_bar_dodge=len(columns)>1;graph.bar_width=.4
    figure=Figure();ax=figure.subplots()
    with pipeline_figures():
        graph._create_bar_plot(ax)
    errors={}
    for item in ax.containers:
        if isinstance(item,ErrorbarContainer):
            for collection in item.lines[2]:
                for segment in collection.get_segments():
                    errors[float(segment[:,1].mean())]=float(np.ptp(segment[:,1])/2.)
    assert frame.to_csv(index=False)==before
    assert graph.df.to_csv(index=False)==before
    assert graph.raw_df.to_csv(index=False)==before
    return graph,errors


@pytest.mark.parametrize('order',[['b','a'],['missing','b','a']])
def test_multi_field_bar_errors_follow_their_labels_even_with_an_empty_category(order):
    """Pristine code pairs all four SDs with the wrong bars; a declared but absent group creates no invented summary or error width."""
    graph,errors=_draw(['late','early'],order)
    assert errors=={3.:2.,21.:1.,10.:0.,11.:4.}
    assert graph.summary_df['Combined Group'].tolist()==['a - early','a - late','b - early','b - late']
    np.testing.assert_array_equal(graph.summary_df['mean'],[10.,3.,11.,21.])
    np.testing.assert_array_equal(graph.summary_df['std'],[0.,2.,4.,1.])
    np.testing.assert_array_equal(graph.summary_df['count'],[3,3,3,3])


def test_duplicate_category_labels_are_rejected_before_bar_pairing():
    """The existing category owner refuses duplicate ordered identities rather than drawing an ambiguous repeated label."""
    with pytest.raises(ValueError,match='unique'):
        spacrGraph(_frame(),'group',['late','early'],graph_type='bar',order=['a','a','b'])


def test_single_field_bar_summary_bytes_and_error_associations_are_unchanged():
    """Keep the exact existing single-field CSV and both known mean/SD pairs; the new label lookup is restricted to multiple fields."""
    graph,errors=_draw(['late'],['b','a'])
    assert errors=={21.:1.,3.:2.}
    assert graph.summary_df.to_csv(index=False)==('group,mean,std,sem,count\n'
        'b,21.0,1.0,0.5773502691896257,3\n'
        'a,3.0,2.0,1.1547005383792515,3\n')
