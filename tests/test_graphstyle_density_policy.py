"""Density appearance preserves original histogram counts and weighted means."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from spacr.figures import style
from spacr.qt.widgets.graph_builder import GraphCanvas
from spacr.qt.widgets.graph_spec import GraphSpec

pytestmark=pytest.mark.qt


@pytest.mark.parametrize('weighted',[False,True])
def test_density_keeps_original_counts_weights_and_masks_and_explains_empty_cells(qtbot,weighted):
    """The same two occupied bins retain counts 2/2 or the original missing-weight-as-zero means 1/8; no masked cell becomes a numeric zero."""
    frame=pd.DataFrame({'x':[.1,.1,.9,.9,np.nan], 'y':[.1,.1,.9,.9,.5],
                        'weight':[2.,np.nan,6.,10.,99.]})
    before=frame.to_csv(index=False)
    view=GraphCanvas();qtbot.addWidget(view)
    view._spec=GraphSpec(x='x',y='y',colour='weight' if weighted else None,kind='scatter',bins=2)
    view._scales=SimpleNamespace(x_edges=np.array([0.,.5,1.]),y_edges=np.array([0.,.5,1.]),
                                colour_limits=(0.,10.) if weighted else None)
    figure=Figure();ax=figure.add_subplot()
    view._draw_density(ax,frame,{})
    image=ax.images[0]
    expected=np.array([[1.,np.nan],[np.nan,8.]]) if weighted else np.array([[2.,np.nan],[np.nan,2.]])
    np.testing.assert_array_equal(image.get_array().filled(np.nan),expected)
    np.testing.assert_array_equal(np.ma.getmaskarray(image.get_array()),[[False,True],[True,False]])
    assert image.origin=='lower' and ax.get_aspect()=='auto'
    assert tuple(image.get_extent())==(0.,1.,0.,1.)
    assert image.get_clim()==((1.,8.) if weighted else (2.,2.))
    assert image.get_alpha()==.6
    np.testing.assert_allclose(image.cmap.get_bad(),image.cmap(0.),rtol=0,atol=1e-12)
    crosses=next(mark for mark in ax.collections if mark.get_gid()=='spacr-missing')
    np.testing.assert_array_equal(crosses.get_offsets(),[[.75,.25],[.25,.75]])
    assert crosses.get_alpha()==.6 and crosses.get_linewidths()[0]==.4
    assert any('finite displayed value' in entry['label'] for entry in ax._spacr_legend_entries)
    assert frame.to_csv(index=False)==before


def test_continuous_density_missing_marks_fit_dense_cells_without_changing_the_matrix():
    """A 40x40 density on the existing 40-mm square needs crosses below half a cell; the ordinary categorical heatmap retains its larger marker."""
    values=np.full((40,40),np.nan);values[20,20]=3.
    figure=Figure();ax=figure.add_subplot()
    image=style.heatmap_image(ax,values,cmap='Blues',vmin=None,vmax=None,aspect='auto')
    cross=next(mark for mark in ax.collections if mark.get_gid()=='spacr-missing')
    width=style.axes_geometry(style.resolve_style('screen','scatter')).width_inches
    assert np.sqrt(cross.get_sizes()[0])/72. <= width/40*.5
    assert image.get_clim()==(3.,3.)
    np.testing.assert_array_equal(image.get_array().filled(np.nan),values)
