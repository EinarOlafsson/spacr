"""Count heatmaps retain exact bins, masks and lower-origin category coordinates."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from spacr.figures import style


@pytest.mark.parametrize('origin', ['lower', 'upper'])
def test_shared_heatmap_keeps_autoscaled_values_and_places_crosses_at_actual_missing_cells(origin):
    """Both Matplotlib image origins retain source order and the original finite normalization endpoints, including an explicitly masked finite cell."""
    figure = Figure(); ax = figure.add_subplot()
    values = np.ma.array([[2., 9., np.nan], [5., 8., 4.]], mask=[[False, True, False], [False]*3])
    before = values.data.tobytes(), values.mask.tobytes()
    image = style.heatmap_image(ax, values, cmap='Blues', vmin=None, vmax=None,
                                origin=origin, extent=(-.5, 2.5, -.5, 1.5))
    assert image.origin == origin
    assert image.get_clim() == (2., 8.)
    np.testing.assert_array_equal(image.get_array().data, values.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(image.get_array()), [[False, True, True], [False]*3])
    crosses = next(item for item in ax.collections if item.get_gid() == 'spacr-missing')
    np.testing.assert_array_equal(crosses.get_offsets(), [[1., 0. if origin == 'lower' else 1.], [2., 0. if origin == 'lower' else 1.]])
    assert image.get_alpha() == .6
    assert (values.data.tobytes(), values.mask.tobytes()) == before


@pytest.mark.qt
def test_graph_canvas_count_heatmap_keeps_crosstab_and_lower_origin_with_explicit_empty_bin_marks(qtbot, monkeypatch):
    """Counts and category order stay exact; empty bins retain the existing mask and receive explained black crosses rather than changing zero-count semantics."""
    from spacr.qt import preferences
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.widgets.graph_builder import GraphCanvas
    from spacr.qt.widgets.graph_spec import CATEGORICAL, GraphSpec

    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    frame = pd.DataFrame({'category':['a','a','b'], 'other':['u','v','v']})
    before = frame.to_csv(index=False)
    widget = GraphCanvas(link=LinkedSelection()); qtbot.addWidget(widget)
    widget._take_frame(frame)
    widget.set_spec(GraphSpec(x='category',y='other',kind='heatmap',
        roles={'category':CATEGORICAL,'other':CATEGORICAL}))
    widget._canvas.draw()
    ax = next(iter(widget.panel_axes().values())); image = ax.images[0]
    assert image.origin == 'lower'
    np.testing.assert_array_equal(image.get_array().compressed(), [1.,1.,1.])
    np.testing.assert_array_equal(np.ma.getmaskarray(image.get_array()), [[False,True],[False,False]])
    np.testing.assert_array_equal(next(item for item in ax.collections if item.get_gid() == 'spacr-missing').get_offsets(), [[1.,0.]])
    assert image.get_clim() == (1.,1.)
    assert image.get_alpha() == .6
    assert not style._heatmap_artist_problems(ax)
    assert not style._quantity_unit_problems(ax._spacr_style_spec)
    assert any('no observations' in entry['label'] for entry in ax._spacr_legend_entries)
    assert widget._visible.to_csv(index=False) == before
    assert frame.to_csv(index=False) == before
