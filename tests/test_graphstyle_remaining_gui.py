"""Remaining GUI creators retain scientific rows while sharing physical presentation."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from matplotlib.text import Text

from spacr.figures import style
from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.widgets.graph_spec import CATEGORICAL, CONTINUOUS, GraphSpec
from spacr.qt.widgets.trellis_spec import SCALE_FREE, TrellisSpec
from spacr.qt.widgets.trellis_view import TrellisCanvas

pytestmark = pytest.mark.qt


@pytest.mark.parametrize('count', [1, 20, 21, 1001])
@pytest.mark.parametrize('scale_mode', ['shared', SCALE_FREE])
def test_trellis_uses_shared_physical_slots_and_external_counts_without_changing_rows(qtbot, monkeypatch, count, scale_mode):
    """The Trellis subclass bypassed final layout; preserve each source row and shared/free value limits while giving both facets .22-inch slots and readable external counts."""
    from spacr import graph_types
    from spacr.qt import preferences

    def automatic(_shape):
        """Use the automatic form independently of any workstation graph preference."""
        return 'auto'

    monkeypatch.setattr(graph_types, 'chosen_for', automatic)
    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    frame = pd.DataFrame({'category':np.repeat(['reference', 'treated', 'third'], 2*count),
        'plateID':np.tile(np.repeat(['p0', 'p1'], count), 3),
        'metric':np.linspace(.5, 2., 6*count)})
    frame.attrs['units'] = {'metric':'a.u.'}
    frame.attrs['reference'] = 'reference'
    before = frame.to_csv(index=False)
    widget = TrellisCanvas(link=LinkedSelection())
    qtbot.addWidget(widget)
    widget._take_frame(frame)
    graph = GraphSpec(x='category', y='metric', facet_col='plateID',
                      roles={'category':CATEGORICAL, 'metric':CONTINUOUS, 'plateID':CATEGORICAL})
    widget.set_trellis_spec(TrellisSpec(graph=graph, scale_x=scale_mode, scale_y=scale_mode))
    widget._canvas.draw()
    axes = list(widget.panel_axes().values())
    assert len(axes) == 2
    for ax in axes:
        assert ax.get_position().width*widget._figure.get_size_inches()[0] == pytest.approx(.66, abs=.5/300)
        assert not ax.get_title()
        assert all(text.get_fontsize() >= 8 for text in ax.findobj(Text) if text.get_visible() and text.get_text())
        notes = [entry['label'] for entry in getattr(ax, '_spacr_legend_entries', ())]
        assert any(str(3*count) in note.replace(',', '') and 'n=' in note for note in notes)
        assert ax._spacr_style_spec.sample_counts == (count, count, count)
        assert not style._condition_presentation_problems(ax, ax._spacr_style_spec)
        points = [artist for artist in ax.collections if isinstance(artist, PathCollection)]
        assert sum(len(artist.get_offsets()) for artist in points) == (3*count if count < 1000 else 0)
    if scale_mode == 'shared':
        assert axes[0].get_ylim() == axes[1].get_ylim()
    else:
        assert axes[0].get_ylim() != axes[1].get_ylim()
    assert widget._visible.to_csv(index=False) == before
    assert frame.to_csv(index=False) == before


@pytest.mark.parametrize('trellis', [False, True])
def test_late_canvas_labels_and_letters_use_shared_font_and_ink(qtbot, monkeypatch, trellis):
    """Actual render review found fallback-font panel letters and GUI-coloured labels created after initial styling; both canvas routes must retain the shared font and black ink."""
    from matplotlib.colors import to_rgba
    from spacr.qt import preferences
    from spacr.qt.widgets.graph_builder import GraphCanvas

    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    frame = pd.DataFrame({'category':['c0','c1','c2']*2, 'plateID':['p0']*3+['p1']*3,
                          'metric':[1.,2.,3.,2.,3.,4.]})
    frame.attrs['units'] = {'metric':'a.u.'}
    graph = GraphSpec(x='category', y='metric', facet_col='plateID',
                      roles={'category':CATEGORICAL, 'metric':CONTINUOUS, 'plateID':CATEGORICAL})
    widget = (TrellisCanvas if trellis else GraphCanvas)(link=LinkedSelection())
    qtbot.addWidget(widget)
    widget._take_frame(frame)
    if trellis:
        widget.set_trellis_spec(TrellisSpec(graph=graph))
    else:
        widget.set_spec(graph)
    widget._canvas.draw()
    letters = [text for ax in widget.panel_axes().values() for text in ax.texts
               if text.get_gid() == 'spacr-panel-letter']
    assert [text.get_text() for text in letters] == ['A','B']
    assert all(text.get_fontfamily() == ['Open Sans'] for text in letters)
    for ax in widget.panel_axes().values():
        assert to_rgba(ax.xaxis.label.get_color()) == to_rgba('#000000')
        assert to_rgba(ax.yaxis.label.get_color()) == to_rgba('#000000')
