"""Live display uses the shared finalizer without repainting or changing scientific values."""
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from spacr.figures import style


def _registered_figure():
    """Create an explicitly described scatter with exact reusable source coordinates."""
    with style.figure_style('print'):
        figure=Figure();ax=figure.add_subplot()
        points=ax.scatter([.2,.8],[.3,.7],color=style.Palette.ACCENT,alpha=.6)
        ax.set_xlabel('Coordinate (a.u.)');ax.set_ylabel('Value (a.u.)')
        spec=style.resolve_style('screen','scatter',legend={'caption':'Synthetic coordinates.',
            'reference':'origin','unit':'a.u.','quantities':{'x':'coordinate','y':'value'}})
        style.apply_axes_style(ax,spec)
    return figure,points


def test_existing_renderer_is_validated_without_another_draw(monkeypatch):
    """After the real Agg draw, even one additional draw is a regression; reuse its actual renderer and preserve canvas, coordinates and limits."""
    figure,points=_registered_figure();canvas=FigureCanvasAgg(figure);canvas.draw()
    renderer=canvas.get_renderer()
    values=points.get_offsets().tobytes()
    limits=(figure.axes[0].get_xlim(),figure.axes[0].get_ylim())

    def forbidden():
        """Reject recursive or duplicate paint requests after the completed draw."""
        raise AssertionError('validator requested another draw')

    monkeypatch.setattr(canvas,'draw',forbidden)
    result=style.finalize_figure(figure,renderer=renderer)
    assert 'text bounds' in result.checks and 'text overlap' in result.checks
    assert figure.canvas is canvas and points.get_offsets().tobytes()==values
    assert (figure.axes[0].get_xlim(),figure.axes[0].get_ylim())==limits


@pytest.mark.qt
def test_shared_qt_canvas_validates_after_exactly_one_actual_draw(qtbot,monkeypatch):
    """The shared Graph Builder canvas must paint once and retain the resulting diagnostics; source offsets remain byte-identical."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from spacr.qt.widgets.graph_builder import _draw_with_user_style

    figure,points=_registered_figure();canvas=FigureCanvasQTAgg(figure);qtbot.addWidget(canvas)
    before=points.get_offsets().tobytes();original=FigureCanvasQTAgg.draw;draws=[]

    def counted(self):
        """Count real Qt draws without replacing their renderer or painting behavior."""
        draws.append(self)
        return original(self)

    monkeypatch.setattr(FigureCanvasQTAgg,'draw',counted)
    _draw_with_user_style(canvas)
    assert draws==[canvas]
    assert 'text bounds' in figure._spacr_style_diagnostics.checks
    assert figure.canvas is canvas and points.get_offsets().tobytes()==before


@pytest.mark.qt
def test_native_grouped_display_records_the_same_semantic_checks(qtbot):
    """A real native display now retains condition form/count checks before any file export, with unchanged input observations."""
    import pandas as pd
    from spacr.qt.widgets.grouped_plot import GroupedPlot,PlotSpec

    frame=pd.DataFrame({'group':['reference','treated'],'value':[1.,2.]})
    source=frame.to_csv(index=False).encode()
    widget=GroupedPlot(PlotSpec(frame,'value',group='group',background='reference'))
    qtbot.addWidget(widget)
    result=widget._spacr_style_diagnostics
    assert 'actual condition forms, counts and identities' in result.checks
    assert widget.spec.frame.to_csv(index=False).encode()==source


@pytest.mark.parametrize('layout', ['fit', 'pack'])
def test_physical_layout_measures_text_without_painting_plot_marks(layout, monkeypatch):
    """Both layout owners repainted all marks up to three times; require zero paints and preserve .22-inch slots within the existing half-pixel 300-dpi margin, with actual final text inside the page."""
    import numpy as np

    with style.figure_style('print'):
        figure = Figure(dpi=300)
        axes = list(figure.subplots(1, 2 if layout == 'pack' else 1, squeeze=False).ravel())
        for ax in axes:
            values = [np.linspace(1., 2., 20), np.linspace(2., 3., 21)]
            style.condition_plot(ax, values)
            ax.set_xticks([0, 1], ['reference_alpha', 'treated_beta'], rotation=45, ha='right')
            ax.set_xlabel('Condition'); ax.set_ylabel('Value (a.u.)')
            spec = style.resolve_style('print', 'categorical', conditions=['reference_alpha', 'treated_beta'],
                                       sample_counts=[20, 21], control=0)
            style.apply_axes_style(ax, spec, layout=False)
        source = [artist.get_offsets().copy() for ax in axes for artist in ax.collections
                  if hasattr(artist, 'get_offsets')]
        actual_draw = FigureCanvasAgg.draw

        def forbidden(self):
            """Reject all plot paints during metric-only physical layout."""
            raise AssertionError('layout repainted data marks')

        monkeypatch.setattr(FigureCanvasAgg, 'draw', forbidden)
        original_canvas = figure.canvas
        if layout == 'fit':
            style.fit_axes_geometry(axes[0], axes[0]._spacr_style_spec)
        else:
            style.pack_axes_geometry(axes)
        assert figure.canvas is original_canvas
        canvas = FigureCanvasAgg(figure)
        actual_draw(canvas)
        renderer = canvas.get_renderer()
        for ax in axes:
            assert ax.get_window_extent(renderer).width / figure.dpi == pytest.approx(.44, abs=.5/300)
            box = ax.get_tightbbox(renderer)
            assert box.x0 >= -.5 and box.y0 >= -.5
            assert box.x1 <= figure.bbox.width+.5 and box.y1 <= figure.bbox.height+.5
        after = [artist.get_offsets() for ax in axes for artist in ax.collections
                 if hasattr(artist, 'get_offsets')]
        for before, current in zip(source, after):
            np.testing.assert_array_equal(before, current)


@pytest.mark.qt
@pytest.mark.parametrize('orientation', ['vertical', 'horizontal'])
@pytest.mark.parametrize('count', [1, 20, 21, 1000])
def test_builder_automatic_conditions_keep_physical_slots_after_viewport_resize(qtbot, monkeypatch, count, orientation):
    """The first after gallery still used boxes for fresh preferences and stretched three conditions across 6.9 inches; require the shared forms and .22-inch slots within one display pixel before and after a real Qt resize, with exact source rows."""
    import numpy as np
    import pandas as pd
    from matplotlib.collections import PathCollection
    from PySide6.QtWidgets import QApplication
    from spacr.qt import preferences
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.widgets.graph_builder import GraphCanvas
    from spacr.qt.widgets.graph_spec import GraphSpec, CATEGORICAL, CONTINUOUS

    def no_choice(shape):
        """Simulate a fresh graph-choice store, whose empty string means automatic."""
        return ''

    monkeypatch.setattr(preferences, 'get_default_graph_type', no_choice)
    frame = pd.DataFrame({'condition': np.repeat(['reference', 'treated'], count),
                          'value': np.tile(np.linspace(1., 2., count), 2)})
    frame.attrs.update(reference='reference', units={'value': 'a.u.'}, experimental_unit='synthetic observation')
    before = frame.to_csv(index=False).encode()
    widget = GraphCanvas(link=LinkedSelection(), source='physical-conditions')
    qtbot.addWidget(widget)
    widget.resize(760, 560)
    widget._take_frame(frame)
    x, y = ('value', 'condition') if orientation == 'horizontal' else ('condition', 'value')
    widget.set_spec(GraphSpec(x=x, y=y, roles={'condition': CATEGORICAL, 'value': CONTINUOUS}))
    widget.show()
    QApplication.processEvents()
    ax = widget.figure().axes[0]
    assert ax._spacr_condition_kinds == (style.condition_kind(np.arange(count)),) * 2
    assert sum(len(artist.get_offsets()) for artist in ax.collections if isinstance(artist, PathCollection)) == (2*count if count < 1000 else 0)
    for size in ((760, 560), (1100, 700)):
        widget.resize(*size)
        QApplication.processEvents()
        widget._canvas.draw()
        area = ax.get_window_extent(widget._canvas.get_renderer())
        length = area.height if orientation == 'horizontal' else area.width
        assert length / widget.figure().dpi == pytest.approx(.44, abs=1./widget.figure().dpi)
        assert widget._visible.to_csv(index=False).encode() == before
        assert widget._render_data.frame.to_csv(index=False).encode() == before
    assert ax._spacr_style_spec.controls == (0,)
    assert frame.to_csv(index=False).encode() == before


def test_live_physical_layout_releases_explicit_later_panel_placement():
    """Automatic viewport placement must not undo a later manual axes rectangle."""
    figure, _ = _registered_figure()
    style._retain_live_axes_geometry(figure, (7., 5.))
    ax = figure.axes[0]
    ax.set_position([.2, .3, .4, .5])
    chosen = tuple(ax.get_position(original=True).bounds)
    assert not style._apply_live_axes_geometry(figure)
    assert tuple(ax.get_position(original=True).bounds) == chosen


@pytest.mark.qt
def test_wide_builder_conditions_scroll_instead_of_shrinking_slots(qtbot, monkeypatch):
    """Sixty .22-inch slots need 13.2 inches; preserve them and expose horizontal scrolling in a 500-pixel viewport instead of clipping scientific marks."""
    import numpy as np
    import pandas as pd
    from PySide6.QtWidgets import QApplication
    from spacr.qt import preferences
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.widgets.graph_builder import GraphCanvas
    from spacr.qt.widgets.graph_spec import GraphSpec, CATEGORICAL

    def no_choice(shape):
        """Use the automatic presentation with no persisted manual form."""
        return ''

    monkeypatch.setattr(preferences, 'get_default_graph_type', no_choice)
    frame = pd.DataFrame({'condition': [f'c{index:02d}' for index in range(60)], 'value': np.arange(60, dtype=float)})
    widget = GraphCanvas(link=LinkedSelection(), source='wide-conditions')
    qtbot.addWidget(widget)
    widget.resize(500, 400)
    widget._take_frame(frame)
    widget.set_spec(GraphSpec(x='condition', y='value', roles={'condition': CATEGORICAL}))
    widget.show()
    QApplication.processEvents()
    widget._canvas.draw()
    ax = widget.figure().axes[0]
    assert ax.get_window_extent(widget._canvas.get_renderer()).width / widget.figure().dpi == pytest.approx(13.2, abs=1./widget.figure().dpi)
    assert widget._plot_scroll.horizontalScrollBar().maximum() > 0


@pytest.mark.parametrize('ordinal, label', [(0, 'A'), (25, 'Z'), (26, 'AA'), (51, 'AZ'), (52, 'BA'), (143, 'EN')])
def test_facet_panel_letters_continue_as_uppercase_identifiers(ordinal, label):
    """The supported 144-panel grid exceeds Z; require complete uppercase labels through EN without punctuation."""
    figure = Figure(); ax = figure.add_subplot()
    style.panel_letter(ax, ordinal)
    assert ax.texts[-1].get_text() == label
    assert ax.texts[-1].get_fontweight() == 'bold'
    assert ax.texts[-1].get_fontsize() == 10


def test_packed_live_legend_has_its_own_column_and_keeps_later_manual_placement():
    """The old upper-right figure key could overlap data; reserve its measured envelope and preserve an explicit later anchor across a viewport resize."""
    figure, points = _registered_figure()
    ax = figure.axes[0]
    legend = figure.legend([points], ['Measured condition'], loc='upper right', frameon=False, fontsize=8)
    style.pack_axes_geometry([ax], legend_space=True)
    style._retain_live_axes_geometry(figure, (7., 5.))
    canvas = FigureCanvasAgg(figure); canvas.draw()
    renderer = canvas.get_renderer()
    assert legend.get_window_extent(renderer).x0 >= ax.get_window_extent(renderer).x1
    legend.set_bbox_to_anchor((.1, .2), transform=figure.transFigure)
    figure.set_size_inches(8., 6.)
    assert style._apply_live_axes_geometry(figure)
    actual = legend.get_bbox_to_anchor().bounds
    assert actual[0] / figure.bbox.width == pytest.approx(.1)
    assert actual[1] / figure.bbox.height == pytest.approx(.2)
