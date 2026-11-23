"""PCA presentation preserves the existing decomposition and exported numerical columns."""
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.text import Text

from spacr.figures import bundle, style
from spacr.qt.widgets import pca_view
from spacr.qt.widgets.pca_model import PCAResult

pytestmark = pytest.mark.qt


def _result(count):
    """Return deterministic synthetic PCA arrays without fitting a decomposition."""
    rng = np.random.default_rng(73)
    variance = np.arange(count, 0, -1, dtype=float)
    return PCAResult(features=tuple(f'feature_{i}' for i in range(12)), rows=np.arange(60),
        scores=rng.normal(size=(60, count)), loadings=np.eye(12, count), correlations=np.eye(12, count),
        explained_variance=variance, explained_variance_ratio=variance/variance.sum(),
        total_variance=float(variance.sum()), centre=np.zeros(12), scale=np.ones(12),
        rank=count, n_rows_in=60, n_features_in=12)


@pytest.mark.parametrize('count', [1, 6, 12])
@pytest.mark.parametrize('theme_name', ['light', 'dark'])
def test_scree_uses_shared_slots_and_linked_axes_with_exact_variance_values(qtbot, monkeypatch, count, theme_name):
    """Keep both original percentage calculations and exported columns while using .22-inch component slots, .6 marks and readable linked-axis labels in both themes."""
    from spacr.qt import preferences, theme

    def palette():
        """Select the actual GUI palette independently of the workstation settings store."""
        return dict(theme._PALETTES[theme_name])

    monkeypatch.setattr(pca_view, 'active_palette', palette)
    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    result = _result(count)
    arrays = [item.tobytes() for item in (result.scores, result.loadings, result.correlations,
                                         result.explained_variance, result.explained_variance_ratio)]
    widget = pca_view.ScreePlot(); qtbot.addWidget(widget)
    widget.set_result(result)
    widget._canvas.draw()
    primary, overlay = widget._figure.axes
    assert primary.get_position().width*widget._figure.get_size_inches()[0] == pytest.approx(count*.22, abs=.5/300)
    assert primary.get_position().bounds == overlay.get_position().bounds
    np.testing.assert_array_equal([bar.get_height() for bar in primary.patches], result.explained_variance_ratio*100.)
    np.testing.assert_array_equal(overlay.lines[0].get_ydata(), np.cumsum(result.explained_variance_ratio)*100.)
    assert overlay.get_ylim() == (0., 105.)
    assert all(bar.get_width() == pytest.approx(.6, abs=1e-12) and bar.get_alpha() == .6 for bar in primary.patches)
    assert overlay.lines[0].get_alpha() == .6
    assert overlay._spacr_style_spec.kind == 'axis_overlay'
    assert all(text.get_fontsize() >= 8 for ax in widget._figure.axes
               for text in ax.findobj(Text) if text.get_visible() and text.get_text())
    assert all(to_rgba(ax.yaxis.label.get_color()) == to_rgba('#000000') for ax in widget._figure.axes)
    frame, _recipe = bundle._figure_record(widget._figure)
    assert list(frame.columns) == ['component', 'variance_percent']
    np.testing.assert_array_equal(frame['variance_percent'], result.explained_variance_ratio*100.)
    assert [item.tobytes() for item in (result.scores, result.loadings, result.correlations,
                                       result.explained_variance, result.explained_variance_ratio)] == arrays
    report = style.finalize_figure(widget._figure)
    assert report.compliant, report.problems


def test_empty_scree_status_uses_readable_shared_text_without_inventing_a_decomposition(qtbot):
    """An empty widget retains its status message with readable shared type and no numerical marks."""
    widget = pca_view.ScreePlot(); qtbot.addWidget(widget)
    widget.set_result(None)
    widget._canvas.draw()
    ax = widget._figure.axes[0]
    assert not ax.patches and not ax.lines
    assert any('no components yet' in text.get_text().lower() and text.get_fontsize() >= 8
               and text.get_gid() == 'spacr-status' for text in ax.texts)


def test_biplot_preserves_every_correlation_vector_and_moves_feature_labels_to_the_legend(qtbot, monkeypatch):
    """Keep every finite projected vector, its shared display scale and all score rows while applying shared opacity and stating feature identities outside the data area."""
    import pandas as pd
    from spacr.qt import preferences
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.widgets.graph_spec import CONTINUOUS, GraphSpec

    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    result = _result(6)
    source = pd.DataFrame(np.random.default_rng(83).normal(size=(60,12)), columns=result.features)
    frame = result.scores_frame(source)
    before = frame.to_csv(index=False)
    correlations = result.correlations.tobytes()
    widget = pca_view.PCAScoresCanvas(link=LinkedSelection()); qtbot.addWidget(widget)
    widget._take_result(result, frame)
    widget.set_spec(pca_view.GraphSpec(x='PC1', y='PC2', kind='scatter', roles={'PC1':CONTINUOUS, 'PC2':CONTINUOUS}))
    widget._canvas.draw()
    ax = next(iter(widget.panel_axes().values()))
    picked = result.plane_features(0, 1, widget._arrow_count)
    scale = pca_view.arrow_scale(result, 0, 1, ax.get_xlim(), ax.get_ylim(), count=widget._arrow_count)
    assert widget.arrow_scale == scale
    arrows = [text for text in ax.texts if getattr(text, 'arrow_patch', None) is not None]
    expected = [(float(result.correlations[index,0])*scale, float(result.correlations[index,1])*scale) for index in picked]
    np.testing.assert_array_equal([arrow.xy for arrow in arrows], expected)
    assert all(arrow.arrow_patch.get_alpha() == .6 for arrow in arrows)
    assert not any(text.get_text() in result.features for text in ax.texts)
    notes = [entry['label'] for entry in getattr(ax, '_spacr_legend_entries', ())]
    assert all(any(result.features[index] in note for note in notes) for index in picked)
    units = dict(dict(ax._spacr_style_spec.legend)['units'])
    assert units['PC1'] == units['PC2'] == 'dimensionless'
    points = next(artist.get_offsets() for artist in ax.collections if len(artist.get_offsets()) == len(result.rows))
    np.testing.assert_array_equal(points, result.scores[:,:2])
    assert widget._visible.to_csv(index=False) == before
    assert result.correlations.tobytes() == correlations


def test_shared_validation_observes_annotation_arrow_opacity_without_changing_it():
    """Annotation arrow patches are actual marks even though Matplotlib stores them outside ax.patches; detect a manual .2 alpha without repairing it."""
    from matplotlib.figure import Figure

    figure = Figure(); ax = figure.add_subplot()
    annotation = ax.annotate('', xy=(.8,.8), xytext=(.2,.2),
        arrowprops={'arrowstyle':'-|>', 'color':style.Palette.ACCENT, 'alpha':.2})
    spec = style.resolve_style('print', 'scatter', legend={'caption':'Synthetic correlation direction.',
        'reference':'origin', 'quantities':{'x':'coordinate','y':'coordinate'}, 'unit':'dimensionless'})
    style.apply_axes_style(ax, spec)
    annotation.arrow_patch.set_alpha(.2)
    position = annotation.xy
    result = style.finalize_figure(figure, diagnostics=False)
    assert any('sixty percent' in problem for problem in result.problems)
    assert annotation.arrow_patch.get_alpha() == .2
    assert annotation.xy == position


@pytest.mark.parametrize('declaration',['common','mixed','missing','explicit'])
def test_raw_pca_units_use_only_declared_common_feature_units_and_preserve_explicit_choices(qtbot,monkeypatch,declaration):
    """Raw score units are known only for consistently declared input features; retain an incompatible explicit component unit and diagnose its mathematical mismatch."""
    from dataclasses import replace
    import pandas as pd
    from spacr.qt import preferences
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.widgets.graph_spec import CONTINUOUS,GraphSpec
    from spacr.qt.widgets.pca_model import SCALE_NONE

    monkeypatch.setattr(preferences,'get_figure_style',dict)
    monkeypatch.setattr(preferences,'get_figure_style_per_graph',dict)
    result=replace(_result(3),scaling=SCALE_NONE)
    source=pd.DataFrame(np.random.default_rng(83).normal(size=(60,12)),columns=result.features)
    frame=result.scores_frame(source)
    units={feature:'µm' for feature in result.features} if declaration!='missing' else {}
    if declaration=='mixed':units[result.features[-1]]='s'
    if declaration=='explicit':units['PC1']='mM'
    frame.attrs['units']=units
    before=frame.to_csv(index=False),dict(frame.attrs['units'])
    widget=pca_view.PCAScoresCanvas(link=LinkedSelection());qtbot.addWidget(widget)
    widget._take_result(result,frame)
    widget.set_spec(GraphSpec(x='PC1',y='PC2',kind='scatter',roles={'PC1':CONTINUOUS,'PC2':CONTINUOUS}))
    ax=next(iter(widget.panel_axes().values()));spec=ax._spacr_style_spec
    declared=dict(dict(spec.legend)['units'])
    if declaration in ('common','explicit'):
        assert declared['PC1']==('mM' if declaration=='explicit' else 'µm')
        assert declared['PC2']=='µm'
    else:
        assert not declared.get('PC1') and not declared.get('PC2')
    assert any('mathematical transformation' in problem for problem in style._quantity_unit_problems(spec))==(declaration=='explicit')
    assert (frame.to_csv(index=False),dict(frame.attrs['units']))==before
