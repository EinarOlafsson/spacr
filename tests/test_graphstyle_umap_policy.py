"""UMAP presentation keeps fixed embeddings, labels, hulls and linked selections."""
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from spacr.figures import style
from spacr.qt import preferences, theme
from spacr.qt.widgets.umap_explorer import ImageUmapExplorer
from spacr.qt.widgets.umap_figure_settings import redraw_umap_figure

pytestmark = pytest.mark.qt


def _payload():
    """Supply fixed separated clusters without fitting or reading images."""
    coords = np.random.default_rng(171).normal(size=(48, 2))
    labels = np.repeat([-1, 0, 1], 16)
    return {'embedding':coords, 'labels':labels, 'records':[{} for _ in labels]}


@pytest.fixture
def isolated_preferences(monkeypatch):
    """Keep workstation appearance preferences outside deterministic producer checks."""
    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)


@pytest.mark.parametrize('name', ['light', 'dark'])
@pytest.mark.parametrize('manual', [False, True])
def test_explorer_uses_shared_policy_and_retains_manual_choices_and_filter_dimming(
        qtbot, monkeypatch, isolated_preferences, name, manual):
    """Retain exact point coordinates and the original filter multiplier while applying defaults before explicit display choices in both themes."""
    def palette():
        """Select the requested GUI theme without reading workstation preferences."""
        return dict(theme._PALETTES[name])

    monkeypatch.setattr(theme, 'active_palette', palette)
    payload = _payload()
    if manual:
        payload['display'] = {'point_color':'#663399', 'point_alpha':.35, 'point_size':37.}
    originals = [payload[key].tobytes() for key in ('embedding', 'labels')]
    widget = ImageUmapExplorer(); qtbot.addWidget(widget)
    widget.set_payload(payload); widget._canvas.draw()
    ax = widget._axes
    assert ax._spacr_style_spec.policy_version == style.POLICY_VERSION
    assert ax.get_box_aspect() == 1
    assert ax.get_title() == ''
    assert to_rgba(ax.get_facecolor()) == to_rgba('#FFFFFF')
    assert to_rgba(ax.xaxis.label.get_color()) == to_rgba('#000000')
    assert widget._scatter.get_alpha() == (.35 if manual else .6)
    np.testing.assert_array_equal(widget._scatter.get_offsets(), payload['embedding'])
    if manual:
        np.testing.assert_allclose(widget._scatter.get_facecolors()[0], to_rgba('#663399', .35), rtol=0, atol=1e-12)
    else:
        colors = widget._scatter.get_facecolors()
        assert set(map(tuple, colors[:, :3])) <= {to_rgba(c)[:3] for c in (*style.SERIES, style.Palette.BACKGROUND)}
    report = style.finalize_figure(widget._figure)
    if manual:
        assert not report.compliant
        assert any('sixty percent' in problem for problem in report.problems)
    else:
        assert report.compliant, report.problems
    widget._point_visible = np.arange(48) % 2 == 0
    widget._apply_point_alpha()
    alpha = .35 if manual else .6
    from spacr.qt.widgets.umap_explorer import DIMMED_ALPHA
    np.testing.assert_array_equal(widget._scatter.get_alpha(), np.where(widget._point_visible, alpha, alpha*DIMMED_ALPHA))
    assert [payload[key].tobytes() for key in ('embedding', 'labels')] == originals
    widget._on_lasso([(-20,-20),(20,-20),(20,20),(-20,20)])
    np.testing.assert_array_equal(widget._selected, np.arange(len(payload['labels'])))


@pytest.mark.parametrize('name', ['light', 'dark'])
@pytest.mark.parametrize('manual', [False, True])
def test_redraw_preserves_all_cluster_coordinates_and_moves_centroid_labels_outside(
        isolated_preferences, name, manual):
    """Retain all original point partitions and hull paths; use shared geometry and external cluster identities without changing manually supplied color, opacity or width."""
    payload = _payload(); figure = Figure()
    values = {'black_background':name == 'dark', 'smooth_lines':False}
    if manual:
        values.update(point_color='#663399', point_alpha=.35, outline_width=1.7)
    originals = [payload[key].tobytes() for key in ('embedding', 'labels')]
    assert redraw_umap_figure(figure, payload, values)
    ax = figure.axes[0]
    assert ax._spacr_style_spec.policy_version == style.POLICY_VERSION
    assert ax.get_box_aspect() == 1
    assert ax.get_legend() is None
    assert not any(text.get_visible() and text.get_text() for text in ax.texts)
    assert all(mark.get_alpha() == (.35 if manual else .6) for mark in ax.collections)
    for mark, label in zip(ax.collections, np.unique(payload['labels'])):
        np.testing.assert_array_equal(mark.get_offsets(), payload['embedding'][payload['labels'] == label])
    assert all(mark.get_alpha() == .6 and mark.get_linewidth() == (1.7 if manual else 1.) for mark in ax.lines)
    notes = [entry['label'] for entry in ax._spacr_legend_entries]
    assert all(any(str(label) in note for note in notes) for label in np.unique(payload['labels']))
    assert [payload[key].tobytes() for key in ('embedding', 'labels')] == originals


@pytest.mark.parametrize('name', ['light','dark'])
def test_explorer_selection_overlays_remain_visible_on_the_shared_ground(qtbot, monkeypatch, isolated_preferences, name):
    """Selected, picked and linked identities remain visible on white figure ground in both GUI themes, without changing their point indices."""
    def palette():
        """Use the actual requested container palette."""
        return dict(theme._PALETTES[name])

    monkeypatch.setattr(theme, 'active_palette', palette)
    payload = _payload()
    view = ImageUmapExplorer(); qtbot.addWidget(view); view.set_payload(payload)
    view._selected = np.array([2,4]); view._picked = 3; view._linked_points = np.array([5])
    view._draw_embedding()
    for artist, indices, colour in [(view._selection_artist,[2,4],'#000000'),
            (view._picked_artist,[3],style.Palette.AMBER),
            (view._linked_artist,[5],style.Palette.ACCENT)]:
        np.testing.assert_array_equal(artist.get_offsets(), payload['embedding'][indices])
        np.testing.assert_allclose(artist.get_edgecolors()[0], to_rgba(colour,.6), rtol=0, atol=1e-12)


def test_categorical_embedding_validation_checks_actual_colors_after_a_manual_colormap_change(qtbot, isolated_preferences):
    """Discrete cluster metadata cannot exempt an incompatible manual colormap from actual palette observation."""
    view=ImageUmapExplorer();qtbot.addWidget(view);view.set_payload(_payload())
    original=view._scatter.get_array().copy()
    view._scatter.set_cmap('viridis')
    report=style.finalize_figure(view._figure)
    assert any('shared data palette' in problem for problem in report.problems)
    np.testing.assert_array_equal(view._scatter.get_array(),original)
    assert view._scatter.get_cmap().name=='viridis'
