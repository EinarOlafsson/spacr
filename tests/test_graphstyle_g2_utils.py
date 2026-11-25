"""Exact scientific-identity checks for the utility figure migration."""

import random

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from spacr import utils as U
from spacr.figures.style import DATA_ALPHA


@pytest.fixture(autouse=True)
def close_utility_figures(monkeypatch):
    """Keep each isolated renderer test free of display callbacks and retained figures."""
    monkeypatch.setattr(U, '_show_if_interactive', lambda: None)
    state = matplotlib.rcParams.copy()
    yield
    plt.close('all')
    matplotlib.rcParams.update(state)


def _exact_array(actual, expected):
    """Compare shape, dtype and numerical bytes rather than a plotting tolerance."""
    actual = np.ascontiguousarray(actual)
    expected = np.ascontiguousarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('background,foreground', [('#ffffff', '#000000'),
                                                  ('#000000', '#ffffff')])
def test_metrics_keep_model_values_and_statistical_coordinates_exact(
        capsys, monkeypatch, tmp_path, background, foreground):
    """Styling leaves the fit, residuals, scale-location coordinates and metrics unchanged."""
    import statsmodels.api as sm
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    store = QSettings(str(tmp_path / 'figure.ini'), QSettings.Format.IniFormat)
    monkeypatch.setattr(preferences, '_settings', lambda: store)
    monkeypatch.delenv('SPACR_PIPELINE_FIGURES', raising=False)
    preferences.set_figure_colors(background, foreground)
    preferences.set_figure_line_colour(foreground)
    preferences.set_figure_style({'background': background, 'foreground': foreground})
    preferences.set_figure_style_per_graph({})

    rng = np.random.default_rng(23)
    x = rng.normal(size=24)
    y = 1.5 + 2.0 * x + rng.normal(scale=0.3, size=len(x))
    model = sm.OLS(y, sm.add_constant(x)).fit()
    fitted, residual = model.fittedvalues.copy(), model.resid.copy()
    scale = np.sqrt(np.abs(model.get_influence().resid_studentized_internal))
    U.model_metrics(model)
    figure = plt.gcf()
    _exact_array(model.fittedvalues, fitted)
    _exact_array(model.resid, residual)
    _exact_array(figure.axes[0].collections[0].get_offsets().data,
                 np.column_stack((fitted, residual)))
    _exact_array(figure.axes[3].collections[0].get_offsets().data,
                 np.column_stack((fitted, scale)))
    assert float(figure._spacr_utils_metrics['rmse']).hex() == float(np.sqrt(model.mse_resid)).hex()
    assert float(figure._spacr_utils_metrics['mae']).hex() == float(np.mean(np.abs(residual))).hex()
    assert str(figure._spacr_utils_metrics['rmse']) in capsys.readouterr().out
    assert all(ax.get_title() == '' for ax in figure.axes)
    assert all(ax.get_box_aspect() == 1 for ax in figure.axes)
    assert figure._spacr_external_legend_layout['outside_panels']
    assert figure._spacr_external_legend_layout['inside_canvas']
    reference = next(line for line in figure.axes[2].lines
                     if line.get_gid() == 'spacr-reference')
    assert matplotlib.colors.to_hex(figure.axes[2].get_facecolor()) == background
    assert matplotlib.colors.to_hex(reference.get_color()) == '#000000'
    assert reference.get_alpha() == DATA_ALPHA
    assert bool(reference.get_path_effects()) == (background == '#000000')


@pytest.mark.parametrize('generator_name', ['SaliencyMapGenerator', 'GradCAMGenerator'])
@pytest.mark.parametrize('channels', [1, 3])
def test_activation_display_preserves_maps_channels_and_predictions(generator_name, channels):
    """Scalar palettes change only display mapping; RGB channel arrays remain exact."""
    torch = pytest.importorskip('torch')
    generator = object.__new__(getattr(U, generator_name))
    inputs = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4) / 100
    maps = torch.arange(2 * channels * 4 * 4, dtype=torch.float32).reshape(2, channels, 4, 4) / 200
    predictions = torch.tensor([0, 1])
    original_inputs = inputs.numpy().copy()
    original_maps = maps.numpy().copy()
    figure = generator.plot_activation_grid(inputs, maps, predictions, overlay=True)
    for index in range(2):
        ax = figure.axes[index]
        expected = (original_maps[index, 0] if channels == 1
                    else original_maps[index].transpose(1, 2, 0))
        _exact_array(ax.images[-1].get_array().data, expected)
        assert ax.images[-1].get_alpha() == DATA_ALPHA
        assert not ax.texts
        comparison = dict(ax._spacr_style_spec.comparison)
        assert comparison['prediction'] == index
        if channels == 1:
            assert ax.images[-1].get_cmap().name == 'Blues'
            assert ax.images[-1].get_clim() == (float(maps.min()), float(maps.max()))
    _exact_array(inputs.numpy(), original_inputs)
    _exact_array(maps.numpy(), original_maps)
    assert all(not ax.axison for ax in figure.axes)
    assert all(not ax.images for ax in figure.axes[2:])
    assert figure._spacr_external_legend_layout['outside_panels']


@pytest.mark.parametrize('points,opacity', [(True, DATA_ALPHA), (True, 0.23), (False, 0.)])
def test_embedding_preserves_each_object_coordinate_and_explicit_visibility(points, opacity):
    """Cluster keys never move, relabel, drop or silently reveal a hidden object."""
    embedding = np.array([[0., 0.], [1., 3.], [2., 4.], [8., 2.], [9., 3.]])
    labels = np.array([-1, 4, 4, 10, 10])
    original = embedding.copy()
    palette = U.generate_colors(3, False)
    payload = {'embedding': embedding, 'labels': labels,
               'plot_labels': labels, 'reduction_method': 'pca',
               'coordinate_units': 'dimensionless',
               'records': [{'object': str(index)} for index in range(len(labels))]}
    state = random.getstate()
    figure = U.plot_embedding(
        embedding, None, labels, 0, .2, palette, True, False, points, False,
        False, True, 3., 12., False, False,
        interactive_payload=payload, point_alpha=opacity if points else .7)
    ax = figure.axes[0]
    for collection, label in zip(ax.collections, np.unique(labels)):
        _exact_array(collection.get_offsets().data, original[labels == label])
        assert collection.get_alpha() == opacity
    _exact_array(embedding, original)
    assert random.getstate() == state
    assert figure._spacr_umap_payload is payload
    assert dict(ax._spacr_style_spec.comparison)['embedding_method'] == 'pca'
    assert ax._spacr_style_spec.controls == ()
    assert ax.get_box_aspect() == 1
    assert not ax.texts
    assert figure._spacr_external_legend_layout['inside_canvas']
    from matplotlib.backends.backend_agg import RendererAgg
    renderer = RendererAgg(8, 8, figure.dpi)
    bounds = ax.get_tightbbox(renderer)
    assert bounds.x0 >= -1 and bounds.y0 >= -1
    assert bounds.x1 <= figure.bbox.width + 1
    assert bounds.y1 <= figure.bbox.height + 1


def test_gallery_keeps_thumbnail_selection_rng_and_cross_view_identity(tmp_path):
    """Wrapping a gallery changes no selected pixels and preserves noncontiguous label colors."""
    labels = np.array([-1, 0, 0, 0, 5, 5, 5])
    paths = []
    for index in range(len(labels)):
        path = tmp_path / f'{index}.png'
        Image.fromarray(np.full((4, 4), index + 1, np.uint8)).save(path)
        paths.append(str(path))
    palette = U.generate_colors(3, False)
    local = random.Random(17)
    selected = [local.sample([1, 2, 3], 2), local.sample([4, 5, 6], 2)]
    limits = (float(min(index + 1 for group in selected for index in group)),
              float(max(index + 1 for group in selected for index in group)))
    state = random.getstate()
    try:
        random.seed(17)
        figure = U.plot_clusters_grid(None, labels, 2, paths, palette, 2., False, False)
        assert random.getstate() == local.getstate()
    finally:
        random.setstate(state)
    for panel, expected, palette_index in zip(figure.axes, selected, [1, 2]):
        actual = [int(ax.images[0].get_array()[0, 0]) for ax in panel.child_axes]
        assert actual == [index + 1 for index in expected]
        np.testing.assert_array_equal(panel.patches[0].get_facecolor()[:3],
                                      palette[palette_index, :3])
        assert [ax.images[0].get_clim() for ax in panel.child_axes] == [limits] * 2
    assert figure._spacr_external_legend_layout['outside_panels']


def test_instance_mask_palette_preserves_original_colors_rng_and_mask_bytes():
    """Instance-image colors, opacity, source labels and random consumption remain exact."""
    mask = np.array([[0, 1, 1], [0, 2, 3]], dtype=np.int32)
    original = mask.copy()
    state = np.random.get_state()
    try:
        np.random.seed(19)
        expected_colors = np.random.rand(4, 4)
        expected_colors[:, 3] = 1
        expected_colors[0, :] = [0, 0, 0, 1]
        expected_state = np.random.get_state()
        np.random.seed(19)
        palette = U._generate_mask_random_cmap(mask)
        actual_state = np.random.get_state()
    finally:
        np.random.set_state(state)
    assert actual_state[0] == expected_state[0]
    np.testing.assert_array_equal(actual_state[1], expected_state[1])
    assert actual_state[2:] == expected_state[2:]
    _exact_array(mask, original)
    assert np.all(np.asarray(palette.colors)[:, 3] == 1)
    np.testing.assert_array_equal(palette.colors[0], [0, 0, 0, 1])
    _exact_array(palette.colors, expected_colors)
