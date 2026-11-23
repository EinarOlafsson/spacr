"""Frozen-data checks for submodule presentation, without model fitting or inference."""

import ast
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spacr import submodules
from spacr.figures import style


def _array_record(values):
    """Include dtype, order, missingness and numerical bytes in the comparison."""
    array = np.asarray(values)
    return (array.shape, array.dtype.str,
            hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
            np.isfinite(array).tobytes())


def _rng_record():
    """Capture the legacy generator used by existing third-party renderers."""
    name, keys, position, gaussian, cached = np.random.get_state()
    return name, keys.tobytes(), position, gaussian, cached


def _nested_plot_owner(parent_name, name):
    """Compile the exact production drawing owner without running its model parent."""
    source = Path(submodules.__file__)
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    parent = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == parent_name)
    node = next(node for node in parent.body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(vars(submodules))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


def _image_appearance_record(image):
    """Freeze numerical pixels and the original display transform and categorical LUT."""
    cmap = image.get_cmap()
    return (_array_record(image.get_array()), image.get_clim(), image.get_alpha(),
            type(image.norm).__name__, image.norm.clip, getattr(image.norm, 'gamma', 1.),
            image.origin, tuple(image.get_extent()), image.get_interpolation(),
            _array_record(cmap(np.linspace(0., 1., cmap.N))))


@pytest.fixture(autouse=True)
def _figures_and_preferences(monkeypatch):
    """Keep tests headless and independent of saved GUI appearance preferences."""
    monkeypatch.setattr(submodules, '_show_if_interactive', lambda: None)
    with style.pipeline_figures():
        yield
    plt.close('all')


@pytest.mark.parametrize('target', ['print', 'screen'])
def test_cellpose_declaration_preserves_image_pixels_norm_and_rng(monkeypatch, target):
    """Image metadata must never apply graph alpha or alter mask label integers."""
    monkeypatch.setattr(submodules, 'theme_target', lambda: target)
    image = np.arange(48, dtype=np.uint16).reshape(6, 8)
    labels = np.zeros((6, 8), dtype=np.int32)
    labels[1:4, 2:6] = 9
    records = [_array_record(image), _array_record(labels)]
    before = _rng_record()
    with style.figure_style(target):
        fig, axes = plt.subplots(1, 2)
        for ax, values in zip(axes, [image, labels]):
            artist = ax.imshow(values, interpolation='nearest')
            limits = artist.get_clim()
            alpha = artist.get_alpha()
            submodules._cellpose_image_style(ax, processing='Frozen diagnostic plane')
            assert artist.get_clim() == limits
            assert artist.get_alpha() == alpha
            assert ax._spacr_style_spec.kind == 'image'
            metadata = dict(ax._spacr_style_spec.comparison)
            assert metadata['gamma'] == 1.0 and metadata['display_range'] == limits
    assert [_array_record(image), _array_record(labels)] == records
    assert _rng_record() == before


@pytest.mark.parametrize('owner', ['training', 'test', 'apply'])
def test_cellpose_layout_preserves_actual_planes_and_panel_inches_within_paper_width(monkeypatch, tmp_path, owner):
    """Actual preview owners fit the complete key without scaling their existing images."""
    image = np.arange(120, dtype=np.float32).reshape(10, 12)
    labels = np.zeros((10, 12), dtype=np.int32)
    labels[2:6, 3:7] = 1
    labels[6:9, 7:11] = 3
    flow = [np.dstack([image / 120] * 3), np.zeros((2, 10, 12), dtype=np.float32), image / 120]
    inputs = [_array_record(value) for value in [image, labels, *flow]]
    observed = []
    original_key = submodules.external_legend

    def checked_key(figure, **kwargs):
        """Check the new layout boundary against its original image and physical geometry."""
        axes = list(kwargs['axes'])
        figure.canvas.draw()
        appearance = [_image_appearance_record(axis.images[0]) for axis in axes]
        sizes = [np.array(axis.get_window_extent().size) / figure.dpi for axis in axes]
        rng = _rng_record()
        legend = original_key(figure, **kwargs)
        figure.canvas.draw()
        assert [_image_appearance_record(axis.images[0]) for axis in axes] == appearance
        for axis, size in zip(axes, sizes):
            np.testing.assert_allclose(np.array(axis.get_window_extent().size) / figure.dpi,
                                       size, rtol=0., atol=1e-10)
        assert _rng_record() == rng
        assert figure.get_size_inches()[0] <= 170 / 25.4 + 1e-10
        assert figure.get_size_inches()[1] <= 275 / 25.4
        bounds = legend.get_window_extent()
        assert bounds.x0 >= 0 and bounds.x1 <= figure.bbox.width
        assert bounds.y0 >= 0 and bounds.y1 <= figure.bbox.height
        assert all(not bounds.overlaps(axis.get_tightbbox(figure.canvas.get_renderer()))
                   for axis in axes)
        assert all(text.get_fontsize() == 8. for text in legend.get_texts())
        observed.append(figure)
        return legend

    monkeypatch.setattr(submodules, 'external_legend', checked_key)
    monkeypatch.setattr(submodules, 'save_figure', lambda figure, path, **kwargs: path)
    if owner == 'training':
        submodules.plot_cellpose_batch([image, image + 10], [labels, labels])
        expected = [image, image + 10, labels, labels]
    elif owner == 'test':
        _nested_plot_owner('test_cellpose_model', 'plot_cellpose_resilts')(
            0, 0, str(tmp_path), image, labels, labels, flow)
        expected = [image, labels, labels, flow[2], flow[0]]
    else:
        _nested_plot_owner('apply_cellpose_model', 'plot_cellpose_result')(
            0, 0, str(tmp_path), image, labels, flow)
        expected = [image, labels, flow[2], flow[0]]
    assert len(observed) == 1
    assert [_array_record(axis.images[0].get_array()) for axis in observed[0].axes] == [
        _array_record(value) for value in expected]
    assert [_array_record(value) for value in [image, labels, *flow]] == inputs


@pytest.mark.parametrize('target', ['print', 'screen'])
def test_scalar_importance_preserves_order_values_and_has_no_invented_ci(monkeypatch, target):
    """Scalar ranks become vertical bars with exact source values and physical slots."""
    monkeypatch.setattr(submodules, 'theme_target', lambda: target)
    table = pd.DataFrame({'feature': ['zeta', 'alpha', 'mu'], 'importance': [.75, -.125, .0]})
    original = table.to_csv(index=False).encode()
    before = _rng_record()
    figure = submodules._feature_importance_figure(table,
        caption='Frozen synthetic importances.', quantity='Importance')
    axis = figure.axes[0]
    assert [tick.get_text() for tick in axis.get_xticklabels()] == list(table['feature'])
    assert _array_record([bar.get_height() for bar in axis.patches]) == _array_record(table['importance'])
    assert all(bar.get_width() == style.BAR_FRAC and bar.get_alpha() == .6 for bar in axis.patches)
    assert not axis.lines and not axis.collections
    width = axis.get_position().width * figure.get_size_inches()[0]
    assert width == pytest.approx(3 * style.SLOT_IN)
    assert axis.get_xlim() == (-.5, 2.5)
    assert table.to_csv(index=False).encode() == original and _rng_record() == before


def test_shap_display_summaries_leave_sample_values_and_rng_unchanged():
    """Display follows the existing SHAP magnitude rank while exported columns remain original."""
    values = np.column_stack([np.linspace(-1., 1., 21), np.linspace(.5, -.25, 21)])
    before = _array_record(values), _rng_record()
    figure = submodules._shap_distribution_figure(values, ['feature_z', 'feature_a'])
    axis = figure.axes[0]
    assert _array_record(figure._spacr_data.to_numpy()) == before[0]
    assert axis._spacr_style_spec.sample_counts == (21, 21)
    assert axis._spacr_condition_kinds == ('box', 'box')
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ['feature_a', 'feature_z']
    assert list(figure._spacr_data.columns) == ['feature_z', 'feature_a']
    for index, source_index in enumerate([1, 0]):
        points = next(item for item in axis.collections
                      if item.get_gid() == f'spacr-condition:{index}:points')
        assert _array_record(points.get_offsets()[:, 1]) == _array_record(values[:, source_index])
    assert (_array_record(values), _rng_record()) == before


@pytest.mark.parametrize('standalone', [False, True])
def test_shap_captured_order_and_wide_source_replay_without_training(standalone):
    """Real captured tick order and a supported melt recipe replay exact frozen contributions."""
    from spacr.figures import bundle

    values = np.column_stack([np.linspace(-1., 1., 21), np.linspace(.5, -.25, 21)])
    summary, original_axis = plt.subplots()
    original_axis.set_yticks([0, 1], ['feature_a', 'feature_z'])
    figure = submodules._shap_distribution_figure(values, ['feature_z', 'feature_a'], figure=summary)
    frame, specification = bundle._figure_record(figure)
    captured = json.loads(json.dumps(bundle._capture_view(figure, specification)))
    assert list(frame.columns) == ['feature_z', 'feature_a']
    assert captured['condition_recipe']['labels'] == ['feature_a', 'feature_z']
    assert captured['melt'] == {'columns': ['feature_z', 'feature_a'],
                                'var_name': 'feature', 'value_name': 'contribution'}
    original = _array_record(frame.to_numpy())
    draw = bundle._draw
    if standalone:
        namespace = {'__name__': 'g2_shap_frozen_replay'}
        exec(compile(bundle._recreate_script(captured), 'g2_shap_replay.py', 'exec'), namespace)
        draw = namespace['_draw']
    replay = plt.figure()
    draw(replay, frame, captured)
    axis = replay.axes[0]
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ['feature_a', 'feature_z']
    for index, source_index in enumerate([1, 0]):
        points = next(item for item in axis.collections
                      if item.get_gid() == f'spacr-condition:{index}:points')
        assert _array_record(points.get_offsets()[:, 1]) == _array_record(values[:, source_index])
    assert _array_record(frame.to_numpy()) == original


@pytest.mark.parametrize('kind', ['replication', 'invasion'])
def test_sparse_composition_retains_counts_empty_statistics_and_categories(kind):
    """Retain exact fractions and the original pandas/Matplotlib stack construction.

    Matplotlib converts bar height through (bottom + height) - bottom, so
    an upper segment's artist bytes can differ from its raw fraction by one
    floating-point step. The immutable original construction is the artist
    oracle; the unrendered fractions are checked independently and exactly.
    """
    column = 'replication_bucket' if kind == 'replication' else 'invasion_class'
    labels = ['1', '2'] if kind == 'replication' else ['attached', 'invaded']
    frame = pd.DataFrame({'condition': ['only'] * 3, 'prc': ['w1'] * 3,
                          column: pd.Categorical([labels[0], labels[1], labels[1]], categories=labels)})
    original = frame.to_csv(index=False).encode()
    baseline_counts = frame.groupby(['condition', column], observed=True).size().unstack(fill_value=0)
    baseline_proportions = baseline_counts.div(baseline_counts.sum(axis=1), axis=0)
    assert _array_record(baseline_counts.to_numpy()) == _array_record(np.array([[1, 2]], dtype=np.int64))
    fraction_record = _array_record(baseline_proportions.to_numpy())
    assert fraction_record == _array_record(np.array([[1 / 3, 2 / 3]], dtype=np.float64))
    baseline_axes = baseline_proportions.plot(kind='bar', stacked=True, colormap='viridis', figsize=(12, 8))
    baseline_heights = _array_record([patch.get_height() for patch in baseline_axes.patches])
    baseline_bottoms = _array_record([patch.get_y() for patch in baseline_axes.patches])
    renderer = submodules._replication_stacked_bars if kind == 'replication' else submodules._invasion_stacked_bars
    kwargs = {'denominators': {'only': 3}} if kind == 'invasion' else {}
    statistics, pairwise, figure = renderer({}, frame, 'condition', 'prc', 'object', 'viridis',
                                            'Frozen composition', **kwargs)
    assert statistics.to_csv(index=False).encode() == b'chi_squared_stat,p_value,degrees_of_freedom\n,,\n'
    assert pairwise.empty
    assert list(pairwise.columns) == ['Group 1', 'Group 2', 'Test Name', 'p-value', 'p-value_adj', 'adj']
    axis = figure.axes[0]
    assert _array_record([patch.get_height() for patch in axis.patches]) == baseline_heights
    assert _array_record([patch.get_y() for patch in axis.patches]) == baseline_bottoms
    assert _array_record(baseline_proportions.to_numpy()) == fraction_record
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ['only']
    assert all(patch.get_alpha() == .6 for patch in axis.patches)
    assert frame.to_csv(index=False).encode() == original
    assert list(frame[column].cat.categories) == labels


@pytest.mark.parametrize('target', ['print', 'screen'])
def test_threshold_qc_keeps_wells_bins_thresholds_and_source_bytes(monkeypatch, target):
    """Appearance does not reselect wells, remove values or replace the actual threshold lines."""
    monkeypatch.setattr(submodules, 'theme_target', lambda: target)
    parasites = pd.DataFrame({'prc': ['w2'] * 6 + ['w1'] * 6,
                             'outside_intensity': [1., 2., 3., 4., 5., np.nan] * 2})
    wells = pd.DataFrame({'prc': ['w2', 'w1'], 'threshold_median': [3., 3.5],
                         'reference_threshold_median': [2.5, 2.], 'threshold_source': ['fixed', 'control'],
                         'bimodality_coefficient': [.6, .8], 'n_total': [6, 6]})
    originals = parasites.to_csv(index=False).encode(), wells.to_csv(index=False).encode(), _rng_record()
    figure = submodules._invasion_threshold_panels(parasites, wells, max_panels=1)
    axis = next(ax for ax in figure.axes if ax.get_title() == 'w1')
    assert [patch.get_height() for patch in axis.patches] == [1.] * 5
    assert [line.get_xdata()[0] for line in axis.lines] == [3.5, 2.]
    assert all(line.get_color() == '#000000' for line in axis.lines)
    assert [line.get_linestyle() for line in axis.lines] == ['--', ':']
    key = figure._spacr_external_legend
    reference_keys = [(text.get_text(), handle.get_linestyle())
                      for text, handle in zip(key.get_texts(), key.legend_handles)
                      if 'threshold' in text.get_text().lower()]
    assert ('Used threshold (control)', '--') in reference_keys
    assert ('Reference threshold', ':') in reference_keys
    assert figure.get_size_inches()[0] <= 170 / 25.4 + 1e-10
    assert all(patch.get_alpha() == .6 for patch in axis.patches)
    assert 'first 1 wells' in figure._spacr_caption
    notes = ' '.join(entry['label'] for entry in axis._spacr_legend_entries)
    assert 'n=6' in notes and '0.80' in notes and 'control' in notes
    assert not axis.texts
    assert (parasites.to_csv(index=False).encode(), wells.to_csv(index=False).encode(), _rng_record()) == originals


def test_threshold_roles_preserve_coincident_cutoff_suppression():
    """A distinct reference style must not create a duplicate coincident numerical cutoff."""
    parasites = pd.DataFrame({'prc': ['w1'] * 3, 'outside_intensity': [1., 2., 3.]})
    wells = pd.DataFrame({'prc': ['w1'], 'threshold_median': [2.],
                         'reference_threshold_median': [2.], 'threshold_source': ['fixed'],
                         'bimodality_coefficient': [.6], 'n_total': [3]})
    original = wells.to_csv(index=False).encode(), _rng_record()
    figure = submodules._invasion_threshold_panels(parasites, wells)
    lines = figure.axes[0].lines
    assert len(lines) == 1 and list(lines[0].get_xdata()) == [2., 2.]
    assert lines[0].get_linestyle() == '--' and lines[0].get_color() == '#000000'
    assert 'Reference threshold' not in [text.get_text() for text in figure._spacr_external_legend.get_texts()]
    assert (wells.to_csv(index=False).encode(), _rng_record()) == original


def test_default_twelve_well_thresholds_retain_all_data_and_fit_paper():
    """The default crowded layout must retain every selected well at true physical size."""
    names = [f'well_{index:02d}' for index in range(12, 0, -1)]
    parasites = pd.DataFrame({'prc': np.repeat(names, 6),
                             'outside_intensity': [1., 2., 3., 4., 5., np.nan] * 12})
    wells = pd.DataFrame({'prc': names, 'threshold_median': [3.5] * 12,
                         'reference_threshold_median': [2.] * 12, 'threshold_source': ['fixed'] * 12,
                         'bimodality_coefficient': [.72] * 12, 'n_total': [6] * 12})
    originals = parasites.to_csv(index=False).encode(), wells.to_csv(index=False).encode(), _rng_record()
    figure = submodules._invasion_threshold_panels(parasites, wells)
    figure.canvas.draw()
    axes = [axis for axis in figure.axes if axis.get_title()]
    assert [axis.get_title() for axis in axes] == sorted(names)
    assert len(axes) == 12
    legend = figure._spacr_external_legend
    labels = [' '.join(text.get_text().split()) for text in legend.get_texts()]
    assert sum(label.count('BC means bimodality coefficient.') for label in labels) == 1
    assert sum(label.count('The histogram retains the original finite signals, bins and thresholds.')
               for label in labels) == 1
    for axis, name in zip(axes, sorted(names)):
        assert [patch.get_height() for patch in axis.patches] == [1.] * 5
        assert [line.get_xdata()[0] for line in axis.lines] == [3.5, 2.]
        assert [line.get_linestyle() for line in axis.lines] == ['--', ':']
        size = np.array(axis.get_window_extent().size) / figure.dpi * 25.4
        np.testing.assert_allclose(size, [40., 40.], rtol=0., atol=1e-8)
        assert f'{name}: n=6; BC=0.72.' in labels
        texts = [axis.title, axis.xaxis.label, axis.yaxis.label,
                 *axis.get_xticklabels(), *axis.get_yticklabels()]
        assert all(text.get_fontsize() == 8. for text in texts if text.get_text())
    assert all(text.get_fontsize() == 8. for text in legend.get_texts())
    assert ('Used threshold (fixed)', '--') in [
        (' '.join(text.get_text().split()), handle.get_linestyle())
        for text, handle in zip(legend.get_texts(), legend.legend_handles)]
    assert ('Reference threshold', ':') in [
        (' '.join(text.get_text().split()), handle.get_linestyle())
        for text, handle in zip(legend.get_texts(), legend.legend_handles)]
    bounds = legend.get_window_extent()
    assert bounds.x0 >= 0 and bounds.x1 <= figure.bbox.width
    assert bounds.y0 >= 0 and bounds.y1 <= figure.bbox.height
    assert all(not bounds.overlaps(axis.get_tightbbox(figure.canvas.get_renderer())) for axis in axes)
    envelopes = [axis.get_tightbbox(figure.canvas.get_renderer()) for axis in axes]
    assert all(not first.overlaps(second) for index, first in enumerate(envelopes)
               for second in envelopes[index + 1:])
    width, height = figure.get_size_inches() * 25.4
    assert width <= 170. + 1e-8 and height <= 275. + 1e-8, (width, height)
    assert figure._spacr_data.to_csv(index=False).encode() == originals[0]
    assert (parasites.to_csv(index=False).encode(), wells.to_csv(index=False).encode(), _rng_record()) == originals
