"""Plate heatmap semantics, missing-cell marks and exact numerical replay."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from spacr.figures import bundle, plates, style


@pytest.mark.parametrize('target', ['print', 'screen'])
def test_plate_cells_use_shared_defaults_without_changing_values_or_limits(target):
    """Retain a measured zero and missing wells while drawing .6-alpha square cells and explicit crosses."""
    frame = pd.DataFrame({'prc': ['p_r1_c1', 'p_r2_c2', 'p_r2_c2'], 'value': [0., 2., 4.]})
    frame.attrs['units'] = {'value': 'a.u.'}
    source = frame.to_csv(index=False)
    figure, panel = plates.build_plates(frame, 'value', target=target, min_max='all')
    ax = next(ax for ax in figure.axes if ax.images)
    image = ax.images[0]
    assert image.get_array()[0, 0] == 0.
    assert image.get_array()[1, 1] == 3.
    assert image.get_clim() == (0., 3.)
    assert image.get_alpha() == .6
    assert image.cmap.name == 'Blues'
    assert not any(spine.get_visible() for spine in ax.spines.values())
    crosses = [item for item in ax.collections if item.get_gid() == 'spacr-missing']
    assert len(crosses) == 1
    missing_rows, missing_columns = np.nonzero(np.ma.getmaskarray(image.get_array()))
    np.testing.assert_array_equal(crosses[0].get_offsets(), np.column_stack((missing_columns+.5, missing_rows+.5)))
    assert crosses[0].get_alpha() == .6
    assert np.allclose(image.cmap.get_bad(), image.cmap(0.))
    assert '×' in panel.caption and 'missing' in panel.caption.lower()
    assert ax._spacr_style_spec.kind == 'heatmap'
    assert dict(dict(ax._spacr_style_spec.legend)['units'])['value'] == 'a.u.'
    assert frame.to_csv(index=False) == source


@pytest.mark.parametrize('target', ['print', 'screen'])
def test_signed_plate_colour_reference_is_zero_with_unchanged_scale_endpoints(target):
    """An asymmetric score range must put the stated zero reference at white without changing values or endpoints."""
    frame = pd.DataFrame({'prc': ['p_r1_c1', 'p_r2_c2'], 'value': [-2., 8.]})
    frame.attrs['units'] = {'value': 'dimensionless'}
    figure, panel = plates.build_plates(frame, 'value', target=target, min_max='all', cmap=plates.score_ramp(target))
    image = next(ax.images[0] for ax in figure.axes if ax.images)
    assert image.get_clim() == (-2., 8.)
    assert image.norm(0.) == .5
    assert all(channel > .99 for channel in image.cmap(image.norm(0.))[:3])
    assert 'zero' in panel.caption.lower()
    assert any(value == 0. for value in figure.axes[-1].get_xticks())


def test_plate_replay_retains_crosses_reference_and_source_bytes():
    """The saved recipe must reproduce signed normalization and missingness without importing current defaults."""
    frame = pd.DataFrame({'prc': ['p_r1_c1', 'p_r2_c2'], 'value': [-2., 8.]})
    frame.attrs['units'] = {'value': 'dimensionless'}
    figure, _panel = plates.build_plates(frame, 'value', target='print', min_max='all', cmap=plates.score_ramp('print'))
    data, spec = bundle._figure_record(figure)
    restored = Figure()
    bundle._draw(restored, data, spec)
    original = next(ax for ax in figure.axes if ax.images)
    actual = next(ax for ax in restored.axes if ax.images)
    np.testing.assert_array_equal(actual.images[0].get_array(), original.images[0].get_array())
    np.testing.assert_array_equal(np.ma.getmaskarray(actual.images[0].get_array()), np.ma.getmaskarray(original.images[0].get_array()))
    assert actual.images[0].norm(0.) == .5
    assert actual.images[0].get_clim() == (-2., 8.)
    assert actual.images[0].get_alpha() == .6
    crosses = [item for item in actual.collections if item.get_gid() == 'spacr-missing']
    assert len(crosses) == 1
    assert crosses[0].get_alpha() == .6
    np.testing.assert_array_equal(crosses[0].get_offsets(), original.collections[0].get_offsets())


@pytest.mark.parametrize('grouping', ['mean', 'count'])
def test_plate_scale_names_the_quantity_and_stays_inside_the_live_canvas(grouping):
    """Tight file cropping hid an off-canvas scale label; keep its full extent visible live and name counts truthfully."""
    frame = pd.DataFrame({'prc': ['p_r1_c1', 'p_r2_c2'], 'value': [2., 8.]})
    frame.attrs['units'] = {'value': 'mM'}
    figure, _panel = plates.build_plates(frame, 'value', target='print', grouping=grouping)
    labels = [text.get_text() for text in figure.texts]
    assert labels == ['Objects per well' if grouping == 'count' else 'value (mM)']
    assert not style.figure_problems(figure)


def test_captured_colormap_keeps_exact_lookup_and_missing_colours():
    """Replaying a custom or locally modified map must preserve its actual lookup, under, over and missing colours."""
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    for ramp in (colormaps['Blues'].copy(), LinearSegmentedColormap.from_list('local_unregistered', ['#283868', '#C878A8'])):
        ramp.set_bad('#0A9878', alpha=.4)
        ramp.set_under('#78AED2')
        ramp.set_over('#F0A830')
        figure = Figure(); ax = figure.add_subplot()
        image = ax.imshow([[0., np.nan], [1., 2.]], cmap=ramp, alpha=.42)
        source = image.get_array().copy()
        values = np.concatenate(([-1.], np.linspace(0., 1., 1001), [2., np.nan]))
        expected = ramp(values)
        appearance = style._capture_appearance(figure)
        image.set_cmap('viridis')
        assert not style._restore_appearance(figure, appearance)
        np.testing.assert_array_equal(image.get_cmap()(values), expected)
        np.testing.assert_array_equal(image.get_array(), source)
        assert image.get_alpha() == .42


@pytest.mark.parametrize('values', [[2., 8.], [-8., -2.], [0., 8.]])
def test_a_zero_reference_outside_the_scale_interior_is_reported_without_changing_limits(values):
    """Preserve the existing endpoints and disclose when a requested zero centre cannot fit inside them."""
    frame = pd.DataFrame({'prc': ['p_r1_c1', 'p_r2_c2'], 'value': values})
    frame.attrs['units'] = {'value': 'dimensionless'}
    figure, panel = plates.build_plates(frame, 'value', min_max='all', target='print', cmap=plates.score_ramp('print'))
    image = next(ax.images[0] for ax in figure.axes if ax.images)
    assert image.get_clim() == (min(values), max(values))
    assert 'not centred on zero' in panel.caption
    result = style.finalize_figure(figure, diagnostics=False)
    assert any('reference is not inside' in problem for problem in result.problems)


def test_distinct_colour_limits_have_distinct_accurate_tick_labels():
    """The unchanged constant-count range 3..3.000001 must not label both endpoints as 3."""
    frame = pd.DataFrame({'prc': ['p_r1_c1']*3, 'value': [1., 2., 3.]})
    figure, _panel = plates.build_plates(frame, 'value', grouping='count', target='print')
    cax = figure.axes[-1]
    labels = [text.get_text() for text in cax.get_xticklabels()]
    assert len(set(labels)) == len(cax.get_xticks())
    np.testing.assert_allclose([float(label) for label in labels], cax.get_xticks(), rtol=0, atol=1e-12)


def test_heatmap_validation_observes_missing_marks_and_actual_colours_without_repair():
    """A missing cross or a map merely named Blues must be diagnosed while manual colours and values stay intact."""
    from matplotlib.colors import LinearSegmentedColormap

    figure = Figure(); ax = figure.add_subplot()
    values = np.array([[0., np.nan], [2., 3.]])
    image = style.heatmap_image(ax, values, cmap=style.heatmap_cmap(), vmin=0., vmax=3.)
    spec = style.resolve_style('print', 'heatmap', legend={'caption':'Synthetic measurements; × means missing.',
        'reference':'zero', 'quantities':{'value':'intensity'}, 'units':{'value':'a.u.'}})
    style.apply_axes_style(ax, spec, layout=False)
    assert not style._semantic_presentation_problems(ax, spec)
    crosses = next(item for item in ax.collections if item.get_gid() == 'spacr-missing')
    crosses.set_visible(False)
    assert any('missing cell' in problem for problem in style._semantic_presentation_problems(ax, spec))
    crosses.set_visible(True)
    fake = LinearSegmentedColormap.from_list('Blues', ['#ff0000', '#00ff00'])
    image.set_cmap(fake)
    before = image.get_array().copy()
    colours = fake(np.linspace(0., 1., 256)).copy()
    assert any('shared heatmap colour map' in problem for problem in style._semantic_presentation_problems(ax, spec))
    np.testing.assert_array_equal(image.get_array(), before)
    np.testing.assert_array_equal(image.cmap(np.linspace(0., 1., 256)), colours)


@pytest.mark.parametrize('kind,quantities,units,valid', [
    ('scatter', {'x':'time', 'y':'intensity'}, {'time':'s', 'intensity':'a.u.'}, True),
    ('scatter', {'x':'time', 'y':'intensity'}, {'time':'s'}, False),
    ('scatter', {'x':'time', 'y':'intensity'}, {'y':'a.u.'}, False),
    ('categorical', {'y':'intensity'}, {'y':'a.u.'}, True),
    ('heatmap', {'value':'intensity'}, {'intensity':'a.u.'}, True),
    ('heatmap', {'value':'intensity'}, {'x':'well'}, False),
])
def test_every_declared_numerical_quantity_requires_its_actual_unit(kind, quantities, units, valid):
    """Pin the new per-axis unit requirement without inventing units from a measurement name."""
    spec = style.resolve_style('print', kind, legend={'quantities':quantities, 'units':units})
    assert bool(style._quantity_unit_problems(spec)) is not valid
