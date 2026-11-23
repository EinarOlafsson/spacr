"""The explorer/pipeline volcano consumes the same presentation owner and retains explicit styles."""
import numpy as np
import pandas as pd
import pytest
from spacr import volcano_style as volcano
from spacr.figures import style
from spacr.qt import preferences


def _results():
    """Supply literal standardized effects and p values; no fitting or control-threshold estimation is needed."""
    return pd.DataFrame({'standardized_marginal_effect': [-2., -.2, .1, 1., 2.],
        'adjusted_p_value': [.001, .5, .8, .04, .002], 'guide': ['a', 'b', 'c', 'd', 'e']})


@pytest.fixture(autouse=True)
def _preferences(monkeypatch):
    """Prevent workstation preference stores from changing fixed renderer checks."""
    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    yield
    from matplotlib import pyplot as plt

    plt.close('all')


def test_explorer_volcano_default_geometry_ink_and_preparation_are_explicit():
    """Keep literal x and negative-log10(p) outputs and selection flags while measuring actual four-square-point marks, .6 alpha, shared colours and a 40-mm square."""
    source = _results()
    before = source.to_csv(index=False)
    presentation = volcano.VolcanoStyle()
    figure, axes = volcano.render_volcano(source, presentation, screen=True)
    axis = axes[0]
    specification = axis._spacr_style_spec
    assert specification.kind == 'continuous'
    assert dict(specification.legend)['unit'] == 'dimensionless'
    assert {artist.get_alpha() for artist in axis.collections} == {.6}
    assert {float(value) for artist in axis.collections for value in artist.get_sizes()} == {4.}
    assert all(spine.get_visible() for spine in axis.spines.values())
    assert axis.get_position().width * figure.get_figwidth() == pytest.approx(40 / 25.4, abs=.5 / figure.dpi)
    assert axis.get_position().height * figure.get_figheight() == pytest.approx(40 / 25.4, abs=.5 / figure.dpi)
    prepared, x, y, raw, selected, cut = volcano._prepare(source, presentation)
    np.testing.assert_array_equal(x, [-2., -.2, .1, 1., 2.])
    np.testing.assert_array_equal(y, -np.log10([.001, .5, .8, .04, .002]))
    np.testing.assert_array_equal(selected, [True, False, False, True, True])
    assert cut is None and source.to_csv(index=False) == before
    assert not style.figure_problems(figure)


def test_explicit_explorer_volcano_styles_survive_validation():
    """Explicit old or new style values remain manual choices, including marker opacity, family, title, grid, dimensions and limits."""
    appearance = volcano.VolcanoStyle.from_dict({'marker_alpha': .35, 'marker_size': 17., 'significant_marker_size': 19.,
        'figure_width': 5., 'figure_height': 3., 'font_family': 'DejaVu Sans', 'tick_font_size': 9.,
        'hide_top_right_spines': True, 'grid': True, 'title': 'Explicit title', 'x_lim': [-3., 3.], 'y_lim': [0., 5.]})
    before = appearance.to_dict()
    figure, axes = volcano.render_volcano(_results(), appearance, screen=True)
    axis = axes[0]
    assert figure.get_size_inches().tolist() == [5., 3.]
    assert {item.get_alpha() for item in axis.collections} == {.35}
    assert {float(size) for item in axis.collections for size in item.get_sizes()} == {17., 19.}
    assert axis.get_xlim() == (-3., 3.) and axis.get_ylim() == (0., 5.)
    assert axis.get_title() == 'Explicit title' and not axis.spines['right'].get_visible()
    assert axis.get_xticklabels()[0].get_fontfamily() == ['DejaVu Sans', 'Open Sans']
    report = style.finalize_figure(figure, diagnostics=False)
    assert 'Draw every data mark at sixty percent opacity' in report.problems
    assert 'Move the plot title into the figure legend' in report.problems
    assert appearance.to_dict() == before


def test_explorer_volcano_broken_axis_preserves_supplied_intervals():
    """A manually split axis keeps both supplied windows and every original point; finalization must not imply an ordinary uninterrupted square view."""
    appearance = volcano.VolcanoStyle(split_axis=True, split_y_lims=((0., 1.), (2., 4.)))
    figure, axes = volcano.render_volcano(_results(), appearance, screen=True)
    assert [axis.get_ylim() for axis in axes] == [(2., 4.), (0., 1.)]
    assert all(dict(axis._spacr_style_spec.comparison)['split_axis'] for axis in axes)
    assert all(dict(axis._spacr_style_spec.comparison)['manual_view'] for axis in axes)
    assert all(sum(len(item.get_offsets()) for item in axis.collections) == 5 for axis in axes)
    assert not style.figure_problems(figure)
    assert axes[0].get_position().y0 > axes[1].get_position().y1
    assert axes[0].get_position().x0 == pytest.approx(axes[1].get_position().x0, abs=1e-12)
    assert figure._supylabel.get_visible()
    for axis, fraction in zip(axes, [.35, .65]):
        width = axis.get_position().width * figure.get_figwidth()
        height = axis.get_position().height * figure.get_figheight()
        assert height / width == pytest.approx(fraction, abs=1e-12)


def test_explorer_volcano_reference_lines_have_actual_shared_black_ink():
    """The source alpha/effect/zero coordinates remain exact, with black .6-alpha reference lines and external explanations."""
    figure, axes = volcano.render_volcano(_results(), volcano.VolcanoStyle(effect_threshold=1., threshold_multiplier=1.), screen=True)
    axis = axes[0]
    assert len(axis.lines) == 4
    assert all(line.get_alpha() == .6 for line in axis.lines)
    assert all(line.get_gid() == 'spacr-reference' for line in axis.lines)
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), [-np.log10(.05)] * 2)
    assert [line.get_xdata()[0] for line in axis.lines[1:]] == [0., -1., 1.]
    assert all(line.get_color() == '#000000' for line in axis.lines)


def test_explicit_control_identity_changes_only_ink_and_has_a_matching_key():
    """A named boolean control column binds grey to those exact rows without inferring labels or changing preparation, masks or point coordinates."""
    from matplotlib.colors import to_rgba

    source = _results().assign(control=[False, True, False, False, True])
    appearance = volcano.VolcanoStyle(control_column='control')
    figure, axes = volcano.render_volcano(source, appearance, screen=True)
    axis = axes[0]
    found = {}
    for artist in axis.collections:
        for point, colour in zip(artist.get_offsets(), artist.get_facecolors()):
            found[float(point[0])] = colour
    for effect in source.loc[source['control'], 'standardized_marginal_effect']:
        np.testing.assert_array_equal(found[effect], to_rgba(style.Palette.CONTROL, .6))
    assert 'Declared controls' in [text.get_text() for text in axis.get_legend().get_texts()]
    np.testing.assert_array_equal(axis.get_legend().legend_handles[-1].get_facecolor(), to_rgba(style.Palette.CONTROL, .6))
    assert sum(len(artist.get_offsets()) for artist in axis.collections) == len(source)
    prepared = volcano._prepare(source, appearance)
    np.testing.assert_array_equal(prepared[1], source['standardized_marginal_effect'])
    np.testing.assert_array_equal(prepared[4], [True, False, False, True, True])


def test_numeric_volcano_key_uses_declared_units_and_checks_actual_label_without_mutation():
    """Keep the scalar values and norm unchanged while showing an explicit micrometre unit; an altered nanometre label is retained and diagnosed."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    source = _results().assign(length=[1., 2., 3., 4., 5.])
    source.attrs['units'] = {'length': 'µm'}
    figure, axes = volcano.render_volcano(source, volcano.VolcanoStyle(color_by='length'), screen=True)
    FigureCanvasAgg(figure).draw()
    axis = axes[0]
    key = style._axis_colourbars(axis)[0]
    assert key.get_ylabel() == 'length (µm)'
    values = axis.collections[0].get_array().copy()
    limits = axis.collections[0].get_clim()
    key.set_ylabel('length (nm)')
    report = style.finalize_figure(figure, diagnostics=False)
    assert 'Match the colour-key label to its declared quantity and unit' in report.problems
    assert key.get_ylabel() == 'length (nm)'
    np.testing.assert_array_equal(axis.collections[0].get_array(), values)
    assert axis.collections[0].get_clim() == limits


def test_default_constant_scalar_colour_stays_visible_without_inventing_a_range():
    """A constant .5 scalar keeps its exact original norm bounds and observations, with one declared blue and one .5 key tick rather than a fabricated gradient range."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import to_rgba

    source = _results().assign(score=.5)
    source.attrs['units'] = {'score': 'dimensionless'}
    figure, axes = volcano.render_volcano(source, volcano.VolcanoStyle(color_by='score'), screen=True)
    FigureCanvasAgg(figure).draw()
    points = axes[0].collections[0]
    assert points.get_clim() == (.5, .5)
    np.testing.assert_array_equal(points.get_array(), [.5] * len(source))
    np.testing.assert_array_equal(points.get_facecolors(), np.tile(to_rgba(style.Palette.ACCENT, .6), (len(source), 1)))
    key = style._axis_colourbars(axes[0])[0]
    assert key.get_yticks().tolist() == [.5]
    assert [text.get_text() for text in key.get_yticklabels()] == ['0.5']
    assert key._colorbar.cmap.N == 1
    assert key.get_ylabel() == 'score (dimensionless)'
