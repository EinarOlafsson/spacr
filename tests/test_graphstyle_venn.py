"""Set overlap uses the shared schematic style and keeps original set/filter semantics."""
import json
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from spacr import plot
from spacr.figures import bundle, style
from spacr.qt import preferences


@pytest.mark.parametrize('threshold', [.1, -.1, None])
def test_venn_declares_and_replays_actual_regions_without_altering_sets(tmp_path, monkeypatch, threshold):
    """Keep distinct-gene counts and exact original patch vertices, and replay .6-alpha regions in local and standalone modes."""
    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    monkeypatch.setattr(plot, '_show_if_interactive', lambda: None)
    paths = [tmp_path / 'first.csv', tmp_path / 'second.csv']
    inputs = [pd.DataFrame({'gene': ['a', 'b', 'c', 'c'], 'coefficient': [.5, .8, -.5, -.5]}),
              pd.DataFrame({'gene': ['a', 'd', 'c', 'e'], 'coefficient': [.5, .8, -.5, -.8]})]
    for frame, path in zip(inputs, paths):
        frame.to_csv(path, index=False)
    before = [path.read_bytes() for path in paths]
    result = plot.create_venn_diagram(*paths, filter_coeff=threshold, save=False)
    expected = {.1: ({'a'}, {'b'}, {'d'}), -.1: ({'c'}, set(), {'e'}), None: ({'a', 'c'}, {'b'}, {'d', 'e'})}[threshold]
    assert tuple(set(result[key]) for key in ('overlap', 'unique_to_file1', 'unique_to_file2')) == expected
    figure = plot.plt.gcf()
    axis = figure.axes[0]
    assert axis._spacr_style_spec.kind == 'schematic'
    assert not axis.axison and axis.get_title() == ''
    assert all(patch.get_alpha() == .6 for patch in axis.patches)
    assert all(text.get_fontsize() >= 8 for text in axis.texts)
    assert not style.figure_problems(figure)
    report = style.finalize_figure(figure, diagnostics=True)
    assert 'registered semantics' in report.checks
    assert not report.problems, report.problems
    data, specification = bundle._figure_record(figure)
    captured = json.loads(json.dumps(bundle._capture_view(figure, specification)))
    assert captured['venn']['alpha'] == .6
    namespace = {'__name__': 'standalone_venn_policy'}
    exec(bundle._recreate_script(captured), namespace)
    for draw in (bundle._draw, namespace['_draw']):
        restored = Figure()
        draw(restored, data, captured)
        for original, copied in zip(axis.patches, restored.axes[0].patches, strict=True):
            np.testing.assert_array_equal(copied.get_path().vertices, original.get_path().vertices)
            np.testing.assert_array_equal(copied.get_facecolor(), original.get_facecolor())
            assert copied.get_alpha() == original.get_alpha()
    assert [path.read_bytes() for path in paths] == before
    plot.plt.close(figure)


def test_axis_off_schematic_is_validated_without_restyling_manual_regions():
    """A declared axis-off chart must undergo validation; measured opacity and colour violations remain unchanged and visible in the report."""
    from matplotlib.patches import Rectangle

    figure = Figure()
    axis = figure.subplots()
    axis.set_axis_off()
    region = Rectangle((0., 0.), 1., 1., facecolor='#C878A8', alpha=.3, gid='spacr-venn-10')
    axis.add_patch(region)
    specification = style.resolve_style('print', 'schematic', comparison={'graph_type': 'venn', 'region_colours': {'10': style.SERIES[0]}},
        legend={'caption': 'Distinct genes in each input and their overlap.', 'unit': 'genes'})
    axis._spacr_style_spec = specification
    before = region.get_facecolor()
    report = style.finalize_figure(figure, diagnostics=False)
    assert 'registered semantics' in report.checks
    assert 'Draw every data mark at sixty percent opacity' in report.problems
    assert 'Match Venn region colours to the declared input and overlap identities' in report.problems
    assert region.get_facecolor() == before and region.get_alpha() == .3


def test_legacy_venn_recipe_without_alpha_keeps_original_opaque_regions():
    """Archives without the additive alpha key retain the original opaque Venn appearance."""
    figure = Figure()
    frame = pd.DataFrame({'input': [0, 0, 1, 1], 'gene': ['a', 'b', 'b', 'c']})
    axis = bundle._draw(figure, frame, {'kind': 'venn', 'venn': {'input_column': 'input', 'gene_column': 'gene',
        'filter_coeff': None, 'inputs': [{'input': 0}, {'input': 1}], 'labels': ['First', 'Second'],
        'colors': {'10': '#0A77B8', '01': '#8C8C8C', '11': '#0A9878'}, 'fontsize': 8, 'ink': '#000000'}})
    assert all(patch.get_alpha() == 1. for patch in axis.patches)


@pytest.mark.parametrize('case', ['empty', 'identical'])
def test_degenerate_venn_keeps_counts_and_has_separate_input_labels(tmp_path, monkeypatch, case):
    """Empty and identical sets expose touching default labels; move only the input captions below the original unmodified region counts and paths."""
    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    monkeypatch.setattr(plot, '_show_if_interactive', lambda: None)
    paths = [tmp_path / 'first.csv', tmp_path / 'second.csv']
    for path in paths:
        pd.DataFrame({'gene': ['a', 'b'], 'coefficient': [.5, .8]}).to_csv(path, index=False)
    result = plot.create_venn_diagram(*paths, filter_coeff=2. if case == 'empty' else None, save=False)
    assert set(result['overlap']) == (set() if case == 'empty' else {'a', 'b'})
    assert result['unique_to_file1'] == result['unique_to_file2'] == []
    figure = plot.plt.gcf()
    axis = figure.axes[0]
    labels = [text for text in axis.texts if text.get_text().startswith('Input')]
    assert [label.get_position() for label in labels] == [(.25, -.08), (.75, -.08)]
    assert all(label.get_transform() is axis.transAxes for label in labels)
    assert not style.figure_problems(figure)
    report = style.finalize_figure(figure)
    assert not report.problems, report.problems
    plot.plt.close(figure)
