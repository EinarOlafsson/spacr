"""Captured bundles preserve real manual styles and portable fonts without changing results."""
import hashlib
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba

from spacr.figures import bundle, style
from spacr.figures.sheet import build_panel
from spacr.tabular import read_table
from tests.test_the_figure_house_style import _results


def test_bundle_replays_actual_manual_fonts_colours_and_geometry(tmp_path):
    """The saved recipe previously lost manual SemiBold title and .42-alpha bars; preserve those exact styles and complete CSV bytes."""
    frame = _results(24)
    figure, _ = build_panel('effect_rank', frame, target='print')
    ax = figure.axes[0]
    ax.set_title('Captured manual title', fontfamily='Open Sans', fontweight='semibold', fontsize=11)
    ax.xaxis.label.set_color('#283868')
    ax.xaxis.label.set_fontsize(9)
    ax.spines['top'].set_linewidth(1.2)
    ax.patches[0].set_facecolor('#C878A8')
    ax.patches[0].set_alpha(.42)
    original = ax.patches[0].get_path().vertices.copy()
    path = bundle._save_zip(figure, str(tmp_path/'saved.zip'), formats=['svg'])
    with zipfile.ZipFile(path) as archive:
        assert archive.read('data.csv').decode() == frame.to_csv(index=False)
        spec = json.loads(archive.read('spec.json'))
        archive.extract('data.csv', tmp_path)
        data = read_table(tmp_path/'data.csv', canonicalise=False, float_precision='round_trip')
        namespace = {'__name__': 'standalone_probe'}
        exec(compile(archive.read('recreate_figure.py'), 'recreate_figure.py', 'exec'), namespace)
    restored = Figure()
    namespace['_draw'](restored, data, spec)
    replay = restored.axes[0]
    assert restored._spacr_replay_problems == ()
    assert replay.title.get_fontfamily() == ['Open Sans']
    assert replay.title.get_fontweight() == 'semibold'
    assert replay.title.get_fontsize() == 11
    assert replay.xaxis.label.get_fontsize() == 9
    assert to_rgba(replay.xaxis.label.get_color()) == to_rgba('#283868')
    assert replay.spines['top'].get_linewidth() == 1.2
    assert replay.patches[0].get_facecolor() == to_rgba('#C878A8', .42)
    assert replay.get_position().bounds == ax.get_position().bounds
    np.testing.assert_array_equal(replay.patches[0].get_path().vertices, original)


def test_relocated_bundle_resolves_font_from_its_own_files_without_spacr(tmp_path):
    """The original standalone export had no fonts; require actual Open Sans resolution from the relocated archive and recorded SHA-256 bytes."""
    figure, _ = build_panel('effect_rank', _results(24), target='print')
    path = bundle._save_zip(figure, str(tmp_path/'saved.zip'), formats=['svg'])
    destination = tmp_path/'relocated'
    with zipfile.ZipFile(path) as archive:
        spec = json.loads(archive.read('spec.json'))
        assert {'OpenSans-Light.ttf','OpenSans-Regular.ttf','OpenSans-SemiBold.ttf'} <= {r['name'] for r in spec['fonts']}
        assert b'SIL OPEN FONT LICENSE' in archive.read('OpenSans-OFL.txt')
        for record in spec['fonts']:
            assert hashlib.sha256(archive.read(record['name'])).hexdigest() == record['sha256']
        archive.extractall(destination)
    probe = '''import json, pathlib, runpy, sys
class RejectSpacr:
    def find_spec(self, fullname, path=None, target=None):
        """Refuse any runtime dependency on the original package."""
        if fullname == 'spacr' or fullname.startswith('spacr.'):
            raise ImportError(fullname)
sys.meta_path.insert(0, RejectSpacr())
namespace = runpy.run_path('recreate_figure.py', run_name='portable_probe')
spec = json.loads(pathlib.Path('spec.json').read_text())
namespace['_register_replay_fonts'](pathlib.Path.cwd(), spec['fonts'])
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
resolved = pathlib.Path(font_manager.findfont(FontProperties(family='Open Sans', weight='light'))).resolve()
assert resolved.parent == pathlib.Path.cwd(), resolved
namespace['main']()
assert pathlib.Path('recreated.png').is_file()
'''
    environment = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(tmp_path/'font-cache'), PYTHONPATH='')
    result = subprocess.run([sys.executable, '-c', probe], cwd=destination, env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_incompatible_capture_reports_the_mismatch_without_recolouring_other_artists():
    """An added line changes the ordered signature; keep both lines untouched and report the mismatch."""
    figure = Figure()
    ax = figure.add_subplot()
    ax.plot([0,1],[1,2], color='#0A77B8', alpha=.6)
    saved = style._capture_appearance(figure)
    second, = ax.plot([0,1],[2,3],color='#C878A8',alpha=.3)
    before = second.get_xydata().copy()
    findings = style._restore_appearance(figure, saved)
    assert 'Captured appearance does not match recreated lines' in findings
    assert second.get_color() == '#C878A8' and second.get_alpha() == .3
    np.testing.assert_array_equal(second.get_xydata(), before)


def test_legacy_recipe_has_no_implicit_style_upgrade():
    """A recipe without captured appearance retains its original face, font and data."""
    figure = Figure(facecolor='#171C22')
    ax = figure.add_subplot()
    ax.title.set_fontsize(12)
    assert style._restore_appearance(figure, None) == ()
    assert figure.get_facecolor() == to_rgba('#171C22')
    assert ax.title.get_fontsize() == 12


@pytest.mark.parametrize('producer', ['grouped', 'graph'])
@pytest.mark.parametrize('count', [1, 20, 21, 1000])
def test_generic_condition_bundle_replays_captured_forms_order_and_observations(
        producer, count, tmp_path, monkeypatch):
    """New auto plots replayed as historical Seaborn boxes; require exact artist types, coordinates and source CSV bytes at each count boundary, including independently edited display labels."""
    import builtins
    import pandas as pd
    import matplotlib.pyplot as plt
    from PySide6.QtCore import QSettings
    from spacr import plot
    from spacr.qt import preferences

    def settings():
        """Keep explicit figure choices inside this regression fixture."""
        return QSettings(str(tmp_path / 'preferences.ini'), QSettings.IniFormat)

    def avoid_display(*args, **kwargs):
        """Leave rendering headless while retaining the real plot creator."""
        return None

    monkeypatch.setattr(preferences, '_settings', settings)
    monkeypatch.setattr(plot, '_show_if_interactive', avoid_display)
    monkeypatch.setattr(plt, 'show', avoid_display)
    frame = pd.DataFrame({'condition': np.repeat(['reference', 'treated'], count),
                          'signal': np.concatenate([np.linspace(1., 2., count), np.linspace(3., 4., count)])})
    order = ['treated', 'reference']
    if producer == 'grouped':
        figure, _ = plot.create_grouped_plot(frame, 'condition', 'signal',
            order=order, colors=[style.Palette.ACCENT, style.Palette.CONTROL])
    else:
        graph = plot.spacrGraph(frame, 'condition', 'signal', order=order,
                                compare_group='reference', all_to_all=False)
        graph.create_plot()
        figure = graph.fig
    try:
        original = figure.axes[0]
        original.set_xticklabels(['Treatment display', 'Reference display'])
        captured_data, _ = bundle._figure_record(figure)
        expected_csv = captured_data.to_csv(index=False).encode()
        output = bundle._save_zip(figure, str(tmp_path / 'conditions.zip'), formats=['svg'])
        with zipfile.ZipFile(output) as archive:
            assert archive.read('data.csv') == expected_csv
            spec = json.loads(archive.read('spec.json'))
            script = archive.read('recreate_figure.py')
            archive.extract('data.csv', tmp_path)
        data = read_table(tmp_path / 'data.csv', canonicalise=False, float_precision='round_trip')
        assert spec['condition_recipe']['labels'] == order
        assert spec['condition_recipe']['kind'] == [style.condition_kind(np.arange(count))] * 2
        assert spec['condition_recipe']['ticklabels'] == ['Treatment display', 'Reference display']
        namespace = {'__name__': 'portable_conditions'}
        native_import = builtins.__import__

        def reject_package(name, *args, **kwargs):
            """Prove this captured renderer has no dependency on the original package."""
            if name == 'spacr' or name.startswith('spacr.'):
                raise ImportError(name)
            return native_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', reject_package)
        exec(compile(script, 'recreate_figure.py', 'exec', dont_inherit=True), namespace)
        restored = Figure()
        replay = namespace['_draw'](restored, data, spec)
        assert restored._spacr_replay_problems == ()
        for name in ('lines', 'collections', 'patches'):
            actual = list(getattr(replay, name))
            expected = list(getattr(original, name))
            assert [type(artist).__name__ for artist in actual] == [type(artist).__name__ for artist in expected]
            for first, second in zip(actual, expected):
                if name == 'lines':
                    np.testing.assert_array_equal(first.get_xydata(), second.get_xydata())
                elif name == 'collections' and hasattr(first, 'get_offsets'):
                    np.testing.assert_array_equal(first.get_offsets(), second.get_offsets())
                    assert len(first.get_paths()) == len(second.get_paths())
                    for apath, epath in zip(first.get_paths(), second.get_paths()):
                        np.testing.assert_array_equal(apath.vertices, epath.vertices)
                else:
                    np.testing.assert_array_equal(first.get_path().vertices, second.get_path().vertices)
        assert [label.get_text() for label in replay.get_xticklabels()] == ['Treatment display', 'Reference display']
        assert replay.get_position().bounds == original.get_position().bounds
        assert replay.get_xlim() == original.get_xlim()
        assert replay.get_ylim() == original.get_ylim()
    finally:
        plt.close(figure)


def test_capture_preserves_manual_limits_inversion_scales_and_side_titles():
    """Captured styles lacked axis views and left/right titles; preserve exact chosen limits and actual title faces/positions while all source coordinates remain unchanged."""
    figure = Figure(); ax = figure.add_subplot()
    line, = ax.plot([1., 10.], [2., 20.])
    ax.set_xscale('log'); ax.set_xlim(12., .8); ax.set_ylim(1., 22.)
    ax.set_title('Left captured title', loc='left', y=1.12, fontfamily='Open Sans', fontweight='semibold', fontsize=11)
    ax.set_title('Right captured title', loc='right', y=1.12, fontfamily='Open Sans', fontweight='regular', fontsize=9)
    saved = style._capture_appearance(figure)
    restored = Figure(); replay = restored.add_subplot()
    copied, = replay.plot([1., 10.], [2., 20.])
    assert not style._restore_appearance(restored, saved)
    assert replay.get_xscale() == 'log'
    assert replay.get_xlim() == ax.get_xlim() and replay.get_ylim() == ax.get_ylim()
    assert replay.get_autoscalex_on() == ax.get_autoscalex_on()
    assert replay.get_autoscaley_on() == ax.get_autoscaley_on()
    for side in ('_left_title', '_right_title'):
        source, target = getattr(ax, side), getattr(replay, side)
        assert target.get_text() == source.get_text()
        assert target.get_position() == source.get_position()
        assert target.get_fontfamily() == source.get_fontfamily()
        assert target.get_fontsize() == source.get_fontsize()
        assert target.get_fontweight() == source.get_fontweight()
    np.testing.assert_array_equal(copied.get_xydata(), line.get_xydata())


@pytest.mark.parametrize('count', [1, 21, 1000])
def test_portable_condition_recipe_keeps_horizontal_binding_and_actual_values(tmp_path, count):
    """Horizontal grouped archives used their historical renderer; captured shared forms must retain exact source CSV, measurement coordinates, full group identities and view limits without importing spaCR."""
    import pandas as pd
    from matplotlib.collections import PathCollection

    frame = pd.DataFrame({'condition': np.repeat(['reference', 'treated'], count),
                          'value': np.concatenate([np.linspace(1., 2., count), np.linspace(2., 3., count)])})
    values = [part.value.to_numpy() for _, part in frame.groupby('condition', sort=False)]
    with style.figure_style('print'):
        figure = Figure(); ax = figure.add_subplot()
        style.condition_plot(ax, values, control=0, labels=['reference', 'treated'], orientation='horizontal')
        ax.set_yticks([0, 1], ['Reference display', 'Treatment display'])
        ax.set_xlabel('Value'); ax.set_ylabel('Condition')
        spec = style.resolve_style('print', 'categorical', conditions=['reference', 'treated'],
            sample_counts=[count, count], control=0, comparison={'orientation': 'horizontal'},
            legend={'caption': 'Synthetic values.', 'unit': 'a.u.', 'quantities': {'x': 'value', 'y': 'condition'}})
        style.apply_axes_style(ax, spec)
    bundle._register_figure_data(figure, frame, x='value', y='condition', kind='box_strip')
    destination = bundle._save_zip(figure, str(tmp_path / 'horizontal.zip'), formats=['svg'])
    with zipfile.ZipFile(destination) as archive:
        assert archive.read('data.csv') == frame.to_csv(index=False).encode()
        captured = json.loads(archive.read('spec.json'))
        script = archive.read('recreate_figure.py')
        archive.extract('data.csv', tmp_path)
    data = read_table(tmp_path / 'data.csv', canonicalise=False, float_precision='round_trip')
    assert captured['condition_recipe']['orientation'] == 'horizontal'
    namespace = {'__name__': 'portable_horizontal'}
    exec(compile(script, 'recreate_figure.py', 'exec', dont_inherit=True), namespace)
    restored = Figure(); replay = namespace['_draw'](restored, data, captured)
    assert not restored._spacr_replay_problems
    assert replay.get_xlim() == ax.get_xlim() and replay.get_ylim() == ax.get_ylim()
    assert replay.get_position().bounds == ax.get_position().bounds
    assert [label.get_text() for label in replay.get_yticklabels()] == ['Reference display', 'Treatment display']
    source_points = [item.get_offsets() for item in ax.collections if isinstance(item, PathCollection)]
    replay_points = [item.get_offsets() for item in replay.collections if isinstance(item, PathCollection)]
    assert len(source_points) == len(replay_points)
    for expected, actual in zip(source_points, replay_points):
        np.testing.assert_array_equal(actual, expected)
    assert [type(item).__name__ for item in replay.patches] == [type(item).__name__ for item in ax.patches]
    for expected, actual in zip(ax.patches, replay.patches):
        np.testing.assert_array_equal(actual.get_path().vertices, expected.get_path().vertices)
    assert replay._spacr_condition_kinds == ax._spacr_condition_kinds
