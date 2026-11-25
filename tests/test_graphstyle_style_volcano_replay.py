"""Explorer volcano archives replay their original transformations and vector primitives."""
import json
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from spacr import volcano_style as volcano
from spacr.figures import bundle, style
from spacr.qt import preferences


@pytest.fixture(autouse=True)
def _close_replay_figures():
    """Keep repeated archive renders isolated and release their owned pyplot figures."""
    from matplotlib import pyplot as plt

    plt.close('all')
    yield
    plt.close('all')


@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize('case', ['ordinary', 'raw', 'numeric', 'constant', 'shapes', 'controls', 'split', 'annotations', 'localization', 'manual'])
def test_explorer_volcano_replays_source_transform_masks_keys_and_annotations(monkeypatch, case, standalone):
    """Keep every source value, scatter coordinate/mask/style, reference vertex, annotation anchor, key and split window exact in local and spaCR-free source replay."""
    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    source = pd.DataFrame({'standardized_marginal_effect': np.linspace(-2.5, 2.5, 16),
        'adjusted_p_value': np.geomspace(.0001, .9, 16), 'guide': ['g' + str(i) for i in range(16)],
        'score': np.linspace(0., 1., 16), 'class': ['a', 'b'] * 8, 'control': [False] * 4 + [True] * 8 + [False] * 4})
    choices = {'ordinary': {}, 'raw': {'y_neg_log10': False}, 'numeric': {'color_by': 'score'},
        'constant': {'color_by': 'score'}, 'shapes': {'color_by': 'class', 'shape_by': 'class'},
        'controls': {'control_column': 'control', 'threshold_method': 'control'},
        'split': {'split_axis': True, 'split_y_lims': ((0., 1.), (2., 4.))},
        'annotations': {'annotations': {'g0': 'A', 'g15': 'B'}},
        'localization': {'localizations': ['A', 'B']},
        'manual': {'marker_size': 17., 'marker_alpha': .35, 'figure_width': 5., 'figure_height': 3.,
            'font_family': 'DejaVu Sans', 'x_lim': (-3., 3.), 'y_lim': (0., 5.)}}[case]
    if case == 'constant':
        source['score'] = .5
    places = np.array(['A', 'B', ''] * 5 + [''])
    monkeypatch.setattr(volcano, 'point_localizations', lambda *args: places)
    if case in ('numeric', 'constant'):
        source.attrs['units'] = {'score': 'µm'}
    figure, axes = volcano.render_volcano(source, volcano.VolcanoStyle(**choices), screen=True)
    if case == 'manual':
        axes[0].set_xlabel('Manually edited effect (ratio)')
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(figure).draw()
    data, specification = bundle._figure_record(figure)
    captured = json.loads(json.dumps(bundle._capture_view(figure, specification)))
    assert captured['volcano_style_recipe']['schema'] == 1
    before = data.to_csv(index=False)
    if standalone:
        namespace = {'__name__': 'standalone_style_volcano'}
        exec(compile(bundle._recreate_script(captured), 'recreate_figure.py', 'exec', dont_inherit=True), namespace)
        draw = namespace['_draw']
    else:
        draw = bundle._draw
    monkeypatch.setattr(volcano, 'point_localizations', lambda *args: (_ for _ in ()).throw(AssertionError('replay must use captured localization identities')))
    restored = Figure()
    draw(restored, data, captured)
    FigureCanvasAgg(restored).draw()
    assert not getattr(restored, '_spacr_replay_problems', ()), getattr(restored, '_spacr_replay_problems', ())
    assert data.to_csv(index=False) == before
    assert len(restored.axes) == len(figure.axes)
    for original, copied in zip(figure.axes, restored.axes, strict=True):
        assert copied.get_xlim() == original.get_xlim() and copied.get_ylim() == original.get_ylim()
        assert copied.axison == original.axison
        assert copied.get_xlabel() == original.get_xlabel()
        assert copied.get_ylabel() == original.get_ylabel()
        assert len(copied.collections) == len(original.collections)
        assert len(copied.lines) == len(original.lines)
        for first, second in zip(original.collections, copied.collections, strict=True):
            np.testing.assert_array_equal(second.get_offsets(), first.get_offsets())
            assert style._artist_appearance(second) == style._artist_appearance(first)
        for first, second in zip(original.lines, copied.lines, strict=True):
            np.testing.assert_array_equal(second.get_xydata(), first.get_xydata())
            assert style._artist_appearance(second) == style._artist_appearance(first)
        assert [text.get_text() for text in copied.texts] == [text.get_text() for text in original.texts]
        for first, second in zip(original.texts, copied.texts, strict=True):
            assert second.get_position() == first.get_position()
            if hasattr(first, 'xy'):
                assert second.xy == first.xy and second.xycoords == first.xycoords
            assert style._text_appearance(second) == style._text_appearance(first)
        if original.get_legend() is not None:
            assert style._legend_appearance(copied.get_legend()) == style._legend_appearance(original.get_legend())


def test_replay_keeps_a_line_collection_explicitly_unfilled():
    """Colorbar line collections expose fill=False with an empty face array; replay must retain the no-fill sentinel as well as every line vertex."""
    from matplotlib.collections import LineCollection

    original = LineCollection([[(0., 0.), (1., 1.)]], facecolors='none', colors='#000000', alpha=.6)
    captured = json.loads(json.dumps(style._artist_appearance(original)))
    assert captured['fill'] is False and captured['facecolor'] == []
    restored = LineCollection(original.get_segments())
    assert not style._restore_artist_appearance(restored, captured)
    assert style._artist_appearance(restored) == style._artist_appearance(original)
    np.testing.assert_array_equal(restored.get_segments(), original.get_segments())


def test_exported_volcano_script_runs_without_spacr_and_reads_exact_float_values(tmp_path, monkeypatch):
    """Exercise the actual archived entry point with spaCR imports rejected; adjacent float64 values must survive CSV loading exactly and vector files must be written."""
    import os
    import subprocess
    import sys
    import zipfile
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    values = [0.1, float(np.nextafter(.1, 1.)), float(np.sqrt(2.)), -2.1]
    source = pd.DataFrame({'standardized_marginal_effect': values,
        'adjusted_p_value': [.001, .5, .8, .04], 'guide': ['a', 'b', 'c', 'd']})
    figure, axes = volcano.render_volcano(source, volcano.VolcanoStyle(annotations={'a': 'A'}), screen=True)
    FigureCanvasAgg(figure).draw()
    archive = bundle._save_zip(figure, str(tmp_path / 'volcano.zip'), formats=['svg'])
    folder = tmp_path / 'portable'
    with zipfile.ZipFile(archive) as saved:
        saved.extractall(folder)
    guard = tmp_path / 'guard'
    guard.mkdir()
    (guard / 'sitecustomize.py').write_text(
        "import sys\n"
        "class RejectSpacr:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'spacr' or fullname.startswith('spacr.'):\n"
        "            raise ImportError('standalone replay must not import spaCR')\n"
        "sys.meta_path.insert(0, RejectSpacr())\n")
    probe = folder / 'probe.py'
    probe.write_text(
        "import json, runpy, sys\nfrom pathlib import Path\nimport pandas as pd\n"
        "original = pd.read_csv\n"
        "def record(*args, **kwargs):\n"
        "    \"\"\"Record the values the real archive main function actually parsed.\"\"\"\n"
        "    frame = original(*args, **kwargs)\n"
        "    Path('parsed.json').write_text(json.dumps(frame['standardized_marginal_effect'].tolist()))\n"
        "    return frame\n"
        "pd.read_csv = record\n"
        "sys.argv = ['recreate_figure.py', 'svg', 'pdf']\n"
        "runpy.run_path('recreate_figure.py', run_name='__main__')\n")
    process = subprocess.run([sys.executable, str(probe)], cwd=folder,
        env=dict(os.environ, PYTHONPATH=str(guard), MPLBACKEND='Agg'), capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    np.testing.assert_array_equal(json.loads((folder / 'parsed.json').read_text()), values)
    assert '<svg' in (folder / 'recreated.svg').read_text()
    assert '<image' not in (folder / 'recreated.svg').read_text()
    assert (folder / 'recreated.pdf').read_bytes().startswith(b'%PDF')
