"""Training-curve exports retain fold spread, source metadata and selections."""
import json
import os
import subprocess
import sys
import zipfile

from matplotlib.figure import Figure
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spacr import train_compare as tc
from spacr.figures import bundle


@pytest.mark.parametrize('band,mark_best,selected', [
    (True, True, False), (True, False, True), (False, True, True),
    (False, False, False),
])
@pytest.mark.parametrize('metric', ['accuracy', 'loss'])
@pytest.mark.parametrize('restyled', [False, True])
def test_training_curve_zip_recreates_fold_bands_and_selected_lines(
        band, mark_best, selected, metric, restyled, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, '_settings', lambda: QSettings(
        str(tmp_path / 'preferences.ini'), QSettings.IniFormat))
    means = pd.DataFrame({
        'epoch': [1., 3., 7.], 'accuracy': [.4, .6, .5],
        'accuracy__sd': [.1, .15, float('nan')],
        'loss': [3., 2., 1.], 'loss__sd': [.3, .2, float('nan')],
        'n_folds': [3, 3, 1],
    })
    fold = pd.DataFrame({'epoch': [1., 3.], 'accuracy': [.7, .9],
                         'loss': [2., 1.]})
    series = [tc.Series('run-a', 'val', 'mean', 'mean', 'a val mean',
                        means, n_folds=3),
              tc.Series('run-a', 'train', 'fold_1', 'fold', 'a train fold1',
                        fold)]
    comparison = tc.Comparison([], series, {}, ['accuracy', 'loss'],
                                fold_mode='both')
    labels = [series[0].label] if selected else None
    figure = tc.plot_curves(comparison, metric, labels=labels,
                            band=band, mark_best=mark_best)
    if restyled:
        for line in figure.axes[0].lines:
            line.set_color('#7817a4')
            line.set_linewidth(2.75)
            line.set_markersize(7.0)
            line.set_marker('^')
            line.set_alpha(.8)
    try:
        path = bundle._save_zip(figure, str(tmp_path / 'training.zip'),
                                formats=['svg'])
        with zipfile.ZipFile(path) as archive:
            source = pd.read_csv(archive.open('data.csv'))
            spec = json.loads(archive.read('spec.json'))
            script = archive.read('recreate_figure.py')
            archive.extractall(tmp_path / 'standalone')
        assert {'accuracy__sd', 'loss', 'n_folds', 'run_id', 'fold', 'kind'} <= set(source.columns)
        assert len(source) == 5
        assert source['series'].tolist() == ['a val mean'] * 3 + ['a train fold1'] * 2
        namespace = {'__name__': 'standalone_training'}
        exec(compile(script, 'recreate_figure.py', 'exec', dont_inherit=True),
             namespace)
        restored = namespace['_draw'](Figure(), source, spec)
        native = figure.axes[0]
        actual_lines = [line for line in restored.lines if len(line.get_xdata())]
        expected_lines = [line for line in native.lines if len(line.get_xdata())]
        assert len(actual_lines) == len(expected_lines)
        for actual, expected in zip(actual_lines, expected_lines):
            np.testing.assert_allclose(actual.get_xdata(), expected.get_xdata())
            np.testing.assert_allclose(actual.get_ydata(), expected.get_ydata())
            np.testing.assert_allclose(to_rgba(actual.get_color()),
                                       to_rgba(expected.get_color()))
            assert actual.get_linestyle() == expected.get_linestyle()
            assert actual.get_marker() == expected.get_marker()
            assert actual.get_alpha() == expected.get_alpha()
            assert actual.get_linewidth() == expected.get_linewidth()
            assert actual.get_markersize() == expected.get_markersize()
        assert len(restored.collections) == len(native.collections)
        for actual, expected in zip(restored.collections, native.collections):
            np.testing.assert_allclose(actual.get_facecolor(), expected.get_facecolor())
            np.testing.assert_allclose(actual.get_linewidth(), expected.get_linewidth())
            assert actual.get_alpha() == expected.get_alpha()
            assert len(actual.get_paths()) == len(expected.get_paths())
            for apath, epath in zip(actual.get_paths(), expected.get_paths()):
                np.testing.assert_allclose(apath.vertices, epath.vertices)
        assert [text.get_text() for text in restored.get_legend().get_texts()] == [
            text.get_text() for text in native.get_legend().get_texts()]
        edited = source.copy()
        selected_rows = edited['series'] == 'a val mean'
        edited.loc[selected_rows, metric] += .25
        if mark_best:
            edited.loc[selected_rows & (edited['epoch'] == 1.), metric] = (
                .99 if metric == 'accuracy' else .01)
        altered = namespace['_draw'](Figure(), edited, spec)
        assert not np.array_equal(altered.lines[0].get_ydata(), restored.lines[0].get_ydata())
        if mark_best:
            assert altered.lines[1].get_xdata()[0] == 1.
        if band and mark_best:
            guard = tmp_path / 'guard'
            guard.mkdir()
            (guard / 'sitecustomize.py').write_text(
                "import sys\n"
                "class RejectSpacr:\n"
                "    def find_spec(self, fullname, path=None, target=None):\n"
                "        if fullname == 'spacr' or fullname.startswith('spacr.'):\n"
                "            raise ImportError('standalone must not import spaCR')\n"
                "sys.meta_path.insert(0, RejectSpacr())\n")
            edited.to_csv(tmp_path / 'standalone/data.csv', index=False)
            process = subprocess.run(
                [sys.executable, str(tmp_path / 'standalone/recreate_figure.py'),
                 'png', 'svg'], cwd=tmp_path / 'standalone',
                env=dict(os.environ, PYTHONPATH=str(guard), MPLBACKEND='Agg'),
                capture_output=True, text=True, timeout=60)
            assert process.returncode == 0, process.stdout + process.stderr
            assert (tmp_path / 'standalone/recreated.png').read_bytes().startswith(
                b'\x89PNG\r\n\x1a\n')
            assert '<svg' in (tmp_path / 'standalone/recreated.svg').read_text()
    finally:
        plt.close(figure)
