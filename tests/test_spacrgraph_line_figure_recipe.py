"""Standalone exports reproduce native grouped numeric lines and SD bands."""
import json
import zipfile

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spacr import plot
from spacr.figures import bundle


@pytest.mark.parametrize('kind,columns,summary', [
    ('line', ['time', 'signal'], 'mean'),
    ('line', ['signal'], 'median'),
    ('line_std', ['time', 'signal'], 'mean'),
])
@pytest.mark.parametrize('representation', ['object', 'well', 'plate'])
@pytest.mark.parametrize('log', [False, True])
def test_native_grouped_line_zip_preserves_source_and_recomputes(
        kind, columns, summary, representation, log, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, '_settings', lambda: QSettings(
        str(tmp_path / 'preferences.ini'), QSettings.IniFormat))
    frame = pd.DataFrame({
        'condition': ['b'] * 8 + ['a'] * 8 + ['a'],
        'time': [1., 1., 3., 3., 9., 9., 27., 27.] * 2 + [float('nan')],
        'signal': [2., 4., 5., 7., 3., 9., 4., 12.,
                   1., 3., 2., 6., 4., 8., 3., 11., float('nan')],
        'sample_name': [f'sample-{index}' for index in range(17)],
        'prc': [f'p{index // 4}_A_{index // 2}' for index in range(17)],
    })
    original = frame.copy(deep=True)
    graph = plot.spacrGraph(frame, 'condition', columns, graph_type=kind,
                            summary_func=summary, order=['b', 'a'],
                            all_to_all=False, representation=representation,
                            log_y=log, log_x=log)
    graph.create_plot()
    try:
        output = bundle._save_zip(graph.fig, str(tmp_path / 'line.zip'), formats=['svg'])
        with zipfile.ZipFile(output) as archive:
            saved = pd.read_csv(archive.open('data.csv'))
            spec = json.loads(archive.read('spec.json'))
            script = archive.read('recreate_figure.py')
        pd.testing.assert_frame_equal(saved, original)
        namespace = {'__name__': 'standalone_grouped_line'}
        exec(compile(script, 'recreate_figure.py', 'exec', dont_inherit=True), namespace)
        restored = Figure()
        axis = namespace['_draw'](restored, saved, spec)
        native = graph.fig.axes[0]
        native_lines = [line for line in native.lines if len(line.get_xdata()) > 0]
        restored_lines = [line for line in axis.lines if len(line.get_xdata()) > 0]
        assert len(restored_lines) == len(native_lines)
        for actual, expected in zip(restored_lines, native_lines):
            np.testing.assert_allclose(np.asarray(actual.get_xdata(), dtype=float),
                                       np.asarray(expected.get_xdata(), dtype=float))
            np.testing.assert_allclose(np.asarray(actual.get_ydata(), dtype=float),
                                       np.asarray(expected.get_ydata(), dtype=float))
        assert len(axis.collections) == len(native.collections)
        for actual, expected in zip(axis.collections, native.collections):
            for apath, epath in zip(actual.get_paths(), expected.get_paths()):
                np.testing.assert_allclose(apath.vertices, epath.vertices)
        edited = saved.copy()
        edited.loc[edited['sample_name'] == 'sample-0', 'signal'] = 42.0
        altered = Figure()
        changed = namespace['_draw'](altered, edited, spec)
        assert not np.array_equal(changed.lines[0].get_ydata(), axis.lines[0].get_ydata())
    finally:
        plt.close(graph.fig)


@pytest.mark.parametrize('kind', ['line', 'line_std'])
@pytest.mark.parametrize('representation', ['object', 'well', 'plate'])
@pytest.mark.parametrize('log', [False, True])
def test_hidden_line_outliers_remain_in_source_and_exported_statistics(
        kind, representation, log, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences
    from spacr.figures import stats

    monkeypatch.setattr(preferences, '_settings', lambda: QSettings(
        str(tmp_path / 'preferences.ini'), QSettings.IniFormat))
    frame = pd.DataFrame({
        'condition': ['a'] * 32 + ['b'] * 32,
        'time': [1., 3., 9., 27.] * 16,
        'signal': [800.] * 4 + [2., 3., 4., 5.] * 7
                  + [3., 4., 5., 6.] * 8,
        'sample_name': [f'sample-{index}' for index in range(64)],
        'prc': [f'p{index // 4}_A_{index}' for index in range(64)],
    })
    graph = plot.spacrGraph(frame, 'condition', ['time', 'signal'],
                            graph_type=kind, summary_func='mean',
                            order=['a', 'b'], all_to_all=False,
                            representation=representation,
                            remove_outliers=True, log_x=log, log_y=log)
    tested = graph.df.copy(deep=True)
    assert tested['signal'].max() == 800.
    graph.create_plot()
    assert len(graph.df) < len(tested)
    assert graph.df['signal'].max() < 800.
    calls = []
    calculate = stats._auto_statistics

    def record(data, *args, **kwargs):
        calls.append(data.copy(deep=True))
        return calculate(data, *args, **kwargs)

    monkeypatch.setattr(stats, '_auto_statistics', record)
    try:
        path = bundle._save_zip(graph.fig, str(tmp_path / 'outliers.zip'),
                                formats=['svg'])
        assert len(calls) == 1
        pd.testing.assert_frame_equal(calls[0], tested)
        with zipfile.ZipFile(path) as archive:
            source = pd.read_csv(archive.open('data.csv'))
            spec = json.loads(archive.read('spec.json'))
            script = archive.read('recreate_figure.py')
            assert not pd.read_csv(archive.open('statistics.csv')).empty
        pd.testing.assert_frame_equal(source, frame)
        namespace = {'__name__': 'standalone_outlier_line'}
        exec(compile(script, 'recreate_figure.py', 'exec', dont_inherit=True),
             namespace)
        restored = namespace['_draw'](Figure(), source, spec)
        native = graph.fig.axes[0]
        assert len(restored.lines) == len(native.lines)
        for actual, expected in zip(restored.lines, native.lines):
            np.testing.assert_allclose(actual.get_xdata(), expected.get_xdata())
            np.testing.assert_allclose(actual.get_ydata(), expected.get_ydata())
        assert len(restored.collections) == len(native.collections)
        for actual, expected in zip(restored.collections, native.collections):
            assert len(actual.get_paths()) == len(expected.get_paths())
            for apath, epath in zip(actual.get_paths(), expected.get_paths()):
                np.testing.assert_allclose(apath.vertices, epath.vertices)
    finally:
        plt.close(graph.fig)
