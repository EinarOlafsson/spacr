"""Plots from spawned mask workers reach the parent without child GUI windows."""

import multiprocessing
import os
import pickle
import queue
from pathlib import Path

import numpy as np
import pytest

from spacr._mask_workers import _read_worker_figure, _run_mask_workers, _worker_figures


def _plot_segmenter(src, settings, object_type, *, batch_paths, on_batch_done, run_qc):
    from matplotlib import pyplot as plt

    from spacr.cancellation import checkpoint
    from spacr.figure_sink import publish

    for index, path in enumerate(batch_paths):
        checkpoint()
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow([[index, 1], [2, 3]], cmap='magma')
        ax.plot([0, 1, 2], [index, index + 1, index + 2], label='mask area')
        ax.set_title(f'worker {os.getpid()} field {index}')
        ax.set_xlabel('Time')
        ax.legend()
        plt.show()
        publish(fig)
        np.save(Path(src) / f'{Path(path).stem}.npy', np.ones((4, 4), np.uint16))
        on_batch_done(path)


def _run(tmp_path, **kwargs):
    source = tmp_path / 'source'
    source.mkdir()
    paths = []
    for index in range(2):
        path = source / f'field{index}.npz'
        path.write_bytes(b'input preserved')
        paths.append(str(path))
    return _run_mask_workers(str(source), {'plot': True}, 'cell', {0: paths},
                             {0: dict(os.environ)}, segmenter=_plot_segmenter, **kwargs)


def test_spawned_figures_preserve_artists_without_registering_parent_windows(tmp_path, monkeypatch):
    import tempfile

    from matplotlib import pyplot as plt

    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    before = plt.get_fignums()
    figures = []
    result = _run(tmp_path, on_figure=figures.append)
    assert len(figures) == 2
    assert len(result['completed_batches']) == 2
    assert plt.get_fignums() == before
    for index, fig in enumerate(figures):
        assert fig.canvas.manager is None
        ax = fig.axes[0]
        assert str(os.getpid()) not in ax.get_title().split()[1:2]
        assert ax.get_title().endswith(f'field {index}')
        assert ax.get_xlabel() == 'Time'
        assert ax.lines[0].get_ydata().tolist() == [index, index + 1, index + 2]
        assert ax.images[0].get_array().tolist() == [[index, 1], [2, 3]]
        assert ax.images[0].get_cmap().name == 'magma'
        assert ax.get_legend().get_texts()[0].get_text() == 'mask area'
    assert not list(tmp_path.glob('spacr-mask-figures-*'))


def test_default_delivery_uses_the_existing_figure_sink(tmp_path):
    from spacr import figure_sink

    figures = []
    previous = figure_sink.set_sink(lambda fig, path: figures.append(fig))
    try:
        _run(tmp_path)
    finally:
        figure_sink.set_sink(previous)
    assert len(figures) == 2
    assert not (tmp_path / 'mask_worker_plots').exists()


def test_headless_figures_are_saved_as_readable_pngs(tmp_path, capsys):
    from PIL import Image

    from spacr import figure_sink

    previous = figure_sink.set_sink(None)
    try:
        _run(tmp_path)
    finally:
        figure_sink.set_sink(previous)
    paths = list((tmp_path / 'mask_worker_plots').glob('*.png'))
    assert len(paths) == 2
    output = capsys.readouterr().out
    for path in paths:
        with Image.open(path) as picture:
            assert picture.size == (200, 200)
            picture.verify()
        assert str(path) in output


def test_failed_delivery_cleans_transport_and_preserves_inputs(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))

    def failed(fig):
        raise RuntimeError('viewer refused figure')

    with pytest.raises(RuntimeError, match='viewer refused figure'):
        _run(tmp_path, on_figure=failed)
    assert not list(tmp_path.glob('spacr-mask-figures-*'))
    inputs = list((tmp_path / 'source').glob('*.npz'))
    assert len(inputs) == 2
    assert all(path.read_bytes() == b'input preserved' for path in inputs)
    assert not [child for child in multiprocessing.active_children()
                if child.name.startswith('spacr-mask-gpu-')]


def test_explicit_publish_cannot_hide_a_serialization_failure(tmp_path, monkeypatch):
    from matplotlib import pyplot as plt

    from spacr import figure_sink

    previous_show = plt.show
    previous_sink = figure_sink.sink()

    def fail(*args, **kwargs):
        raise OSError('plot storage full')

    monkeypatch.setattr(pickle, 'dump', fail)
    with pytest.raises(RuntimeError, match='plot storage full'):
        with _worker_figures(str(tmp_path), queue.Queue(), 0):
            fig = plt.figure()
            figure_sink.publish(fig)
    assert plt.show is previous_show
    assert figure_sink.sink() is previous_sink


def test_transport_rejects_external_paths_and_nonfigure_payloads(tmp_path):
    outside = tmp_path / 'outside.pickle'
    outside.write_bytes(b'not a pickle')
    folder = tmp_path / 'owned'
    folder.mkdir()
    with pytest.raises(ValueError, match='outside'):
        _read_worker_figure(outside, folder)
    assert outside.read_bytes() == b'not a pickle'
    inside = folder / 'invalid.pickle'
    inside.write_bytes(pickle.dumps({'not': 'a figure'}))
    with pytest.raises(ValueError, match='not a Matplotlib Figure'):
        _read_worker_figure(inside, folder)
    assert not inside.exists()
