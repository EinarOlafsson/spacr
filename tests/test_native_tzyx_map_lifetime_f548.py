"""Native normalization diagnostics retain safe mapped pixels until released."""

import gc
from pathlib import Path

import numpy as np
import pytest

from spacr import io
from tests.test_native_tzyx_batch_f548 import _settings
from tests.test_watch_timelapse_series_f548 import _converted_series


def _frame_array(error, function, name):
    trace = error.__traceback__
    while trace is not None:
        if trace.tb_frame.f_code.co_name == function:
            return trace.tb_frame.f_locals[name]
        trace = trace.tb_next
    raise AssertionError(f'Missing {function} diagnostic frame')


def _map_handle(array):
    owner = array
    while owner is not None:
        if isinstance(owner, np.memmap):
            return owner._mmap
        owner = getattr(owner, 'base', None)
    raise AssertionError('Diagnostic array lost its mapping owner')


@pytest.mark.parametrize('operation', ['percentile', 'rescale', 'reserve'])
def test_actual_native_failure_retains_readable_pixels_then_removes_workspace(
        tmp_path, monkeypatch, operation):
    source, _rows = _converted_series(tmp_path)

    def fail(values, *args, **kwargs):
        raise RuntimeError(f'{operation} diagnostic')

    if operation == 'percentile':
        monkeypatch.setattr(io.np, 'percentile', fail)
    elif operation == 'rescale':
        monkeypatch.setattr(io.exposure, 'rescale_intensity', fail)
    else:
        monkeypatch.setattr(io, '_reserve_private_memmap', fail)
    with pytest.raises(RuntimeError, match=f'{operation} diagnostic') as failure:
        io.preprocess_img_data(_settings(source))
    borrowed = _frame_array(failure.value, 'fail', 'values')
    handle = _map_handle(borrowed)
    assert not handle.closed
    assert np.isfinite(borrowed).all()
    assert borrowed.size > 0
    stage = list(source.glob('.spacr-volume-series-*'))
    assert len(stage) == 1
    assert not (source / 'stack').exists()
    assert not (source / 'masks').exists()
    del borrowed, failure
    gc.collect()
    assert handle.closed
    assert not stage[0].exists()


@pytest.mark.parametrize('order', ['C', 'F'])
def test_native_workspace_retained_axis_view_closes_before_file_removal(
        tmp_path, monkeypatch, order):
    observed = []
    unlink = io.os.unlink
    with io._native_map_workspace(prefix='.native-test-', dir=tmp_path) as workspace:
        path = Path(workspace['name']) / 'channel.npy'
        owner = io._retain_native_memmap(
            np.lib.format.open_memmap(path, mode='w+', dtype=np.uint16,
                                      shape=(2, 3, 4), fortran_order=order == 'F'),
            workspace['owners'], remove_path=path)
        owner[:] = np.arange(24).reshape(2, 3, 4)
        retained = np.asarray(owner).swapaxes(0, 1)
        handle = owner._mmap
        io._close_private_memmap(owner)
        del owner
    assert path.exists() and not handle.closed
    assert retained.sum() == 276

    def observe_removal(name, *args, **kwargs):
        if str(name) == str(path):
            observed.append(handle.closed)
        return unlink(name, *args, **kwargs)

    monkeypatch.setattr(io.os, 'unlink', observe_removal)
    del retained
    gc.collect()
    assert observed == [True]
    assert handle.closed and not path.parent.exists()


def test_nested_native_workspace_preserves_source_until_last_borrower(tmp_path):
    with io._native_map_workspace(prefix='.outer-', dir=tmp_path) as outer:
        source = Path(outer['name']) / 'source.npy'
        np.save(source, np.arange(24, dtype=np.float32))
        with io._native_map_workspace(prefix='.inner-', dir=outer['name'],
                                    parent=outer) as inner:
            mapped = io._retain_native_memmap(
                np.load(source, mmap_mode='r'), inner['owners'])
            view = mapped[::2]
            handle = mapped._mmap
            del mapped
    assert source.exists() and not handle.closed
    assert view.sum() == 132
    del view
    gc.collect()
    assert handle.closed and not source.parent.exists()


def test_native_success_removes_all_private_workspaces_before_return(tmp_path):
    source, _rows = _converted_series(tmp_path)
    io.preprocess_img_data(_settings(source))
    assert (source / 'stack').is_dir() and (source / 'masks').is_dir()
    assert not list(source.glob('.spacr-volume-series-*'))


def test_failed_quantile_map_constructor_removes_partial_private_file(
        tmp_path, monkeypatch):
    channel = np.arange(24, dtype=np.uint16).reshape(2, 2, 2, 3)

    def incomplete_file(path, *args, **kwargs):
        Path(path).write_bytes(b'incomplete private map')
        raise OSError('allocation failed after creating file')

    monkeypatch.setattr(io.np, 'memmap', incomplete_file)
    with pytest.raises(OSError, match='allocation failed'):
        io._native_nonzero_vector(channel, str(tmp_path), 1, 0, False)
    assert not list(tmp_path.glob('channel-1-quantiles.bin'))
