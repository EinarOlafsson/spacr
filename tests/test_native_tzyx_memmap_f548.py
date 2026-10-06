"""Native mapped normalization retains scalar statistics and archive bytes."""

import hashlib
import errno
import os
import warnings

import numpy as np
import pytest

from spacr import cancellation, io


@pytest.mark.skipif(not hasattr(os, 'posix_fallocate'),
                    reason='POSIX disk reservation is unavailable')
def test_native_private_map_reservation_falls_back_without_changing_npy_header(
        tmp_path, monkeypatch):
    mapped = np.lib.format.open_memmap(
        tmp_path / 'selected.npy', mode='w+', dtype=np.float32,
        shape=(2, 3, 4))

    def unsupported(*_args):
        raise OSError(errno.EOPNOTSUPP, 'filesystem cannot preallocate')

    monkeypatch.setattr(io.os, 'posix_fallocate', unsupported)
    io._reserve_private_memmap(mapped)
    mapped[:] = np.arange(mapped.size, dtype=np.float32).reshape(mapped.shape)
    io._close_private_memmap(mapped)
    np.testing.assert_array_equal(
        np.load(tmp_path / 'selected.npy', allow_pickle=False),
        np.arange(24, dtype=np.float32).reshape(2, 3, 4))


@pytest.mark.skipif(not hasattr(os, 'posix_fallocate') or not hasattr(os, 'pwrite'),
                    reason='POSIX reservation and positional writes are unavailable')
@pytest.mark.parametrize('failure', ['cancel', 'disk_full'])
def test_native_fallback_reservation_stops_between_chunks_and_closes_map(
        tmp_path, monkeypatch, failure):
    path = tmp_path / 'selected.npy'
    mapped = np.lib.format.open_memmap(
        path, mode='w+', dtype=np.float32, shape=(1024, 1024))
    token = cancellation.CancellationToken()
    original_write = io.os.pwrite
    writes = 0

    def unsupported(*_args):
        raise OSError(errno.EOPNOTSUPP, 'filesystem cannot preallocate')

    def interrupted_write(*args):
        nonlocal writes
        writes += 1
        if writes == 2 and failure == 'disk_full':
            raise OSError(errno.ENOSPC, 'disk full during reservation')
        result = original_write(*args)
        if writes == 1 and failure == 'cancel':
            token.cancel()
        return result

    monkeypatch.setattr(io.os, 'posix_fallocate', unsupported)
    monkeypatch.setattr(io.os, 'pwrite', interrupted_write)
    try:
        with cancellation.installed_token(token):
            if failure == 'cancel':
                with pytest.raises(cancellation.PipelineCancelled):
                    io._reserve_private_memmap(mapped)
            else:
                with pytest.raises(OSError) as error:
                    io._reserve_private_memmap(mapped)
                assert error.value.errno == errno.ENOSPC
    finally:
        io._close_private_memmap(mapped)
        path.unlink()
    assert writes == (1 if failure == 'cancel' else 2)
    assert not path.exists()


def _settings():
    return dict(background=100, Signal_to_noise=10,
                remove_background=False, lower_percentile=17,
                nucleus_channel=3, nucleus_background=200,
                nucleus_signal_to_noise=2, remove_background_nucleus=True,
                cell_channel=1, cell_background=90,
                cell_signal_to_noise=4, remove_background_cell=False,
                pathogen_channel=None, organelle_channel=2,
                organelle_background=150, organelle_signal_to_noise=3,
                remove_background_organelle=True)


def _mapped_output(raw, selected, settings, folder):
    output = np.lib.format.open_memmap(
        folder / 'selected.npy', mode='w+', dtype=np.float32,
        shape=(*raw.shape[:-1], len(selected)))

    def load_channel(channel):
        channel_map = np.lib.format.open_memmap(
            folder / f'channel-{channel}.npy', mode='w+',
            dtype=raw.dtype, shape=raw.shape[:-1])
        channel_map[:] = raw[..., channel]
        return channel_map

    return io._normalize_img_channels(
        output, selected, np.float32, settings, load_channel,
        output_columns={channel: index for index, channel in enumerate(selected)},
        workspace_dir=str(folder))


@pytest.mark.parametrize('dtype', [np.uint16, np.float32, np.float64])
@pytest.mark.parametrize('selected', [[3, 1], [2, 0, 3]])
@pytest.mark.parametrize('lower', [0, 17, 67])
def test_mapped_native_output_and_compressed_archive_match_scalar_oracle(
        dtype, selected, lower, tmp_path):
    rng = np.random.default_rng(20261006)
    raw = rng.integers(0, 4096, size=(3, 2, 41, 37, 4),
                       dtype=np.uint16).astype(dtype)
    raw[::2, :, ::9, ::7] = 0
    raw[..., 0] = 0
    settings = _settings()
    settings['lower_percentile'] = lower
    reference = io._normalize_img_batch(
        raw.copy(), selected, np.float32, settings)[..., selected]
    mapped = _mapped_output(raw, selected, settings, tmp_path)
    assert mapped.tobytes() == reference.tobytes()
    assert not list(tmp_path.glob('channel-*.npy'))
    assert not list(tmp_path.glob('*quantiles.bin'))
    names = ['first.npy', 'second.npy', 'third.npy']
    io._save_npz_atomic(tmp_path / 'scalar.npz',
                        data=reference, filenames=names)
    io._save_npz_atomic(tmp_path / 'mapped.npz',
                        data=mapped, filenames=names)
    assert hashlib.sha256((tmp_path / 'scalar.npz').read_bytes()).digest() == (
        hashlib.sha256((tmp_path / 'mapped.npz').read_bytes()).digest())
    io._close_private_memmap(mapped)


def test_private_nonfinite_quantiles_keep_scalar_behavior(tmp_path):
    rng = np.random.default_rng(42)
    raw = rng.integers(0, 4096, size=(2, 2, 19, 23, 4),
                       dtype=np.uint16).astype(np.float32)
    raw[0, 0, 0, 0, 1] = np.nan
    raw[0, 0, 0, 1, 1] = np.inf
    raw[0, 0, 0, 2, 1] = -np.inf
    selected = [3, 1]
    settings = _settings()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        reference = io._normalize_img_batch(
            raw.copy(), selected, np.float32, settings)[..., selected]
        mapped = _mapped_output(raw, selected, settings, tmp_path)
    assert mapped.tobytes() == reference.tobytes()
    io._close_private_memmap(mapped)


def test_stop_during_quantile_fill_closes_its_private_map(tmp_path, monkeypatch):
    channel = np.lib.format.open_memmap(
        tmp_path / 'source.npy', mode='w+', dtype=np.uint16,
        shape=(2, 2, 8, 8))
    channel[:] = 3
    calls = 0
    closed = []
    original_close = io._close_private_memmap

    def stop_on_first_fill():
        nonlocal calls
        calls += 1
        if calls == 5:
            raise cancellation.PipelineCancelled('stopped during quantile fill')

    def record_close(mapped):
        closed.append(mapped.filename)
        original_close(mapped)

    monkeypatch.setattr(cancellation, 'checkpoint', stop_on_first_fill)
    monkeypatch.setattr(io, '_close_private_memmap', record_close)
    with pytest.raises(cancellation.PipelineCancelled):
        io._native_nonzero_vector(channel, str(tmp_path), 1, 0, False)
    assert calls == 5
    assert any(str(path).endswith('channel-1-quantiles.bin')
               for path in closed)
    assert not list(tmp_path.glob('*quantiles.bin'))
    original_close(channel)


def test_quantile_workspace_allocation_failure_leaves_no_partial_vector(
        tmp_path, monkeypatch):
    channel = np.lib.format.open_memmap(
        tmp_path / 'source.npy', mode='w+', dtype=np.uint16,
        shape=(2, 2, 8, 8))
    channel[:] = 3

    def unavailable(*_args, **_kwargs):
        raise OSError('private workspace is full')

    monkeypatch.setattr(io.np, 'memmap', unavailable)
    with pytest.raises(OSError, match='workspace is full'):
        io._native_nonzero_vector(channel, str(tmp_path), 1, 0, False)
    assert not list(tmp_path.glob('*quantiles.bin'))
    io._close_private_memmap(channel)
