"""Native mapped normalization retains scalar statistics and archive bytes."""

import hashlib
import warnings

import numpy as np
import pytest

from spacr import cancellation, io


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
