"""Opt-in raw TIFF volumes retain Z; legacy plane ingestion stays separate."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import io
from spacr.settings import set_default_settings_preprocess_generate_masks


def _settings(folder, **changes):
    settings = dict(src=str(folder), channels=[0], nucleus_channel=0,
                    cell_channel=None, pathogen_channel=None,
                    z_stack=True, z_segmentation_mode='volumetric', z_axis=0,
                    anisotropy=2, metadata_type='cellvoyager', plot=False,
                    test_mode=False, n_jobs=1, batch_size=8, verbose=False,
                    normalize=True, masks=True, save_original_images=False,
                    remove_background_nucleus=False)
    settings.update(changes)
    return set_default_settings_preprocess_generate_masks(settings)


def _write(folder, channel=1, field=1, z=1, axes='ZYX', shape=(4, 8, 9)):
    image = (np.arange(np.prod(shape)).reshape(shape) + channel*10).astype(np.uint16)
    name = f'plate1_A01_T0001F{field:03d}L01A01Z{z:02d}C{channel:02d}.tif'
    path = folder / name
    tifffile.imwrite(path, image, metadata={'axes': axes}, photometric='minisblack')
    return path, image


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_tiffs_preserve_raw_values_z_and_channel_selection(tmp_path):
    first, a = _write(tmp_path, 1)
    second, b = _write(tmp_path, 2)
    before = {p.name: _digest(p) for p in (first, second)}
    settings, src = io.preprocess_img_data(_settings(tmp_path, nucleus_channel=1, channels=[0, 1]))
    stack = np.load(tmp_path/'stack'/'plate1_A01_1_1.npy')
    assert stack.shape == (*a.shape, 2)
    np.testing.assert_array_equal(stack[..., 0], a)
    np.testing.assert_array_equal(stack[..., 1], b)
    with np.load(next((tmp_path/'masks').glob('*.npz'))) as archive:
        assert archive['data'].shape == (1, *a.shape, 1)
        assert np.isfinite(archive['data']).all()
        assert archive['data'].min() >= 0 and archive['data'].max() <= 1
    assert settings['cellpose_nucleus_channel'] == 0
    assert src == str(tmp_path)
    assert {p.name: _digest(p) for p in (first, second)} == before
    receipt = json.loads((tmp_path/'stack'/'.spacr_volume_ingest.json').read_text())
    assert receipt['axes'] == 'ZYXC' and receipt['channels'] == ['1', '2']
    assert len(receipt['inputs']) == 2
    # Reuse must verify both the original TIFF bytes and derived stack bytes.
    io.preprocess_img_data(_settings(tmp_path, nucleus_channel=1, channels=[0, 1]))
    assert {p.name: _digest(p) for p in (first, second)} == before


@pytest.mark.parametrize('case', ['duplicate', 'missing_channel', 'wrong_axes', 'mismatched_shape'])
def test_ambiguous_or_incomplete_volume_layouts_fail_before_writes(tmp_path, case):
    first, _ = _write(tmp_path)
    if case == 'duplicate':
        _write(tmp_path, z=2)
    elif case == 'missing_channel':
        _write(tmp_path, channel=2)
        _write(tmp_path, field=2)
    elif case == 'wrong_axes':
        tifffile.imwrite(first, np.zeros((4, 8, 9), np.uint16),
                         metadata={'axes': 'QYX'}, photometric='minisblack')
    else:
        _write(tmp_path, channel=2, shape=(5, 8, 9))
    before = {p.name: _digest(p) for p in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        io.preprocess_img_data(_settings(tmp_path))
    assert {p.name: _digest(p) for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize('changed', ['raw', 'stack', 'legacy', 'recipe'])
def test_resume_never_reuses_changed_or_unbound_stacks(tmp_path, changed):
    path, _ = _write(tmp_path)
    if changed == 'legacy':
        (tmp_path/'stack').mkdir()
        np.save(tmp_path/'stack'/'plate1_A01_1_1.npy', np.zeros((8, 9, 1)))
    else:
        io.preprocess_img_data(_settings(tmp_path))
        if changed == 'raw':
            _write(tmp_path, shape=(5, 8, 9))
        elif changed == 'stack':
            target = tmp_path/'stack'/'plate1_A01_1_1.npy'
            arr = np.load(target); arr[0, 0, 0, 0] += 1; np.save(target, arr)
    existing = {str(p.relative_to(tmp_path)): _digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        io.preprocess_img_data(_settings(tmp_path, **({'lower_percentile': 8} if changed == 'recipe' else {})))
    assert {str(p.relative_to(tmp_path)): _digest(p) for p in tmp_path.rglob('*') if p.is_file()} == existing


def test_publish_failure_removes_only_owned_staging(tmp_path, monkeypatch):
    path, _ = _write(tmp_path)
    before = _digest(path)
    real_link = io.os.link

    def fail_receipt(source, destination):
        if str(destination).endswith('.spacr_volume_ingest.json'):
            raise OSError('simulated completion failure')
        return real_link(source, destination)

    monkeypatch.setattr(io.os, 'link', fail_receipt)
    with pytest.raises(OSError, match='completion failure'):
        io.preprocess_img_data(_settings(tmp_path))
    assert list(tmp_path.iterdir()) == [path]
    assert _digest(path) == before


def test_late_destination_collision_is_not_overwritten(tmp_path, monkeypatch):
    path, _ = _write(tmp_path)
    real_mkdir = io.os.mkdir

    def late_directory(destination, *args, **kwargs):
        if str(destination) == str(tmp_path/'stack'):
            real_mkdir(destination)
            (Path(destination)/'other.txt').write_text('unrelated')
        return real_mkdir(destination, *args, **kwargs)

    monkeypatch.setattr(io.os, 'mkdir', late_directory)
    with pytest.raises(FileExistsError):
        io.preprocess_img_data(_settings(tmp_path))
    assert (tmp_path/'stack'/'other.txt').read_text() == 'unrelated'
    assert path.exists() and not list(tmp_path.glob('.spacr-volume-*'))


def test_blank_volume_normalizes_to_zero_and_multiple_fields_are_not_padded(tmp_path):
    path, _ = _write(tmp_path)
    tifffile.imwrite(path, np.zeros((4, 8, 9), np.uint16),
                     metadata={'axes': 'ZYX'}, photometric='minisblack')
    _write(tmp_path, field=2, shape=(5, 10, 11))
    io.preprocess_img_data(_settings(tmp_path))
    arrays = []
    for filename in sorted((tmp_path/'masks').glob('*.npz')):
        with np.load(filename) as archive:
            arrays.append(archive['data'])
    assert [a.shape for a in arrays] == [(1,4,8,9,1), (1,5,10,11,1)]
    assert not arrays[0].any()


def test_legacy_2d_organizer_keeps_its_max_projection(tmp_path):
    _, first = _write(tmp_path, shape=(8, 9), axes='YX')
    second_path, second = _write(tmp_path, z=2, shape=(8, 9), axes='YX')
    second += 100
    tifffile.imwrite(second_path, second, photometric='minisblack')
    io.preprocess_img_data(_settings(tmp_path, z_stack=False))
    stack = np.load(tmp_path/'stack'/'plate1_A01_1_1.npy')
    assert stack.shape == (8, 9, 1)
    np.testing.assert_array_equal(stack[..., 0], np.maximum(first, second))
