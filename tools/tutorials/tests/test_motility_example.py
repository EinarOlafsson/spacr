import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_motility_example import build_archive


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    (root / 'merged').mkdir(parents=True)
    for i in range(1, 9):
        np.save(root / f'merged/plate1_A01_1_{i}.npy', np.full((256, 256, 4), i, np.uint16))
    (root / 'merged/.spacr_plane_layout.json').write_text(json.dumps({
        'intensity_channels': [0, 1], 'mask_dims': {'cell': 2, 'nucleus': 3}}))
    (root / 'old.db').write_bytes(b'existing old database, must not be packaged')
    (root / 'merged/old.npy').write_bytes(b'unselected existing input')
    return root


def test_exact_members_are_preserved_and_actual_old_files_are_not_included(source, tmp_path):
    target = tmp_path / 'example.zip'
    result = build_archive(source, target)
    assert len(result['members']) == 11
    assert build_archive(source, target) == result
    with zipfile.ZipFile(target) as archive:
        names = {n.removeprefix('SYNTHETIC_motility_tracks/') for n in archive.namelist()}
        assert names == set(result['members'])
        assert 'old.db' not in names and 'merged/old.npy' not in names
        for name in names:
            if name.startswith('merged/'):
                assert archive.read('SYNTHETIC_motility_tracks/' + name) == (source / name).read_bytes()
        manifest = json.loads(archive.read('SYNTHETIC_motility_tracks/manifest.json'))
        assert manifest['acquired_calibration'] is False
        assert manifest['pathogen_mask'] is None
    original = target.read_bytes()
    np.save(source / 'merged/plate1_A01_1_1.npy', np.zeros((256, 256, 4), np.uint16))
    with pytest.raises(FileExistsError, match='different'):
        build_archive(source, target)
    assert target.read_bytes() == original


@pytest.mark.parametrize('shape,dtype', [((8, 8, 4), np.uint16), ((256, 256, 4), np.int16)])
def test_wrong_shape_or_dtype_is_rejected_after_a_positive_counterpart(source, tmp_path, shape, dtype):
    build_archive(source, tmp_path / 'good.zip')
    np.save(source / 'merged/plate1_A01_1_1.npy', np.zeros(shape, dtype))
    with pytest.raises(ValueError, match='uint16'):
        build_archive(source, tmp_path / 'bad.zip')


def test_external_file_that_really_exists_cannot_replace_input(source, tmp_path):
    build_archive(source, tmp_path / 'good.zip')
    path = source / 'merged/plate1_A01_1_1.npy'
    outside = tmp_path / 'outside.npy'
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match='links'):
        build_archive(source, tmp_path / 'bad.zip')


@pytest.mark.parametrize('key,value', [('intensity_channels', [0, 1, 2]), ('mask_dims', {'cell': 3, 'nucleus': 2})])
def test_wrong_plane_sidecar_is_rejected_after_the_real_good_one(source, tmp_path, key, value):
    build_archive(source, tmp_path / 'good.zip')
    path = source / 'merged/.spacr_plane_layout.json'
    mapping = json.loads(path.read_text())
    mapping[key] = value
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match='mapping'):
        build_archive(source, tmp_path / 'bad.zip')


def test_oversized_input_rejected_before_loading(source, tmp_path):
    build_archive(source, tmp_path / 'good.zip')
    path = source / 'merged/plate1_A01_1_1.npy'
    with path.open('ab') as stream:
        stream.write(b'x' * (1024 * 1024))
    with pytest.raises(ValueError, match='large'):
        build_archive(source, tmp_path / 'bad.zip')
