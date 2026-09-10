from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_timelapse_example import build_archive


def fixture(tmp_path):
    source = tmp_path / 'source'
    for kind in ('image_sequences', 'label_sequences'):
        folder = source / kind / 'field_A01'
        folder.mkdir(parents=True)
        for i in range(8):
            np.save(folder / f'frame_{i:02}.npy', np.full((256, 256), i, dtype=np.uint16))
    (source / 'old.db').write_bytes(b'actual old result: excluded')
    return source


def test_deterministic_archive_preserves_all_sixteen_planes_and_excludes_existing_old_results(tmp_path):
    source = fixture(tmp_path)
    target = tmp_path / 'example.zip'
    result = build_archive(source, target)
    assert len(result['members']) == 18
    assert build_archive(source, target) == result
    with zipfile.ZipFile(target) as archive:
        assert all(not name.endswith('.db') for name in archive.namelist())
        for name in result['members']:
            if name.endswith('.npy'):
                assert archive.read('SYNTHETIC_timelapse_preview/' + name) == (source / name).read_bytes()
    original = target.read_bytes()
    np.save(source / 'image_sequences/field_A01/frame_00.npy', np.ones((256, 256), dtype=np.uint16))
    with pytest.raises(FileExistsError, match='different'):
        build_archive(source, target)
    assert target.read_bytes() == original


def test_wrong_shape_is_not_packaged_as_a_recorded_plane(tmp_path):
    source = fixture(tmp_path)
    build_archive(source, tmp_path / 'good.zip')
    np.save(source / 'image_sequences/field_A01/frame_00.npy', np.zeros((8, 8), dtype=np.uint16))
    with pytest.raises(ValueError, match='256'):
        build_archive(source, tmp_path / 'bad.zip')


def test_actual_external_symlink_is_refused(tmp_path):
    source = fixture(tmp_path)
    build_archive(source, tmp_path / 'good.zip')
    outside = tmp_path / 'actual_outside.npy'
    np.save(outside, np.zeros((256, 256), dtype=np.uint16))
    target = source / 'image_sequences/field_A01/frame_00.npy'
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        build_archive(source, tmp_path / 'bad.zip')
