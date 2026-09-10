import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_plaque_example import build_archive


def source_fixture(tmp_path):
    source = tmp_path/'source'; source.mkdir(); (source/'masks').mkdir()
    records = [{'file': f'{n}.tif'} for n in range(4)]
    (source/'manifest.json').write_text(json.dumps({'records': records}))
    for row in records:
        (source/row['file']).write_bytes(b'image-'+row['file'].encode())
        (source/'masks'/row['file']).write_bytes(b'mask-'+row['file'].encode())
    (source/'masks/old.db').write_bytes(b'old result must not be packed')
    return source


def test_archive_exactly_preserves_inputs_excludes_results_and_is_deterministic(tmp_path):
    source = source_fixture(tmp_path); target = tmp_path/'example.zip'
    result = build_archive(source, target)
    assert len(result['members']) == 10
    assert build_archive(source, target) == result
    with zipfile.ZipFile(target) as archive:
        assert archive.read('SYNTHETIC_plaque/0.tif') == b'image-0.tif'
        assert archive.read('SYNTHETIC_plaque/masks/0.tif') == b'mask-0.tif'
        assert 'SYNTHETIC_plaque/masks/old.db' not in archive.namelist()
    assert (source/'masks/old.db').read_bytes() == b'old result must not be packed'
    target.write_bytes(b'other archive')
    with pytest.raises(FileExistsError):
        build_archive(source, target)
    assert target.read_bytes() == b'other archive'


def test_archive_refuses_a_traversal_in_manifest(tmp_path):
    source = source_fixture(tmp_path)
    assert build_archive(source, tmp_path/'valid.zip')['synthetic']
    records = [{'file': n} for n in ('../outside.tif', '1.tif', '2.tif', '3.tif')]
    (tmp_path/'outside.tif').write_bytes(b'real outside file')
    (source/'manifest.json').write_text(json.dumps({'records': records}))
    with pytest.raises(ValueError, match='plain TIFF'):
        build_archive(source, tmp_path/'invalid.zip')
