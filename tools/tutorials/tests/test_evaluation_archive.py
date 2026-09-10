"""Small archive checks with actual positive counterparts for each refusal."""
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import package_evaluation_example as pack


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / 'recorded'
    source.mkdir()
    for name in pack.FILES:
        (source / name).write_bytes(b'recorded bytes')
    manifest = dict(files={n: n for n in pack.FILES}, leakage_passed=False)
    (source / 'evaluation_manifest.json').write_text(json.dumps(manifest))
    rows = [dict(sample='/private/original/plate1_r1_c1_f1_o%d.png' % i,
                 basename='plate1_r1_c1_f1_o%d.png' % i, confidence='0.866123',
                 true_label='1') for i in range(234)]
    put_rows(source, rows)
    return source


def put_rows(source, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    (source / 'oof_predictions.csv').write_text(stream.getvalue())


def test_exact_archive_is_deterministic_readable_and_preserves_numeric_cells(bundle, tmp_path):
    before = {p.name: p.read_bytes() for p in bundle.iterdir()}
    target = tmp_path / 'example.zip'
    receipt = pack.build_archive(bundle, target)
    assert receipt == pack.build_archive(bundle, target)
    assert receipt['independent_validation'] is False
    assert receipt['crops_included'] is False
    with zipfile.ZipFile(target) as z:
        content = {k.removeprefix(pack.ROOT + '/'): z.read(k) for k in z.namelist()}
    assert set(content) == set(pack.FILES) | {'README.txt', 'tutorial_provenance.json'}
    original = list(csv.DictReader(io.StringIO(before['oof_predictions.csv'].decode())))
    copied = list(csv.DictReader(io.StringIO(content['oof_predictions.csv'].decode())))
    assert len(copied) == len(original) == 234
    for a, b in zip(original, copied):
        assert b.pop('sample') == 'crops-not-included/' + a['basename']
        a.pop('sample')
        assert a == b
    for name in set(pack.FILES) - {'oof_predictions.csv'}:
        assert content[name] == before[name]
    assert {p.name: p.read_bytes() for p in bundle.iterdir()} == before
    assert b'/private/original/' not in content['oof_predictions.csv']


def test_different_existing_archive_is_not_overwritten(bundle, tmp_path):
    target = tmp_path / 'existing.zip'
    target.write_bytes(b'important existing file')
    with pytest.raises(FileExistsError):
        pack.build_archive(bundle, target)
    assert target.read_bytes() == b'important existing file'


def test_real_external_symlink_is_rejected(bundle, tmp_path):
    assert pack.members(bundle)
    target = tmp_path / 'external.csv'
    target.write_bytes((bundle / 'calibration.csv').read_bytes())
    (bundle / 'calibration.csv').unlink()
    (bundle / 'calibration.csv').symlink_to(target)
    assert target.is_file()
    with pytest.raises(ValueError, match='without links'):
        pack.members(bundle)


@pytest.mark.parametrize('changed', ['inventory', 'leakage'])
def test_manifest_guard(bundle, changed):
    assert pack.members(bundle)
    path = bundle / 'evaluation_manifest.json'
    data = json.loads(path.read_text())
    if changed == 'inventory':
        data['files']['calibration.csv'] = '../calibration.csv'
    else:
        data['leakage_passed'] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='inventory|leakage'):
        pack.members(bundle)


@pytest.mark.parametrize('changed', ['count', 'duplicate', 'path'])
def test_prediction_identity_guards(bundle, changed):
    assert pack.members(bundle)
    rows = list(csv.DictReader((bundle / 'oof_predictions.csv').open()))
    if changed == 'count':
        rows.pop()
    elif changed == 'duplicate':
        rows[0]['basename'] = rows[1]['basename']
    else:
        rows[0]['basename'] = '../plate1_outside.png'
    put_rows(bundle, rows)
    with pytest.raises(ValueError, match='234 distinct|Unsafe'):
        pack.members(bundle)


def test_capture_extracts_verified_files_and_rejects_a_changed_member(bundle, tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    bundle.rename(source / 'evaluation')
    (source / 'preparation.json').write_text(json.dumps(dict(input_prepared=True)))
    archive = tmp_path / 'example.zip'
    pack.build_archive(source / 'evaluation', archive)
    root = pack.prepare_capture(source, archive, tmp_path)
    proof = json.loads((root / 'preparation.json').read_text())
    assert proof['portable_inspection_only'] is True
    assert proof['crops_included'] is False
    assert {p.name: p.read_bytes() for p in (root / 'evaluation').iterdir()} == pack.members(source / 'evaluation')
    with zipfile.ZipFile(archive) as z:
        entries = {n: z.read(n) for n in z.namelist()}
    entries[pack.ROOT + '/calibration.csv'] = b'changed'
    with zipfile.ZipFile(archive, 'w') as z:
        for n, data in entries.items():
            z.writestr(n, data)
    with pytest.raises(ValueError, match='Archive member differs'):
        pack.prepare_capture(source, archive, tmp_path)
