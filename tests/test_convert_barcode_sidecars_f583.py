"""Convert binds local barcode sidecars to scanned source and output plate IDs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import convert as cv
from spacr.errors import ConfigurationError


def _input(tmp_path, multi=False):
    """Create genuine TIFF source plates and a local records CSV."""
    root = tmp_path / 'input'
    folders = [root / name / 'A01' for name in ['Alpha plate', 'Zulu plate']] if multi else [root / 'A01']
    for folder in folders:
        folder.mkdir(parents=True)
        tifffile.imwrite(folder / 'field01_C1.tif', np.arange(64, dtype=np.uint16).reshape(8, 8))
    records = tmp_path / 'samples.csv'
    records.write_text('barcode,well,strain,compound,concentration\n00123,A01,RH,DMSO,0\n00456,A01,ME49,drug,5\n00123,A02,RH,DMSO,0\n')
    return dict(src=str(root), dst=str(tmp_path / 'output'), plate_barcode_source=str(records), preview_rows=0)


def _bundle(result):
    """Load the immutable completion receipt and generated plate map."""
    directory = Path(result.dst) / 'plate_barcode_linkage'
    return json.loads((directory / 'complete.json').read_text()), pd.read_csv(directory / 'plate_map_lims.csv', dtype=str)


def test_single_plate_sidecar_preserves_zeros_and_records_provenance(tmp_path):
    settings = _input(tmp_path)
    sidecar = Path(settings['src']) / 'barcode.txt'
    sidecar.write_text('00123\n')
    before = {path: path.read_bytes() for path in Path(settings['src']).rglob('*') if path.is_file()}
    result = cv.convert_folder(settings)
    receipt, frame = _bundle(result)
    assert receipt['plate_barcodes'] == {'plate1': '00123'}
    assert receipt['barcode_origins']['plate1'] == [{'kind': 'barcode.txt', 'path': str(sidecar), 'source_plate': 'input'}]
    assert receipt['input_sha256'][str(sidecar)] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert frame['plate_barcode'].tolist() == ['00123']
    assert frame['strain'].tolist() == ['RH']
    assert receipt['mismatches'] == 1
    assert all(path.read_bytes() == content for path, content in before.items())
    assert 'plate_barcodes' not in settings


def test_multiple_plate_folders_map_original_names_to_output_ids(tmp_path):
    settings = _input(tmp_path, multi=True)
    root = Path(settings['src'])
    for name, barcode in [('Alpha plate', '00123'), ('Zulu plate', '00456')]:
        (root / name / 'barcode.txt').write_text(barcode)
    receipt, frame = _bundle(cv.convert_folder(settings))
    assert receipt['plate_barcodes'] == {'plate1': '00123', 'plate2': '00456'}
    assert frame[['plateID', 'strain']].values.tolist() == [['plate1', 'RH'], ['plate2', 'ME49']]
    assert receipt['barcode_origins']['plate2'][0]['source_plate'] == 'Zulu plate'
    assert receipt['barcode_sidecars'][str(root / 'barcode.txt')] is None


def test_root_keyed_sidecar_uses_source_names_and_explicit_remainder(tmp_path):
    settings = _input(tmp_path, multi=True)
    root = Path(settings['src'])
    (root / 'barcode.txt').write_text('Alpha plate=00123\n')
    settings['plate_barcodes'] = {'plate2': '00456'}
    receipt, _ = _bundle(cv.convert_folder(settings))
    assert receipt['plate_barcodes'] == {'plate1': '00123', 'plate2': '00456'}
    assert receipt['barcode_origins']['plate2'] == [{'kind': 'explicit', 'output_plate': 'plate2'}]


def test_matching_explicit_and_sidecar_evidence_are_both_retained(tmp_path):
    settings = _input(tmp_path)
    (Path(settings['src']) / 'barcode.txt').write_text('00123')
    settings['plate_barcodes'] = {'plate1': '00123'}
    receipt, _ = _bundle(cv.convert_folder(settings))
    assert [entry['kind'] for entry in receipt['barcode_origins']['plate1']] == ['explicit', 'barcode.txt']


@pytest.mark.parametrize('text', ['00123', 'plate1=00123', 'Alpha plate=00123;Alpha plate=00123',
                                  'Alpha plate=00123;Alpha plate=00456', 'Alpha plate=', '', 'bad\x00text'])
def test_ambiguous_or_bad_root_sidecar_refused_before_image_writes(tmp_path, text):
    settings = _input(tmp_path, multi=True)
    (Path(settings['src']) / 'barcode.txt').write_text(text)
    with pytest.raises(ConfigurationError, match='sidecar'):
        cv.convert_folder(settings)
    assert not Path(settings['dst']).exists()


@pytest.mark.parametrize('conflict', ['explicit', 'parent'])
def test_conflicting_evidence_refused_before_conversion(tmp_path, conflict):
    settings = _input(tmp_path, multi=True)
    root = Path(settings['src'])
    (root / 'Alpha plate/barcode.txt').write_text('00123')
    if conflict == 'explicit':
        settings['plate_barcodes'] = {'plate1': '999', 'plate2': '00456'}
    else:
        (root / 'barcode.txt').write_text('Alpha plate=999;Zulu plate=00456')
    with pytest.raises(ConfigurationError, match='Conflicting barcode assignments'):
        cv.convert_folder(settings)
    assert not Path(settings['dst']).exists()


@pytest.mark.parametrize('kind', ['changed', 'deleted', 'introduced'])
def test_late_sidecar_change_never_publishes_success(tmp_path, monkeypatch, kind):
    settings = _input(tmp_path)
    path = Path(settings['src']) / 'barcode.txt'
    settings['plate_barcodes'] = {'plate1': '00123'}
    if kind != 'introduced':
        path.write_text('00123')
    original = cv.convert

    def convert_then_change(*args, **kwargs):
        """Modify discovery evidence after real TIFF conversion, before linkage."""
        result = original(*args, **kwargs)
        if kind == 'deleted':
            path.unlink()
        else:
            path.write_text('00999')
        return result

    monkeypatch.setattr(cv, 'convert', convert_then_change)
    with pytest.raises(ConfigurationError, match='changed|local file'):
        cv.convert_folder(settings)
    assert list(Path(settings['dst']).glob('*.tif'))
    assert not (Path(settings['dst']) / 'plate_barcode_linkage').exists()


@pytest.mark.parametrize('kind', ['symlink', 'directory', 'oversized', 'invalid_utf8'])
def test_unsafe_sidecar_refused(tmp_path, kind):
    settings = _input(tmp_path)
    path = Path(settings['src']) / 'barcode.txt'
    if kind == 'symlink':
        other = tmp_path / 'external.txt'
        other.write_text('00123')
        path.symlink_to(other)
    elif kind == 'directory':
        path.mkdir()
    elif kind == 'oversized':
        path.write_bytes(b'0' * (64 * 1024 + 1))
    else:
        path.write_bytes(b'\xff\xfe')
    with pytest.raises(ConfigurationError, match='sidecar'):
        cv.convert_folder(settings)
    assert not Path(settings['dst']).exists()


def test_blank_source_keeps_sidecar_irrelevant_and_preview_writes_nothing(tmp_path):
    settings = _input(tmp_path)
    path = Path(settings['src']) / 'barcode.txt'
    path.write_text('00123')
    cv.convert_folder(settings, preview_only=True)
    assert not Path(settings['dst']).exists()
    path.write_text('invalid\x00sidecar')
    result = cv.convert_folder(settings, plate_barcode_source='')
    assert result.is_complete
    assert not (Path(result.dst) / 'plate_barcode_linkage').exists()


def test_real_cli_import_discovers_source_barcode(tmp_path):
    from spacr import cli

    settings = _input(tmp_path)
    (Path(settings['src']) / 'barcode.txt').write_text('00123')
    config = tmp_path / 'settings.json'
    config.write_text(json.dumps(settings))
    assert cli.main(['convert', '--settings', str(config)]) == cli.EXIT_OK
    assert (Path(settings['dst']) / 'plate_barcode_linkage/complete.json').is_file()
