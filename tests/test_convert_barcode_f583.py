"""Convert links explicit imported wells to CSV metadata without source writes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import convert as cv
from spacr.errors import ConfigurationError


@pytest.fixture
def acquisition(tmp_path):
    """Two real TIFF wells with zero-prefixed barcode and independent user map."""
    raw = tmp_path / 'raw'
    for well in ['A01', 'A02']:
        folder = raw / well
        folder.mkdir(parents=True)
        tifffile.imwrite(folder / 'field01_C1.tif', np.arange(64, dtype=np.uint16).reshape(8, 8))
    records = tmp_path / 'records.csv'
    records.write_text('barcode,well,strain,compound,concentration\n00123,A01,RH,DMSO,0\n00123,A03,ME49,drug,5\n')
    own = tmp_path / 'own.csv'
    own.write_text('plateID,well,compound,concentration\nplate1,A01,DMSO,1\n')
    settings = {'src': str(raw), 'dst': str(tmp_path / 'converted'),
                'plate_barcode_source': str(records),
                'plate_barcodes': {'plate1': '00123'},
                'profiling_metadata': str(own), 'preview_rows': 0}
    return settings


def _snapshot(settings):
    """Capture input bytes to prove conversion only adds destination artifacts."""
    paths = list(Path(settings['src']).rglob('*.tif'))
    paths += [Path(settings['plate_barcode_source']), Path(settings['profiling_metadata'])]
    return {path: path.read_bytes() for path in paths}


def test_convert_settings_entry_links_only_imported_wells_with_mismatches(acquisition):
    original = _snapshot(acquisition)
    settings_before = dict(acquisition)
    result = cv.convert_folder(acquisition)
    bundle = Path(result.dst) / 'plate_barcode_linkage'
    linked = pd.read_csv(bundle / 'plate_map_lims.csv', dtype=str)
    assert linked[['plateID', 'rowID', 'columnID', 'plate_barcode', 'strain']].values.tolist() == [
        ['plate1', 'r1', 'c1', '00123', 'RH']]
    mismatches = pd.read_csv(bundle / 'plate_barcode_mismatches.csv')
    assert set(mismatches['kind']) == {'well_not_in_lims', 'well_not_imaged', 'plate_map_differs'}
    receipt = json.loads((bundle / 'complete.json').read_text())
    assert receipt['complete'] and receipt['linked_wells'] == 1
    assert receipt['converted_or_existing_files'] == 2
    for path, digest in receipt['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    for name, digest in receipt['files'].items():
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest
    assert all(path.read_bytes() == content for path, content in original.items())
    assert acquisition == settings_before
    assert not (bundle / '.complete.json.pending').exists()
    np.testing.assert_array_equal(tifffile.imread(result.written[0].source),
                                  tifffile.imread(Path(result.dst) / result.written[0].target))


def test_normal_cli_settings_path_performs_the_import_linkage(acquisition, tmp_path):
    from spacr import cli
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps(acquisition))
    assert cli.main(['convert', '--settings', str(settings)]) == cli.EXIT_OK
    assert (Path(acquisition['dst']) / 'plate_barcode_linkage/complete.json').is_file()


def test_blank_opt_in_and_preview_leave_linkage_unwritten(acquisition):
    preview = cv.convert_folder(acquisition, preview_only=True)
    assert not Path(preview.dst).exists()
    result = cv.convert_folder(acquisition, plate_barcode_source='')
    assert result.is_complete
    assert not (Path(result.dst) / 'plate_barcode_linkage').exists()


@pytest.mark.parametrize('override,match', [
    ({'plate_barcode_source': 'https://invalid.example/records.csv'}, 'local CSV'),
    ({'plate_barcodes': None}, 'every output plate'),
    ({'plate_barcodes': {'plate1': '00123', 'extra': 'B'}}, 'no others'),
    ({'map_name': '../records.csv'}, 'filename in the destination'),
])
def test_bad_configuration_fails_before_image_output(acquisition, override, match):
    original = _snapshot(acquisition)
    with pytest.raises(ConfigurationError, match=match):
        cv.convert_folder(acquisition, **override)
    assert not Path(acquisition['dst']).exists()
    assert all(path.read_bytes() == content for path, content in original.items())


def test_bad_csv_schema_fails_before_image_output(acquisition):
    Path(acquisition['plate_barcode_source']).write_text('barcode,not_a_well\n00123,x\n')
    with pytest.raises(ValueError):
        cv.convert_folder(acquisition)
    assert not Path(acquisition['dst']).exists()


@pytest.mark.parametrize('kind', ['directory', 'file', 'symlink'])
def test_existing_bundle_is_refused_without_touching_any_output(acquisition, kind, tmp_path):
    bundle = Path(acquisition['dst']) / 'plate_barcode_linkage'
    bundle.parent.mkdir()
    if kind == 'directory':
        bundle.mkdir()
    elif kind == 'file':
        bundle.write_text('existing bundle')
    else:
        bundle.symlink_to(tmp_path / 'absent')
    before = list(bundle.parent.iterdir())
    with pytest.raises(ConfigurationError, match='already exists'):
        cv.convert_folder(acquisition)
    assert list(bundle.parent.iterdir()) == before


def test_source_or_user_map_cannot_be_a_conversion_output(acquisition):
    original = _snapshot(acquisition)
    with pytest.raises(ConfigurationError, match='overwrite a plate barcode input'):
        cv.convert_folder(acquisition, db_path=acquisition['profiling_metadata'])
    with pytest.raises(ConfigurationError, match='separate from the source tree'):
        cv.convert_folder(acquisition, dst=str(Path(acquisition['src']) / 'converted'))
    assert all(path.read_bytes() == data for path, data in original.items())


def test_incomplete_conversion_never_publishes_a_success_receipt(acquisition, monkeypatch):
    actual = cv.convert

    def incomplete(*args, **kwargs):
        """Keep real outputs but simulate an additional unreadable acquisition."""
        result = actual(*args, **kwargs)
        result.skipped.append(('extra.tif', 'unreadable'))
        return result

    monkeypatch.setattr(cv, 'convert', incomplete)
    with pytest.raises(ConfigurationError, match='incomplete'):
        cv.convert_folder(acquisition)
    assert not (Path(acquisition['dst']) / 'plate_barcode_linkage').exists()


def test_changed_records_during_conversion_refuse_stale_metadata(acquisition, monkeypatch):
    actual = cv.convert

    def changed(*args, **kwargs):
        """Model an independent metadata edit after preflight."""
        result = actual(*args, **kwargs)
        Path(acquisition['plate_barcode_source']).write_text('barcode,well,strain\n00123,A01,changed\n')
        return result

    monkeypatch.setattr(cv, 'convert', changed)
    with pytest.raises(ConfigurationError, match='changed during conversion'):
        cv.convert_folder(acquisition)
    assert not (Path(acquisition['dst']) / 'plate_barcode_linkage').exists()


def test_late_bundle_collision_is_preserved(acquisition, monkeypatch):
    actual = cv.convert
    marker = Path(acquisition['dst']) / 'plate_barcode_linkage/other.txt'

    def collided(*args, **kwargs):
        """Create a competing bundle at the real publication boundary."""
        result = actual(*args, **kwargs)
        marker.parent.mkdir()
        marker.write_text('other run')
        return result

    monkeypatch.setattr(cv, 'convert', collided)
    with pytest.raises(FileExistsError):
        cv.convert_folder(acquisition)
    assert marker.read_text() == 'other run'
    assert list(marker.parent.iterdir()) == [marker]


def test_failed_receipt_publication_cleans_only_our_bundle(acquisition, monkeypatch):
    def failed(*args, **kwargs):
        """Inject failure at the final atomic hard-link receipt publication."""
        raise OSError('publication failed')

    monkeypatch.setattr(cv.os, 'link', failed)
    with pytest.raises(OSError, match='publication failed'):
        cv.convert_folder(acquisition)
    assert not (Path(acquisition['dst']) / 'plate_barcode_linkage').exists()
    assert len(list(Path(acquisition['dst']).glob('*.tif'))) == 2


def test_checkpoint_cannot_replace_an_original_image(acquisition):
    original = _snapshot(acquisition)
    image = next(Path(acquisition['src']).rglob('*.tif'))
    with pytest.raises(ConfigurationError, match='source image tree'):
        cv.convert_folder(acquisition, checkpoint_path=str(image))
    assert all(path.read_bytes() == data for path, data in original.items())
    assert not Path(acquisition['dst']).exists()


def test_cancel_during_bundle_write_removes_unfinished_metadata(acquisition, monkeypatch):
    from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token
    actual = pd.DataFrame.to_csv
    token = CancellationToken()

    def cancel_after_first_csv(frame, path, *args, **kwargs):
        """Cancel after the first owned bundle CSV has actually been written."""
        result = actual(frame, path, *args, **kwargs)
        if str(getattr(path, 'name', '')).endswith('/plate_barcode_linkage/plate_map_lims.csv'):
            token.cancel()
        return result

    monkeypatch.setattr(pd.DataFrame, 'to_csv', cancel_after_first_csv)
    with installed_token(token), pytest.raises(PipelineCancelled):
        cv.convert_folder(acquisition)
    assert not (Path(acquisition['dst']) / 'plate_barcode_linkage').exists()
    assert len(list(Path(acquisition['dst']).glob('*.tif'))) == 2
