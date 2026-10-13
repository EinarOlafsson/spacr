"""Small positive and adversarial checks for the tutorial's real TIFF audit."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch_data import digest, verify_outputs


@pytest.fixture
def job(tmp_path):
    source, destination = tmp_path / 'input', tmp_path / 'output'
    source.mkdir()
    destination.mkdir()
    records, rows = [], []
    for channel in range(1, 5):
        path = source / f'fov01_C{channel}.tif'
        pixels = np.arange(72, dtype=np.uint16).reshape(8, 9) + channel * 100
        tifffile.imwrite(path, pixels)
        target = destination / f'plate1_A01_T0001F001L01A01Z01C{channel:02d}.tif'
        tifffile.imwrite(target, pixels)
        records.append(dict(source=str(path), original=str(path), sha256=digest(path),
                            channel=channel, shape=[8, 9]))
        rows.append(dict(target=target.name, target_path=str(target), source=str(path),
                         source_relpath=path.name, plate='plate1', well='A01', field='1',
                         channel=str(channel), z='1', t='1', source_field='fov01',
                         source_channel=f'C{channel}', source_well='A01', plateID='plate1',
                         rowID='r1', columnID='c1', fieldID='f1', prc='plate1_r1_c1',
                         prcf='plate1_r1_c1_f1', z_handling='keep', n_z_planes='1',
                         n_timepoints='1', status='converted'))
    write_rows(destination / 'conversion_map.csv', rows)
    (destination / 'conversion_map.run_status.json').write_text(json.dumps([{
        'run_id': 'positive-fixture', 'name': 'convert_to_yokogawa_plan',
        'status': 'complete', 'n_attempted': 4, 'n_succeeded': 4, 'n_failed': 0,
        'failures': [], 'success_by_stage': {'convert': 4}}]))
    return dict(destination=str(destination), records=records, plate='plate1')


def write_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_exact_four_channels_pass(job):
    result = verify_outputs(job)
    assert result['accepted'] is True
    assert result['fields'] == 1 and result['channels'] == 4
    assert {x['channel'] for x in result['outputs']} == {1, 2, 3, 4}


@pytest.mark.parametrize('column,value', [('field', '2'), ('channel', '1'),
    ('source_channel', 'C1'), ('well', 'A02'), ('prcf', 'plate1_r1_c1_f2'),
    ('source_relpath', 'different.tif'), ('status', 'skipped')])
def test_map_tampering_is_rejected(job, column, value):
    path = Path(job['destination']) / 'conversion_map.csv'
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    rows[1][column] = value
    write_rows(path, rows)
    with pytest.raises(ValueError, match='map changed'):
        verify_outputs(job)


def test_missing_output_rejected(job):
    path = next(Path(job['destination']).glob('*.tif'))
    path.rename(path.with_suffix('.held'))
    with pytest.raises(ValueError, match='four distinct channel outputs'):
        verify_outputs(job)


def test_changed_pixel_rejected(job):
    path = next(Path(job['destination']).glob('*.tif'))
    pixels = tifffile.imread(path)
    pixels[0, 0] += 1
    tifffile.imwrite(path, pixels)
    with pytest.raises(ValueError, match='every acquired pixel'):
        verify_outputs(job)


def test_changed_source_rejected(job):
    path = Path(job['records'][0]['source'])
    pixels = tifffile.imread(path)
    pixels[0, 0] += 1
    tifffile.imwrite(path, pixels)
    with pytest.raises(ValueError, match='source changed'):
        verify_outputs(job)


def test_duplicate_expected_channel_rejected(job):
    job['records'][1]['channel'] = 1
    with pytest.raises(ValueError, match='exactly once'):
        verify_outputs(job)


@pytest.mark.parametrize('key,value', [('status', 'partial'), ('n_attempted', 3),
    ('n_succeeded', 3), ('n_failed', 1)])
def test_incomplete_ledger_rejected(job, key, value):
    path = Path(job['destination']) / 'conversion_map.run_status.json'
    ledger = json.loads(path.read_text())
    ledger[0][key] = value
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError):
        verify_outputs(job)
