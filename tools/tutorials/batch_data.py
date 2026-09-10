"""Private real TIFF copies and independent Batch conversion acceptance checks.

The old recorder's W...C names were treated as four single-channel fields.
Here explicit fov01_Cn names preserve four channels of one image field per
job. A01 is a demonstration coordinate, never the original acquired well.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(stage):
    import numpy as np
    import tifffile
    original = Path('/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/spacr/tutorials/orig')
    prefixes = ('W0127F0004T0001Z000', 'W0315F0003T0001Z000', 'W0025F0001T0001Z000')
    parent = Path(stage) / 'batch_runs'
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='real-channels-', dir=parent))
    jobs = []
    for number, prefix in enumerate(prefixes, 1):
        source, destination = root / f'source_{number:02d}', root / f'converted_{number:02d}'
        source.mkdir()
        records = []
        for channel in range(1, 5):
            path = original / f'{prefix}C{channel}.tif'
            if not path.is_file() or path.stat().st_size > 64 * 1024**2:
                raise ValueError('Missing or unexpectedly large original TIFF')
            image = tifffile.imread(path)
            if image.ndim != 2 or image.dtype != np.uint16:
                raise ValueError('Expected an acquired two-dimensional uint16 channel')
            target = source / f'fov01_C{channel}.tif'
            before = digest(path)
            shutil.copyfile(path, target)
            if digest(target) != before or path.stat().st_ino == target.stat().st_ino:
                raise ValueError('The prepared TIFF is not an independent exact copy')
            records.append({'original': str(path), 'source': str(target),
                            'sha256': before, 'channel': channel, 'shape': list(image.shape)})
        if len({tuple(r['shape']) for r in records}) != 1:
            raise ValueError('Acquired channels have different dimensions')
        settings = dict(src=str(source), dst=str(destination), layout='auto',
                        z_handling='keep', plate_naming='index', overwrite=False,
                        map_name='conversion_map.csv', preview_only=False, resume=False)
        path = root / f'convert_job_{number:02d}.json'
        path.write_text(json.dumps(settings, indent=2) + '\n')
        jobs.append({'number': number, 'source': str(source), 'destination': str(destination),
                     'settings_file': str(path), 'settings': settings, 'records': records,
                     'plate': 'source-02' if number == 2 else 'plate1'})
    return {'root': str(root), 'jobs': jobs, 'acquired_data': True,
            'synthetic_pixels': False, 'original_well_metadata_preserved': False,
            'demonstration_well': 'A01', 'fields_per_job': 1, 'channels_per_field': 4}


def verify_outputs(job):
    """Require all four channel identities, all pixels and source preservation."""
    import numpy as np
    import tifffile
    destination = Path(job['destination'])
    records = job['records']
    if len(records) != 4 or {r['channel'] for r in records} != {1, 2, 3, 4}:
        raise ValueError('Every original channel must be checked exactly once')
    mapping = destination / 'conversion_map.csv'
    with mapping.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected_names = {f"{job['plate']}_A01_T0001F001L01A01Z01C{c:02d}.tif" for c in range(1, 5)}
    if (len(rows) != 4 or {r['target'] for r in rows} != expected_names
            or {p.name for p in destination.glob('*.tif')} != expected_names):
        raise ValueError('Expected exactly one field with four distinct channel outputs')
    by_channel = {str(r['channel']): r for r in records}
    checked = []
    for row in rows:
        if row['channel'] not in by_channel:
            raise ValueError('The map changed a channel identity')
        source = by_channel[row['channel']]
        image_path = Path(source['source'])
        name = f"{job['plate']}_A01_T0001F001L01A01Z01C{source['channel']:02d}.tif"
        target = destination / name
        expected = dict(target=name, target_path=str(target), source=str(image_path),
                        source_relpath=image_path.name, plate=job['plate'], well='A01',
                        field='1', channel=str(source['channel']), z='1', t='1',
                        source_field='fov01', source_channel=f"C{source['channel']}",
                        source_well='A01', plateID=job['plate'], rowID='r1', columnID='c1',
                        fieldID='f1', prc=f"{job['plate']}_r1_c1",
                        prcf=f"{job['plate']}_r1_c1_f1", z_handling='keep',
                        n_z_planes='1', n_timepoints='1', status='converted')
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError('The conversion map changed identity or channel metadata')
        for path in (source['original'], source['source']):
            if digest(path) != source['sha256']:
                raise ValueError('An original or prepared source changed')
        raw, output = tifffile.imread(image_path), tifffile.imread(target)
        if (list(raw.shape) != source['shape'] or raw.ndim != 2
                or raw.dtype != np.uint16 or output.dtype != raw.dtype
                or output.shape != raw.shape or not np.array_equal(output, raw)):
            raise ValueError('An output does not preserve every acquired pixel')
        checked.append({'target': name, 'sha256': digest(target), 'channel': source['channel'],
                        'shape': list(output.shape), 'all_pixels_exact': True,
                        'map_fields_checked': len(expected)})
    from capture_converter import check_converter_run_status
    ledger_path = mapping.with_suffix('.run_status.json')
    ledger = json.loads(ledger_path.read_text())
    ledger_check = check_converter_run_status(
        ledger, operation='convert', planned_targets=sorted(expected_names),
        verified_outputs={entry['target']: entry for entry in checked},
        planned_fields=[f"{job['plate']}_A01_f1"], resumed_fields=[],
        n_sources=4, n_written=4, n_existing=0)
    return {'accepted': True, 'fields': 1, 'channels': 4, 'outputs': checked,
            'map_sha256': digest(mapping), 'original_unchanged': True,
            'ledger_check': ledger_check, 'run_status_sha256': digest(ledger_path)}
