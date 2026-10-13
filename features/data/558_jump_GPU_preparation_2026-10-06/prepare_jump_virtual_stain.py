from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
import hashlib
import json
import shutil
import time

import numpy as np
import requests
import tifffile

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '558-jump-current-r1'
root.mkdir(exist_ok=False)
(root / 'originals').mkdir()
(root / 'fields').mkdir()
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
selected = []
metadata = {}
channels = ('URL_OrigDNA', 'URL_OrigBrightfield_L', 'URL_OrigBrightfield_H', 'URL_OrigBrightfield')
for plate, wells, role in [('BR00117035', 24, 'train'), ('BR00117036', 10, 'held_out')]:
    source = scratch / (f'jump-{plate}-load_data.csv')
    shutil.copyfile(source, root / source.name)
    metadata[plate] = {'path': str(root / source.name), 'sha256': digest(source),
                       'url': f'https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_4/workspace/load_data_csv/2021_04_26_Batch1/{plate}/load_data.csv'}
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    chosen = [r for r in rows if r['Metadata_Well'] in {f'A{i:02d}' for i in range(1, wells + 1)} and int(r['Metadata_Site']) in (1, 2)]
    chosen.sort(key=lambda r: (r['Metadata_Well'], int(r['Metadata_Site'])))
    assert len(chosen) == wells * 2
    assert {(r['Metadata_Well'], int(r['Metadata_Site'])) for r in chosen} == {(f'A{i:02d}', j) for i in range(1, wells + 1) for j in (1, 2)}
    for row in chosen:
        assert row['Metadata_Source'] == 'source_4' and row['Metadata_Batch'] == '2021_04_26_Batch1'
        assert row['Metadata_Plate'] == plate and row['Metadata_ImageSizeX'] == row['Metadata_ImageSizeY'] == '1080'
        selected.append({'plate': plate, 'well': row['Metadata_Well'], 'site': int(row['Metadata_Site']),
                         'role': role, 'metadata': row, 'urls': [row[c] for c in channels]})
jobs = []
for row in selected:
    stem = f"{row['plate']}_{row['well']}_s{row['site']:02d}"
    for channel, uri in enumerate(row['urls']):
        assert uri.startswith('s3://cellpainting-gallery/cpg0016-jump/source_4/images/2021_04_26_Batch1/')
        path = root / 'originals' / f'{stem}_c{channel}.tiff'
        jobs.append((uri, path))

def fetch(job):
    uri, path = job
    url = 'https://cellpainting-gallery.s3.amazonaws.com/' + uri.removeprefix('s3://cellpainting-gallery/')
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            path.write_bytes(response.content)
            etag = response.headers.get('ETag', '').strip('"')
            if len(etag) == 32:
                assert hashlib.md5(response.content).hexdigest() == etag
            image = tifffile.imread(path)
            assert image.shape == (1080, 1080) and image.dtype == np.uint16
            return {'path': str(path), 'url': url, 's3_uri': uri, 'etag': etag,
                    'sha256': digest(path), 'bytes': path.stat().st_size,
                    'shape': list(image.shape), 'dtype': str(image.dtype)}
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))

with ThreadPoolExecutor(max_workers=8) as pool:
    downloaded = list(pool.map(fetch, jobs))
by_uri = {r['s3_uri']: r for r in downloaded}
assert len(downloaded) == len(by_uri) == 272
for row in selected:
    arrays = [tifffile.imread(by_uri[uri]['path']) for uri in row['urls']]
    field = np.stack(arrays, axis=-1)
    stem = f"{row['plate']}_{row['well']}_s{row['site']:02d}"
    path = root / 'fields' / (stem + '.npy')
    np.save(path, field)
    readback = np.load(path, allow_pickle=False)
    for index, original in enumerate(arrays):
        np.testing.assert_array_equal(readback[..., index], original)
    row['field_path'] = str(path)
    row['field_sha256'] = digest(path)
    row['shape'] = list(field.shape)
    row['channels'] = list(channels)
train = [r for r in selected if r['role'] == 'train']
test = [r for r in selected if r['role'] == 'held_out']
assert len(train) == 48 and len(test) == 20
assert {r['plate'] for r in train}.isdisjoint({r['plate'] for r in test})
parameters = {'sources': [1, 2, 3], 'target': 0, 'scale': 2, 'crop': 128,
              'per_field': 32, 'epochs': 20, 'base': 16, 'depth': 3, 'seed': 0,
              'device': 'cuda'}
plan = {'prepared': True, 'source': 'Cell Painting Gallery cpg0016-jump source_4, 2021_04_26_Batch1',
        'primary_documentation': 'https://broadinstitute.github.io/cellpainting-gallery/data_structure.html',
        'selection_policy': 'Predeclared first 24 train wells / first 10 held-out wells in row A, sites 1 and 2; no selection on images, predictions or scores',
        'training': train, 'held_out': test, 'original_downloads': downloaded,
        'metadata': metadata, 'exact_original_to_multichannel_pixels_verified': True,
        'channel_order': list(channels), 'training_parameters': parameters,
        'scorecard_policy': 'Same current Cellpose-SAM default segmenter on real/predicted/first brightfield; real stain predictions are reference, not human gold',
        'no_claim_to_reuse_unavailable_historical_trained_model': True,
        'new_reproducible_training_and_GPU_scorecard_pending': True,
        'script_sha256': digest(__file__),
        'source_sha256': {p: digest(p) for p in ['spacr/deep_spacr.py', 'spacr/scorecard.py', 'spacr/tabular.py']}}
(root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
print('PASS: 272 original acquired channel images, exact 68 paired fields, independent metadata and plate-disjoint 48/20 split prepared; actual GPU training and Cellpose scorecard remain pending.', flush=True)
