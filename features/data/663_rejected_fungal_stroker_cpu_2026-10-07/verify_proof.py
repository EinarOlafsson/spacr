"""Verify payload bytes, frozen renderer bindings, and recorded rejection."""
import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--git', metavar='REV', help='Read committed blobs instead of worktree files.')
args = parser.parse_args()

def read(name):
    """Read one archive payload from the filesystem or requested Git tree."""
    if args.git:
        return subprocess.check_output(['git', 'show', args.git + ':features/data/' + root.name + '/' + name],
                                       cwd=root)
    return (root / name).read_bytes()

manifest = json.loads(read('manifest.json'))
for name, expected in manifest['files'].items():
    data = read(name)
    assert len(data) == expected['bytes'], name
    assert hashlib.sha256(data).hexdigest() == expected['sha256'], name
receipt = json.loads(read('receipt.json'))
assert receipt['status'] == 'complete/rejected'
assert not receipt['app_source_changed'] and not receipt['hard_24_fps_accepted']
for name, expected in receipt['source_sha256'].items():
    assert hashlib.sha256(gzip.decompress(read(name + '_ambient.py.gz'))).hexdigest() == expected
raw = json.loads(read('unrestricted_receipt.json'))
assert raw['ambient_sha256'] == receipt['source_sha256']['before']
assert [row['different_pixels'] for row in raw['rows']] == [0, 61750]
assert not raw['accepted']
cached = json.loads(read('cached_receipt.json'))
assert cached['source_sha256'] == receipt['source_sha256']
pairs = [row for row in cached['records'] if 'different_pixels' in row]
assert len(pairs) == 12 and all(row['different_pixels'] == 0 for row in pairs)
assert sum(row['strokes'] == 0 for row in pairs) == 3
admission = json.loads(read('admission_receipt.json'))
for row in admission['rows']:
    assert row['source_sha256'] == receipt['source_sha256'][row['source']]
    assert row['counts']['raster_evictions'] == 0
    assert row['counts'].get('headroom_refusal', 0) == 0
    assert max(sample['raster_entries'] for sample in row['samples']) <= 64
    assert max(sample['raster_array_bytes'] for sample in row['samples']) <= 8 * 1024 ** 2
stroke = next(row for row in admission['rows'] if row['source'] == 'candidate')['stroke_operations']
assert stroke['lookup'] == stroke['key_miss'] == stroke['insert'] == 12964
assert stroke.get('key_hit', 0) == 0 and stroke['delete'] == 12900
print(f"Verified {len(manifest['files'])} payloads; frozen source/parity/counter bindings agree; rejected.")
