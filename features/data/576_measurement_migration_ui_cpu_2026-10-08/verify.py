"""Verify the frozen F576 migration UI source and focused evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path)
parser.add_argument('--git', action='store_true')
args = parser.parse_args()
sha = lambda data: hashlib.sha256(data).hexdigest()
manifest = json.loads((here / 'MANIFEST.json').read_text())['files']
for name, expected in manifest.items():
    data = (here / name).read_bytes()
    assert sha(data) == expected['sha256'] and len(data) == expected['bytes'], name
    if name.endswith('.gz'):
        gzip.decompress(data)
receipt = json.loads((here / 'receipt.json').read_text())
old = json.loads(gzip.decompress((here / 'baseline-inventory.json.gz').read_bytes()))
new = json.loads(gzip.decompress((here / 'final-inventory.json.gz').read_bytes()))
delta = json.loads((here / 'inventory-delta.json').read_text())
assert old['api'] == new['api']
assert len(old['api']) == len(new['api']) == delta['api_count_after'] == 13232
for bucket, before in old['runtime'].items():
    after = new['runtime'][bucket]
    row = delta['runtime'][bucket]
    assert (len(before), len(after)) == (row['count_before'], row['count_after'])
    if isinstance(before, dict):
        assert before == after
    else:
        assert sorted(set(after) - set(before)) == row['added']
        assert sorted(set(before) - set(after)) == row['removed']
assert len(delta['runtime']['ui']['added']) == 22
if args.source:
    for path, item in receipt['changed_files'].items():
        data = (args.source / path).read_bytes()
        assert sha(data) == item['final_sha256'], path
        if path.startswith('spacr/'):
            assert new['source_hashes'][path] == item['final_sha256'], path
if args.git:
    repo = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'],
                                   cwd=here, text=True).strip()
    for name, expected in manifest.items():
        relative = (here / name).relative_to(repo)
        data = subprocess.check_output(['git', 'show', 'HEAD:' + str(relative)],
                                       cwd=repo)
        assert sha(data) == expected['sha256'], name
print('F576 source and evidence verified:', len(manifest), 'payloads, 22 UI arrivals')
