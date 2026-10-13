import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--git', metavar='REV')
parser.add_argument('--bind-current', action='store_true')
args = parser.parse_args()
folder = Path(__file__).resolve().parent
repo = folder.parents[2]
relative = folder.relative_to(repo).as_posix()

def read(name):
    if args.git:
        return subprocess.check_output(['git', 'show', f'{args.git}:{relative}/{name}'], cwd=repo)
    return (folder / name).read_bytes()

manifest = json.loads(read('MANIFEST.json'))
for item in manifest['files']:
    data = read(item['path'])
    assert len(data) == item['bytes'], item['path']
    assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
receipt = json.loads(read('receipt.json'))
assert receipt['head_sha'] == '55badc57ff25ef6a7e14721561773a412b515318'
assert receipt['run_id'] == 37720484173 and receipt['run_attempt'] == 1
assert len(receipt['jobs']) == 6
assert len({job['job_id'] for job in receipt['jobs']}) == 6
all_failed = set()
for job in receipt['jobs']:
    name = job['phase']
    metadata = json.loads(read(f'{name}-job.json'))
    assert metadata['status'] == 'completed' and metadata['conclusion'] == job['conclusion']
    assert metadata['id'] == job['job_id'] and metadata['head_sha'] == receipt['head_sha']
    assert metadata['run_id'] == receipt['run_id'] and metadata['run_attempt'] == 1
    raw = gzip.decompress(read(f'{name}.log.gz'))
    assert len(raw) == job['raw_log_bytes']
    assert hashlib.sha256(raw).hexdigest() == job['raw_log_sha256']
    failed = sorted(set(re.findall(r'\bFAILED (tests/\S+)', raw.decode('utf-8'))))
    assert failed == job['blocking_failed_nodes'], name
    all_failed.update(failed)
assert sorted(all_failed) == receipt['unique_blocking_nodes']
assert len(all_failed) == 4
if args.bind_current:
    revision = args.git or 'HEAD'
    for path in receipt['unchanged_production_files']:
        data = subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=repo)
        expected = receipt['source_bindings'][path]
        assert expected['before_sha256'] == expected['corrected_sha256']
        assert hashlib.sha256(data).hexdigest() == expected['corrected_sha256'], path
print(f"Verified {len(manifest['files'])} payloads, six exact-source terminal jobs, four unique blocking nodes.")
