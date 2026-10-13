from pathlib import Path
import gzip
import hashlib
import json
import re
import subprocess
import zipfile

root = Path.cwd()
folder = root / 'features/data/43_0b_ordinary_qt_terminal_2026-10-08'
manifest = json.loads((folder / 'MANIFEST.json').read_text())
digest = lambda value: hashlib.sha256(value).hexdigest()
source = manifest['source_sha']
tree = subprocess.check_output(['git', 'rev-parse', source + '^{tree}'], text=True).strip()
assert tree == manifest['source_tree']
payloads = []
raw_run = (folder / manifest['run_metadata']).read_bytes()
assert digest(raw_run) == manifest['run_metadata_sha256']
run = json.loads(raw_run)
assert run['head_sha'] == source
for row in manifest['jobs']:
    raw_job = (folder / row['job_metadata']).read_bytes()
    assert digest(raw_job) == row['job_metadata_sha256']
    job = json.loads(raw_job)
    assert job['head_sha'] == source and job['id'] == row['job_id']
    assert job['status'] == 'completed' and job['conclusion'] == 'success'
    packed = (folder / row['log']).read_bytes()
    assert len(packed) == row['log_gzip_bytes']
    assert digest(packed) == row['log_gzip_sha256']
    raw = gzip.decompress(packed)
    assert len(raw) == row['log_original_bytes']
    assert digest(raw) == row['log_original_sha256']
    assert source.encode() in raw
    summaries = [line for line in raw.decode().splitlines() if re.search(r'=+.*\bin [0-9.]+s', line)]
    outcomes = [match for line in summaries for match in re.findall(r'(\d+) (passed|skipped|xfailed|xpassed|failed|errors?)\b', line)]
    assert not any(kind in {'failed', 'error', 'errors', 'xpassed'} for _, kind in outcomes)
    for kind, key in [('passed', 'passed'), ('skipped', 'skipped'), ('xfailed', 'expected_xfailed')]:
        assert sum(int(count) for count, label in outcomes if label == kind) == row[key]
    archive = folder / row['memory_archive']
    assert digest(archive.read_bytes()) == row['memory_archive_sha256']
    with zipfile.ZipFile(archive) as handle:
        assert handle.testzip() is None
        assert handle.namelist()
        for name in handle.namelist():
            handle.read(name)
    assert row['failed_or_error_nodes'] == 0
assert sum(row['passed'] for row in manifest['jobs']) == manifest['total_passed'] == 32569
assert sum(row['skipped'] for row in manifest['jobs']) == manifest['total_skipped'] == 110
assert sum(row['expected_xfailed'] for row in manifest['jobs']) == manifest['total_expected_xfailed'] == 11
for path in sorted(folder.iterdir()):
    if path.name != 'MANIFEST.json':
        raw = path.read_bytes()
        payloads.append({'path': str(path.relative_to(root)), 'bytes': len(raw), 'sha256': digest(raw)})
assert len(payloads) == 11
receipt = {
    'verdict': 'PASS', 'source_sha': source, 'source_tree': tree,
    'archive_manifest_sha256': digest((folder / 'MANIFEST.json').read_bytes()),
    'payloads': payloads, 'ordinary_qt_shards': 3,
    'selected_passed': 32569, 'selected_skipped': 110, 'expected_xfailed': 11,
    'scope': 'Completed exact-source ordinary Qt shard archives only; overlapping selected cohorts are not a unique full-suite count.',
    'limitations': ['Required workflow failed on the known profile assertions.', 'Original-order serial and corrected successor acceptance remain separate.', 'Successful shards do not establish native crash causality.'],
}
out = Path('/media/carruthers/mnt3/codex/scratch/current-source-ci-20261007/home-current-0b-qt-independent-readback-r2.json')
assert not out.exists()
out.write_text(json.dumps(receipt, indent=2) + '\n')
print('PASS: 11 payloads, 3 full raw/gzip Qt logs and 3 ZIPs; exact source; 32569 passed, 110 skipped, 11 expected xfails')
