"""Verify the self-contained hosted failure archive and frozen job set."""

import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'MANIFEST.json').read_text())
snapshot = gzip.decompress((root / 'pre-cancel-snapshot.json.gz').read_bytes())
pre = json.loads(snapshot)
assert pre['headSha'] == manifest['source_sha']
failed = {job['databaseId'] for job in pre['jobs'] if job['conclusion'] == 'failure'}
assert len(failed) == manifest['failed_job_count'] == 21
log_ids = set()
for entry in manifest['files']:
    data = (root / entry['file']).read_bytes()
    assert len(data) == entry['bytes'], entry['file']
    assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['file']
    if 'uncompressed_sha256' in entry:
        plain = gzip.decompress(data)
        assert len(plain) == entry['uncompressed_bytes'], entry['file']
        assert hashlib.sha256(plain).hexdigest() == entry['uncompressed_sha256'], entry['file']
    if 'job_id' in entry:
        log_ids.add(entry['job_id'])
assert failed == log_ids
print(f"Verified {len(log_ids)} failed-job logs and {len(manifest['files'])} archive files")
