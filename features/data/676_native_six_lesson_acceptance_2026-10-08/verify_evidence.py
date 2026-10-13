"""Verify the portable archive and, optionally, every private candidate byte."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--candidate', type=Path)
args = parser.parse_args()
directory = Path(__file__).resolve().parent
receipt = json.loads((directory / 'receipt.json').read_text())
record = receipt['archive']
archive_path = directory / record['name']
assert archive_path.stat().st_size == record['bytes']
assert digest(archive_path) == record['sha256']
with tarfile.open(archive_path, 'r:gz') as archive:
    members = archive.getmembers()
    assert len(members) == record['members']
    actual = {}
    for member in members:
        assert member.isfile() and not member.name.startswith('/')
        assert '..' not in Path(member.name).parts and member.name not in actual
        actual[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    assert actual == record['member_sha256']
    manifest_bytes = archive.extractfile('candidate/release-manifest.json').read()
    browser = json.load(archive.extractfile('candidate/checks/candidate-browser-checks.json'))
    assert hashlib.sha256(manifest_bytes).hexdigest() == receipt['candidate_manifest_sha256']
    assert browser['passed'] and browser['manifest_sha256'] == receipt['candidate_manifest_sha256']
    assert len(browser['ready_playback_cases']) == 84
if args.candidate:
    candidate = args.candidate.resolve()
    assert digest(candidate / 'release-manifest.json') == receipt['candidate_manifest_sha256']
    manifest = json.loads(manifest_bytes)
    assert len(manifest['files']) == receipt['candidate_files']
    for item in manifest['files']:
        path = candidate / item['path']
        assert path.is_file() and path.stat().st_size == item['bytes'], item['path']
        assert digest(path) == item['sha256'], item['path']
    print('Candidate full-byte readback PASS:', len(manifest['files']), 'files')
print('Portable archive full-byte readback PASS:', len(actual), 'members;', record['sha256'])
