import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--git')
parser.add_argument('--bind-current', action='store_true')
args = parser.parse_args()
here = Path(__file__).resolve().parent
root = next(parent for parent in here.parents if (parent / '.git').exists())
prefix = here.relative_to(root).as_posix()

def read(path):
    if args.git:
        return subprocess.check_output(['git', 'show', args.git + ':' + path], cwd=root)
    return (root / path).read_bytes()

def payload(name):
    return read(prefix + '/' + name)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

manifest = json.loads(payload('MANIFEST.json'))
for name, expected in manifest['payloads'].items():
    raw = payload(name)
    assert len(raw) == expected['bytes'] and digest(raw) == expected['sha256'], name
receipt = json.loads(payload('receipt.json'))
assert receipt['head_sha'] == '5636eac60938f5d6021eaf73546ff8a30ae17c32'
assert receipt['run_id'] == 37763414112 and receipt['run_attempt'] == 1
assert len(receipt['jobs']) == 6
assert receipt['verdict'] == {'success': 6, 'failure': 0}
observed = {'success': 0, 'failure': 0}
failed = set()
for row in receipt['jobs']:
    job = json.loads(payload(row['phase'] + '-job.json'))
    assert job['id'] == row['id'] and job['run_id'] == receipt['run_id']
    assert job['head_sha'] == receipt['head_sha'] and job['run_attempt'] == 1
    assert job['status'] == 'completed' and job['conclusion'] == row['conclusion']
    observed[job['conclusion']] += 1
    raw = gzip.decompress(payload(row['phase'] + '.log.gz'))
    assert digest(raw) == row['raw_log_sha256'] and len(raw) == row['raw_log_bytes']
    text = raw.decode()
    nodes = sorted(set(re.findall(r'\bFAILED (tests/\S+)', text)))
    assert nodes == row['blocking_failed_nodes']
    assert not re.search(r'Fatal Python error|worker .*crashed|Segmentation fault', text)
    lines = text.splitlines()
    all_rows = []
    for number, line in enumerate(lines, 1):
        match = re.search(r'=+\s+(.*?)\s+=+\s*$', line)
        if match and re.search(r'\b\d+ (passed|failed|skipped|xfailed|xpassed|error|errors)\b', match.group(1)):
            all_rows.append({'line': number, 'text': line})
    assert all_rows == row['pytest_summaries']
    counts = {}
    for summary in row['pytest_summaries']:
        assert lines[summary['line'] - 1] == summary['text']
        for count, outcome in re.findall(r'(\d+) (passed|failed|skipped|xfailed|xpassed|error|errors)\b', summary['text']):
            counts[outcome] = counts.get(outcome, 0) + int(count)
    assert counts == row['summed_phase_outcomes_not_unique_tests']
    failed.update(nodes)
assert observed == receipt['verdict']
assert sorted(failed) == receipt['unique_blocking_nodes']
assert sorted(failed) == []
run = json.loads(payload('run-snapshot.json'))
assert run['id'] == receipt['run_id'] and run['head_sha'] == receipt['head_sha']
binding = receipt['source_bindings']['tests/test_local_ci_replay.py']
assert binding['measured_sha256'] == binding['repair_reference_sha256']
assert digest(gzip.decompress(payload('measured-profile-test.py.gz'))) == binding['measured_sha256']
assert digest(gzip.decompress(payload('repaired-profile-test.py.gz'))) == binding['repair_reference_sha256']
for path in receipt['unchanged_production_files']:
    expected = receipt['source_bindings'][path]
    assert expected['measured_sha256'] == expected['repair_reference_sha256']
    if args.bind_current:
        assert digest(read(path)) == expected['measured_sha256'], path
print('Verified', len(manifest['payloads']), 'payloads, six complete logs, six genuine successes; bind_current=', args.bind_current)
