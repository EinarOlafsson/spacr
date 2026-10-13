import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

root = Path('/mnt/wd4tb/spacr-worktrees/codex-fresh-home-contracts-20261008')
scratch = Path('/mnt/wd4tb/scratch/ci-69ba-fast-min-20261008')
dest = root / 'features/data/43_69ba_fast_min_cancelled_partial_2026-10-08'
dest.mkdir(parents=True, exist_ok=True)
source = '69ba4e462f2ea60924a977b192d006ce9b70f42c'
run_id = 37737743925
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
jobs = []
for phase in ('fast0', 'fast1', 'fast2', 'min0', 'min1', 'min2'):
    metadata = json.loads((scratch / (phase + '-job.json')).read_text())
    assert metadata['head_sha'] == source and metadata['run_id'] == run_id and metadata['run_attempt'] == 1
    assert metadata['status'] == 'completed' and metadata['conclusion'] == 'cancelled'
    raw = (scratch / (phase + '.log')).read_bytes()
    text = raw.decode('utf-8', errors='replace')
    findings = [line for line in text.splitlines() if re.search(r'FAILED tests/|ERROR tests/|batches failed|Fatal Python error|Segmentation fault|worker .*crashed', line)]
    jobs.append({'phase': phase, 'id': metadata['id'], 'name': metadata['name'], 'status': metadata['status'], 'conclusion': metadata['conclusion'], 'started_at': metadata['started_at'], 'completed_at': metadata['completed_at'], 'raw_log_bytes': len(raw), 'raw_log_sha256': digest(raw), 'observed_failure_text': findings})
    (dest / (phase + '.log.gz')).write_bytes(gzip.compress(raw, mtime=0))
    (dest / (phase + '-job.json')).write_bytes((scratch / (phase + '-job.json')).read_bytes())
for name in ('run-initial.json', 'run-terminal.json', 'jobs-terminal.json', 'watch.jsonl'):
    (dest / name).write_bytes((scratch / name).read_bytes())
bindings = {}
for relative in ('spacr/plaque.py', 'spacr/submodules.py', 'spacr/qt/mask_engine.py', 'spacr/qt/screens/plate_view.py', 'spacr/qt/widgets/ambient.py', 'spacr/qt/app.py', 'spacr/qt/screens/app_screen.py'):
    old = subprocess.check_output(['git', 'show', source + ':' + relative], cwd=root)
    current = (root / relative).read_bytes()
    assert old == current, relative
    bindings[relative] = digest(old)
receipt = {'head_sha': source, 'run_id': run_id, 'run_attempt': 1, 'event': 'workflow_dispatch', 'scope': 'Six Fast/Minimum jobs only, cancelled before full phase completion', 'verdict': {'cancelled': 6, 'success': 0, 'failure': 0}, 'supersession': {'actor': 'Parent/root explicitly authorized and requested cancellation', 'reason': 'Source predates positive persisted-session fixture correctness repairs; latest required source is 0b8c2c4120a0ede4de568d484ea3fc9d517f1de3', 'replacement_run_id': 37741615330}, 'jobs': jobs, 'unchanged_production_sha256': bindings, 'limits': 'All six job cancellations are PARTIAL, never test-phase acceptance or true test failure conclusions. No matching failed-node/native-fault text was observed in these complete cancellation logs; that absence is not a pass, whole Qt acceptance, or historical crash-causation closure. Existing old4133 Qt1 and latest protected required/serial remain outside this archive.'}
(dest / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(dest / 'README.txt').write_text('Cancelled/partial 69ba Fast/Minimum phase, 2026-10-08\n\nExact manual required run37737743925 source69ba4e462f2ea60924a977b192d006ce9b70f42c attempt1. Parent/root explicitly superseded it after verified fresh/Home/first-build test fixtures were published; latest manual required37741615330 binds0b8c2c4120a0ede4de568d484ea3fc9d517f1de3. All six old Fast/Minimum jobs end CANCELLED, so this archive proves only partial execution and provenance, never acceptance. It preserves all six complete available cancellation logs, exact job metadata, initial/final run snapshots and timestamped watch observations. The seven declared production files remain byte-identical to current context; test fixture changes are intentional and are not bound as unchanged.\n\nNo named failure/native-fault markers were observed in the six complete cancellation logs. That absence cannot prove a pass, full Qt/native/memory acceptance or resolve any historical crash. Other old/current Qt/serial runs are outside this scope. No gates, limits, selection, recovery rules or application bytes were changed by this archive.\n\nRun verify_manifest.py --git HEAD --bind-current after integration. No hosted dispatch/cancel/push was performed by this lane.\n')
verify = '''import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
p = argparse.ArgumentParser()
p.add_argument('--git')
p.add_argument('--bind-current', action='store_true')
a = p.parse_args()
here = Path(__file__).resolve().parent
root = next(path for path in here.parents if (path / '.git').exists())
prefix = here.relative_to(root).as_posix()
def read(path):
    return subprocess.check_output(['git', 'show', a.git + ':' + path], cwd=root) if a.git else (root / path).read_bytes()
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
m = json.loads(read(prefix + '/MANIFEST.json'))
for name, record in m['payloads'].items():
    raw = read(prefix + '/' + name)
    assert len(raw) == record['bytes'] and digest(raw) == record['sha256'], name
r = json.loads(read(prefix + '/receipt.json'))
assert len(r['jobs']) == 6 and r['verdict'] == {'cancelled': 6, 'success': 0, 'failure': 0}
for row in r['jobs']:
    metadata = json.loads(read(prefix + '/' + row['phase'] + '-job.json'))
    assert metadata['status'] == 'completed' and metadata['conclusion'] == 'cancelled'
    assert metadata['id'] == row['id'] and metadata['run_id'] == r['run_id']
    assert metadata['head_sha'] == r['head_sha'] and metadata['run_attempt'] == 1
    raw = gzip.decompress(read(prefix + '/' + row['phase'] + '.log.gz'))
    assert len(raw) == row['raw_log_bytes'] and digest(raw) == row['raw_log_sha256']
run = json.loads(read(prefix + '/run-terminal.json'))
assert run['id'] == r['run_id'] and run['head_sha'] == r['head_sha'] and run['conclusion'] == 'cancelled'
if a.bind_current:
    for path, expected in r['unchanged_production_sha256'].items():
        assert digest(read(path)) == expected, path
print('Verified', len(m['payloads']), 'payloads and six cancelled/partial full raw logs; bind_current=', a.bind_current)
'''
(dest / 'verify_manifest.py').write_text(verify)
(dest / 'archive_phase.py').write_bytes(Path(__file__).read_bytes())
payloads = {path.name: {'bytes': path.stat().st_size, 'sha256': digest(path.read_bytes())} for path in sorted(dest.iterdir()) if path.name != 'MANIFEST.json'}
(dest / 'MANIFEST.json').write_text(json.dumps({'schema': 1, 'payloads': payloads}, indent=2) + '\n')
print(len(payloads), sum(record['bytes'] for record in payloads.values()))
