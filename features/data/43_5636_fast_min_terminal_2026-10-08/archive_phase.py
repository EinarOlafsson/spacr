import argparse
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--scratch', type=Path, required=True)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
head = '5636eac60938f5d6021eaf73546ff8a30ae17c32'
corrected = '110920c706f941712ebb7a3488a969ae176eb228'
sha = lambda data: hashlib.sha256(data).hexdigest()
jobs = []
for name in ['fast0', 'fast1', 'fast2', 'min0', 'min1', 'min2']:
    metadata = (args.scratch / (name + '-job.json')).read_bytes()
    job = json.loads(metadata)
    assert job['head_sha'] == head and job['run_id'] == 37763414112
    assert job['run_attempt'] == 1 and job['status'] == 'completed'
    assert job['conclusion'] == 'success'
    raw = (args.scratch / (name + '.log')).read_bytes()
    text = raw.decode()
    lines = text.splitlines()
    (args.output / (name + '.log.gz')).write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    (args.output / (name + '-job.json')).write_bytes(metadata)
    summaries = [{'line': n, 'text': line} for n, line in enumerate(lines, 1)
                 if re.search(r'=+ .*\b(?:passed|failed|skipped|xfailed|xpassed|error|errors)\b.*=+', line)]
    totals = {}
    for summary in summaries:
        for count, outcome in re.findall(r'(\d+) (passed|failed|skipped|xfailed|xpassed|error|errors)\b', summary['text']):
            totals[outcome] = totals.get(outcome, 0) + int(count)
    jobs.append({'phase': name, 'id': job['id'], 'url': job['html_url'],
        'head_sha': head, 'run_id': job['run_id'], 'run_attempt': job['run_attempt'],
        'conclusion': job['conclusion'], 'started_at': job['started_at'], 'completed_at': job['completed_at'],
        'raw_log_bytes': len(raw), 'raw_log_sha256': sha(raw),
        'blocking_failed_nodes': sorted(set(re.findall(r'\bFAILED (tests/\S+)', text))),
        'pytest_summary_count': len(summaries), 'pytest_summaries': summaries,
        'summed_phase_outcomes_not_unique_tests': totals,
        'batch_result_lines': [line for line in lines if re.search(r'\d+ of \d+ batches failed|all \d+ batches', line)],
        'watchdog_dump_lines': [{'line': n, 'text': line} for n, line in enumerate(lines, 1) if 'Timeout (0:05:00)!' in line],
        'native_fault_markers': [line for line in lines if re.search(r'Fatal Python error|worker .*crashed|Segmentation fault', line)]})
failed = sorted({node for job in jobs for node in job['blocking_failed_nodes']})
assert failed == []
assert not any(job['native_fault_markers'] for job in jobs)
run_raw = (args.scratch / 'run-latest.json').read_bytes()
run = json.loads(run_raw)
assert run['head_sha'] == head and run['id'] == 37763414112
(args.output / 'run-snapshot.json').write_bytes(run_raw)
production = ['spacr/plaque.py', 'spacr/submodules.py', 'spacr/qt/mask_engine.py', 'spacr/qt/screens/plate_view.py', 'spacr/qt/widgets/ambient.py', 'spacr/qt/app.py', 'spacr/qt/screens/app_screen.py']
bindings = {}
for path in production + ['tests/test_local_ci_replay.py']:
    old = subprocess.check_output(['git', 'show', head + ':' + path], cwd=args.repo)
    new = subprocess.check_output(['git', 'show', corrected + ':' + path], cwd=args.repo)
    bindings[path] = {'measured_sha256': sha(old), 'repair_reference_sha256': sha(new)}
    if path == 'tests/test_local_ci_replay.py':
        (args.output / 'measured-profile-test.py.gz').write_bytes(gzip.compress(old, mtime=0))
        (args.output / 'repaired-profile-test.py.gz').write_bytes(gzip.compress(new, mtime=0))
    if path in production:
        assert old == new == (args.repo / path).read_bytes()
receipt = {'scope': 'Six complete terminal Fast/Minimum jobs on exact 5636; whole required run not accepted.',
    'head_sha': head, 'run_id': 37763414112, 'run_attempt': 1,
    'verdict': {'success': 6, 'failure': 0}, 'unique_blocking_nodes': failed,
    'jobs': jobs, 'source_bindings': bindings, 'unchanged_production_files': production,
    'corrected_reference_head': corrected, 'repair_reference_scope': 'Later 110920 metadata/reference snapshot has unchanged seven bound production files and profile test. These logs measure 5636 only; whole required/Qt/serial acceptance remains separate.',
    'accepted_profile_contract': 'Corrected profile pins exactly one dedicated first-open timing exclusion in Qt and coverage while retaining every prior exclusion count, unchanged budgets and original routing guards. Both formerly failing profile phases genuinely pass here.',
    'run_snapshot_policy': 'The saved run JSON is a contemporaneous snapshot; later parent cancellation never changes these six terminal job verdicts.',
    'count_policy': 'Sum of raw pytest phase summary counts within each job, not unique tests and not an additive multi-shard total.',
    'native_scope': 'No native-fault text markers in these six logs; this does not close historical Qt causation or prove every Qt job passed.'}
(args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(args.output / 'archive_phase.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'jobs': len(jobs), 'verdict': receipt['verdict'], 'failed': failed, 'phase_outcomes': {j['phase']: j['summed_phase_outcomes_not_unique_tests'] for j in jobs}}, indent=2))
