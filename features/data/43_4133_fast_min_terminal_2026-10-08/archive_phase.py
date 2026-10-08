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
head = '4133beafcd0ae427795617a4a01e295fd40539b7'
corrected = '86d98624993ebcc356c70277028a0d3256d3ddcd'
sha = lambda data: hashlib.sha256(data).hexdigest()
jobs = []
for name in ['fast0', 'fast1', 'fast2', 'min0', 'min1', 'min2']:
    metadata = (args.scratch / (name + '-job.json')).read_bytes()
    job = json.loads(metadata)
    assert job['head_sha'] == head and job['run_id'] == 37728397690
    assert job['run_attempt'] == 1 and job['status'] == 'completed'
    assert job['conclusion'] == ('failure' if name in ['fast1', 'min1'] else 'success')
    raw = (args.scratch / (name + '.log')).read_bytes()
    text = raw.decode()
    lines = text.splitlines()
    (args.output / (name + '.log.gz')).write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    (args.output / (name + '-job.json')).write_bytes(metadata)
    summaries = [{'line': n, 'text': line} for n, line in enumerate(lines, 1)
                 if re.search(r'=+ .*\b(?:passed|failed|skipped|xfailed)\b.*=+', line)]
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
assert failed == ['tests/test_cellpose_api_contract.py::test_every_converted_double_declares_the_installed_signature']
assert not any(job['native_fault_markers'] for job in jobs)
run_raw = (args.scratch / 'run.json').read_bytes()
run = json.loads(run_raw)
assert run['head_sha'] == head and run['id'] == 37728397690
(args.output / 'run-snapshot.json').write_bytes(run_raw)
production = ['spacr/plaque.py', 'spacr/submodules.py', 'spacr/qt/mask_engine.py', 'spacr/qt/screens/plate_view.py', 'spacr/qt/widgets/ambient.py']
bindings = {}
for path in production + ['tests/test_plaque_segmentation_diagnostics.py', 'tests/test_cellpose_api_contract.py', 'tests/test_test_suite_hygiene.py']:
    old = subprocess.check_output(['git', 'show', head + ':' + path], cwd=args.repo)
    new = subprocess.check_output(['git', 'show', corrected + ':' + path], cwd=args.repo)
    bindings[path] = {'4133_sha256': sha(old), '86d_sha256': sha(new)}
    if path in production:
        assert old == new == (args.repo / path).read_bytes()
partial = (args.scratch / 'fast0-partial-request.log').read_bytes()
full = (args.scratch / 'fast0.log').read_bytes()
partial_info = {'bytes': len(partial), 'sha256': sha(partial), 'full_terminal_starts_with_partial_bytes': full.startswith(partial), 'last_observed_completed_batch': 94, 'last_observed_started_batch': 95, 'selected_batches': 122, 'prefix_last_timestamp': '2026-10-08T06:16:05.5612476Z', 'scope': 'One authorized live prefix request; no pass, imminence or hang inference from its stale endpoint.'}
if not full.startswith(partial):
    (args.output / 'fast0-partial.log.gz').write_bytes(gzip.compress(partial, mtime=0))
receipt = {'scope': 'Six complete terminal Fast/Minimum jobs on exact 4133; whole required run not accepted.',
    'head_sha': head, 'run_id': 37728397690, 'run_attempt': 1,
    'verdict': {'success': 4, 'failure': 2}, 'unique_blocking_nodes': failed,
    'jobs': jobs, 'source_bindings': bindings, 'unchanged_production_files': production,
    'corrected_reference_head': corrected, 'corrected_reference_required_run': 37735178663,
    'repair': 'Complete literal Cellpose eval signature replaces partial **kwargs double; axis sentinel/check and original fail-before-model assertion preserved. Existing installed-signature sweep and both hygiene guards unchanged.',
    'accepted_focused_proof': 'features/data/43_plaque_fail_fast_signature_cpu_2026-10-08/receipt.json',
    'run_snapshot_policy': 'The saved run JSON is a contemporaneous snapshot; later parent cancellation never changes these six terminal job verdicts.',
    'count_policy': 'Sum of raw pytest phase summary counts within each job, not unique tests and not an additive multi-shard total.',
    'native_scope': 'No native-fault text markers in these six logs; this does not close historical Qt causation or prove every Qt job passed.',
    'partial_fast0_observation': partial_info}
(args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(args.output / 'archive_phase.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'jobs': len(jobs), 'verdict': receipt['verdict'], 'failed': failed, 'partial': partial_info, 'phase_outcomes': {j['phase']: j['summed_phase_outcomes_not_unique_tests'] for j in jobs}}, indent=2))
