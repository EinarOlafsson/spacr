import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--scratch', type=Path, required=True)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
before = '55badc57ff25ef6a7e14721561773a412b515318'
after = '4133beafcd0ae427795617a4a01e295fd40539b7'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def git_bytes(revision, path):
    return subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=args.repo)

jobs = []
for name in ('fast0', 'fast1', 'fast2', 'min0', 'min1', 'min2'):
    metadata = (args.scratch / f'{name}-job.json').read_bytes()
    job = json.loads(metadata)
    assert job['head_sha'] == before and job['run_id'] == 37720484173
    assert job['run_attempt'] == 1 and job['status'] == 'completed'
    raw = (args.scratch / f'{name}.log').read_bytes()
    lines = raw.decode('utf-8').splitlines()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    (args.output / f'{name}.log.gz').write_bytes(compressed)
    (args.output / f'{name}-job.json').write_bytes(metadata)
    summaries = [{'line': number, 'text': line} for number, line in enumerate(lines, 1)
                 if re.search(r'=+ .*\b(?:passed|failed|skipped|xfailed)\b.*=+', line)]
    totals = {}
    for item in summaries:
        for count, label in re.findall(r'(\d+) (passed|failed|skipped|xfailed|xpassed|error|errors)\b', item['text']):
            totals[label] = totals.get(label, 0) + int(count)
    jobs.append({'phase': name, 'job_id': job['id'], 'url': job['html_url'],
                 'source_head': job['head_sha'], 'run_id': job['run_id'],
                 'run_attempt': job['run_attempt'], 'conclusion': job['conclusion'],
                 'started_at': job['started_at'], 'completed_at': job['completed_at'],
                 'raw_log_sha256': digest(raw), 'raw_log_bytes': len(raw),
                 'blocking_failed_nodes': sorted(set(re.findall(r'\bFAILED (tests/\S+)', '\n'.join(lines)))),
                 'pytest_summary_count': len(summaries),
                 'summed_phase_outcomes_not_unique_test_counts': totals,
                 'pytest_summaries': summaries,
                 'batch_result_lines': [line for line in lines if re.search(r'\d+ of \d+ batches failed|all \d+ batches', line)],
                 'watchdog_dump_lines': [{'line': number, 'text': line} for number, line in enumerate(lines, 1) if 'Timeout (0:05:00)!' in line],
                 'native_fault_markers': [line for line in lines if re.search(r'Fatal Python error|worker .*crashed|Segmentation fault', line)]})
shutil.copyfile(args.scratch / 'run.json', args.output / 'run.json')
production = ('spacr/plaque.py', 'spacr/submodules.py', 'spacr/qt/mask_engine.py',
              'spacr/qt/screens/plate_view.py', 'spacr/qt/widgets/ambient.py')
tests = ('tests/test_no_two_packaged_files_differ_only_in_case.py',
         'tests/test_plaque_segmentation_diagnostics.py', 'tests/test_test_suite_hygiene.py',
         'tests/test_docstring_correctness.py')
bindings = {path: {'before_sha256': digest(git_bytes(before, path)),
                   'corrected_sha256': digest(git_bytes(after, path))} for path in (*production, *tests)}
assert all(bindings[path]['before_sha256'] == bindings[path]['corrected_sha256'] for path in production)
receipt = {'scope': 'Six terminal original-55 Fast/Minimum-dependencies jobs; no corrected-run acceptance claimed.',
           'head_sha': before, 'run_id': 37720484173, 'run_attempt': 1,
           'corrected_head_reference': after, 'corrected_required_run_reference': 37728397690,
           'source_bindings': bindings, 'unchanged_production_files': list(production),
           'unique_blocking_nodes': sorted({node for job in jobs for node in job['blocking_failed_nodes']}),
           'jobs': jobs,
           'repair_checks': {'case_collision': 'Original bytes renamed to snapshot-receipt.json; existing collision guard passed locally (1 node; parent subsequently verified both file nodes).',
                             'plaque_mock': 'Named sentinel channel_axis and existing helper; invalid-return ValueError still precedes eval. The original plaque rejection plus two hygiene guards passed (3 nodes).',
                             'background_boundary': 'Root explicitly subtracted the new background-picker arrival while preserving 13182. One owning boundary node passed.',
                             'limits': 'These focused checks are not a whole-suite, GPU, or corrected-hosted acceptance claim.'},
           'count_policy': 'Each outcome count is a sum of raw pytest phase summaries within one job. Different Python/dependency shards and repeated phases must not be summed into unique tests.',
           'watchdog_policy': 'Nonfatal five-minute thread dumps are preserved. All six jobs completed; only the reported batch-98/87/107 failures caused their final exit. Dumps are not relabeled as failed tests or native faults.'}
assert len(receipt['unique_blocking_nodes']) == 4
(args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
for source, target in ((args.scratch / 'plaque-mock-after.log', 'plaque-mock-after.log.gz'),
                       (args.scratch / 'case-collision-repair' / 'after-pytest.log', 'case-guard-after.log.gz')):
    (args.output / target).write_bytes(gzip.compress(source.read_bytes(), compresslevel=9, mtime=0))
shutil.copyfile(args.scratch / 'case-collision-repair' / 'repair-receipt.json', args.output / 'case-repair-receipt.json')
shutil.copyfile(Path(__file__), args.output / 'archive_phase.py')
print(json.dumps({'jobs': len(jobs), 'unique_blocking_nodes': receipt['unique_blocking_nodes'],
                  'native_fault_markers': sum(len(job['native_fault_markers']) for job in jobs),
                  'outcomes': {job['phase']: job['summed_phase_outcomes_not_unique_test_counts'] for job in jobs}}, indent=2))
