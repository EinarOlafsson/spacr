"""Verify a freshly fetched GitHub run snapshot before the approved promotion."""
import argparse
import collections
import datetime
import hashlib
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('snapshot', type=Path)
parser.add_argument('--head', required=True)
args = parser.parse_args()
raw = args.snapshot.read_bytes()
run = json.loads(raw)
problems = []
for key, expected in [('headSha', args.head), ('headBranch', 'nightly'),
                      ('workflowName', 'tests'), ('event', 'workflow_dispatch'),
                      ('status', 'completed'), ('conclusion', 'success')]:
    if run.get(key) != expected:
        problems.append(f'{key}: expected {expected!r}, received {run.get(key)!r}')
jobs = run.get('jobs', [])
names = [job.get('name', '') for job in jobs]
if len(set(names)) != len(names):
    problems.append('duplicate job names in run snapshot')
required = ['Lint / types / complexity',
            'Coverage / every module at 90% and none loses coverage',
            'Release gate / zero unexpected failures', 'Publish passed/total badge']
for name in required:
    matches = [job for job in jobs if job.get('name') == name]
    if len(matches) != 1 or matches[0].get('status') != 'completed' or matches[0].get('conclusion') != 'success':
        problems.append(f'required successful job missing: {name}')
groups = [
    ('coverage', r'^Coverage shard (\d+) of 12$', set(range(12))),
    ('fast', r'^Fast / Full suite control .* shard (\d+) of 3$', set(range(3))),
    ('minimum', r'^Minimum dependencies .* shard (\d+) of 3$', set(range(3))),
    ('Qt', r'Qt shard (\d+) of 3 ', set(range(3))),
]
for label, pattern, expected in groups:
    matched = [(int(match.group(1)), job) for job in jobs
               if (match := re.search(pattern, job.get('name', '')))]
    if len(matched) != len(expected) or {i for i, _ in matched} != expected:
        problems.append(f'{label}: incomplete or duplicate matrix')
    if any(job.get('status') != 'completed' or job.get('conclusion') != 'success'
           for _, job in matched):
        problems.append(f'{label}: matrix has unfinished or unsuccessful jobs')
for label in ['Integration', 'Slow', 'GPU', 'Network', 'NAS']:
    matched = [job for job in jobs if job.get('name', '').startswith(label + ' / ')]
    if len(matched) != 1 or matched[0].get('status') != 'completed' or matched[0].get('conclusion') != 'success':
        problems.append(f'{label}: required successful suite missing')
if len(jobs) != 30:
    problems.append(f'expected 30 final jobs from pinned workflow, received {len(jobs)}')
if any(job.get('status') != 'completed' or job.get('conclusion') != 'success' for job in jobs):
    problems.append('run still has unfinished or unsuccessful jobs')
print(json.dumps({'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'expected_head': args.head, 'snapshot_sha256': hashlib.sha256(raw).hexdigest(),
                  'snapshot_file': str(args.snapshot), 'verified_full_success': not problems,
                  'job_counts': dict(collections.Counter(job.get('conclusion') or job.get('status') for job in jobs)),
                  'problems': problems}, indent=2))
raise SystemExit(0 if not problems else 2)
