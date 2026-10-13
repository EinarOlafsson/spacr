from pathlib import Path
import datetime
import json
import subprocess
import time

folder = Path('/media/carruthers/mnt3/codex/scratch/current-source-ci-20261007/current-ci-watch-20261009')
folder.mkdir(exist_ok=False)
runs = (37881951815, 37881951511, 37881951526)
source = '6fb0115f807c351382f6d6b20625e00ba05dd0dc'
repository = 'EinarOlafsson/spacr'
start = time.monotonic()
finished = set()
for attempt in range(97):
    for run in runs:
        if run in finished:
            continue
        raw = subprocess.check_output(['gh', 'api',
            f'repos/{repository}/actions/runs/{run}'], timeout=60)
        state = json.loads(raw)
        if state['head_sha'] != source:
            raise RuntimeError(f'Unexpected source for run {run}')
        temporary = folder / f'current-{run}.tmp'
        temporary.write_bytes(raw)
        temporary.replace(folder / f'current-{run}.json')
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(), run,
              state['status'], state['conclusion'], flush=True)
        jobs = json.loads(subprocess.check_output(['gh', 'api', '--paginate', '--slurp',
            f'repos/{repository}/actions/runs/{run}/jobs?per_page=100'], timeout=60))
        for page in jobs:
            for job in page['jobs']:
                output = folder / f'terminal-job-{job["id"]}.log'
                if job['status'] != 'completed' or job['conclusion'] != 'failure' or output.exists():
                    continue
                data = subprocess.check_output(['gh', 'api', '--allow-escape-sequences',
                    f'repos/{repository}/actions/jobs/{job["id"]}/logs'], timeout=60)
                if not data:
                    raise RuntimeError('Empty terminal failure log')
                output.write_bytes(data)
                (folder / f'terminal-job-{job["id"]}.json').write_text(
                    json.dumps({'run': run, 'source': source, 'job': job}, indent=2) + '\n')
        if state['status'] == 'completed':
            finished.add(run)
    if len(finished) == len(runs):
        print('All watched runs terminal; inspect conclusions and exact job evidence.', flush=True)
        break
    if time.monotonic() - start >= 8 * 3600:
        print('Observation window ended; unfinished runs remain unverified.', flush=True)
        break
    time.sleep(300)
