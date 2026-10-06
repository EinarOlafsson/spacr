from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
worktree = Path('/media/carruthers/mnt3/codex/spacr-worktrees/docs-completion-20261005')
root = scratch / 'subcell-docs-37478293839'
main_python = '/home/carruthers/anaconda3/envs/spacr/bin/python'
browser_python = str(scratch / 'render-py312/bin/python')
run = '37478293839'
expected = 'e259c2d4ecfa34b8228a8e8eab4520ff06edcf04'
expectations = json.loads((scratch / 'subcell-publication-expectations-r1.json').read_text())
assert expectations['requested_nightly_commit'] == expected
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=worktree, text=True).strip() == expected
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=worktree, text=True).strip()
for relative, row in expectations['expected_files'].items():
    payload = (worktree / relative).read_bytes()
    assert len(payload) == row['bytes'] and hashlib.sha256(payload).hexdigest() == row['sha256'], relative
counter = 0
while True:
    payload = subprocess.check_output(['gh', 'run', 'view', run, '--json', 'status,conclusion,headSha,jobs,url'], cwd=worktree)
    state = json.loads(payload)
    counter += 1
    snapshot = root / ('workflow-observation-' + str(counter).zfill(4) + '.json')
    snapshot.write_text(json.dumps({'observed_UTC': datetime.now(timezone.utc).isoformat(), 'actual_run': state}, indent=2) + '\n')
    assert state['headSha'] == expected
    steps = [(job['name'], job['status'], [step['name'] for step in job['steps'] if step['status'] == 'in_progress']) for job in state['jobs']]
    print(datetime.now(timezone.utc).isoformat(), state['status'], state['conclusion'], steps, flush=True)
    if state['status'] == 'completed':
        assert state['conclusion'] == 'success', state
        break
    time.sleep(30)

stage_log = []
def execute(label, command):
    started = datetime.now(timezone.utc).isoformat()
    print('START', label, started, command, flush=True)
    result = subprocess.run(command, cwd=worktree, env=os.environ)
    stage_log.append({'stage': label, 'started_UTC': started, 'finished_UTC': datetime.now(timezone.utc).isoformat(), 'command': command, 'exit_code': result.returncode})
    (root / 'publication-stage-journal.json').write_text(json.dumps(stage_log, indent=2) + '\n')
    assert result.returncode == 0, (label, result.returncode)
for branch in ('main', 'nightly'):
    execute('Download normal ' + branch + ' artifact', ['gh', 'run', 'download', run, '-n', 'docs-channel-' + branch, '-D', str(root / ('docs-channel-' + branch))])
execute('Assemble with normal publisher', [main_python, 'tools/publish_docs_channels.py', 'assemble', '--main', str(root / 'docs-channel-main'), '--nightly', str(root / 'docs-channel-nightly'), '--output', str(root / 'assembled'), '--base-path', '/spacr'])
channels = json.loads((root / 'assembled/channels.json').read_text())
assert channels['channels']['nightly']['commit'] == expected
for relative, row in expectations['expected_files'].items():
    route = relative.removeprefix('docs/source/').replace('_extra/tutorials/', 'tutorials/')
    payload = (root / 'assembled/nightly' / route).read_bytes()
    assert len(payload) == row['bytes'] and hashlib.sha256(payload).hexdigest() == row['sha256'], relative
print('PASS actual normal artifact resolves exact pushed source and all frozen API/tutorial bytes.', flush=True)
execute('Actual deployed catalog/player roots', [main_python, str(scratch / 'verify_subcell_current_deployment_r1.py')])
execute('Actual all-nine API browser readback', [browser_python, str(scratch / 'verify_subcell_deployed_API_browser_r1.py')])
execute('Actual all-nine puncta guide readback', [browser_python, str(scratch / 'verify_subcell_deployed_puncta_guides_r1.py')])
(root / 'publication-completion.json').write_text(json.dumps({'passed': True, 'finished_UTC': datetime.now(timezone.utc).isoformat(), 'workflow': int(run), 'actual_resolved_nightly_source': expected, 'actual_terminal_workflow_snapshot': str(snapshot), 'actual_stages': stage_log, 'all_frozen_API_and_tutorial_bytes_exact': True, 'no_application_or_catalog_source_changed': True, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
print('PASS complete normal workflow/artifact/deployed API/guide/tutorial publication acceptance.', flush=True)
