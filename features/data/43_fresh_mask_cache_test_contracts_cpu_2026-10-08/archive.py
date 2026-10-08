import gzip
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/mnt/wd4tb/spacr-worktrees/codex-fresh-home-contracts-20261008')
scratch = Path('/mnt/wd4tb/scratch/fresh-mask-cache-contracts-20261008')
dest = root / 'features/data/43_fresh_mask_cache_test_contracts_cpu_2026-10-08'
dest.mkdir(parents=True, exist_ok=True)
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
transitions = json.loads((scratch / 'source-transition.json').read_text())
lint = []
for record in transitions:
    outcomes = []
    for phase in ('before', 'after'):
        name = phase + '-' + Path(record['path']).name + '.gz'
        raw = (scratch / name).read_bytes()
        (dest / name).write_bytes(raw)
        run = subprocess.run(['/home/olafsson/.local/bin/ruff', 'check', '--output-format=json', '--stdin-filename', record['path'], '-'], input=gzip.decompress(raw), capture_output=True, cwd=root)
        outcomes.append(json.loads(run.stdout))
    assert outcomes[0] == outcomes[1], record['path']
    lint.append({'path': record['path'], 'unchanged_findings': outcomes[0]})
for phase in ('before', 'after'):
    (dest / (phase + '.log.gz')).write_bytes(gzip.compress((scratch / (phase + '.log')).read_bytes(), mtime=0))
    (dest / ('seeded-' + phase + '.json')).write_bytes((scratch / ('seeded-' + phase + '.json')).read_bytes())
for name in ('seed_mask_targets.py', 'source-transition.json', 'fatal-ruff.log'):
    (dest / name).write_bytes((scratch / name).read_bytes())
(dest / 'baseline-lint.json').write_text(json.dumps(lint, indent=2) + '\n')
production = {}
for relative in ('spacr/qt/app.py', 'spacr/qt/screens/app_screen.py', 'spacr/qt/preferences.py', 'spacr/restart_state.py'):
    raw = (root / relative).read_bytes()
    assert raw == subprocess.check_output(['git', 'show', 'ffed9b993d868203b45b4bb2e293c53b307c2805:' + relative], cwd=root)
    production[relative] = digest(raw)
receipt = {'parent': 'dac0fdc96f', 'implementation_commit': '7d44278c67', 'python': '/home/olafsson/anaconda3/envs/spacr/bin/python', 'pytest': '8.4.2', 'pytest_overlay': '/mnt/wd4tb/scratch/ci-55bad-20261008/pytest842-overlay', 'RAM_cap': '4G', 'CUDA_VISIBLE_DEVICES': '', 'QT_QPA_PLATFORM': 'offscreen', 'SPACR_DEVICE': 'cpu', 'MPLBACKEND': 'Agg', 'coverage': False, 'xdist': False, 'before': {'failed': 2, 'seconds': 7.33, 'exit_code': 1}, 'after': {'passed': 2, 'seconds': 5.53, 'exit_code': 0}, 'production_sha256': production, 'test_transition': transitions, 'limits': 'Two targeted persisted-Mask correctness counterexamples only; no application mutation, altered assertion, changed limit, whole-suite/native-crash/memory/performance acceptance. Deliberate restore tests preserved. Existing F401 and import-layout Ruff findings unchanged.'}
(dest / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(dest / 'README.txt').write_text('Fresh Mask cache contracts, 2026-10-08\n\nTwo existing tests require Mask to be unbuilt: main-window navigation must add one stack page, and rebuild-before-built must have no previous Mask screen. A normal constructor can restore a persisted Mask first. The archived plugin writes and verifies a real saved session immediately before each constructor; both original nodes fail before and pass after explicitly requesting Home. App, original assertions, timing limits and deliberate session-restore tests remain unchanged.\n\nActual pytest8.4.2; CUDA-hidden/offscreen CPU;4 GiB cap: before2 failed7.33s; after2 passed5.53s. Both processes retired. Each after record selects actual Home with empty module cache despite persisted Mask. Both runs use normal strict pytest.ini. Existing full-Ruff findings match exactly, including two baseline F401 unused imports in test_main_window.py; no incidental import cleanup performed. git diff check passed. Durations are correctness-run observations, not performance acceptance.\n\nVerify with verify_manifest.py --git HEAD --bind-current. reproduce_after.sh replays both real persisted-session assertions in an isolated checkout using the retained pytest8.4.2 overlay.\n')
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
for record in r['test_transition']:
    name = Path(record['path']).name
    before = gzip.decompress(read(prefix + '/before-' + name + '.gz'))
    after = gzip.decompress(read(prefix + '/after-' + name + '.gz'))
    assert digest(before) == record['before_sha256']
    assert digest(after) == record['after_sha256']
    assert before.replace(b'MainWindow()', b'MainWindow(initial_app="__home__")', 1) == after
    if a.bind_current:
        assert digest(read(record['path'])) == record['after_sha256'], record['path']
for phase, marker in [('before', b'2 failed in 7.33s'), ('after', b'2 passed in 5.53s')]:
    assert marker in gzip.decompress(read(prefix + '/' + phase + '.log.gz'))
rows = json.loads(read(prefix + '/seeded-after.json'))
assert len(rows) == 2
assert all(row['persisted_target'] == 'mask' and row['initial_app'] == '__home__' and row['starts_at_home'] and not row['screens_after_constructor'] for row in rows)
if a.bind_current:
    for path, expected in r['production_sha256'].items():
        assert digest(read(path)) == expected, path
print('Verified', len(m['payloads']), 'payloads, two exact test-only transitions and two real Home constructors; bind_current=', a.bind_current)
'''
(dest / 'verify_manifest.py').write_text(verify)
(dest / 'reproduce_after.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
proof_dir=features/data/43_fresh_mask_cache_test_contracts_cpu_2026-10-08
replay_dir=$(mktemp -d /mnt/wd4tb/scratch/fresh-mask-cache-replay-XXXXXX)
export CUDA_VISIBLE_DEVICES="" QT_QPA_PLATFORM=offscreen SPACR_DEVICE=cpu MPLBACKEND=Agg
export PYTHONPATH=.:/mnt/wd4tb/scratch/ci-55bad-20261008/pytest842-overlay:"$proof_dir"
tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q --tb=short -p no:randomly -p seed_mask_targets --basetemp="$replay_dir/after" \\
  tests/qt/test_main_window.py::test_main_window_constructs_and_switches \\
  tests/qt/test_cov_wf_qt_app.py::test_a_screen_can_be_rebuilt_before_it_was_ever_built
''')
(dest / 'archive.py').write_bytes(Path(__file__).read_bytes())
payloads = {path.name: {'bytes': path.stat().st_size, 'sha256': digest(path.read_bytes())} for path in sorted(dest.iterdir()) if path.name != 'MANIFEST.json'}
(dest / 'MANIFEST.json').write_text(json.dumps({'schema': 1, 'payloads': payloads}, indent=2) + '\n')
print(len(payloads), sum(record['bytes'] for record in payloads.values()))
