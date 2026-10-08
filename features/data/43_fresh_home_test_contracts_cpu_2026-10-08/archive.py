import gzip
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/mnt/wd4tb/spacr-worktrees/codex-fresh-home-contracts-20261008')
scratch = Path('/mnt/wd4tb/scratch/fresh-home-contracts-20261008')
dest = root / 'features/data/43_fresh_home_test_contracts_cpu_2026-10-08'
dest.mkdir(parents=True, exist_ok=True)
transition = json.loads((scratch / 'source-transition.json').read_text())
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
for record in transition:
    name = Path(record['path']).name
    for phase in ('before', 'after'):
        (dest / (phase + '-' + name + '.gz')).write_bytes((scratch / (phase + '-' + name + '.gz')).read_bytes())
for phase in ('before', 'after'):
    (dest / (phase + '.log.gz')).write_bytes(gzip.compress((scratch / (phase + '.log')).read_bytes(), mtime=0))
    (dest / ('seeded-' + phase + '.json')).write_bytes((scratch / ('seeded-' + phase + '.json')).read_bytes())
for name in ('seed_saved_targets.py', 'source-transition.json', 'fatal-ruff.log'):
    (dest / name).write_bytes((scratch / name).read_bytes())
production = {}
for relative in ('spacr/qt/app.py', 'spacr/qt/screens/app_screen.py', 'spacr/qt/theme.py', 'spacr/qt/preferences.py', 'spacr/restart_state.py'):
    raw = (root / relative).read_bytes()
    before = subprocess.check_output(['git', 'show', 'ffed9b993d868203b45b4bb2e293c53b307c2805:' + relative], cwd=root)
    assert raw == before
    production[relative] = digest(raw)
baseline_lint = []
for record in transition:
    name = Path(record['path']).name
    results = []
    for phase in ('before', 'after'):
        raw = gzip.decompress((dest / (phase + '-' + name + '.gz')).read_bytes())
        p = subprocess.run(['/home/olafsson/.local/bin/ruff', 'check', '--output-format=json', '--stdin-filename', record['path'], '-'], input=raw, capture_output=True, cwd=root)
        results.append(json.loads(p.stdout))
    assert results[0] == results[1], record['path']
    baseline_lint.append({'path': record['path'], 'unchanged_findings': results[0]})
(dest / 'baseline-lint.json').write_text(json.dumps(baseline_lint, indent=2) + '\n')
receipt = {
    'scope': 'Six test-only fresh/Home/first-construction contracts; no production changes',
    'parent': 'ffed9b993d868203b45b4bb2e293c53b307c2805',
    'implementation_commit': 'd798ee80fd',
    'python': '/home/olafsson/anaconda3/envs/spacr/bin/python',
    'pytest': '8.4.2',
    'pytest_overlay': '/mnt/wd4tb/scratch/ci-55bad-20261008/pytest842-overlay',
    'controls': {'RAM_cap': '4G', 'CUDA_VISIBLE_DEVICES': '', 'QT_QPA_PLATFORM': 'offscreen', 'SPACR_DEVICE': 'cpu', 'MPLBACKEND': 'Agg', 'randomly': 'disabled', 'xdist': 'unused', 'coverage': 'unused'},
    'before': {'failed': 6, 'passed': 1, 'seconds': 36.62, 'exit_code': 1, 'warnings': 6},
    'after': {'passed': 7, 'seconds': 28.10, 'exit_code': 0, 'constructor_observations': 10},
    'production_sha256': production,
    'test_transition': transition,
    'negative_scope': 'The closed-category original comparison already passed before; only the other six assertions demonstrate failure-to-pass. Its explicit-Home fix ensures the mode is applied before target construction.',
    'harness_note': 'Before ran before cone checkout materialized root pytest.ini and emitted six marker warnings; after used the normal strict pytest.ini. Actual original assertion failures and persisted-state constructor records are retained, and no guard changed.',
    'limits': 'These are targeted CPU correctness checks, not full Qt/CI/native-crash/memory acceptance or isolated timing benchmarks. Existing lock cleanup messages and QApplication widget census remain in raw logs. No deliberate session/restart tests changed.',
}
(dest / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(dest / 'README.txt').write_text('Fresh Home test contracts, 2026-10-08\n\nSeven constructor substitutions in six existing tests explicitly request Home.\nReal persisted target records are written before each constructor by the archived plugin; production session-restore code remains active. Six original assertions fail before and pass after; the seventh closed-category comparison passes in both states and is not claimed as a reproduced failure. Every after constructor starts at Home with an empty module cache. Original assertion bodies, time limits, mode flags and settings/row comparisons are unchanged. Deliberate restore tests and both Qt-owned first-open fixture files are outside this patch.\n\nActual pytest8.4.2, CUDA-hidden/offscreen CPU, 4 GiB cap: before6 failed/1 passed36.62s; after7 passed28.10s. These durations are correctness-run observations, not performance claims. The initial sparse checkout omitted pytest.ini and emitted marker warnings; after uses normal strict configuration. Both raw logs retain that distinction. Three full-Ruff I001 findings are byte-identical before/after; fatal F/E9 and diff checks pass. No full suite or memory/native-fault acceptance is claimed.\n\nRun verify_manifest.py --git HEAD --bind-current after cherry-picking this proof and implementation. reproduce_after.sh replays the seven saved-session cases in an isolated checkout with the retained pytest8.4.2 overlay; it does not mutate production files.\n')
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
    assert before.count(b'MainWindow()') == record['replacements']
    assert before.replace(b'MainWindow()', b'MainWindow(initial_app="__home__")') == after
    if a.bind_current:
        assert digest(read(record['path'])) == record['after_sha256'], record['path']
for phase, marker in [('before', b'6 failed, 1 passed'), ('after', b'7 passed in 28.10s')]:
    assert marker in gzip.decompress(read(prefix + '/' + phase + '.log.gz'))
observations = json.loads(read(prefix + '/seeded-after.json'))
assert len(observations) == 10
assert all(row['initial_app'] == '__home__' and row['starts_at_home'] and not row['screens_after_constructor'] for row in observations)
if a.bind_current:
    for path, expected in r['production_sha256'].items():
        assert digest(read(path)) == expected, path
print('Verified', len(m['payloads']), 'payloads, six exact test-only transitions and ten real Home constructors; bind_current=', a.bind_current)
'''
(dest / 'verify_manifest.py').write_text(verify)
nodes = [
    'tests/qt/test_641_screen_prewarm.py::test_a_half_built_screen_finishes_inside_its_open',
    'tests/qt/test_opening_things_does_not_hang.py::test_a_navigation_that_arrives_mid_open_waits_for_it',
    'tests/qt/test_a_module_screen_is_sheeted_once_before_it_is_seen.py::test_opening_a_module_repolishes_its_screen_once[measure]',
    'tests/qt/test_a_module_screen_is_sheeted_once_before_it_is_seen.py::test_classify_controls_exist_before_the_first_screen_sheet',
    'tests/qt/test_a_tile_that_cannot_open_says_so.py::test_a_missing_package_is_named_rather_than_silent',
    'tests/qt/test_home_navigation_acceptance.py::test_visible_home_tile_opens_real_screen[measure]',
    'tests/qt/test_a_closed_category_builds_nothing_until_opened.py::test_the_run_is_given_exactly_what_a_window_that_built_everything_gives',
]
script = '#!/usr/bin/env bash\nset -euo pipefail\nproof_dir=features/data/43_fresh_home_test_contracts_cpu_2026-10-08\nreplay_dir=$(mktemp -d /mnt/wd4tb/scratch/fresh-home-replay-XXXXXX)\nexport CUDA_VISIBLE_DEVICES="" QT_QPA_PLATFORM=offscreen SPACR_DEVICE=cpu MPLBACKEND=Agg\nexport PYTHONPATH=.:/mnt/wd4tb/scratch/ci-55bad-20261008/pytest842-overlay:"$proof_dir"\ntools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q --tb=short -p no:randomly -p seed_saved_targets --basetemp="$replay_dir/after"'
script += ''.join(" \\\n  '" + node + "'" for node in nodes) + '\n'
(dest / 'reproduce_after.sh').write_text(script)
(dest / 'archive.py').write_bytes(Path(__file__).read_bytes())
payloads = {path.name: {'bytes': path.stat().st_size, 'sha256': digest(path.read_bytes())} for path in sorted(dest.iterdir()) if path.name != 'MANIFEST.json'}
(dest / 'MANIFEST.json').write_text(json.dumps({'schema': 1, 'payloads': payloads}, indent=2) + '\n')
print(len(payloads), sum(record['bytes'] for record in payloads.values()))
