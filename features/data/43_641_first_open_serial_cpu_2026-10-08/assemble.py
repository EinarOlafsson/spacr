import ast
import gzip
import hashlib
import json
import platform
import subprocess
from pathlib import Path

root = Path('/mnt/wd4tb/spacr-worktrees/codex-first-open-serial-20261008')
scratch = Path('/mnt/wd4tb/scratch/first-open-serial-20261008')
archive = root / 'features/data/43_641_first_open_serial_cpu_2026-10-08'
archive.mkdir(parents=True, exist_ok=True)
base = '4b5c699f1e9a2550985c04f48d44d6bcdbb0cd33'
paths = ['.github/workflows/_pytest-suite.yml', '.github/workflows/tests.yml',
         'tests/qt/test_641_module_first_open_budgets.py',
         'tests/qt/test_641_module_first_open_timing.py',
         'tests/test_ci_suite_classification.py']
sha = lambda b: hashlib.sha256(b).hexdigest()
for phase in ['before', 'after']:
    (archive / phase).mkdir(exist_ok=True)
    for path in paths:
        if phase == 'before' and path.endswith('_timing.py'):
            continue
        data = subprocess.check_output(['git', 'show', f'{base}:{path}'], cwd=root) if phase == 'before' else (root / path).read_bytes()
        (archive / phase / (Path(path).name + '.gz')).write_bytes(gzip.compress(data, mtime=0))
for name in ['timing.log', 'structural.log', 'workflow.log', 'structural-coverage.json']:
    (archive / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
app = (root / 'spacr/qt/app.py').read_bytes()
(archive / 'app-source.py.gz').write_bytes(gzip.compress(app, mtime=0))
cov = json.loads((scratch / 'structural-coverage.json').read_text())
file_cov = cov['files']['spacr/qt/app.py']
tree = ast.parse(app)
methods = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in {'_on_nav_selected', '_open_a_module_screen'}:
        methods[node.name] = {'start': node.lineno, 'end': node.end_lineno,
            'executed_lines': [n for n in file_cov['executed_lines'] if node.lineno <= n <= node.end_lineno],
            'executed_branches': [a for a in file_cov['executed_branches'] if node.lineno <= a[0] <= node.end_lineno]}
receipt = {
    'base_commit': base,
    'production_changed': False,
    'budget_seconds': 10.0,
    'keys_per_navigation_phase': 48,
    'timing': {'passed': 48, 'pytest_seconds': 14.29, 'coverage': False, 'xdist': False},
    'structural': {'passed': 54, 'pytest_seconds': 25.03, 'coverage': 'spacr.qt.app branch coverage', 'xdist': False},
    'workflow': {'passed': 21, 'pytest_seconds': 2.21},
    'scope': 'Three separate focused phases; no whole-suite, hosted acceptance, or ratchet closure claim.',
    'routing': {'reusable_parallel_ignores': 2, 'serial_tail_mentions': 1, 'coverage_exclusions': 1,
                'original_structural_file_remains_coverage_eligible': True},
    'contracts': ['fresh screen per key', 'actual _on_nav_selected', 'three QApplication.processEvents turns',
                  'currentWidget is created screen', 'elapsed strictly below unchanged 10.0 seconds'],
    'startup_coverage': methods,
    'after_bindings': {p: sha((root / p).read_bytes()) for p in paths},
    'production_bindings': {'spacr/qt/app.py': sha(app)},
    'environment': {'python': platform.python_version(), 'CUDA_VISIBLE_DEVICES': '', 'SPACR_DEVICE': 'cpu',
                    'QT_QPA_PLATFORM': 'offscreen', 'MPLBACKEND': 'Agg', 'memory_cap': '4G', 'pytest': '9.1.1'},
    'observations': ['Both Qt phases exit 0. Each logs a nonfatal already-removed chaining lock message after its passing summary.',
                     'No before timing replay was performed; exact prior test/workflow bytes are retained.',
                     'All six corrected Fast/Minimum jobs remained running during this focused repair.']
}
(archive / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(archive / 'README.txt').write_text('First-open timing separation, 2026-10-08\n\n'
    'No production bytes, budget, skip policy, ratchet ceiling or test timeout changed.\n'
    'The original file retains all 48 untimed fresh-screen navigation cases and all six\n'
    'structural guards. The dedicated file retains the original timing contract exactly\n'
    'and has its own explicit window fixture. The existing reusable Qt serial tail runs\n'
    'it once after xdist exits; both parallel branches and coverage exclude only it.\n'
    'The existing classification guard pins those routes and continued original-file\n'
    'eligibility. Actual local results: timing 48 pass, traced structural 54 pass,\n'
    'workflow 21 pass. Counts describe separate phases, not whole acceptance.\n'
    'Before source snapshots are retained without a redundant before timing replay.\n'
    'The app snapshot/hash and focused branch JSON bind retained real startup execution;\n'
    'this partial coverage is not a full-module numerical-ratchet receipt.\n'
    'All logs are losslessly compressed. No native or hosted green acceptance inferred.\n'
    'Run verify.py from this directory, optionally --git HEAD --bind-current.\n')
(archive / 'reproduce.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg
python_bin=${SPACR_PROOF_PYTHON:-python}
tools/run_capped.sh 4G "$python_bin" -m pytest -q -p no:randomly tests/test_ci_suite_classification.py
tools/run_capped.sh 4G "$python_bin" -m pytest -v --tb=short -p no:randomly tests/qt/test_641_module_first_open_timing.py --durations=0
tools/run_capped.sh 4G "$python_bin" -m pytest -v --tb=short -p no:randomly tests/qt/test_641_module_first_open_budgets.py --cov=spacr.qt.app --cov-branch --cov-report=json:first-open-structural-coverage.json
''')
(archive / 'verify.py').write_text('''"""Verify immutable evidence and optionally bind the accepted files to a Git ref."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--git')
parser.add_argument('--bind-current', action='store_true')
args = parser.parse_args()
here = Path(__file__).resolve().parent
root = next(p for p in here.parents if (p / '.git').exists())
prefix = here.relative_to(root).as_posix()

def read(path):
    if args.git:
        return subprocess.check_output(['git', 'show', f'{args.git}:{path}'], cwd=root)
    return (root / path).read_bytes()

manifest = json.loads(read(prefix + '/MANIFEST.json'))
for name, record in manifest['payloads'].items():
    data = read(prefix + '/' + name)
    assert len(data) == record['bytes'], name
    assert hashlib.sha256(data).hexdigest() == record['sha256'], name
if args.bind_current:
    receipt = json.loads(read(prefix + '/receipt.json'))
    for path, expected in {**receipt['after_bindings'], **receipt['production_bindings']}.items():
        assert hashlib.sha256(read(path)).hexdigest() == expected, path
print(f"Verified {len(manifest['payloads'])} payloads; bind_current={args.bind_current}")
''')
(archive / 'assemble.py').write_bytes(Path(__file__).read_bytes())
payloads = {p.relative_to(archive).as_posix(): {'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size}
            for p in sorted(archive.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json'}
(archive / 'MANIFEST.json').write_text(json.dumps({'schema': 1, 'payloads': payloads}, indent=2) + '\n')
print(json.dumps({'payload_count': len(payloads), 'bytes': sum(v['bytes'] for v in payloads.values()), 'startup_coverage': methods}))
