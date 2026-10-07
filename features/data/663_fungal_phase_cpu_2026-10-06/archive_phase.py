import gzip
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/mnt/wd4tb/spacr-worktrees/codex-cpu-final-20261006')
target = root / 'features/data/663_fungal_phase_cpu_2026-10-06'
target.mkdir(parents=True, exist_ok=True)
bindings = {}
for name in ('spacr/qt/widgets/ambient.py', 'tests/qt/test_ambient.py',
             'tests/qt/test_fungal_growth_engine.py'):
    data = (root / name).read_bytes()
    bindings[name] = hashlib.sha256(data).hexdigest()
    (target / (Path(name).name + '.gz')).write_bytes(gzip.compress(data, mtime=0))
log = Path('/mnt/wd4tb/scratch/ci-final-repairs-20261006/fungal-phases-integrated.log').read_bytes()
(target / 'integrated-phases.log.gz').write_bytes(gzip.compress(log, mtime=0))
receipt = {
    'source_commit': subprocess.check_output(['git', 'rev-parse', '81c102068b'], text=True).strip(),
    'source_sha256': bindings,
    'result': '59 passed in 9.23s; CUDA hidden, offscreen Qt, 4GiB cap',
    'renderer_changed': False,
    'reason': 'The old fixed minimum ink census predates connected fine common-origin filaments and measured only the young-growth phase.',
    'checks': ['nonempty, expanding, upward and widening early growth at2/9/30s',
               'substantial mature and hour-later motion at90/97s and3600.33/3607.33s',
               'at most30percent screen ink at all observed phases',
               'dark/light at320x200 and960x600',
               'empty, frozen-clock, mature-only-frozen and overfilled injected defects rejected',
               'existing linked-parent/fork/bright-tip/density guards preserved'],
    'remaining': 'Final hosted acceptance and native4K hard24FPS remain open',
}
(target / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(target / 'README.md').write_text('Mycelium rendered phase acceptance, 2026-10-06\n\n'
    'The renderer is unchanged. The obsolete style-specific minimum painted '
    'fraction is replaced by rendered growth/motion checks across early, '
    'mature and hour-later phases, with the user\'s maximum30percent screen '
    'coverage. Four injected defects prove the checks reject empty/frozen/'
    'overfilled output. All59 selected existing/new cases pass under4GiB. '
    'Source snapshots, full log and hashes are retained here. This accepts '
    'these CPU rendering checks; final hosted CI and hard24FPS remain open.\n')
(target / 'archive_phase.py').write_bytes(Path(__file__).read_bytes())
files = {}
for path in sorted(target.iterdir()):
    if path.name == 'manifest.json':
        continue
    data = path.read_bytes()
    files[path.name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
(target / 'manifest.json').write_text(json.dumps({'payloads': files}, indent=2) + '\n')
print(len(files), 'phase payloads generated')
