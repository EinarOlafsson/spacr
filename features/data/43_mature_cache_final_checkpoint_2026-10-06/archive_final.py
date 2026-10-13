import gzip
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('/mnt/wd4tb/spacr-worktrees/codex-cpu-final-20261006')
target = root / 'features/data/43_mature_cache_final_checkpoint_2026-10-06'
target.mkdir(parents=True, exist_ok=True)
for name in ('cache-integrated.log', 'cached-widget-journal.log'):
    data = Path('/mnt/wd4tb/scratch/ci-final-repairs-20261006', name).read_bytes()
    (target / (name + '.gz')).write_bytes(gzip.compress(data, mtime=0))
paths = ('spacr/qt/widgets/ambient.py', 'spacr/qt/app.py',
         'spacr/qt/preferences.py', 'spacr/io.py', 'spacr/object.py',
         'spacr/core.py', 'tools/pytest_plugins/qt_serial_rss_journal.py',
         'tests/test_qt_serial_rss_journal.py')
receipt = {
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in paths},
    'results': {'cache_ownership_geometry_phases': '49 passed in 13.89s',
                'passive_cached_widget_journal': '16 passed in 5.79s'},
    'limits': '4GiB cap, CUDA hidden, offscreen Qt; no whole local suite',
    'remaining': 'Hosted green CI, numerical coverage, full serial acceptance, installed Save crash and hard24FPS',
}
(target / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(target / 'README.md').write_text('Final integrated mature-cache CPU checkpoint\n\n'
    'Full focused logs bind the final renderer and passive cached-widget '
    'journal source hashes. The journal reads existing fixture integers; it '
    'does not query/import Qt, collect garbage or change test order. '
    'The actual native MainWindow Save/GC proof and native pixel/worker proof '
    'remain in their separate source-bound archives. No full-suite or '
    'reported crash closure is claimed.\n')
(target / 'archive_final.py').write_bytes(Path(__file__).read_bytes())
files = {}
for path in sorted(target.iterdir()):
    if path.name == 'manifest.json':
        continue
    data = path.read_bytes()
    files[path.name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
(target / 'manifest.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
print(len(files), 'final checkpoint payloads generated')
