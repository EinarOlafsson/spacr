import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path.cwd().resolve()
sys.meta_path[:] = [finder for finder in sys.meta_path
                    if '_Editable' not in getattr(finder, '__name__', type(finder).__name__)
                    and not (getattr(finder, '__module__', '') or '').startswith('__editable__')]
sys.path.insert(0, str(root / 'tools'))
sys.path.insert(0, str(root))
import build_documentation_i18n as docs
import build_i18n_catalogs as runtime

baseline_path = Path('/mnt/wd4tb/scratch/root-ci-final-wave-current-inventory-20261008.json')
baseline = json.loads(baseline_path.read_text())
current = {'api': docs.public_docstrings(), 'runtime': runtime.canonical_sources()}
for snapshot in (baseline, current):
    for key, value in snapshot['runtime'].items():
        if isinstance(value, (list, tuple)):
            assert len(value) == len(set(value)), key
            snapshot['runtime'][key] = sorted(value)
foreign = {name: str(Path(module.__file__).resolve())
           for name, module in sys.modules.items()
           if name == 'spacr' or name.startswith('spacr.')
           if getattr(module, '__file__', None)
           and not Path(module.__file__).resolve().is_relative_to(root)}
assert not foreign, foreign
out = Path('/mnt/wd4tb/scratch/n663-spaceout-prefs-current-20261008')
with gzip.open(out / 'after-full-inventory.json.gz', 'wt') as handle:
    json.dump(current, handle, sort_keys=True)
counts = {key: len(value) for key, value in current['runtime'].items()}
changed_api = sorted(key for key in baseline['api'].keys() | current['api'].keys()
                     if baseline['api'].get(key) != current['api'].get(key))
changed_runtime = {name: sorted(key for key in baseline['runtime'][name].keys() | current['runtime'][name].keys()
                                if baseline['runtime'][name].get(key) != current['runtime'][name].get(key))
                   if isinstance(current['runtime'][name], dict)
                   else ([] if baseline['runtime'][name] == current['runtime'][name] else ['<list>'])
                   for name in current['runtime']}
report = {
    'source_root': str(root),
    'source_head': os.popen('git rev-parse HEAD').read().strip(),
    'source_sha256': {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                      for p in ('spacr/qt/app.py', 'spacr/qt/preferences.py', 'spacr/qt/screens/app_screen.py', 'spacr/qt/widgets/ambient.py')},
    'baseline_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
    'full_equal': current == baseline,
    'api_entries': len(current['api']),
    'runtime_counts': counts,
    'changed_api': changed_api,
    'changed_runtime': changed_runtime,
    'loaded_spacr_modules': len([name for name in sys.modules if name == 'spacr' or name.startswith('spacr.')]),
    'foreign_spacr_modules': foreign,
}
(out / 'after-full-inventory-report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
assert report['full_equal']
