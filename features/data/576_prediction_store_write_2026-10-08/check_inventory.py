import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path.cwd().resolve()
sys.meta_path[:] = [
    finder for finder in sys.meta_path
    if '_Editable' not in getattr(finder, '__name__', type(finder).__name__)
    and not (getattr(finder, '__module__', '') or '').startswith('__editable__')
]
sys.path.insert(0, str(root / 'tools'))
sys.path.insert(0, str(root))
import build_documentation_i18n as docs
import build_i18n_catalogs as runtime

proof = Path(__file__).resolve().parent
with gzip.open(proof / 'baseline-api.json.gz', 'rt') as handle:
    api_before = json.load(handle)
with gzip.open(proof / 'baseline-runtime.json.gz', 'rt') as handle:
    runtime_before = json.load(handle)
api_after = docs.public_docstrings()
runtime_after = runtime.canonical_sources()
for maps in (runtime_before, runtime_after):
    for key, value in maps.items():
        if isinstance(value, (list, tuple)):
            maps[key] = sorted(value)
foreign = {
    name: str(Path(module.__file__).resolve())
    for name, module in sys.modules.items()
    if (name == 'spacr' or name.startswith('spacr.'))
    and getattr(module, '__file__', None)
    and not Path(module.__file__).resolve().is_relative_to(root)
}
report = {
    'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'source_root': str(root),
    'api_before': len(api_before),
    'api_after': len(api_after),
    'api_changed': [
        key for key in sorted(set(api_before) | set(api_after))
        if api_before.get(key) != api_after.get(key)
    ],
    'api_delta': {
        key: {'before': api_before.get(key), 'after': api_after.get(key)}
        for key in sorted(set(api_before) | set(api_after))
        if api_before.get(key) != api_after.get(key)
    },
    'runtime_counts_before': {key: len(value) for key, value in runtime_before.items()},
    'runtime_counts_after': {key: len(value) for key, value in runtime_after.items()},
    'runtime_changed': [
        key for key in sorted(set(runtime_before) | set(runtime_after))
        if runtime_before.get(key) != runtime_after.get(key)
    ],
    'foreign_imports': foreign,
    'source_sha256': {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            'spacr/predictions.py', 'spacr/tabular.py',
            'tests/test_predictions_merge.py', 'tests/test_measurement_backends.py'
        )
    },
}
print(json.dumps(report, sort_keys=True, indent=2))
assert report['api_changed'] == [
    'spacr.predictions.merge_prediction_results',
    'spacr.predictions.migrate_prediction_columns',
]
assert not report['runtime_changed'] and not foreign
