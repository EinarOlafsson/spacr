import ast
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root))
revision = sys.argv[1]
if revision != 'current':
    import spacr

    source = subprocess.check_output(['git', 'show', revision + ':spacr/settings.py'], text=True)
    settings = types.ModuleType('spacr.settings')
    settings.__package__ = 'spacr'
    settings.__file__ = str(root / 'spacr/settings.py')
    sys.modules['spacr.settings'] = settings
    spacr.settings = settings
    exec(compile(source, 'spacr/settings.py', 'exec'), settings.__dict__)
else:
    import spacr.settings as settings
spec = importlib.util.spec_from_file_location('default_claim_test', root / 'tests/test_settings_tooltip_quality.py')
test = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = test
spec.loader.exec_module(test)
from spacr.qt.app import APPS
from spacr.qt.screens.settings_model import resolve_default_settings

pairs = {}
variants = {}
for entry in APPS:
    app = entry[0]
    for key, actual in resolve_default_settings(app).items():
        parsed = []
        for match in test.DEFAULT_LITERAL.finditer(settings.tooltips.get(key, '')):
            raw = match.group('value')
            try:
                value = '' if raw.lower() in {'empty', 'blank'} else ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                continue
            parsed.append((raw, value))
        if not parsed:
            continue
        raw, claimed = parsed[-1]
        pair = app + '/' + key
        pairs[pair] = {'claimed': raw, 'actual': repr(actual)}
        if not test._same_default(actual, claimed):
            variants[pair] = pairs[pair]
print(json.dumps({'settings_revision': revision, 'resolver_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'count': len(pairs), 'pairs': pairs, 'variants': variants}, indent=2, sort_keys=True))
