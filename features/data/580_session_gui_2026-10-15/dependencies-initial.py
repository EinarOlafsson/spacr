"""Bounded exact GUI/source dependency census; no global collector or writer."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path.cwd()
OUT = Path('/mnt/wd4tb/scratch/f580-session-ui-20261015')
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT)]
import api_visibility
import build_documentation_i18n as docs
import nested_helper_docs
from spacr.qt.screens.settings_model import _humanize, _strip_type_prefix

sha = lambda value: hashlib.sha256(value).hexdigest()
BASE = '2bb17bb917'
BACKEND_BEFORE = '659dee56545820c8e475f374bfef6af7eae77c0b'
FILES = ['spacr/qt/screens/settings_model.py', 'spacr/qt/widgets/calibration_preview.py']
KEYS = ['intensity_calibration_sessions', 'intensity_calibration_session_column',
        'intensity_calibration_beads', 'intensity_calibration_reference_session']
CHANGED = ['intensity_calibration', 'intensity_calibration_wells',
           'intensity_calibration_reference_plate', 'intensity_calibration_statistic']

def old(ref, path):
    return subprocess.check_output(['git', 'show', f'{ref}:{path}'])

def tips(raw):
    tree = ast.parse(raw)
    value = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'tooltips' for t in n.targets))
    return {k.value: ast.literal_eval(v) for k, v in zip(value.keys, value.values)
            if isinstance(k, ast.Constant) and k.value in [*KEYS, *CHANGED]}

current = tips((ROOT / 'spacr/settings.py').read_bytes())
previous = tips(old(BACKEND_BEFORE, 'spacr/settings.py'))
rows = [{'table': 'setting_labels', 'key': key, 'source': _humanize(key)} for key in KEYS]
for key in [*KEYS, *CHANGED]:
    row = {'table': 'setting_tooltips', 'key': key,
           'source': ' '.join(_strip_type_prefix(current[key]).split())}
    if key in previous:
        row['old_source'] = ' '.join(_strip_type_prefix(previous[key]).split())
        row['old_source_sha256'] = sha(row['old_source'].encode())
    rows.append(row)

def literals(raw):
    return {n.args[0].value for n in ast.walk(ast.parse(raw))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'tr'
            and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)}

new_ui = set()
removed_ui = set()
for path in FILES:
    before = literals(old(BASE, path)); after = literals((ROOT / path).read_bytes())
    new_ui.update(after - before); removed_ui.update(before - after)
en = ast.parse((ROOT / 'spacr/qt/i18n_catalogs/en.py').read_bytes())
en_ui = next(ast.literal_eval(n.value) for n in en.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'UI' for t in n.targets))
reused_ui = sorted(new_ui.intersection(en_ui))
new_ui.difference_update(en_ui)
rows += [{'table': 'ui', 'key': value, 'source': value} for value in sorted(new_ui)]
assert len(rows) == 18, (len(rows), new_ui)
for row in rows:
    row['source_sha256'] = sha(row['source'].encode())

review_paths = subprocess.check_output(['rg', '-l',
    '"key": "intensity_calibration(_wells|_reference_plate|_statistic)?"',
    'docs/i18n/reviewed/runtime'], cwd=ROOT, text=True).splitlines()
retire = []
for path in review_paths:
    data = json.loads((ROOT / path).read_text())
    for index, record in enumerate(data.get('records', [])):
        if record.get('table') != 'setting_tooltips' or record.get('key') not in CHANGED:
            continue
        source = next(row for row in rows if row['table'] == 'setting_tooltips' and row['key'] == record['key'])
        assert record['source_sha256'] == source['old_source_sha256'], (path, record['key'])
        retire.append({'path': path, 'language': data['language'], 'record_index': index,
                       'old_record': record, 'file_sha256': sha((ROOT / path).read_bytes()),
                       'replacement_source_sha256': source['source_sha256']})
assert len(retire) == 36, len(retire)

scopes = {}
private = []
for label in ('before', 'after'):
    scope = OUT / ('api-' + label)
    for path in FILES:
        dest = scope / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(old(BASE, path) if label == 'before' else (ROOT / path).read_bytes())
    definitions = nested_helper_docs.inventory(scope, ignore_patterns=docs.AUTOAPI_IGNORE)
    modules = {entry.module for entry in definitions} & nested_helper_docs.ENABLED_MODULES
    entries = nested_helper_docs.entries(definitions, modules=modules)
    scopes[label] = {entry.qualified_key: {'docstring': entry.docstring, 'signatures': entry.signatures} for entry in entries}
assert scopes['before'] == scopes['after']
tree = ast.parse((ROOT / FILES[0]).read_bytes())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_CalibrationPathEdit')
for node in [cls, *[n for n in cls.body if isinstance(n, ast.FunctionDef)]]:
    symbol = 'spacr.qt.screens.settings_model.' + cls.name
    if node is not cls:
        symbol += '.' + node.name
    assert not api_visibility.public_symbol(symbol)
    private.append({'symbol': symbol, 'public_symbol': False, 'docstring': docs._clean_doc(node)})
module = 'spacr.qt.widgets.calibration_preview'
assert api_visibility.public_symbol(module)
api = {'symbol': module, 'before': docs._clean_doc(ast.parse(old(BASE, FILES[1]))),
       'after': docs._clean_doc(ast.parse((ROOT / FILES[1]).read_bytes()))}
api['source_sha256'] = sha(api['after'].encode())
packet = {'backend_owner_source': '0d81431062978ca6db5fb731d95bb97ed3b819e6',
    'private_backend_adoption': BASE, 'prior_root_merge': BACKEND_BEFORE,
    'runtime_sources': rows, 'runtime_message_count': len(rows), 'nine_locale_target_count': len(rows) * 9,
    'reused_existing_ui_literals': reused_ui,
    'removed_gui_literals_pending_normal_canonical_retirement_check': sorted(removed_ui),
    'four_prior_tooltip_review_retirements': retire,
    'registry': {'existing_backend_alpha_entry': 580,
                 'settings': KEYS,
                 'literal_widget_names': ['IntensityCalibrationSessions', 'IntensityCalibrationSessionColumn',
                                         'IntensityCalibrationBeads', 'IntensityCalibrationReferenceSession'],
                 'additional_registry_change_required': False},
    'api': {'changed_existing_module': api, 'excluded_private_class_bodies': private,
            'enabled_nested_before': len(scopes['before']), 'enabled_nested_after': len(scopes['after']),
            'enabled_nested_docs_and_signatures_unchanged': True,
            'scope': 'Actual normal visibility and nested inventory over the two owner source files only; no global source inventory.'},
    'source_sha256': {path: sha((ROOT / path).read_bytes()) for path in [*FILES, 'spacr/settings.py', 'spacr/measure.py', 'spacr/intensity_rescale.py']},
    'pending': 'Independent source peer; exact runtime/API authoring, normal writers/retirements/derived artifacts, and resolving the retained dark/light viewport failure before coherent owner publication.'}
(OUT / 'dependency-packet.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'runtime_sources': len(rows), 'targets': len(rows) * 9, 'old_tooltip_records': len(retire),
                  'api_module_blocks': 1, 'private_bodies': len(private), 'unchanged_enabled_nested': len(scopes['after'])}), flush=True)
