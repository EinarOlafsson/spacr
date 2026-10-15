"""Validate exact reviewed inputs without a full canonical collector or writer."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT), str(OUT)]
import build_i18n_catalogs as runtime
import build_documentation_i18n as docs
import build_guide_i18n as guide
from targets import TARGETS

sha = lambda raw: hashlib.sha256(raw).hexdigest()
packet = json.loads((ROOT / 'features/data/580_session_gui_2026-10-15/dependency-packet.json').read_text())
for path, expected in packet['source_sha256'].items():
    assert sha((ROOT / path).read_bytes()) == expected, path
rows = packet['runtime_sources']
assert len(rows) == 18 and set(TARGETS) == set(runtime.MODEL_SPECS)
for row in rows:
    assert sha(row['source'].encode()) == row['source_sha256']

sources = {'setting_labels': {}, 'setting_tooltips': {}, 'ui': []}
for row in rows:
    if row['table'] == 'ui':
        sources['ui'].append(row['source'])
    else:
        sources[row['table']][row['key']] = row['source']
sources['ui'] = tuple(sources['ui'])
runtime_inputs = OUT / 'runtime_inputs'
api_inputs = OUT / 'api_inputs'
for language, values in TARGETS.items():
    assert len(values) == 19
    records = [{key: row[key] for key in ('table', 'key', 'source', 'source_sha256')} |
               {'translation': target} for row, target in zip(rows, values[:18])]
    path = runtime_inputs / language / '2026-10-15-f580-session-controls-home-b.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'schema': 1, 'language': language,
        'review_kind': 'AI technical authoring; independent technical peer required; no native-speaker acceptance',
        'note': 'Exact eighteen owner sources; normal scoped source-context gates, no catalog generation. Backend API scope is separate.',
        'records': records}, ensure_ascii=False, indent=2) + '\n')

runtime.REVIEWED_RUNTIME_DIR = runtime_inputs
runtime.canonical_sources = lambda: sources
runtime._translate_batches = lambda *a, **k: (_ for _ in ()).throw(AssertionError('No model decoding'))
checks = {'runtime': {}, 'api': {}}
for language in runtime.MODEL_SPECS:
    try:
        loaded = runtime.reviewed_runtime_translations(language)
        assert loaded == {row['source']: target for row, target in zip(rows, TARGETS[language][:18])}
        checks['runtime'][language] = {'status': 'PASS', 'count': len(loaded)}
    except Exception as error:
        checks['runtime'][language] = {'status': 'FAIL', 'error': str(error)}

module = packet['api']['changed_existing_module']
node = ast.parse((ROOT / 'spacr/qt/widgets/calibration_preview.py').read_text())
body = docs._clean_doc(node)
assert body == module['after']
blocks, layout = docs.translatable_blocks(body)
assert blocks == [module['after']]
source = blocks[0]
context = docs._api_translation_source(source)
api_row = {'label': module['symbol'] + '#0', 'source': source, 'context': context,
           'source_sha256': sha(source.encode()), 'previous_same_source_indexes': []}
retire_api = []
for language in runtime.MODEL_SPECS:
    target = TARGETS[language][18]
    try:
        assert docs._reviewed_api_block_valid(source, target, language), 'source candidate gate'
        assert docs._reviewed_api_block_valid(context, target, language), 'context candidate gate'
        assert not guide.inline_markup_problems(target), 'inline markup'
        checks['api'][language] = {'status': 'PASS', 'count': 1}
    except Exception as error:
        checks['api'][language] = {'status': 'FAIL', 'error': str(error)}
    path = api_inputs / language / '2026-10-15-f580-session-preview-home-b.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'schema': 1, 'language': language,
        'review_kind': 'AI technical authoring; independent technical peer required; no native-speaker acceptance',
        'note': 'One existing module paragraph; no method, helper or backend API scope expansion.',
        'records': [api_row | {'translation': target, 'review_basis': 'Exact read-only plate/acquisition-session preview semantics; no claim that the preview changes files.'}]},
        ensure_ascii=False, indent=2) + '\n')
    for old_path in (ROOT / 'docs/i18n/reviewed/api' / language).glob('*.json'):
        old_payload = json.loads(old_path.read_text())
        for index, old_row in enumerate(old_payload.get('records', [])):
            if old_row.get('label') == api_row['label'] and not old_row.get('retired'):
                assert old_row['source'] == module['before']
                retire_api.append({'path': str(old_path.relative_to(ROOT)), 'record_index': index,
                    'file_sha256': sha(old_path.read_bytes()), 'old_record': old_row,
                    'replacement_source_sha256': api_row['source_sha256']})
assert len(retire_api) == 9
(OUT / 'input-check.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n')
(OUT / 'api-source.json').write_text(json.dumps({'source_commit': '6a1b51bfd7', 'body': body,
    'changed_block': api_row, 'retirements': retire_api}, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
assert all(row['status'] == 'PASS' for category in checks.values() for row in category.values())
print('PASS 162 runtime targets through nine normal scoped review loaders and nine API source/context/inline checks. No full canonical collection or catalog writer.', flush=True)
