"""Only the Portuguese loader and nine previously unexecuted inline checks."""
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
for path, digest in packet['source_sha256'].items():
    assert sha((ROOT / path).read_bytes()) == digest
rows = packet['runtime_sources']
sources = {'setting_labels': {}, 'setting_tooltips': {}, 'ui': []}
for row in rows:
    if row['table'] == 'ui':
        sources['ui'].append(row['source'])
    else:
        sources[row['table']][row['key']] = row['source']
sources['ui'] = tuple(sources['ui'])
runtime.REVIEWED_RUNTIME_DIR = OUT / 'runtime_inputs'
runtime.canonical_sources = lambda: sources
runtime._translate_batches = lambda *a, **k: (_ for _ in ()).throw(AssertionError('No model decoding'))
path = runtime.REVIEWED_RUNTIME_DIR / 'pt/2026-10-15-f580-session-controls-home-b.json'
payload = json.loads(path.read_text())
before = payload['records'][4]['translation']
after = TARGETS['pt'][4]
assert after.replace('"filename"', 'filename') == before
payload['records'][4]['translation'] = after
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
loaded = runtime.reviewed_runtime_translations('pt')
assert loaded == {row['source']: target for row, target in zip(rows, TARGETS['pt'][:18])}
print('PASS Portuguese normal scoped loader: 18 targets; only filename quotation changed.', flush=True)

checks = json.loads((OUT / 'initial-input-check.json').read_text())
checks['runtime']['pt'] = {'status': 'PASS', 'count': 18, 'followup': 'Only required column identifier quoted; normal loader unchanged.'}
for language in runtime.MODEL_SPECS:
    api = json.loads((OUT / 'api_inputs' / language / '2026-10-15-f580-session-preview-home-b.json').read_text())
    row = api['records'][0]
    assert row['translation'] == TARGETS[language][18]
    assert checks['api'][language]['error'] == 'inline_markup_problems() takes 1 positional argument but 2 were given'
    assert not guide.inline_markup_problems(row['translation'])
    checks['api'][language] = {'status': 'PASS', 'count': 1,
        'source_context': 'Passed before initial harness reached the wrong-arity inline-parser call.',
        'inline': 'Actual one-argument parser now invoked; no product change.'}
print('PASS all nine actual API inline parsers; unchanged source/context gates retained from first run.', flush=True)
for language in runtime.MODEL_SPECS:
    payload = json.loads((runtime.REVIEWED_RUNTIME_DIR / language / '2026-10-15-f580-session-controls-home-b.json').read_text())
    for index, row in enumerate(payload['records']):
        assert row['translation'] == TARGETS[language][index]
assert all(row['status'] == 'PASS' for category in checks.values() for row in category.values())
(OUT / 'input-check.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n')
follow = json.loads((OUT / 'portuguese-followup.json').read_text())
follow.update({'old_target': before, 'new_target': after, 'old_target_sha256': sha(before.encode()),
    'new_target_sha256': sha(after.encode()), 'source_sha256': rows[4]['source_sha256'],
    'normal_loader_status': 'PASS'})
(OUT / 'portuguese-followup.json').write_text(json.dumps(follow, ensure_ascii=False, indent=2) + '\n')
