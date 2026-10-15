"""Normal one-module API generation with complete foreign-record preservation."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT)]
import build_documentation_i18n as docs
import build_i18n_catalogs as runtime

sha = lambda raw: hashlib.sha256(raw).hexdigest()
packet = json.loads((ROOT / 'features/data/580_session_gui_2026-10-15/dependency-packet.json').read_text())
source_path = ROOT / 'spacr/qt/widgets/calibration_preview.py'
source_hash = sha(source_path.read_bytes())
assert source_hash == packet['source_sha256']['spacr/qt/widgets/calibration_preview.py']
module = packet['api']['changed_existing_module']
key = module['symbol']
body = docs._clean_doc(ast.parse(source_path.read_text()))
assert body == module['after']
selected = {key: body}
base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
before = {language: json.loads((docs.API_DIR / (language + '.json')).read_text())
          for language in ['en', *docs.MODEL_SPECS]}
all_docs = {name: record['text'] for name, record in before['en']['symbols'].items()}
assert all_docs[key] == module['before']
all_docs[key] = body
history_path = ROOT / 'docs/i18n/api_source_history.json'
history_before = history_path.read_bytes() if history_path.exists() else None
guards = {'runtime_collector_requests': 0, 'api_collector_requests': 0, 'model_decode_requests': 0}

def no_runtime(*args, **kwargs):
    guards['runtime_collector_requests'] += 1
    raise RuntimeError('No full runtime collector authorized for this scoped API operation')

def no_api(*args, **kwargs):
    guards['api_collector_requests'] += 1
    raise RuntimeError('No full API collector authorized for this scoped API operation')

def no_model(*args, **kwargs):
    guards['model_decode_requests'] += 1
    raise RuntimeError('No model decoding authorized for reviewed API targets')

runtime.canonical_sources = no_runtime
docs.public_docstrings = no_api
docs._translate_blocks = no_model
translations = {}
for language in docs.MODEL_SPECS:
    reviewed = docs.reviewed_api_block_translations(selected, language)
    expected_file = ROOT / 'docs/i18n/reviewed/api' / language / '2026-10-15-f580-session-preview-home-b.json'
    expected = json.loads(expected_file.read_text())['records'][0]['translation']
    assert reviewed == {body: expected}
    translations[language] = docs._translate_api_documents(
        selected, language, OUT / 'no-models', SimpleNamespace(), reuse_history=False)
    assert translations[language] == {key: expected}
    print(language, 'normal reviewed API preflight PASS', flush=True)

receipts = {}
for language in docs.MODEL_SPECS:
    docs.write_language(selected, language, translations[language], preserve_other_symbols=True)
    after = json.loads((docs.API_DIR / (language + '.json')).read_text())
    assert {name: value for name, value in before[language]['symbols'].items() if name != key} == {
        name: value for name, value in after['symbols'].items() if name != key}
    assert {name: value for name, value in before[language].items() if name != 'symbols'} == {
        name: value for name, value in after.items() if name != 'symbols'}
    assert after['symbols'][key]['text'] == translations[language][key]
    assert after['symbols'][key]['source_sha256'] == docs._source_hash(body)
    assert after['symbols'][key]['source_blocks_sha256'] == docs._source_block_hashes(body)
    assert after['symbols'][key]['translation_source_blocks_sha256'] == docs._translation_source_block_hashes(body)
    receipts[language] = {'reviewed_target_exact': True,
        'foreign_records_preserved': len(before[language]['symbols']) - 1,
        'metadata_preserved': True, 'catalog_sha256': sha((docs.API_DIR / (language + '.json')).read_bytes())}

docs._write_english_api_manifest(all_docs)
after = json.loads((docs.API_DIR / 'en.json').read_text())
assert {name: value for name, value in before['en']['symbols'].items() if name != key} == {
    name: value for name, value in after['symbols'].items() if name != key}
assert {name: value for name, value in before['en'].items() if name != 'symbols'} == {
    name: value for name, value in after.items() if name != 'symbols'}
assert after['symbols'][key]['text'] == body
assert (history_path.read_bytes() if history_path.exists() else None) == history_before
assert sha(source_path.read_bytes()) == source_hash
assert all(value == 0 for value in guards.values())
report = {'status': 'PASS', 'base_commit': base, 'source_commit': '6a1b51bfd7',
    'module': key, 'source_file_sha256': source_hash, 'source_body': body,
    'normal_functions': ['_clean_doc', 'reviewed_api_block_translations', '_translate_api_documents(reuse_history=False)',
        'write_language(preserve_other_symbols=True)', '_write_english_api_manifest'],
    'history_choice': 'The entire single module block has a current exact review, so no historical paragraph is needed; unrelated records and complete history bytes remain exact.',
    'english_foreign_records_preserved': len(before['en']['symbols']) - 1,
    'history_bytes_preserved': True, 'guard_calls': guards, 'locales': receipts,
    'scope': 'One existing module in ten API catalogs; current source claim only for the exact owner module. No runtime/catalog-arrival/UI-behavior/scientific/native publication claim.'}
(OUT / 'api-generation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print('PASS normal one-module writer; nine reviewed targets exact, unrelated API records/metadata and all history bytes preserved.', flush=True)
