"""Check F534's existing reviews and catalogs against current source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
sys.path.insert(0, str(root / 'tools'))
import build_documentation_i18n as api
import build_i18n_catalogs as runtime

languages = ('sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr')
api_review_name = '2026-10-11-spatial-submodules-and-zoo-ownership.json'
runtime_review_name = '2026-10-11-spatial-submodules.json'
errors = []

def check(condition, description):
    if not condition:
        errors.append(description)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

baseline = json.loads((api.REVIEWED_API_DIR / 'sv' / api_review_name).read_text())
symbols = {r['label'].rpartition('#')[0] for r in baseline['records'] if not r.get('retired')}
print('Extract current public API source once', flush=True)
all_docs = api.public_docstrings()
docs = {key: all_docs[key] for key in symbols if key in all_docs}
check(set(docs) == symbols, 'Missing reviewed symbols: ' + repr(sorted(symbols - set(docs))))
contract_path = root / 'tests/data/release_contracts/534_spatial_api_arrivals_2026-10-11.json'
contract = json.loads(contract_path.read_text())
check(len(contract) == 22, 'F534 contract no longer has22 symbols')
for key, source in contract.items():
    check(all_docs.get(key) == source, 'Current source differs from F534 contract: ' + key)
print('Extract current canonical runtime source once', flush=True)
sources = runtime.canonical_sources()
report = {
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'scope': 'Existing 2026-10-11 F534 review batch only, including its shared model-zoo/API records',
    'no_catalogs_or_reviews_written': True,
    'no_model_decode_network_gpu_or_data_execution': True,
    'review_provenance': 'Existing AI technical review retained; this is source/hash/gate/catalog reconciliation, not new native-speaker or scientific review.',
    'contract': {'path': str(contract_path.relative_to(root)), 'sha256': sha(contract_path), 'symbols': len(contract), 'current_source_matches': all(all_docs.get(k) == v for k, v in contract.items())},
    'api_symbols': {key: {'source_sha256': api._source_hash(source), 'source_blocks_sha256': api._source_block_hashes(source), 'context_blocks_sha256': api._translation_source_block_hashes(source)} for key, source in sorted(docs.items())},
    'source_file_hashes': {},
    'locales': {},
}
for key in docs:
    parts = key.split('.')
    for end in range(len(parts), 0, -1):
        path = root.joinpath(*parts[:end]).with_suffix('.py')
        if path.is_file():
            report['source_file_hashes'][str(path.relative_to(root))] = sha(path)
            break

english = json.loads((api.API_DIR / 'en.json').read_text())['symbols']
for key, source in docs.items():
    check(english.get(key) == api._english_manifest({key: source})['symbols'][key], 'Stale English API catalog: ' + key)

table_names = {'setting_labels': 'SETTING_LABELS', 'setting_tooltips': 'SETTING_TOOLTIPS', 'ui': 'UI'}
for language in languages:
    api_path = api.REVIEWED_API_DIR / language / api_review_name
    runtime_path = runtime.REVIEWED_RUNTIME_DIR / language / runtime_review_name
    api_payload = json.loads(api_path.read_text())
    runtime_payload = json.loads(runtime_path.read_text())
    active = [r for r in api_payload['records'] if not r.get('retired')]
    check(len(active) == 80, language + ': expected80 active API records')
    try:
        admitted = api.reviewed_api_block_translations(docs, language)
    except ValueError as exc:
        admitted = {}
        errors.append(language + ': normal API loader: ' + str(exc))
    api_catalog = json.loads((api.API_DIR / f'{language}.json').read_text())['symbols']
    for key, source in docs.items():
        record = api_catalog.get(key, {})
        check(record.get('source_sha256') == api._source_hash(source), language + ': stale API source hash: ' + key)
        check(record.get('source_blocks_sha256') == api._source_block_hashes(source), language + ': stale API block hashes: ' + key)
        check(record.get('translation_source_blocks_sha256') == api._translation_source_block_hashes(source), language + ': stale API context hashes: ' + key)
    for record in active:
        key, _, raw_index = record['label'].rpartition('#')
        index = int(raw_index)
        blocks, _ = api.translatable_blocks(docs[key])
        target_blocks, _ = api.translatable_blocks(api_catalog[key]['text'])
        check(index < len(blocks) and blocks[index] == record['source'], language + ': stale active API review: ' + record['label'])
        check(admitted.get(record['source']) == record['translation'], language + ': API review not admitted: ' + record['label'])
        check(index < len(target_blocks) and target_blocks[index] == record['translation'], language + ': API target differs at current block: ' + record['label'])

    retired_api = [r for r in api_payload['records'] if r.get('retired')]
    for record in retired_api:
        key, _, raw_index = record['label'].rpartition('#')
        index = int(raw_index)
        blocks, _ = api.translatable_blocks(docs[key])
        target_blocks, _ = api.translatable_blocks(api_catalog[key]['text'])
        check(blocks[index] != record['source'], language + ': retired API paragraph unexpectedly current: ' + record['label'])
        check(admitted.get(blocks[index]) == target_blocks[index], language + ': replacement API review/catalog not current: ' + record['label'])

    runtime_catalog_path = runtime.CATALOG_DIR / f'{language}.py'
    runtime_catalog = runpy.run_path(str(runtime_catalog_path))
    rows = runtime_payload['records']
    check(len(rows) == 59, language + ': expected59 runtime records')
    replacement_path = runtime.REVIEWED_RUNTIME_DIR / language / '2026-10-12-n692-spatial-hints-arrivals.json'
    replacements = [r for r in json.loads(replacement_path.read_text())['records']
                    if r.get('table') == 'setting_tooltips' and r.get('key') == 'spatial_platform']
    check(len(replacements) == 1, language + ': expected one current spatial_platform replacement')
    runtime._REVIEWED_RUNTIME_LOADING.add(language)
    try:
        for record in rows + replacements:
            table, key, source, target = (record[k] for k in ('table', 'key', 'source', 'translation'))
            current = sources[table].get(key) if isinstance(sources[table], dict) else key if key in sources[table] else None
            label = language + ':' + table + '/' + key
            check(current == source, 'Stale runtime source: ' + label)
            check(record['source_sha256'] == runtime._source_hash(source), 'Stale runtime review hash: ' + label)
            check(runtime._contextualize(target, language, source) == target, 'Runtime target not context-idempotent: ' + label)
            reasons = runtime._translation_rejection_reasons(source, target, language, force=runtime._looks_translatable(source))
            check(not reasons, 'Runtime target rejected: ' + label + ' ' + repr(sorted(reasons)))
            catalog_table = table_names[table]
            check(runtime_catalog[catalog_table].get(key) == target, 'Runtime catalog target differs: ' + label)
            check(runtime_catalog['SOURCE_HASHES'].get((catalog_table, key)) == runtime._source_hash(source), 'Stale runtime catalog source hash: ' + label)
    finally:
        runtime._REVIEWED_RUNTIME_LOADING.discard(language)
    report['locales'][language] = {
        'active_api_records': len(active),
        'retired_api_records_retained': len(api_payload['records']) - len(active),
        'runtime_records': len(rows),
        'retired_runtime_replacement_records': len(replacements),
        'runtime_replacement_review_sha256': sha(replacement_path),
        'retired_api_replacement_blocks_verified': len(retired_api),
        'api_review_sha256': sha(api_path),
        'runtime_review_sha256': sha(runtime_path),
        'api_catalog_sha256': sha(api.API_DIR / f'{language}.json'),
        'runtime_catalog_sha256': sha(runtime_catalog_path),
    }
    print(language, 'checked80 API and59 runtime records; cumulative errors', len(errors), flush=True)

inventory_path = root / 'tests/test_docstring_correctness.py'
inventory_text = inventory_path.read_text()
for name, pin in (
    ('test_public_callable_inventory_is_source_derived_not_docstring_derived', 10026),
    ('test_callable_boundary_is_cross_checked_with_i18n_extractor', 9675),
):
    node = next(n for n in ast.parse(inventory_text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    constants = [n.value for n in ast.walk(node) if isinstance(n, ast.Constant)]
    check(contract_path.name in constants, name + ': missing exact F534 contract')
    check(pin in constants, name + ': historical pin changed')
report['inventory'] = {'path': str(inventory_path.relative_to(root)), 'sha256': sha(inventory_path), 'inspection_only_not_full_test_run': True, 'dated_f534_contract_present_at_both_pins': True, 'unchanged_pins': [10026, 9675]}
report['errors'] = errors
report['passed'] = not errors
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
for error in errors:
    print('FAIL', error, flush=True)
print('PASS' if not errors else 'FAIL', 'scoped source-bound F534 reconciliation', flush=True)
raise SystemExit(bool(errors))
