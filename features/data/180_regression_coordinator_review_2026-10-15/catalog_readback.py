import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[3]
baseline = '92d44cca97e8fe0f0c4a3ed80d0dfa53d9500cd6'
batches = [('678_job_runner_api_catalogs', 'ba0a0a894'),
           ('180_regression_api_catalogs', '3a594466a')]

def at(ref, path):
    return subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=root)

report = {'baseline': baseline, 'batches': {}, 'catalogs': {}}
owned = set()
for name, ref in batches:
    receipt = json.loads((root / 'features/data' / (name + '_2026-10-15') / 'receipt.json').read_text())
    symbols = set(receipt['english_changed_symbols'])
    assert not owned & symbols
    owned.update(symbols)
    for path, sha in {**receipt['owned_source_files'], **receipt['review_files']}.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == sha, path
    report['batches'][name] = {'artifact': ref, 'symbols': sorted(symbols), 'source_review_hashes_exact': True}

catalogs = []
for locale in ['en', 'de', 'es', 'fr', 'hi', 'is', 'ko', 'pt', 'sv', 'zh_CN']:
    path = f'docs/source/_static/i18n/api/{locale}.json'
    before = json.loads(at(baseline, path))
    actual = json.loads((root / path).read_text())
    for name, ref in batches:
        expected = json.loads(at(ref, path))
        for symbol in report['batches'][name]['symbols']:
            assert actual['symbols'][symbol] == expected['symbols'][symbol], (locale, symbol)
    unrelated_before = {k: v for k, v in before['symbols'].items() if k not in owned}
    unrelated_after = {k: v for k, v in actual['symbols'].items() if k not in owned}
    assert unrelated_before == unrelated_after, locale
    for key in before.keys() | actual.keys():
        if key != 'symbols':
            assert before.get(key) == actual.get(key), (locale, key)
    report['catalogs'][locale] = {'owned_symbols_exact': len(owned), 'unrelated_records_preserved': len(unrelated_before)}
    catalogs.append(actual)

path = 'docs/i18n/api_source_history.json'
before = json.loads(at(baseline, path))
actual = json.loads((root / path).read_text())
receipt = json.loads((root / 'features/data/678_job_runner_api_catalogs_2026-10-15/receipt.json').read_text())
removed = set(receipt['source_history']['removed_unreferenced_hashes'])
assert actual['sources'] == {k: v for k, v in before['sources'].items() if k not in removed}
assert {k: v for k, v in actual.items() if k != 'sources'} == {k: v for k, v in before.items() if k != 'sources'}
references = set()
for catalog in catalogs:
    for row in catalog['symbols'].values():
        if isinstance(row, dict):
            references.add(row.get('source_sha256'))
            references.update(row.get('source_blocks_sha256', []))
            references.update(row.get('translation_source_blocks_sha256', []))
assert not removed & references
report['source_history'] = {'normal_pruned_unreferenced': sorted(removed), 'all_other_records_exact': True}
(Path(__file__).parent / 'catalog-readback.json').write_text(json.dumps(report, indent=2) + '\n')
print(f'PASS {len(owned)} symbols in 10 catalogs, all unrelated incoming records preserved; normal history pruning verified')
