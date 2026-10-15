import ast
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[3]

def at(ref, path):
    return subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=root)

batches = [
    ('678_row_exclusion_api_catalogs', 'c3f8909b28d'),
    ('180_graph_api_catalogs', 'bca8cc7eb0c'),
    ('678_register_stage_api_catalogs', '46bfb196edd'),
]
owned = set()
report = {'baseline': '1d90f7c921b', 'batches': {}, 'catalogs': {}}
for directory, ref in batches:
    receipt = json.loads((root / 'features/data' / (directory + '_2026-10-15') / 'receipt.json').read_text())
    symbols = set(receipt['english_changed_symbols'])
    assert not symbols & owned
    owned.update(symbols)
    for path, expected in receipt['owned_source_files'].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, path
    for path, expected in receipt['review_files'].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, path
    report['batches'][directory] = {'reference': ref, 'symbols': sorted(symbols), 'source_and_review_hashes_exact': True}

for locale in ['en', 'de', 'es', 'fr', 'hi', 'is', 'ko', 'pt', 'sv', 'zh_CN']:
    path = f'docs/source/_static/i18n/api/{locale}.json'
    actual = json.loads((root / path).read_text())
    before = json.loads(at('1d90f7c921b', path))
    for directory, ref in batches:
        expected = json.loads(at(ref, path))
        for symbol in report['batches'][directory]['symbols']:
            assert actual['symbols'][symbol] == expected['symbols'][symbol], (locale, symbol)
    remaining_before = {k: v for k, v in before['symbols'].items() if k not in owned}
    remaining_after = {k: v for k, v in actual['symbols'].items() if k not in owned}
    assert remaining_before == remaining_after, locale
    for key in before.keys() | actual.keys():
        if key != 'symbols':
            assert actual.get(key) == before.get(key), (locale, key)
    report['catalogs'][locale] = {'owned_symbols_exact': len(owned), 'unrelated_records_preserved': len(remaining_before)}

for path in ['spacr/qt/widgets/row_exclusion.py', 'spacr/qt/app.py']:
    def executable(data):
        tree = ast.parse(data)
        for node in ast.walk(tree):
            if hasattr(node, 'body') and isinstance(node.body, list) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
        return ast.dump(tree, include_attributes=False)
    assert executable(at('1d90f7c921b', path)) == executable((root / path).read_bytes()), path

target = root / 'features/data/180_graph_coordinator_readback_2026-10-15'
target.mkdir(exist_ok=True)
(target / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS: 11 symbols exact in all 10 catalogs, all incoming unrelated records retained, doc-only executable AST unchanged')
