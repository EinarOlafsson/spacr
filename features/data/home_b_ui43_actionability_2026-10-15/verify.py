"""Verify this read-only audit against its pinned Git objects, without running Qt."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
receipt = json.loads((HERE / 'receipt.json').read_text())
pin = receipt['source_pin']
cache = {}

def source(path):
    if path not in cache:
        cache[path] = subprocess.check_output(['git', 'show', f'{pin}:{path}'], cwd=ROOT)
    return cache[path]

def verify_hash(path, expected):
    assert hashlib.sha256(source(path)).hexdigest() == expected, path

for row in receipt['items']:
    verify_hash(row['item_file'], row['item_sha256'])
    for evidence in row['source_evidence']:
        verify_hash(evidence['path'], evidence['sha256'])
        lines = source(evidence['path']).decode().splitlines()
        for anchor in evidence['anchors']:
            assert anchor['literal'] in lines[anchor['first_line'] - 1], anchor

captions = json.loads((HERE / 'caption-debt-check.json').read_text())
assert captions['source_pin'] == pin
verify_hash('spacr/qt/i18n_catalogs/en.py', captions['english_catalog_sha256'])
for locale in captions['locales']:
    path = f"spacr/qt/i18n_catalogs/{locale['locale']}.py"
    verify_hash(path, locale['sha256'])
    assignments = {node.targets[0].id: ast.literal_eval(node.value)
                   for node in ast.parse(source(path)).body
                   if isinstance(node, ast.Assign)
                   and isinstance(node.targets[0], ast.Name)
                   and isinstance(node.value, ast.Dict)}
    for caption in captions['captions']:
        text = caption['text']
        assert assignments['UI'].get(text), (locale['locale'], text)
        assert assignments['SOURCE_HASHES'][('UI', text)] == caption['source_sha256']
print(f"Verified {len(receipt['items'])} item dispositions and {len(captions['captions'])} captions in nine locales at {pin}; no application tests run.")
