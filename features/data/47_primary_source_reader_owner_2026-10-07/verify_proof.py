"""Verify archive hashes and strictly insertion-only coverage inheritance."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
for name, record in manifest['files'].items():
    data = (root / name).read_bytes()
    assert len(data) == record['bytes']
    assert hashlib.sha256(data).hexdigest() == record['sha256'], name
before = (root / 'before_primary_mask_selector.py').read_bytes()
after = (root / 'after_primary_mask_selector.py').read_bytes()
assert after == before + b'            if worker.isRunning():\n                worker.setParent(None)\n'
old = json.loads((root / 'hosted_module_coverage.json').read_text())['file']
new = json.loads((root / 'coverage.json').read_text())['files']['spacr/qt/widgets/primary_mask_selector.py']
old_lines = set(old['executed_lines'] + old['missing_lines'])
new_lines = set(new['executed_lines'] + new['missing_lines'])
old_arcs = set(map(tuple, old['executed_branches'] + old['missing_branches']))
new_arcs = set(map(tuple, new['executed_branches'] + new['missing_branches']))
assert new_lines == old_lines | {227, 228}
assert new_arcs == old_arcs | {(227, -218), (227, 228)}
assert {227, 228} <= set(new['executed_lines'])
assert {(227, -218), (227, 228)} <= set(map(tuple, new['executed_branches']))
assert not new_lines - set(old['executed_lines'] + new['executed_lines'])
assert not new_arcs - set(map(tuple, old['executed_branches'] + new['executed_branches']))
print(json.dumps({'verified_payloads': len(manifest['files']), 'union_uncovered_statements': 0, 'union_uncovered_branches': 0}))
