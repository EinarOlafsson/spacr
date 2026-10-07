"""Verify declared payloads and insertion-only source-bound coverage."""
import difflib
import gzip
import hashlib
import json
from pathlib import Path

archive = Path(__file__).resolve().parent
manifest = json.loads((archive / 'manifest.json').read_text())
for name, record in manifest['files'].items():
    data = (archive / name).read_bytes()
    assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], name
before = gzip.decompress((archive / 'before_annotate.py.gz').read_bytes())
after = gzip.decompress((archive / 'after_annotate.py.gz').read_bytes())
expected = before
for kind in ('similar', 'retrain', 'suggest'):
    needle = ('            if stopped:\n                _retire(' + kind + ')\n').encode()
    assert expected.count(needle) == 1
    expected = expected.replace(needle, needle + ('            else:\n                ' + kind + '.setParent(None)\n').encode())
assert expected == after
path = 'spacr/qt/screens/annotate.py'
old = json.loads(gzip.decompress((archive / 'hosted_e1.json.gz').read_bytes()))['files'][path]
new = json.loads(gzip.decompress((archive / 'focused.json.gz').read_bytes()))['files'][path]
left, right = before.decode().splitlines(), after.decode().splitlines()
mapping = {a + i + 1: b + i + 1 for a, b, size in difflib.SequenceMatcher(None, left, right, autojunk=False).get_matching_blocks() for i in range(size)}
assert len(mapping) == len(left)
def mapped(n): return mapping[abs(n)] * (1 if n > 0 else -1)
lines = set(new['executed_lines'] + new['missing_lines'])
arcs = set(map(tuple, new['executed_branches'] + new['missing_branches']))
old_arcs = {tuple(mapped(n) for n in edge) for edge in old['executed_branches'] + old['missing_branches']}
assert lines == {mapped(n) for n in old['executed_lines'] + old['missing_lines']} | {7666, 7681, 7699}
assert old_arcs - arcs == {(7663, 7667), (7678, 7682), (7696, 7700)}
assert arcs - old_arcs == {(7663, 7666), (7678, 7681), (7696, 7699)}
assert {7666, 7681, 7699} <= set(new['executed_lines'])
assert {(7663, 7666), (7678, 7681), (7696, 7699), (7663, 7664), (7678, 7679), (7696, 7697)} <= set(map(tuple, new['executed_branches']))
executed = {mapped(n) for n in old['executed_lines']} | set(new['executed_lines'])
taken = ({tuple(mapped(n) for n in edge) for edge in old['executed_branches']} & arcs) | set(map(tuple, new['executed_branches']))
r = json.loads((archive / 'receipt.json').read_text())
assert sorted(lines - executed) == r['missing_lines']
assert sorted(map(list, arcs - taken)) == r['missing_branches']
assert len(lines - executed) <= r['original_allowance']['uncovered_statements']
assert len(arcs - taken) <= r['original_allowance']['uncovered_branches']
print(json.dumps({'payloads_verified': len(manifest['files']), 'uncovered_statements': len(lines - executed), 'uncovered_branches': len(arcs - taken)}))
