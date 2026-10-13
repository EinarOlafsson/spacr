"""Verify bounded artifact hashes and conservative insertion-only coverage."""
import difflib
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

archive = Path(__file__).resolve().parent
repo = archive.parents[2]
manifest = json.loads((archive / 'manifest.json').read_text())
for name, record in manifest['files'].items():
    data = (archive / name).read_bytes()
    assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], name
before = gzip.decompress((archive / 'before_make_masks.py.gz').read_bytes())
after = gzip.decompress((archive / 'after_make_masks.py.gz').read_bytes())
needle = b'            drain_thread(worker, timeout_ms=5000)\n        self._loading = False\n'
replacement = b'            if not drain_thread(worker, timeout_ms=5000):\n                worker.setParent(None)\n        self._loading = False\n'
assert after == before.replace(needle, replacement)
receipt = json.loads((archive / 'receipt.json').read_text())
path = 'spacr/qt/screens/make_masks.py'
records = {label: json.loads(gzip.decompress((archive / (label + '.json.gz')).read_bytes()))['files'][path]
           for label in ('focused_now', 'focused222', 'hosted_e1')}
hosted_source = subprocess.check_output(['git', 'show', receipt['hosted_sha'] + ':' + path], cwd=repo)
sources = {'focused_now': after, 'focused222': before, 'hosted_e1': hosted_source}
current = records['focused_now']
lines = set(current['executed_lines'] + current['missing_lines'])
arcs = set(map(tuple, current['executed_branches'] + current['missing_branches']))
executed, taken = set(), set()
for label, record in records.items():
    left, right = sources[label].decode().splitlines(), after.decode().splitlines()
    mapping = {a + i + 1: b + i + 1
               for a, b, size in difflib.SequenceMatcher(None, left, right, autojunk=False).get_matching_blocks()
               for i in range(size)}
    changed = sorted(set(range(1, len(left) + 1)) - set(mapping))
    if label == 'focused_now':
        assert changed == []
    else:
        assert len(changed) == 1 and left[changed[0] - 1] == '            drain_thread(worker, timeout_ms=5000)'
        assert 16643 not in mapping.values()
    def mapped(n):
        value = mapping.get(abs(n))
        return None if value is None else value * (1 if n > 0 else -1)
    executed |= {mapped(n) for n in record['executed_lines'] if mapped(n) is not None}
    taken |= {tuple(mapped(n) for n in edge) for edge in record['executed_branches'] if all(mapped(n) is not None for n in edge)} & arcs
assert sorted(lines - executed) == receipt['missing_lines']
assert sorted(map(list, arcs - taken)) == receipt['missing_branches']
assert {16643, 16644} <= set(current['executed_lines'])
assert {(16643, 16644), (16643, 16645)} <= set(map(tuple, current['executed_branches']))
assert len(lines - executed) <= receipt['original_allowance']['uncovered_statements']
assert len(arcs - taken) <= receipt['original_allowance']['uncovered_branches']
print(json.dumps({'payloads_verified': len(manifest['files']), 'uncovered_statements': len(lines - executed), 'uncovered_branches': len(arcs - taken)}))
