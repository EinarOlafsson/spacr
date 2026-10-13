"""Verify the archived native watcher added-source branch receipt."""

import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
receipt = json.loads((ROOT / 'receipt.json').read_text())
source = gzip.decompress((ROOT / 'core.py.gz').read_bytes())
assert hashlib.sha256(source).hexdigest() == receipt['core_sha256']
added, line = set(), None
for text in gzip.decompress((ROOT / 'core.patch.gz').read_bytes()).decode().splitlines():
    match = re.match(r'@@ .*\+(\d+)', text)
    if match:
        line = int(match[1])
        continue
    if line is None or text.startswith('+++'):
        continue
    if text.startswith('+'):
        added.add(line)
        line += 1
    elif not text.startswith(('-', '\\')):
        line += 1
file = json.loads((ROOT / 'core-branch-trace.json').read_text())['files']['spacr/core.py']
executed, missing = set(file['executed_lines']), set(file['missing_lines'])
arcs = {tuple(arc) for arc in file['executed_branches']}
absent = {tuple(arc) for arc in file['missing_branches']}
assert len(added & (executed | missing)) == len(added & executed) == 106
assert not added & missing
assert len([arc for arc in arcs if arc[0] in added]) == 58
assert len([arc for arc in arcs if set(arc) & added]) == 63
assert not [arc for arc in absent if set(arc) & added]
print('Native watch frozen-source coverage: 106/106 added statements; 63/63 touching arcs.')
