"""Recompute the frozen pooled-source receipt from its archived branch traces.

Run under tools/run_capped.sh 2G with CUDA hidden. This script reads archived
source and traces; it does not reinterpret them against a later core.py.
The two JSON reports come from 53 and 37 passing final-source cases.
The earlier 176-case trace predates the final publication guard and is excluded.
"""

import gzip
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_SOURCE = '5c7bb2ea4072381cb06132ee79ca46bd043ec504104044086d5db26606607625'
source = ROOT / 'core-source-8409ff7c3.py.gz'
assert hashlib.sha256(gzip.decompress(source.read_bytes())).hexdigest() == EXPECTED_SOURCE
added, line = set(), None
for text in gzip.decompress((ROOT / 'core-diff-8409ff7c3.patch.gz').read_bytes()).decode().splitlines():
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

names = ('coverage-final.json', 'coverage-core-final.json')
files = [json.loads(gzip.decompress((ROOT / (name + '.gz')).read_bytes()))['files']['spacr/core.py']
         for name in names]
executed = set().union(*(set(file['executed_lines']) for file in files))
missing = set().union(*(set(file['missing_lines']) for file in files)) - executed
ex_arcs = set().union(*({tuple(arc) for arc in file['executed_branches']}
                       for file in files))
absent = set().union(*({tuple(arc) for arc in file['missing_branches']}
                      for file in files)) - ex_arcs
executable = added & (executed | missing)
report = {
    'source_commit': '8409ff7c30a4f35ab5e92a83dee9c90e250d925e',
    'source_sha256': EXPECTED_SOURCE,
    'source_artifact': source.name,
    'diff_artifact': 'core-diff-8409ff7c3.patch.gz',
    'trace_inputs_sha256': {name: hashlib.sha256(gzip.decompress((ROOT / (name + '.gz')).read_bytes())).hexdigest()
                           for name in names},
    'new_executable_statements': len(executable),
    'new_executed_statements': len(executable - missing),
    'missing_added_lines': sorted(added & missing),
    'executed_touching_branch_arcs': len([arc for arc in ex_arcs if set(arc) & added]),
    'missing_touching_branch_arcs': sorted(arc for arc in absent if set(arc) & added),
}
assert report['new_executed_statements'] == report['new_executable_statements'] == 128
assert report['executed_touching_branch_arcs'] == 70
assert not report['missing_added_lines'] and not report['missing_touching_branch_arcs']
(ROOT / 'new-source-coverage-reproduced.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
