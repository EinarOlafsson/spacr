import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / 'io-source-0ee2f0053.py.gz'
assert hashlib.sha256(gzip.decompress(source.read_bytes())).hexdigest() == '993c6161391e28f6239360bae16bdeac9b1f068cd99c24d42083d610699bc970'
added, line = set(), None
for text in gzip.decompress((ROOT / 'io-diff-0ee2f0053.patch.gz').read_bytes()).decode().splitlines():
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
trace = ROOT / 'native-batch-coverage.json'
file = json.loads(trace.read_text())['files']['spacr/io.py']
executed, missing = set(file['executed_lines']), set(file['missing_lines'])
arcs = {tuple(arc) for arc in file['executed_branches']}
absent = {tuple(arc) for arc in file['missing_branches']}
report = {
    'source_commit': '0ee2f0053268a2a0bda0caab52af7cf1ebe6eb62',
    'source_sha256': hashlib.sha256(gzip.decompress(source.read_bytes())).hexdigest(),
    'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
    'new_executable_statements': len(added & (executed | missing)),
    'new_executed_statements': len(added & executed),
    'missing_added_lines': sorted(added & missing),
    'executed_touching_branch_arcs': len([arc for arc in arcs if set(arc) & added]),
    'missing_touching_branch_arcs': sorted(arc for arc in absent if set(arc) & added),
    'scope': 'Added native batch source only; no full-module coverage claim.',
    'trace_cohort': 'Agent-reported 40 passing bounded CPU cases. Standalone command transcript not retained.',
}
assert report['new_executable_statements'] == report['new_executed_statements'] == 157
assert report['executed_touching_branch_arcs'] == 84
assert not report['missing_added_lines'] and not report['missing_touching_branch_arcs']
(ROOT / 'native-batch-new-source-coverage.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
