"""Check unchanged-source inheritance and directly observed new ancestry code."""
import argparse
import difflib
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

folder = Path(__file__).resolve().parent
root = folder.parents[2]
prefix = folder.relative_to(root).as_posix()
parser = argparse.ArgumentParser()
parser.add_argument('--git', action='store_true')
arguments = parser.parse_args()


def read(name):
    """Read exact accepted proof bytes from disk or committed blobs."""
    if arguments.git:
        return subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{prefix}/{name}'])
    return (folder / name).read_bytes()


old = gzip.decompress(read('ambient-before.py.gz'))
new = gzip.decompress(read('ambient-after.py.gz'))
receipt = json.loads(read('acceptance.json'))
assert hashlib.sha256(old).hexdigest() == receipt['before_sha256']
assert hashlib.sha256(new).hexdigest() == receipt['after_sha256']
path = 'spacr/qt/widgets/ambient.py'
actual = subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{path}']) if arguments.git else (root / path).read_bytes()
assert actual == new
assert subprocess.check_output(['git', '-C', str(root), 'show', receipt['parent_revision'] + ':' + path]) == old
expected = old.replace(b'        selected = set()\n', b'        parents = [by_endpoint.get(edge[:2]) for edge in candidates]\n        selected = set()\n', 1)
expected = expected.replace(b'                    cursor = by_endpoint.get(candidates[cursor][:2])\n', b'                    cursor = parents[cursor]\n', 1)
assert expected == new
prior = json.loads(read('prior-ambient-union.json'))
assert prior['module_receipt']['current_source_sha256'] == receipt['before_sha256']
assert prior['focused']['measured_source_sha256'] == receipt['before_sha256']
removed = {6135, 6136, 6137}


def old_coordinate(value):
    """Map only the prior proved unreachable fallback deletion."""
    assert abs(value) not in removed
    sign = -1 if value < 0 else 1
    return sign * (abs(value) - 3 if abs(value) > 6137 else abs(value))


prior_missing = {old_coordinate(n) for n in prior['hosted_missing']['missing_lines'] if n not in removed}
prior_arcs = {tuple(old_coordinate(n) for n in arc) for arc in prior['hosted_missing']['missing_branches'] if not any(abs(n) in removed for n in arc)}
assert not prior_missing - set(prior['focused']['executed_lines'])
assert not prior_arcs - {tuple(a) for a in prior['focused']['executed_branches']}
assert prior['module_receipt']['remaining_lines'] == []
assert prior['module_receipt']['remaining_branches'] == []
line_map = {}
for tag, start_old, end_old, start_new, end_new in difflib.SequenceMatcher(None, old.splitlines(), new.splitlines(), autojunk=False).get_opcodes():
    if tag == 'equal':
        for offset in range(end_old - start_old):
            line_map[start_new + offset + 1] = start_old + offset + 1
coverage = json.loads(gzip.decompress(read('final-focused-coverage.json.gz')))
row = next(value for key, value in coverage['files'].items() if key.endswith(path))
changed = {i for i in range(1, len(new.splitlines()) + 1) if i not in line_map}
assert len(changed) == 2
assert changed <= set(row['executed_lines'])
remaining_lines = sorted(set(row['missing_lines']) - set(line_map))
remaining_arcs = sorted(tuple(arc) for arc in row['missing_branches'] if any(abs(n) not in line_map for n in arc))
allowance = prior['module_receipt']['original_allowance']
assert not remaining_lines
assert not remaining_arcs
assert allowance['uncovered_statements'] == 1 and allowance['uncovered_branches'] == 0
assert old.count(b'pragma: no cover') == new.count(b'pragma: no cover')
test_path = 'tests/qt/test_fungal_growth_engine.py'
test_bytes = subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{test_path}']) if arguments.git else (root / test_path).read_bytes()
assert test_bytes == gzip.decompress(read('test_fungal_growth_engine.py.gz'))
print('Prior exact-source union0/0; changed lines', sorted(changed), 'directly observed; current union0/0 within unchanged1/0 allowance')
