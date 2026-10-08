"""Verify exact source insertion and directly covered guard outcomes."""
import argparse
import ast
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
args = parser.parse_args()


def read(name):
    """Read accepted payload bytes from files or committed blobs."""
    if args.git:
        return subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{prefix}/{name}'])
    return (folder / name).read_bytes()


receipt = json.loads(read('acceptance.json'))
old = gzip.decompress(read('ambient-before.py.gz'))
new = gzip.decompress(read('ambient-after.py.gz'))
assert hashlib.sha256(old).hexdigest() == receipt['before_sha256']
assert hashlib.sha256(new).hexdigest() == receipt['after_sha256']
path = 'spacr/qt/widgets/ambient.py'
assert subprocess.check_output(['git', '-C', str(root), 'show', receipt['parent_revision'] + ':' + path]) == old
actual = subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{path}']) if args.git else (root / path).read_bytes()
assert actual == new
key = b'        for index in reversed(range(len(candidates))):\n'
assert old.count(key) == 1
addition = b'            if index not in selected and costs[index] > budget:\n                continue\n'
assert old.replace(key, key + addition) == new
legacy = gzip.decompress(read('ambient-parent-index-a309.py.gz'))
legacy_class = next(n for n in ast.parse(legacy).body if isinstance(n, ast.ClassDef) and n.name == '_FungalGrowthEngine')
current_class = next(n for n in ast.parse(old).body if isinstance(n, ast.ClassDef) and n.name == '_FungalGrowthEngine')
assert ast.dump(legacy_class, include_attributes=False) == ast.dump(current_class, include_attributes=False)
current_parity = json.loads(gzip.decompress(read('current-parity.json.gz')))
assert current_parity['source_sha256'] == receipt['before_sha256']
assert len(current_parity['frames']) == 4 and all(row['differing_pixels'] == 0 for row in current_parity['frames'])
line = new.decode().splitlines().index(addition.decode().splitlines()[0]) + 1
row = next(value for key, value in json.loads(gzip.decompress(read('final-focused-coverage.json.gz')))['files'].items() if key.endswith(path))
assert {line, line + 1} <= set(row['executed_lines'])
assert {(line, line + 1), (line, line + 2)} <= {tuple(a) for a in row['executed_branches']}
assert not [arc for arc in row['missing_branches'] if any(line <= abs(n) <= line + 1 for n in arc)]
assert old.count(b'pragma: no cover') == new.count(b'pragma: no cover')
source_contracts = json.loads(gzip.decompress(read('source-contracts.json.gz')))
assert source_contracts['callable_names_signatures_docstrings_exact']
test_path = 'tests/qt/test_fungal_growth_engine.py'
test = subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{test_path}']) if args.git else (root / test_path).read_bytes()
assert test == gzip.decompress(read('test_fungal_growth_engine.py.gz'))
for entry in receipt['workers']:
    worker = json.loads(gzip.decompress(read(entry['file'])))
    assert worker['source_sha256'] == receipt['before_sha256' if worker['variant'] == 'before' else 'after_sha256']
    assert worker['viewport'] == worker['buffer'] == [3840, 2160]
    assert worker['controls']['density'] == 3 and worker['controls']['resolution'] == 2 and worker['controls']['size'] == 2.5
    assert worker['worker_stopped_immediately_after_hide'] and worker['worker_stopped_200ms_after_hide']
print('Exact2-line source insertion; both new lines/arcs observed;4native workers source-bound and retired')
