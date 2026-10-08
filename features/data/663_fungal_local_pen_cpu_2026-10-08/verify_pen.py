"""Check exact source, observed pen statements and native worker provenance."""
import argparse
import ast
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
    """Read frozen filesystem payloads or committed sparse Git blobs."""
    if arguments.git:
        return subprocess.check_output(['git', '-C', str(root), 'show',
                                        f'HEAD:{prefix}/{name}'])
    return (folder / name).read_bytes()


def unpack(name):
    """Read one compressed proof payload without application imports."""
    return gzip.decompress(read(name + '.gz'))


def signatures(source):
    """Compare all existing callable arguments and class/function prose."""
    return [(type(node).__name__, node.name,
             ast.dump(node.args, include_attributes=False)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None,
             ast.get_docstring(node))
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


receipt = json.loads(read('acceptance.json'))
before, after = unpack('before.py'), unpack('after.py')
assert hashlib.sha256(before).hexdigest() == receipt['before_source_sha256']
assert hashlib.sha256(after).hexdigest() == receipt['after_source_sha256']
assert hashlib.sha256(unpack('test_fungal_mature_raster_cache.py')).hexdigest() == receipt['after_test_sha256']
for path, expected in (
        ('spacr/qt/widgets/ambient.py', receipt['after_source_sha256']),
        ('tests/qt/test_fungal_mature_raster_cache.py', receipt['after_test_sha256'])):
    current = subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{path}']) \
        if arguments.git else (root / path).read_bytes()
    assert hashlib.sha256(current).hexdigest() == expected, path
assert signatures(before) == signatures(after)
coverage = json.loads(unpack('focused-coverage.json'))
ambient = next(value for name, value in coverage['files'].items()
               if name.endswith('spacr/qt/widgets/ambient.py'))
changed = {line for tag, i, j, k, end in difflib.SequenceMatcher(
    None, before.decode().splitlines(), after.decode().splitlines(),
    autojunk=False).get_opcodes() if tag != 'equal'
    for line in range(k + 1, end + 1)}
assert changed == set(receipt['all_changed_statements_executed'])
assert changed <= set(ambient['executed_lines'])
dependencies = []
for palette in ('spacr', 'random'):
    for phase in ('before-A', 'after-B', 'after-C', 'before-D'):
        variant = phase.split('-')[0]
        data = json.loads(unpack(f'data_art_fungal_growth-{variant}-{palette}-3.0-{phase}.json'))
        assert data['source_sha256'] == hashlib.sha256(before if variant == 'before' else after).hexdigest()
        assert data['viewport'] == data['buffer'] == [3840, 2160]
        assert data['controls']['fps'] == 24
        assert data['controls']['density'] == 3 and data['controls']['resolution'] == 2
        assert data['worker_stopped_immediately_after_hide']
        assert data['worker_stopped_200ms_after_hide']
        assert data['records'][0]['published_frames'] == data['records'][0]['distinct_published_clocks']
        dependencies.append(data['dependency_source_sha256'])
assert all(value == dependencies[0] for value in dependencies)
assert b'19 passed in 43.20s' in unpack('focused.log')
print('Verified frozen native pen source, tests, all changed statements and eight retired workers')
