"""Verify deferred frozen costs, exact-source receipts and owned worker bounds."""
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
parser.add_argument('--check-production', action='store_true')
arguments = parser.parse_args()


def read(name):
    """Read a filesystem payload or committed sparse Git blob."""
    if arguments.git:
        return subprocess.check_output(['git', '-C', str(root), 'show',
                                        f'HEAD:{prefix}/{name}'])
    return (folder / name).read_bytes()


def unpack(name):
    """Read one compressed original source or probe receipt."""
    return gzip.decompress(read(name + '.gz'))


def signatures(source):
    """Return every existing callable argument and class/function docstring."""
    return [(type(node).__name__, node.name,
             ast.dump(node.args, include_attributes=False)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None,
             ast.get_docstring(node))
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


receipt = json.loads(read('receipt.json'))
before, after = unpack('before.py'), unpack('after.py')
assert hashlib.sha256(before).hexdigest() == receipt['baseline_source_sha256']
assert hashlib.sha256(after).hexdigest() == receipt['final_candidate_source_sha256']
assert hashlib.sha256(unpack('initial-candidate.py')).hexdigest() == receipt['initial_candidate_sha256']
assert signatures(before) == signatures(after)
if arguments.check_production:
    current = subprocess.check_output(['git', '-C', str(root), 'show',
                                       'HEAD:spacr/qt/widgets/ambient.py']) \
        if arguments.git else (root / 'spacr/qt/widgets/ambient.py').read_bytes()
    assert hashlib.sha256(current).hexdigest() == receipt['baseline_source_sha256']
functional = json.loads(unpack('functional-receipt.json'))
timing = json.loads(unpack('timing-receipt.json'))
for data in (functional, timing):
    assert data['before_source_sha256'] == receipt['baseline_source_sha256']
    assert data['after_source_sha256'] == receipt['final_candidate_source_sha256']
assert len(functional['states']) == 24 and len(functional['boundary']) == 17
assert len(functional['extra_hues']) == 3 and functional['identity_mutant_detected']
assert all(row['entries'] <= 8 for row in functional['cache_memory'])
assert functional['max_metadata_python_bytes'] <= 3539712
mutation = json.loads(unpack('mutation-receipt.json'))
assert mutation['final_candidate_sha256'] == receipt['final_candidate_source_sha256']
assert mutation['initial_candidate_defect_reproduced']
assert mutation['snapshot_fallback_matches_original']
assert mutation['selected_counts'] == [22060, 22303, 22060]
assert len(timing['native_frames']) == 4
assert all(row['differing_pixels'] == 0 for row in timing['native_frames'])
assert all(row['stages']['_fungal_paths']['path_element_bit_sha256']
           for row in timing['rows'])
dependencies = []
for phase in ('before-A', 'after-B', 'after-C', 'before-D'):
    variant = phase.split('-')[0]
    data = json.loads(unpack(f'data_art_fungal_growth-{variant}-spacr-3.0-{phase}.json'))
    expected = receipt['baseline_source_sha256'] if variant == 'before' \
        else receipt['final_candidate_source_sha256']
    assert data['source_sha256'] == expected
    assert data['viewport'] == data['buffer'] == [3840, 2160]
    assert data['controls']['fps'] == data['controls']['active_rate_cap'] == 24
    assert data['controls']['density'] == 3 and data['controls']['resolution'] == 2
    assert data['controls']['size'] == 2.5 and data['controls']['blur'] == 0
    assert data['worker_stopped_immediately_after_hide']
    assert data['worker_stopped_200ms_after_hide']
    record = data['records'][0]
    assert record['published_frames'] == record['distinct_published_clocks']
    dependencies.append(data['dependency_source_sha256'])
assert all(value == dependencies[0] for value in dependencies)
print('Verified deferred source/functional/native/cold/warm proof and four retired workers')
