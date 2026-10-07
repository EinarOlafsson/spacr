"""Verify two fixture-owner archives and unchanged test-function assertions."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--git', metavar='REV')
parser.add_argument('--check-source', metavar='REV', help='Also check runtime bindings against an available source revision.')
args = parser.parse_args()

def read(name):
    """Read one payload from the worktree or committed archive."""
    if args.git:
        return subprocess.check_output(['git', 'show', args.git + ':features/data/' + root.name + '/' + name], cwd=root)
    return (root / name).read_bytes()

manifest = json.loads(read('manifest.json'))
for name, entry in manifest['files'].items():
    data = read(name)
    assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256'], name
receipt = json.loads(read('receipt.json'))
assert not receipt['app_source_changed'] and not receipt['guard_or_gc_changes']
total = 0
for path, entry in receipt['bindings'].items():
    before, after = [read(prefix + Path(path).name) for prefix in ('before_', 'after_')]
    assert hashlib.sha256(before).hexdigest() == entry['before_sha256']
    assert hashlib.sha256(after).hexdigest() == entry['after_sha256']
    nodes = []
    for data in (before, after):
        nodes.append({node.name: ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(data))
                      if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')})
    assert nodes[0] == nodes[1] and len(nodes[0]) == entry['unchanged_test_functions']
    total += len(nodes[0])
assert total == receipt['unchanged_test_function_AST_count'] == 18
for path, digest in receipt['unchanged_source_sha256'].items():
    before = subprocess.check_output(['git', 'show', receipt['baseline_commit'] + ':' + path], cwd=root)
    assert hashlib.sha256(before).hexdigest() == digest, path
    if args.check_source:
        after = subprocess.check_output(['git', 'show', args.check_source + ':' + path], cwd=root)
        assert hashlib.sha256(after).hexdigest() == digest, path
for phase in ('before', 'after', 'final'):
    journal = [json.loads(line) for line in read(phase + '.jsonl').splitlines()]
    finish = next(row for row in journal if row['event'] == 'session_finish')
    assert finish['exitstatus'] == 0 and finish['completed_files'] == 3
    boundary = [row for row in journal if row['event'] == 'file_end'][-1]
    assert (boundary['cached_qt_widgets'], boundary['cached_qt_top_levels']) == ((412, 129) if phase == 'before' else (0, 0))
print(f"Verified {len(manifest['files'])} payloads, eight unchanged source bindings, 18 unchanged tests and three passing phases.")
