"""Verify this compact review archive from files or sparse Git blobs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PREFIX = 'features/data/664_665_lifecycle_independent_readback_2026-10-08'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='.')
    parser.add_argument('--git')
    args = parser.parse_args()
    def read(name):
        if args.git:
            return subprocess.check_output(['git', 'show', args.git + ':' + PREFIX + '/' + name], cwd=args.repo)
        return (Path(__file__).parent / name).read_bytes()
    manifest = json.loads(read('MANIFEST.json'))
    if args.git:
        tree = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', args.git, PREFIX], cwd=args.repo).decode().splitlines()
        paths = {path[len(PREFIX) + 1:] for path in tree}
    else:
        paths = {path.name for path in Path(__file__).parent.iterdir() if path.is_file()}
    assert paths == set(manifest['payloads']) | {'MANIFEST.json'}
    for path, expected in manifest['payloads'].items():
        raw = read(path)
        assert len(raw) == expected['bytes']
        assert hashlib.sha256(raw).hexdigest() == expected['sha256']
    result = json.loads(read('readback.json'))
    assert result['passed'] and result['report_totals']['missed_lines'] == 32 and result['report_totals']['missed_arcs'] == 18
    assert len(result['payloads_verified']) == 52
    print(json.dumps({'passed': True, 'payload_count': len(manifest['payloads']), 'referenced_payloads_verified': 52, 'remaining_lines': 32, 'remaining_arcs': 18}))


if __name__ == '__main__':
    main()
