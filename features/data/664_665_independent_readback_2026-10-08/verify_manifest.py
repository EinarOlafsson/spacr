"""Verify this compact review archive from files or sparse Git blobs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PREFIX = 'features/data/664_665_independent_readback_2026-10-08'


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
    assert result['passed'] and result['remaining_changed_lines'] == 38 and result['remaining_changed_arcs'] == 21
    assert sum(archive['payload_count'] for archive in result['archives'].values()) == 228
    print(json.dumps({'passed': True, 'payload_count': len(manifest['payloads']), 'referenced_payloads_verified': 228, 'remaining_lines': 38, 'remaining_arcs': 21}))


if __name__ == '__main__':
    main()
