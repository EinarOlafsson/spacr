"""Verify archived payloads and optional current source without importing spaCR."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    """Read frozen or Git-backed payloads and validate all declared bindings."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--git', action='store_true')
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--current', action='store_true')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    prefix = directory.relative_to(root).as_posix()

    def read(name):
        """Read one owned proof payload from disk or current Git tree."""
        if args.git:
            return subprocess.check_output(['git', 'show', 'HEAD:' + prefix + '/' + name], cwd=root)
        return (directory / name).read_bytes()

    manifest = json.loads(read('manifest.json'))
    for name, expected in manifest['payloads'].items():
        data = read(name)
        assert len(data) == expected['bytes'], name
        assert hashlib.sha256(data).hexdigest() == expected['sha256'], name
        if name.endswith('.gz'):
            gzip.decompress(data)
    receipt = json.loads(read('receipt.json'))
    for path, binding in receipt['source_bindings'].items():
        for phase in ('before', 'after'):
            payload = binding.get(phase + '_payload')
            if payload:
                data = gzip.decompress(read(payload))
                assert hashlib.sha256(data).hexdigest() == binding[phase + '_sha256'], path
        if args.current:
            data = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=root) if args.git else (root / path).read_bytes()
            assert hashlib.sha256(data).hexdigest() == binding['after_sha256'], path
    inventory = json.loads(read('inventory.json'))
    assert inventory['full_before_api_count'] == inventory['full_after_api_count'] == 13215
    assert not inventory['scoped_api_added'] and not inventory['scoped_api_removed']
    assert inventory['scoped_api_changed'] == ['spacr.qt.shortcuts']
    assert all(inventory['new_caption_membership'].values())
    delta = json.loads(read('runtime-delta.json'))
    assert len(delta['ui']['added']) == 22 and not delta['ui']['removed']
    assert all(not item['added'] and not item['removed'] and not item['changed']
               for key, item in delta.items() if key != 'ui')
    changes = json.loads(read('changed-coverage.json'))
    assert sum(len(row['changed_executable']) for row in changes.values()) == 234
    assert all(not row['changed_missing_lines'] and not row['changed_missing_arcs'] for row in changes.values())
    print(f"PASS {len(manifest['payloads'])} payloads, {len(receipt['source_bindings'])} source/test bindings; frozen N656 scope only")


if __name__ == '__main__':
    main()
