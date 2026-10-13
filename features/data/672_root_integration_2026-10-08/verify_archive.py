"""Verify source handoff integrity and available frozen Git identities."""

import gzip
import hashlib
import json
import pathlib
import subprocess


def main():
    """Check all payload hashes, terminal results and source-bound inventories."""
    archive = pathlib.Path(__file__).resolve().parent
    root = archive.parents[2]
    receipt = json.loads((archive / 'receipt.json').read_text())
    compared = 0
    for name, row in receipt['sources'].items():
        packed = (archive / row['payload']).read_bytes()
        raw = gzip.decompress(packed)
        assert hashlib.sha256(packed).hexdigest() == row['sha256']
        assert hashlib.sha256(raw).hexdigest() == row['raw_sha256']
        try:
            frozen = subprocess.check_output(
                ['git', 'show', f"{receipt['checkpoint']}:{name}"],
                cwd=root, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue
        assert raw == frozen, name
        compared += 1
    for row in [receipt['patch'], receipt['inventory_diff'], *receipt['phases']]:
        packed = (archive / row['payload']).read_bytes()
        raw = gzip.decompress(packed)
        assert hashlib.sha256(packed).hexdigest() == row['sha256']
        assert hashlib.sha256(raw).hexdigest() == row['raw_sha256']
        if 'terminal_summary' in row:
            assert row['terminal_summary'].encode() in raw
            assert row['passing'] == ('failed' not in row['terminal_summary'])
    inventory = json.loads(gzip.decompress(
        (archive / receipt['inventory_diff']['payload']).read_bytes()))
    assert inventory['checkpoint'] == receipt['checkpoint']
    for name, row in receipt['sources'].items():
        if name.startswith('spacr/'):
            assert inventory['source_hashes'][name] == row['raw_sha256']
    print(f"PASS: {len(receipt['sources'])} payloads, patch/inventories and "
          f"{len(receipt['phases'])} terminal logs; {compared} frozen Git comparisons")
    if compared != len(receipt['sources']):
        print('Private checkpoint unavailable: Git comparisons remain incomplete.')


if __name__ == '__main__':
    main()
