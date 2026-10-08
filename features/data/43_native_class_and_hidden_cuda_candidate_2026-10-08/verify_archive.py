"""Check all bounded diagnostic candidate payloads and source identities."""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    """Validate every payload and compare available immutable Git objects."""
    archive = Path(__file__).resolve().parent
    root = archive.parents[2]
    receipt = json.loads((archive / 'receipt.json').read_text())
    rows = [*receipt['sources'].values(), receipt['patch'], receipt['forensics'],
            *receipt['phases']]
    for row in rows:
        packed = (archive / row['payload']).read_bytes()
        raw = gzip.decompress(packed)
        assert hashlib.sha256(packed).hexdigest() == row['sha256']
        assert hashlib.sha256(raw).hexdigest() == row['raw_sha256']
        if 'terminal_summary' in row:
            assert row['terminal_summary'].encode() in raw
    compared = 0
    for name, row in receipt['sources'].items():
        try:
            expected = subprocess.check_output(
                ['git', 'show', f"{receipt['checkpoint']}:{name}"],
                cwd=root, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue
        assert gzip.decompress((archive / row['payload']).read_bytes()) == expected
        compared += 1
    print(f'PASS: {len(rows)} payloads; {compared} frozen Git comparisons')
    if compared != len(receipt['sources']):
        print('Private source unavailable: Git comparisons remain incomplete.')


if __name__ == '__main__':
    main()
