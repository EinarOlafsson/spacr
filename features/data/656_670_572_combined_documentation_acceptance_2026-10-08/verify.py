from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import tempfile

archive = Path(__file__).resolve().parent
receipt = json.loads((archive / 'receipt.json').read_text())
for row in receipt['payloads']:
    stored = (archive / row['path']).read_bytes()
    assert len(stored) == row['bytes'] and hashlib.sha256(stored).hexdigest() == row['sha256']
    if row['encoding'] == 'gzip-unified-diff':
        basis = subprocess.check_output(['git', 'show', row['basis_git_object']])
        assert hashlib.sha256(basis).hexdigest() == row['basis_raw_sha256']
        patch = gzip.decompress(stored)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            before, after = directory / 'accepted.json', directory / 'original.json'
            before.write_bytes(basis)
            if patch:
                result = subprocess.run(['patch', '--silent', '--output', str(after), str(before)], input=patch, capture_output=True)
                assert result.returncode == 0, result.stderr
                raw = after.read_bytes()
            else:
                raw = basis
        if row['reconstruct_original_gzip']:
            raw = gzip.compress(raw, mtime=0)
            raw = bytes.fromhex(row["original_gzip_header_hex"]) + raw[10:]
    else:
        raw = gzip.decompress(stored) if row['encoding'] == 'gzip' else stored
    assert len(raw) == row['raw_bytes'] and hashlib.sha256(raw).hexdigest() == row['raw_sha256'], row['path']
print('PASS', len(receipt['payloads']), 'portable payloads and exact original snapshot reconstructions.')
