from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
for name, record in manifest['payloads'].items():
    data = (root / name).read_bytes()
    assert len(data) == record['bytes'], name
    assert hashlib.sha256(data).hexdigest() == record['sha256'], name
print(len(manifest['payloads']), 'source-bound archive payloads verified')
