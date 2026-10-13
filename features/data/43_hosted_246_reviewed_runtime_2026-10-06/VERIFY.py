"""Verify the exact-source reviewed-runtime evidence payloads."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
for relative, record in manifest["files"].items():
    path = ROOT / relative
    compressed = path.read_bytes()
    assert len(compressed) == record["stored_size"], relative
    assert hashlib.sha256(compressed).hexdigest() == record["stored_sha256"], relative
    raw = gzip.decompress(compressed) if relative.endswith(".gz") else compressed
    assert len(raw) == record["raw_size"], relative
    assert hashlib.sha256(raw).hexdigest() == record["raw_sha256"], relative
print(f"verified {len(manifest['files'])} files for {manifest['source_sha']}")
