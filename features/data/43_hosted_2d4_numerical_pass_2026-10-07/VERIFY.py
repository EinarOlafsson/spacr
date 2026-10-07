"""Verify hosted 2d4 payload bytes and the numerical ratchet conclusion."""

from __future__ import annotations

import gzip
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
for relative, record in manifest["files"].items():
    stored = (ROOT / relative).read_bytes()
    assert len(stored) == record["stored_size"], relative
    assert hashlib.sha256(stored).hexdigest() == record["stored_sha256"], relative
    raw = gzip.decompress(stored) if relative.endswith(".gz") else stored
    assert len(raw) == record["raw_size"], relative
    assert hashlib.sha256(raw).hexdigest() == record["raw_sha256"], relative
run = json.loads((ROOT / "run-37574709966-terminal.json").read_text(encoding="utf-8"))
jobs = json.loads((ROOT / "jobs-37574709966-terminal.json").read_text(encoding="utf-8"))["jobs"]
assert run["head_sha"] == manifest["source_sha"]
assert (run["status"], run["conclusion"]) == ("completed", "failure")
assert len([job for job in jobs if job["conclusion"] == "success"]) == 15
assert len([job for job in jobs if job["conclusion"] == "failure"]) == 14
archive = ROOT / "11471310982-coverage-report.zip"
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["artifact_sha256"]
with zipfile.ZipFile(archive) as zf:
    ratchet = json.loads(zf.read("module-coverage-ratchet.json"))
assert ratchet["status"] == manifest["numerical_ratchet"] == "pass"
assert ratchet["measurement_integrity"]["status"] == "complete"
print(f"verified {len(manifest['files'])} hosted payloads for {manifest['source_sha']}")
