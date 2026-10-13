"""Verify every frozen payload hash, optionally using committed Git blobs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

base = Path(__file__).resolve().parent
manifest = json.loads((base / "manifest.json").read_text())
for name, expected in manifest["payloads"].items():
    if "--git" in sys.argv:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=base, text=True).strip()
        relative = (base / name).relative_to(root).as_posix()
        data = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=base)
    else:
        data = (base / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, name
print(f"Verified {len(manifest['payloads'])} payloads")
