"""Verify frozen F572 merged-source evidence and optional current source."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    for name, expected in manifest["payloads"].items():
        path = HERE / name
        assert path.is_file() and sha256(path.read_bytes()) == expected, name
    if "--git" in sys.argv:
        for name, expected in manifest["payloads"].items():
            relative = (HERE / name).relative_to(ROOT)
            data = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"HEAD:{relative}"])
            assert sha256(data) == expected, str(relative)
    if "--source" in sys.argv:
        receipt = json.loads((HERE / "receipt.json").read_text())
        for name, expected in receipt["source_sha256"].items():
            assert sha256((ROOT / name).read_bytes()) == expected, name
    print(f"verified {len(manifest['payloads'])} payloads")


if __name__ == "__main__":
    main()
