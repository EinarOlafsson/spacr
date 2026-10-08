"""Verify the immutable Spaceout Preferences hotfix receipt."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    receipt = json.loads((HERE / "receipt.json").read_text(encoding="utf-8"))
    manifest = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        data = (HERE / name).read_bytes()
        assert _sha(data) == expected, name
        if "--git" in sys.argv:
            relative = (HERE / name).relative_to(ROOT).as_posix()
            tracked = subprocess.check_output(
                ["git", "show", f"HEAD:{relative}"], cwd=ROOT)
            assert data == tracked, relative
    for name, record in receipt["payloads"].items():
        data = gzip.decompress((HERE / name).read_bytes())
        assert _sha(data) == record["raw_sha256"], name
        if "git_blob" in record:
            actual = subprocess.check_output(
                ["git", "hash-object", "--stdin"], input=data, cwd=ROOT,
                text=False).decode().strip()
            assert actual == record["git_blob"], name
    print(f"verified {len(manifest)} tracked payloads and "
          f"{len(receipt['payloads'])} raw snapshots")


if __name__ == "__main__":
    main()
