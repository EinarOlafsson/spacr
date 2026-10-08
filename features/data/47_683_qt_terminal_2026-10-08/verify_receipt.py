"""Verify the archived Qt job bytes and their Git source objects."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Reject missing or changed evidence and mismatched source blobs."""
    directory = Path(__file__).resolve().parent
    repository = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    receipt = json.loads((directory / "receipt.json").read_text())
    for item in receipt["files"]:
        source = directory / item["path"]
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise SystemExit(f"archive hash mismatch: {item['path']}")
        if "raw_sha256" in item:
            raw = gzip.decompress(payload)
            if hashlib.sha256(raw).hexdigest() != item["raw_sha256"]:
                raise SystemExit(f"raw log hash mismatch: {item['path']}")
    for item in receipt["git_blobs"]:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse",
             f"{item['commit']}:{item['path']}"],
            check=True, capture_output=True, text=True,
        )
        if result.stdout.strip() != item["blob"]:
            raise SystemExit(f"Git blob mismatch: {item['commit']}:{item['path']}")
    print(f"verified {len(receipt['files'])} files and "
          f"{len(receipt['git_blobs'])} Git blobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
