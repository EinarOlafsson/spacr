"""Verify the frozen F288 floor audit without rerunning a full suite."""

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
            relative = path.relative_to(ROOT)
            data = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"HEAD:{relative}"])
            assert sha256(data) == expected, str(relative)
    receipt = json.loads((HERE / "receipt.json").read_text())
    report = ROOT / receipt["hosted_report"]
    assert sha256(report.read_bytes()) == receipt["hosted_report_sha256"]
    if "--source" in sys.argv:
        for name, expected in receipt["current_file_sha256"].items():
            assert sha256((ROOT / name).read_bytes()) == expected, name
    print(f"verified {len(manifest['payloads'])} payloads and hosted report")


if __name__ == "__main__":
    main()
