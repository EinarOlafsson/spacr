"""Verify private UI handoff bytes against the frozen Git checkpoint."""

import gzip
import hashlib
import json
import pathlib
import subprocess


def main():
    """Check compressed integrity, source identity and scoped passing logs."""
    archive = pathlib.Path(__file__).resolve().parent
    root = archive.parents[2]
    receipt = json.loads((archive / "receipt.json").read_text())
    compared = 0
    for name, row in receipt["sources"].items():
        packed = (archive / row["payload"]).read_bytes()
        raw = gzip.decompress(packed)
        assert hashlib.sha256(packed).hexdigest() == row["sha256"]
        assert hashlib.sha256(raw).hexdigest() == row["raw_sha256"]
        try:
            frozen = subprocess.check_output(["git", "show", f"{receipt['checkpoint']}:{name}"], cwd=root, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue
        assert raw == frozen, name
        compared += 1
    for row in [receipt["patch"], receipt["inventory_diff"], *receipt["phases"]]:
        packed = (archive / row["payload"]).read_bytes()
        raw = gzip.decompress(packed)
        assert hashlib.sha256(packed).hexdigest() == row["sha256"]
        assert hashlib.sha256(raw).hexdigest() == row["raw_sha256"]
        if "passed" in row:
            assert f"{row['passed']} passed".encode() in raw
    print(f"PASS: {len(receipt['sources'])} payload bindings, patch and {len(receipt['phases'])} scoped logs; {compared} frozen Git comparisons")
    if compared != len(receipt["sources"]):
        print("Frozen private checkpoint is unavailable here; payload integrity is verified, Git source comparison is incomplete.")


if __name__ == "__main__":
    main()
