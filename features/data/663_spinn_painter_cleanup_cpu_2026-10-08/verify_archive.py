"""Verify the frozen N663 painter-cleanup source and test receipts."""

import hashlib
import gzip
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _digest(data):
    """Return the SHA-256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def _git_file(commit, path):
    """Read one source-bound Git blob without changing a checkout."""
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT
    )


def main():
    """Reject a changed log, test snapshot, or source binding."""
    receipt = json.loads((HERE / "receipt.json").read_text())
    for name, expected in receipt["logs"].items():
        compressed = (HERE / name).read_bytes()
        assert _digest(compressed) == expected["sha256"], name
        content = gzip.decompress(compressed)
        assert _digest(content) == expected["raw_sha256"], name
        assert expected["result"].encode() in content, name
    test = (HERE / "test_snapshot.py").read_bytes()
    assert _digest(test) == receipt["test_sha256"]
    assert test == _git_file(receipt["fixed_source_commit"], receipt["test_path"])
    for kind in ("original", "fixed"):
        source = _git_file(receipt[f"{kind}_source_commit"], receipt["source_path"])
        assert _digest(source) == receipt[f"{kind}_source_sha256"]
    print("N663 painter archive: four logs and three source/test bindings verified")


if __name__ == "__main__":
    main()
