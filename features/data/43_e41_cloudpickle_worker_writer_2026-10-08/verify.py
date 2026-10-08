"""Verify the e41 worker-writer cloudpickle compatibility evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = HERE.relative_to(ROOT).as_posix()


def payload(name: str, git: bool) -> bytes:
    if git:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"HEAD:{PREFIX}/{name}"]
        )
    return (HERE / name).read_bytes()


def main() -> None:
    git = "--git" in sys.argv[1:]
    current = "--current" in sys.argv[1:]
    receipt = json.loads(payload("receipt.json", git))
    for name, expected in receipt["manifest"].items():
        packed = payload(name, git)
        assert len(packed) == expected["bytes"], name
        assert hashlib.sha256(packed).hexdigest() == expected["sha256"], name
        original = gzip.decompress(packed)
        assert len(original) == expected["original_bytes"], name
        assert hashlib.sha256(original).hexdigest() == expected["original_sha256"], name

    for source in receipt["source_paths"]:
        key = source.replace("/", "__")
        for label, revision in (("hosted", receipt["hosted_revision"]),
                                ("fixed", receipt["fixed_revision"])):
            archived = gzip.decompress(payload(f"sources/{label}__{key}.gz", git))
            committed = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"{revision}:{source}"]
            )
            assert archived == committed, (label, source)
            if label == "fixed" and current:
                assert archived == (ROOT / source).read_bytes(), source
    print(f"verified {len(receipt['manifest'])} payloads; git={git}; current={current}")


if __name__ == "__main__":
    main()
