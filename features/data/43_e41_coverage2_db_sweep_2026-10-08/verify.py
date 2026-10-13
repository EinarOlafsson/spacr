"""Verify the frozen e41 Coverage2 database and sweep evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = HERE.relative_to(ROOT).as_posix()


def _payload(name: str, git: bool) -> bytes:
    if git:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"HEAD:{PREFIX}/{name}"]
        )
    return (HERE / name).read_bytes()


def main() -> None:
    git = "--git" in sys.argv[1:]
    current = "--current" in sys.argv[1:]
    receipt = json.loads(_payload("receipt.json", git))
    for name, expected in receipt["manifest"].items():
        stored = _payload(name, git)
        assert len(stored) == expected["bytes"], name
        assert hashlib.sha256(stored).hexdigest() == expected["sha256"], name
        original = gzip.decompress(stored)
        assert len(original) == expected["original_bytes"], name
        assert hashlib.sha256(original).hexdigest() == expected["original_sha256"], name

    for source in receipt["source_paths"]:
        key = source.replace("/", "__")
        old = gzip.decompress(_payload(f"sources/e41__{key}.gz", git))
        expected = subprocess.check_output(
            ["git", "-C", str(ROOT), "show",
             f"{receipt['hosted_revision']}:{source}"]
        )
        assert old == expected, source
        if current:
            final = gzip.decompress(_payload(f"sources/final__{key}.gz", git))
            assert final == (ROOT / source).read_bytes(), source
    print(f"verified {len(receipt['manifest'])} payloads; git={git}; current={current}")


if __name__ == "__main__":
    main()
