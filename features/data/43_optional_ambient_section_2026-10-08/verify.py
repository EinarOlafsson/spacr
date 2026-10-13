"""Verify the frozen optional-ambient Section evidence."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import zipfile


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
        original = gzip.decompress(stored) if name.endswith(".gz") else stored
        assert len(original) == expected["original_bytes"], name
        assert hashlib.sha256(original).hexdigest() == expected["original_sha256"], name

    archive = _payload("logs/hosted-e41-cancelled-serial.zip", git)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        assert zf.testzip() is None
        provenance = json.loads(zf.read("source-provenance.json"))
    assert provenance["source_revision"] == receipt["hosted_source_revision"]
    assert (provenance["source_sha256"]["spacr/qt/widgets/section.py"]
            == receipt["hosted_section_sha256"])
    assert (receipt["manifest"]["sources/section-hosted-e41.py.gz"]
            ["original_sha256"] == receipt["hosted_section_sha256"])

    if current:
        for name, source in (
            ("sources/section-current.py.gz", "spacr/qt/widgets/section.py"),
            ("sources/test-ambient-wiring-current.py.gz", "tests/qt/test_ambient_wiring.py"),
            ("sources/preferences-current.py.gz", "spacr/qt/preferences.py"),
        ):
            assert gzip.decompress(_payload(name, git)) == (ROOT / source).read_bytes(), source
    print(f"verified {len(receipt['manifest'])} payloads; git={git}; current={current}")


if __name__ == "__main__":
    main()
