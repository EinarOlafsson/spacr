"""Archive the exact hosted puncta crash and bounded negative local replays."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
SCRATCH = Path("/mnt/wd4tb/scratch")
HOSTED = SCRATCH / "ci-224-failures-20261007/112567504138-392.log.gz"
REPLAY = SCRATCH / "qt392-puncta-replay"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Copy complete logs and write a deterministic content manifest."""
    shutil.copyfile(HOSTED, HERE / "hosted-112567504138.log.gz")
    for source, target in (
        ("exact-batch59.log", "batch59.log.gz"),
        ("order58-59.log", "batches58-59.log.gz"),
    ):
        (HERE / target).write_bytes(
            gzip.compress((REPLAY / source).read_bytes(), compresslevel=9, mtime=0)
        )
    source_receipt = json.loads((REPLAY / "replay-receipt.json").read_text())
    (HERE / "replay-receipt.json").write_text(
        json.dumps(source_receipt, indent=2, sort_keys=True) + "\n"
    )
    payloads = ("hosted-112567504138.log.gz", "batch59.log.gz",
                "batches58-59.log.gz", "replay-receipt.json", "README.md")
    manifest = {
        "schema": "spacr.qt.puncta-negative-replay-archive/v1",
        "hosted_source": "392ca4d6a2319fd88c2d352b1a0b021682123d6b",
        "hosted_run": 37550980965,
        "hosted_job": 112567504138,
        "payloads": {
            name: {"sha256": _digest(HERE / name), "bytes": (HERE / name).stat().st_size}
            for name in payloads
        },
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
