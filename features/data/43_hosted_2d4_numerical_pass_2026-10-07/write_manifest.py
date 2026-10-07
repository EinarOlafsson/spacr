"""Generate the source-bound hosted coverage receipt from preserved payloads."""

from __future__ import annotations

import gzip
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_SHA = "2d4f8f1c2bae3cf8b3914be3b1c670147a9f3ccf"
RUN_ID = 37574709966
ARTIFACT_SHA = "1088ca8cd4b3ca1a2eabd98f65af1574570ffa886e507eeca1a9c327bf34cef5"
PAYLOADS = (
    "11471310982-coverage-report.zip",
    "run-37574709966-terminal.json",
    "jobs-37574709966-terminal.json",
    "logs/112682260766-fast1.log.gz",
    "logs/112682260779-mindeps1.log.gz",
    "logs/112682260783-fast0.log.gz",
    "logs/112682260855-coverage6.log.gz",
    "logs/112682260858-mindeps0.log.gz",
    "logs/112682260873-coverage11.log.gz",
    "logs/112682260894-coverage4.log.gz",
    "logs/112682260933-coverage9.log.gz",
    "logs/112682260962-coverage0.log.gz",
    "logs/112682260981-coverage7.log.gz",
    "logs/112682261014-qt0.log.gz",
    "logs/112682261038-qt2.log.gz",
    "logs/112712053695-aggregate.log.gz",
    "logs/112734409824-release.log.gz",
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    """Bind every archived byte stream to the exact hosted source and result."""
    run = json.loads((ROOT / "run-37574709966-terminal.json").read_text(encoding="utf-8"))
    jobs = json.loads((ROOT / "jobs-37574709966-terminal.json").read_text(encoding="utf-8"))["jobs"]
    assert run["id"] == RUN_ID and run["head_sha"] == SOURCE_SHA
    assert (run["status"], run["conclusion"]) == ("completed", "failure")
    assert len(jobs) == 29 and all(job["status"] == "completed" for job in jobs)
    failures = {job["id"] for job in jobs if job["conclusion"] == "failure"}
    successes = {job["id"] for job in jobs if job["conclusion"] == "success"}
    assert len(failures) == 14 and len(successes) == 15
    archived_failures = {
        int(Path(path).name.split("-")[0]) for path in PAYLOADS if path.endswith(".log.gz")
    }
    assert archived_failures == failures

    archive = ROOT / "11471310982-coverage-report.zip"
    assert _digest(archive.read_bytes()) == ARTIFACT_SHA
    with zipfile.ZipFile(archive) as zf:
        ratchet = json.loads(zf.read("module-coverage-ratchet.json"))
    summary = ratchet["summary"]
    integrity = ratchet["measurement_integrity"]
    assert ratchet["status"] == "pass"
    assert summary["modules_checked"] == summary["shipped_modules"] == 664
    assert summary["failed_modules"] == summary["unconfirmed_modules"] == 0
    assert summary["stale_baseline_entries"] == 0
    assert integrity["status"] == "complete" and integrity["shards_with_records"] == 12

    files = {}
    for relative in PAYLOADS:
        stored = (ROOT / relative).read_bytes()
        raw = gzip.decompress(stored) if relative.endswith(".gz") else stored
        files[relative] = {
            "stored_size": len(stored),
            "stored_sha256": _digest(stored),
            "raw_size": len(raw),
            "raw_sha256": _digest(raw),
        }
    manifest = {
        "schema": 1,
        "source_sha": SOURCE_SHA,
        "run_id": RUN_ID,
        "run_conclusion": "failure",
        "artifact_id": 11471310982,
        "artifact_sha256": ARTIFACT_SHA,
        "numerical_ratchet": "pass",
        "jobs": {"success": 15, "failure": 14},
        "files": files,
    }
    (ROOT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
