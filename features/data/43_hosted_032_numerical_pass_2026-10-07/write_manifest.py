"""Generate the source-bound hosted coverage receipt from preserved payloads."""

from __future__ import annotations

import gzip
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_SHA = "0324166b59da0ba4c31e01cf108087f05d3e5992"
RUN_ID = 37573419011
ARTIFACT_SHA = "59843ffbbe9f15f9b5db3c814ab32a1d386fead2afa77669e4585be72553a06e"
PAYLOADS = (
    "11464421456-coverage-report.zip",
    "run-37573419011-terminal.json",
    "jobs-37573419011-terminal.json",
    "logs/112637317392-032.log.gz",
    "logs/112637317452-032.log.gz",
    "logs/112637317499-032.log.gz",
    "logs/112637317611-032.log.gz",
    "logs/112637317654-032.log.gz",
    "logs/112637317676-032.log.gz",
    "logs/112637317732-032.log.gz",
    "logs/112637317836-032.log.gz",
    "logs/112656808371-032.log.gz",
    "logs/112682224404-032.log.gz",
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    """Bind every archived byte stream to the exact hosted source and result."""
    run = json.loads((ROOT / "run-37573419011-terminal.json").read_text(encoding="utf-8"))
    jobs = json.loads((ROOT / "jobs-37573419011-terminal.json").read_text(encoding="utf-8"))["jobs"]
    assert run["id"] == RUN_ID and run["head_sha"] == SOURCE_SHA
    assert (run["status"], run["conclusion"]) == ("completed", "failure")
    assert len(jobs) == 29 and all(job["status"] == "completed" for job in jobs)
    failures = {job["id"] for job in jobs if job["conclusion"] == "failure"}
    successes = {job["id"] for job in jobs if job["conclusion"] == "success"}
    assert len(failures) == 10 and len(successes) == 19
    archived_failures = {
        int(Path(path).name.split("-")[0]) for path in PAYLOADS if path.endswith(".log.gz")
    }
    assert archived_failures == failures

    archive = ROOT / "11464421456-coverage-report.zip"
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
        "artifact_id": 11464421456,
        "artifact_sha256": ARTIFACT_SHA,
        "numerical_ratchet": "pass",
        "jobs": {"success": 19, "failure": 10},
        "files": files,
    }
    (ROOT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
