"""Verify the source-bound two-attempt Qt diagnostic receipt from raw evidence."""

from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import gzip
import json
import re
import sys


RUN_ID = 37782034011
JOB_ID = 113327271680
WORKFLOW_SOURCE = "1d7da13f6aa9d0c14333eb7d1ab5331fd70d1e5c"
TEST_SOURCE = "7a51b6c921ea0d9b51ca3f5e6d28a68904ce2cab"
COLLECTOR_SOURCE = "9e7b35c8abb4cf7d72cf1fa546322472e0c18819"
BATCH_SHA256 = "d5f347bdb25b692cf207f18526b333d01e58a952a923842881367ad77ff65826"
COVERAGE_ARTIFACT_ID = 11553571256
COVERAGE_ARTIFACT_SHA256 = "874989fb9854da8e4e3043c4299ca1601640a3ba1cb5ceb5a2e7442f2d3557c0"


def _sha(value: bytes) -> str:
    """Return a stable digest for raw archive bytes."""
    return sha256(value).hexdigest()


def build_manifest(directory: Path) -> dict:
    """Check both attempts, terminal status, source identity and integrity."""
    job = json.loads((directory / "job.json").read_text())
    run = json.loads((directory / "run.json").read_text())
    assert job["id"] == JOB_ID and job["run_id"] == RUN_ID
    assert run["id"] == RUN_ID
    assert job["head_sha"] == run["head_sha"] == WORKFLOW_SOURCE
    assert job["status"] == run["status"] == "completed"
    assert job["conclusion"] == run["conclusion"] == "success"
    steps = {step["name"]: step["conclusion"] for step in job["steps"]}
    assert steps["Replay the original-order coverage batch twice independently"] == "success"
    assert steps["Recover a failed worker's bounded native evidence"] == "skipped"
    assert steps["Upload native failure evidence"] == "skipped"
    compressed = (directory / "job.log.gz").read_bytes()
    raw = gzip.decompress(compressed)
    text = raw.decode("utf-8", errors="replace")
    summaries = [(int(passed), int(warnings), float(seconds)) for passed, warnings, seconds
                 in re.findall(r"(\d+) passed, (\d+) warnings in ([\d.]+)s", text)]
    assert len(summaries) == 2
    assert [row[:2] for row in summaries] == [(503, 38), (503, 38)]
    assert "Fatal Python error: Segmentation fault" not in text
    identity_bytes = (directory / "identity.zip").read_bytes()
    with ZipFile(directory / "identity.zip") as archive:
        batch = archive.read("batch-3-files.txt")
        assert _sha(batch) == BATCH_SHA256 and len(batch.splitlines()) == 32
        provenance = archive.read("source-and-runtime.txt").decode()
        assert f"replay_source_sha={TEST_SOURCE}" in provenance
        assert f"diagnostic_collector_sha={COLLECTOR_SOURCE}" in provenance
        assert f"workflow_dispatch_sha={WORKFLOW_SOURCE}" in provenance
        assert "Python 3.12.15" in provenance and "PySide6 6.12.0 Qt 6.12.0" in provenance
        attempts = archive.read("attempts.txt").decode().splitlines()
        assert attempts == [
            f"attempt={attempt} status={status} source={TEST_SOURCE}"
            for attempt in (1, 2) for status in ("started", "success")
        ]
    integrity = []
    files = ("job.log.gz", "identity.zip", "job.json", "run.json")
    for attempt in (1, 2):
        integrity_name = f"attempt-{attempt}-integrity.json"
        ledger_name = f"attempt-{attempt}-ledger.json"
        files += (integrity_name, ledger_name)
        record = json.loads((directory / integrity_name).read_text())
        ledger = json.loads((directory / ledger_name).read_text())
        assert record["shard_index"] == 0 and record["shard_count"] == 1
        assert record["batches_total"] == record["batches_finished"] == 1
        batch_record = record["batches"][0]
        assert batch_record["batch"] == 1 and batch_record["exit_code"] == 0
        assert all(batch_record[name] == [] for name in (
            "lost", "lost_files", "recovered_files", "unrecovered_files", "discarded_data_files"))
        assert ledger["schema"].startswith("spacr.")
        integrity.append({"attempt": attempt, "batch_exit_code": 0,
                          "pytest_passed": summaries[attempt - 1][0],
                          "pytest_warnings": summaries[attempt - 1][1],
                          "pytest_seconds": summaries[attempt - 1][2]})
    return {
        "run_id": RUN_ID,
        "job_id": JOB_ID,
        "job_started_at": job["started_at"],
        "job_completed_at": job["completed_at"],
        "job_conclusion": job["conclusion"],
        "workflow_sha": WORKFLOW_SOURCE,
        "test_source_sha": TEST_SOURCE,
        "collector_sha": COLLECTOR_SOURCE,
        "batch_files_sha256": BATCH_SHA256,
        "attempts": integrity,
        "native_recovery_and_upload": "skipped: no diagnostic fault",
        "coverage_artifact_id": COVERAGE_ARTIFACT_ID,
        "coverage_artifact_original_sha256": COVERAGE_ARTIFACT_SHA256,
        "coverage_artifact_scope": "Original raw coverage ZIP remains in GitHub Actions; compact integrity and ledger JSON are preserved here.",
        "files_sha256": {name: _sha((directory / name).read_bytes()) for name in files},
        "log_original_sha256": _sha(raw),
        "log_original_bytes": len(raw),
        "identity_original_sha256": _sha(identity_bytes),
        "interpretation": "Two successful isolated diagnostic attempts are intermittent controls, not a native-cause fix or required-suite acceptance.",
    }


def main() -> None:
    """Write a manifest once, then verify every raw receipt against it."""
    directory = Path(__file__).resolve().parent
    computed = build_manifest(directory)
    path = directory / "MANIFEST.json"
    if len(sys.argv) == 2 and sys.argv[1] == "--write":
        path.write_text(json.dumps(computed, indent=2, sort_keys=True) + "\n")
    else:
        assert len(sys.argv) == 1
        assert json.loads(path.read_text()) == computed
    print("two attempts: 503 passed and 38 warnings each; diagnostic success only")


if __name__ == "__main__":
    main()
