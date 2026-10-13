"""Verify the source-bound complete coverage phase and its separate verdicts."""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from io import BytesIO


ARCHIVE = Path(__file__).resolve().parent
SOURCE = "0b8c2c4120a0ede4de568d484ea3fc9d517f1de3"
RUN = 37741615330


def _git(*args):
    return subprocess.check_output(("git", *args), cwd=ARCHIVE).decode().strip()


def _read(name, from_git):
    if not from_git:
        return (ARCHIVE / name).read_bytes()
    root = Path(_git("rev-parse", "--show-toplevel"))
    relative = ARCHIVE.relative_to(root).as_posix()
    return subprocess.check_output(("git", "show", f"HEAD:{relative}/{name}"), cwd=ARCHIVE)


def main():
    from_git = sys.argv[1:] == ["--git"]
    if sys.argv[1:] and not from_git:
        raise SystemExit("usage: python verify.py [--git]")
    manifest = json.loads(_read("MANIFEST.json", from_git))
    assert manifest["source_sha"] == SOURCE
    names = set()
    for row in manifest["payloads"]:
        data = _read(row["path"], from_git)
        assert len(data) == row["bytes"]
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        names.add(row["path"])
    assert len(names) == len(manifest["payloads"])
    if not from_git:
        assert {path.name for path in ARCHIVE.iterdir() if path.is_file()} == names | {"MANIFEST.json"}

    receipt = json.loads(_read("receipt.json", from_git))
    assert receipt["source_sha"] == SOURCE
    assert receipt["source_tree"] == _git("rev-parse", SOURCE + "^{tree}")
    assert receipt["run_id"] == RUN and receipt["run_attempt"] == 1
    for path, digest in receipt["workflow_blobs"].items():
        assert digest == _git("rev-parse", f"{SOURCE}:{path}")
    for path, digest in receipt["guard_blobs"].items():
        assert digest == _git("rev-parse", f"{SOURCE}:{path}")
    run = json.loads(_read("run-final.json", from_git))
    jobs = json.loads(_read("jobs-final.json", from_git))["jobs"]
    assert run["id"] == RUN and run["head_sha"] == SOURCE
    assert run["status"] == receipt["run_snapshot_status"]
    jobs_by_id = {job["id"]: job for job in jobs}
    assert len(jobs_by_id) == len(jobs)

    shards = receipt["coverage_shards"]
    assert {row["number"] for row in shards} == set(range(12))
    assert sum(row["conclusion"] == "success" for row in shards) == receipt["coverage_success"] == 11
    assert sum(row["conclusion"] == "failure" for row in shards) == receipt["coverage_failure"] == 1
    for row in shards:
        number = row["number"]
        job = json.loads(_read(f"coverage{number}-job.json", from_git))
        assert job["id"] == row["job_id"]
        assert job["run_id"] == RUN and job["head_sha"] == SOURCE
        assert job["conclusion"] == row["conclusion"] == jobs_by_id[job["id"]]["conclusion"]
        assert job["completed_at"] == row["completed_at"]
        raw = gzip.decompress(_read(f"coverage{number}.log.gz", from_git))
        assert hashlib.sha256(raw).hexdigest() == row["raw_log_sha256"]
        assert f"spacr-coverage-data-{RUN}-1-{number}".encode() in raw
    failed = gzip.decompress(_read("coverage9.log.gz", from_git))
    assert b"assert 8 == 7" in failed and b"assert 6 == 5" in failed
    for node in receipt["known_failure_tests"]:
        assert node.encode() in failed
    assert b"Fatal Python error" not in failed

    combine = json.loads(_read("combine-job.json", from_git))
    assert combine["id"] == receipt["combine_job_id"]
    assert combine["run_id"] == RUN and combine["head_sha"] == SOURCE
    assert combine["conclusion"] == receipt["combine_conclusion"] == "failure"
    assert jobs_by_id[combine["id"]]["conclusion"] == "failure"
    steps = {step["name"]: step["conclusion"] for step in combine["steps"]}
    assert {name: steps[name] for name in receipt["combine_step_results"]} == receipt["combine_step_results"]
    assert steps["Combine process data and write coverage.py JSON"] == "success"
    assert steps["Gate on the per-module coverage ratchet baseline"] == "success"
    assert steps["Require every coverage shard to have passed its tests"] == "failure"
    combine_log = gzip.decompress(_read("combine.log.gz", from_git))
    assert b"spaCR shipped-module coverage ratchet: PASS" in combine_log
    assert b"coverage shards finished with failure" in combine_log

    archive = _read("module-coverage-report.zip", from_git)
    assert hashlib.sha256(archive).hexdigest() == receipt["report_artifact"]["zip_sha256"]
    assert receipt["report_artifact"]["id"] == 11539995050
    with zipfile.ZipFile(BytesIO(archive)) as report_zip:
        assert report_zip.testzip() is None
        assert set(report_zip.namelist()) == {
            "coverage.json", "module-coverage-ratchet.json", "module-coverage-ratchet.txt"
        }
        numerical_bytes = report_zip.read("module-coverage-ratchet.json")
    assert hashlib.sha256(numerical_bytes).hexdigest() == receipt["report_artifact"]["ratchet_json_sha256"]
    numerical = json.loads(numerical_bytes)
    assert numerical["status"] == receipt["numerical_status"] == "pass"
    assert numerical["summary"] == receipt["numerical_summary"]
    assert numerical["summary"]["shipped_modules"] == numerical["summary"]["modules_checked"] == 664
    assert numerical["summary"]["failed_modules"] == numerical["summary"]["unconfirmed_modules"] == 0
    assert numerical["measurement_integrity"]["status"] == "complete"
    assert numerical["measurement_integrity"]["shard_count"] == numerical["measurement_integrity"]["shards_with_records"] == 12
    assert numerical["measurement_integrity"]["recovered"] == []
    assert numerical["measurement_integrity"]["issues"] == []
    print("0b8 coverage: complete 12-shard numerical PASS; 11 selected-test shards pass, one known-profile shard fails")


if __name__ == "__main__":
    main()
