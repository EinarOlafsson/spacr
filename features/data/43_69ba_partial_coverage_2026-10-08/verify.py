"""Verify the frozen source and cancellation evidence for the 69ba run."""

import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ARCHIVE = Path(__file__).resolve().parent
SOURCE = "69ba4e462f2ea60924a977b192d006ce9b70f42c"
RUN = 37737743925


def _git(*args):
    return subprocess.check_output(("git", *args), cwd=ARCHIVE)


def _read(name, from_git):
    if not from_git:
        return (ARCHIVE / name).read_bytes()
    root = Path(_git("rev-parse", "--show-toplevel").decode().strip())
    relative = ARCHIVE.relative_to(root).as_posix()
    return _git("show", f"HEAD:{relative}/{name}")


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
        files = {path.name for path in ARCHIVE.iterdir() if path.is_file()}
        assert files == names | {"MANIFEST.json"}

    receipt = json.loads(_read("receipt.json", from_git))
    assert receipt["source_sha"] == SOURCE and receipt["run_id"] == RUN
    before = json.loads(_read("run-before.json", from_git))
    after = json.loads(_read("run-cancelled.json", from_git))
    assert before["id"] == after["id"] == RUN
    assert before["head_sha"] == after["head_sha"] == SOURCE
    assert after["status"] == "completed" and after["conclusion"] == "cancelled"
    assert receipt["coverage_success"] == 1
    assert receipt["coverage_cancelled"] == 11
    assert receipt["combine"] == "failed on missing shard 5 and 11 inputs"
    final_jobs = json.loads(_read("jobs-final.json", from_git))["jobs"]
    by_id = {job["id"]: job for job in final_jobs}
    combine_job = json.loads(_read("combine-job.json", from_git))
    release_job = json.loads(_read("release-job.json", from_git))
    for job in (combine_job, release_job):
        assert job["run_id"] == RUN and job["head_sha"] == SOURCE
        assert job["conclusion"] == "failure"
        assert by_id[job["id"]]["conclusion"] == "failure"
    assert combine_job["id"] == receipt["combine_job_id"]
    assert release_job["id"] == receipt["release_job_id"]
    assert next(step for step in combine_job["steps"] if step["name"] == "Gate on the per-module coverage ratchet baseline")["conclusion"] == "skipped"
    combine_log = gzip.decompress(_read("combine.log.gz", from_git))
    assert b"coverage shard 5 produced no coverage data" in combine_log
    assert b"coverage shard 11 produced no coverage data" in combine_log
    assert b"coverage shards finished with cancelled" in combine_log
    release_log = gzip.decompress(_read("release.log.gz", from_git))
    assert b"coverage-combine finished with failure" in release_log
    assert b"qt finished with cancelled" in release_log
    for row in receipt["coverage_shards"]:
        number = row["number"]
        job = json.loads(_read(f"coverage{number}-job.json", from_git))
        assert job["id"] == row["job_id"]
        assert job["run_id"] == RUN and job["head_sha"] == SOURCE
        assert job["conclusion"] == row["conclusion"]
    assert {row["number"] for row in receipt["coverage_shards"]} == set(range(12))
    assert receipt["available_log_shards"] == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    good = gzip.decompress(_read("coverage6.log.gz", from_git))
    batches = re.findall(rb"=+ (\d+) passed(?:, (\d+) skipped)?[^\n]* =+", good)
    assert len(batches) == 10
    assert sum(int(passed) for passed, _ in batches) == 6875
    assert sum(int(skipped or 0) for _, skipped in batches) == 30
    assert b"spacr-coverage-data-37737743925-1-6" in good
    native = gzip.decompress(_read("coverage0.log.gz", from_git))
    assert b"Fatal Python error: Segmentation fault" in native
    assert b"test_native_sources_changed_during_detection_cannot_publish_a_result[parent]" in native
    replay = gzip.decompress(_read("coverage9.log.gz", from_git))
    assert b"assert 8 == 7" in replay and b"assert 6 == 5" in replay
    print("69ba coverage: one passed shard, eleven cancelled, combine failed on missing data, no numerical verdict")


if __name__ == "__main__":
    main()
