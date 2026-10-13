"""Verify the frozen source and hosted payloads for the 4133 coverage phase."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile


ARCHIVE = Path(__file__).resolve().parent
SOURCE = "4133beafcd0ae427795617a4a01e295fd40539b7"
RUN = 37728397690


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
    for item in manifest["payloads"]:
        name = item["path"]
        data = _read(name, from_git)
        assert len(data) == item["bytes"], name
        assert hashlib.sha256(data).hexdigest() == item["sha256"], name
        names.add(name)
    assert len(names) == len(manifest["payloads"])
    if not from_git:
        files = {path.name for path in ARCHIVE.iterdir() if path.is_file()}
        assert files == names | {"MANIFEST.json"}

    receipt = json.loads(_read("receipt.json", from_git))
    assert receipt["source_sha"] == SOURCE and receipt["run_id"] == RUN
    blob = _git("rev-parse", f"{SOURCE}:spacr/qt/widgets/ambient.py").decode().strip()
    assert receipt["source_ambient_blob"] == blob
    run = json.loads(_read("tests-run-coverage-phase.json", from_git))
    assert run["id"] == RUN and run["head_sha"] == SOURCE
    assert run["status"] == "in_progress" and run["conclusion"] is None
    assert len(receipt["coverage_shards"]) == 12
    assert receipt["coverage_success"] == 8
    assert receipt["coverage_failure"] == 4
    for row in receipt["coverage_shards"]:
        number = row["number"]
        job = json.loads(_read(f"coverage{number}-job.json", from_git))
        assert job["id"] == row["job_id"]
        assert job["head_sha"] == SOURCE and job["run_id"] == RUN
        assert job["status"] == "completed"
        assert job["conclusion"] == row["conclusion"]
        assert job["completed_at"] == row["completed_at"]
    assert {int(key) for key in receipt["failed_nodes"]} == {0, 1, 2, 7}

    combine = json.loads(_read("combine-job.json", from_git))
    assert combine["id"] == receipt["combine"]["job_id"]
    assert combine["head_sha"] == SOURCE and combine["run_id"] == RUN
    assert combine["conclusion"] == "failure"
    steps = {step["name"]: step["conclusion"] for step in combine["steps"]}
    assert steps["Gate on the per-module coverage ratchet baseline"] == "success"
    assert steps["Require every coverage shard to have passed its tests"] == "failure"
    with zipfile.ZipFile(io.BytesIO(_read("module-report.zip", from_git))) as archive:
        report = json.loads(archive.read("module-coverage-ratchet.json"))
    assert report["status"] == "pass"
    assert report["summary"]["modules_checked"] == 664
    assert report["summary"]["failed_modules"] == 0
    assert report["summary"]["unconfirmed_modules"] == 0
    assert report["measurement_integrity"]["status"] == "complete"
    assert report["measurement_integrity"]["shard_count"] == 12
    assert report["measurement_integrity"]["shards_with_records"] == 12
    assert report["measurement_integrity"]["recovered"] == []
    assert receipt["numerical"]["status"] == report["status"]
    assert receipt["numerical"]["integrity"] == report["measurement_integrity"]
    print("4133 coverage phase: numerical pass, 12/12 complete, 8 selected-test passes, 4 failures")


if __name__ == "__main__":
    main()
