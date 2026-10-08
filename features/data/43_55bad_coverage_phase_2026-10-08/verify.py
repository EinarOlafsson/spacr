"""Verify the frozen source and hosted payloads for the 55bad coverage phase."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile


ARCHIVE = Path(__file__).resolve().parent
SOURCE = "55badc57ff25ef6a7e14721561773a412b515318"


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
    actual_names = set()
    for item in manifest["payloads"]:
        name = item["path"]
        data = _read(name, from_git)
        assert len(data) == item["bytes"], name
        assert hashlib.sha256(data).hexdigest() == item["sha256"], name
        actual_names.add(name)
    assert len(actual_names) == len(manifest["payloads"])
    if not from_git:
        files = {p.name for p in ARCHIVE.iterdir() if p.is_file()}
        assert files == actual_names | {"MANIFEST.json"}
    receipt = json.loads(_read("receipt.json", from_git))
    assert receipt["source_sha"] == SOURCE
    blob = _git("rev-parse", f"{SOURCE}:spacr/qt/widgets/ambient.py").decode().strip()
    assert receipt["source_ambient_blob"] == blob
    assert receipt["coverage_success"] == 8
    assert receipt["coverage_failure"] == 4
    assert receipt["combine"]["conclusion"] == "failure"
    with zipfile.ZipFile(io.BytesIO(_read("module-report.zip", from_git))) as z:
        report = json.loads(z.read("module-coverage-ratchet.json"))
    assert report["status"] == "fail"
    assert report["summary"]["modules_checked"] == 664
    assert report["summary"]["failed_modules"] == 1
    assert report["summary"]["unconfirmed_modules"] == 0
    assert report["measurement_integrity"]["status"] == "complete"
    modules = {row["path"]: row for row in report["modules"]}
    assert modules["spacr/qt/widgets/ambient.py"]["missing_branches"] == [[5957, 5960]]
    assert modules["spacr/qt/widgets/ambient.py"]["missing_lines"] == []
    print("55bad coverage phase: 12/12 complete, 8 success, 4 failed, one ambient arc")


if __name__ == "__main__":
    main()
