"""Verify the source-bound Fast0 OPS resume failure and focused repair."""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = "f0b81dacefe089c54b1d22a4aebd363a29d53cfb"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(name, from_git):
    if not from_git:
        return (HERE / name).read_bytes()
    relative = HERE.relative_to(ROOT).as_posix()
    return subprocess.check_output(
        ("git", "show", f"HEAD:{relative}/{name}"), cwd=ROOT)


def main():
    arguments = set(sys.argv[1:])
    if arguments - {"--git", "--source"}:
        raise SystemExit("usage: python verify.py [--git] [--source]")
    from_git = "--git" in arguments
    receipt = json.loads(_read("receipt.json", from_git))
    manifest = json.loads(_read("MANIFEST.json", from_git))
    assert receipt["hosted_source"] == manifest["hosted_source"] == BASE
    assert receipt["hosted_job_id"] == 113582650451
    names = set()
    for row in manifest["payloads"]:
        name = row["path"]
        assert name not in names
        names.add(name)
        data = _read(name, from_git)
        assert len(data) == row["bytes"] and _sha(data) == row["sha256"]
        if name.endswith(".gz"):
            plain = gzip.decompress(data)
            assert len(plain) == row["raw_bytes"]
            assert _sha(plain) == row["raw_sha256"]
    assert names == set(receipt["payloads"])
    if not from_git:
        assert {path.name for path in HERE.iterdir() if path.is_file()} == names | {"MANIFEST.json"}

    before = gzip.decompress(_read("test-before.py.gz", from_git))
    after = gzip.decompress(_read("test-after.py.gz", from_git))
    assert _sha(before) == receipt["test_before_sha256"]
    assert _sha(after) == receipt["test_after_sha256"]
    assert before == subprocess.check_output(
        ("git", "show", f"{BASE}:tests/test_the_ops_engine_runs_a_well_end_to_end.py"),
        cwd=ROOT)
    assert after != before
    assert gzip.decompress(_read("resource-log.py.gz", from_git)) == subprocess.check_output(
        ("git", "show", f"{BASE}:spacr/resource_log.py"), cwd=ROOT)
    assert gzip.decompress(_read("ops-engine.py.gz", from_git)) == subprocess.check_output(
        ("git", "show", f"{BASE}:spacr/ops_engine.py"), cwd=ROOT)

    hosted = gzip.decompress(_read("hosted-fast0.log.gz", from_git))
    assert b"worker 'gw1' crashed while running 'tests/test_the_ops_engine_runs_a_well_end_to_end.py::test_two_wells_with_stored_reads_resume_without_rerunning'" in hosted
    assert b"1 failed, 800 passed, 1 skipped" in hosted
    assert b"timeout: 300.0s" in hosted
    targeted = gzip.decompress(_read("targeted-pass.log.gz", from_git))
    assert b"1 passed in 275.32s" in targeted

    def report(name):
        return json.loads(gzip.decompress(_read(name, from_git)))

    fixture = report("fixture-five-site-report.json.gz")
    assert fixture["stitch"]["sites"] == fixture["stitch"]["placed"] == 5
    assert fixture["decode"]["fields"] == 5
    assert fixture["decode"]["field_seconds"]["read"] == 150.0
    for well in ("A1", "A2"):
        measured = report(f"candidate-{well}-report.json.gz")
        assert measured["stitch"]["sites"] == measured["stitch"]["placed"] == 2
        assert measured["decode"]["fields"] == 2
        assert measured["decode"]["ops_reads_rows"] == 232
        assert measured["decode"]["field_seconds"]["read"] == 60.0
    if "full-file-pass.log.gz" in names:
        assert b"10 passed in" in gzip.decompress(_read("full-file-pass.log.gz", from_git))
    if "--source" in arguments:
        assert (ROOT / "tests/test_the_ops_engine_runs_a_well_end_to_end.py").read_bytes() == after
        assert (ROOT / "spacr/resource_log.py").read_bytes() == gzip.decompress(
            _read("resource-log.py.gz", from_git))
        assert (ROOT / "spacr/ops_engine.py").read_bytes() == gzip.decompress(
            _read("ops-engine.py.gz", from_git))
    print("OPS resume proof: two real stitched/decoded wells; hosted worker failure retained")


if __name__ == "__main__":
    main()
