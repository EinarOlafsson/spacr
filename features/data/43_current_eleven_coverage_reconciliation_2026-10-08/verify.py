"""Verify the current eleven-module result from two frozen coverage archives."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = HERE.relative_to(ROOT).as_posix()


def _git(spec):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", spec])


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _archive(path, frozen):
    for name, expected in frozen.items():
        assert _sha(_git(f"HEAD:{path}/{name}")) == expected, (path, name)
    manifest = json.loads(_git(f"HEAD:{path}/MANIFEST.json"))
    manifest = manifest.get("sha256", manifest)
    for name, expected in manifest.items():
        assert _sha(_git(f"HEAD:{path}/{name}")) == expected, (path, name)
    source = _git(f"HEAD:{path}/verify_union.py")
    namespace = {
        "__file__": str(ROOT / path / "verify_union.py"),
        "__name__": "_frozen_coverage_verifier",
    }
    exec(compile(source, namespace["__file__"], "exec"), namespace)
    return namespace


def verify(git=False):
    manifest_raw = (
        _git(f"HEAD:{PREFIX}/MANIFEST.json")
        if git else (HERE / "MANIFEST.json").read_bytes()
    )
    for name, expected in json.loads(manifest_raw).items():
        data = _git(f"HEAD:{PREFIX}/{name}") if git else (HERE / name).read_bytes()
        assert _sha(data) == expected, name
    raw = _git(f"HEAD:{PREFIX}/receipt.json") if git else (HERE / "receipt.json").read_bytes()
    receipt = json.loads(raw)
    assert receipt["source_revision"] == "7f321bf3d9ca663ce9e76c16aaa9c32105fbd4e4"
    old_path = receipt["original_archive"]
    new_path = receipt["changed_archive"]
    old = _archive(old_path, receipt["original_archive_sha256"])
    new = _archive(new_path, receipt["changed_archive_sha256"])
    old["_verify"](git=True, revision=receipt["original_target_revision"])
    new["verify"](git=True, revision="HEAD")
    old_receipt = json.loads(_git(f"HEAD:{old_path}/receipt.json"))
    new_receipt = json.loads(_git(f"HEAD:{new_path}/receipt.json"))
    assert set(old_receipt["modules"]) == set(receipt["modules"])
    assert set(receipt["changed_modules"]) == {
        "spacr/qt/preferences.py", "spacr/qt/widgets/ambient.py"
    }
    for path, expected in receipt["modules"].items():
        current_hash = _sha(_git("HEAD:" + path))
        assert current_hash == expected["source_sha256"], path
        old_row = old_receipt["modules"][path]
        allowance = old_row["original_allowance"]
        assert expected["original_allowance"] == [
            allowance["uncovered_statements"], allowance["uncovered_branches"]
        ]
        if path in receipt["changed_modules"]:
            assert current_hash == new_receipt["target_source_sha256"][path]
            remaining = new_receipt["expected_remaining"][path]
            assert new_receipt["unchanged_original_allowances"][path] == expected[
                "original_allowance"
            ]
        else:
            assert current_hash == old_row["current_source_sha256"]
            remaining = [len(old_row["remaining_lines"]), len(old_row["remaining_branches"])]
        assert remaining == expected["remaining"], path
        assert all(n <= maximum for n, maximum in zip(remaining, expected["original_allowance"]))
        print(path, *remaining, "within unchanged allowance")
    print("Eleven source-matched modules; no ceiling changes or new hosted verdict")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--git", action="store_true")
    verify(parser.parse_args().git)
