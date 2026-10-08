"""Verify the eleven exact-source numerical coverage repairs without rerunning tests."""

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path


FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[2]
PREFIX = FOLDER.relative_to(ROOT).as_posix()
REMOVED_AMBIENT_LINES = {6135, 6136, 6137}


def _read(name, *, git=False):
    """Read an archived payload from disk or the committed Git tree."""
    if git:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"HEAD:{PREFIX}/{name}"]
        )
    return (FOLDER / name).read_bytes()


def _source(revision, path):
    """Read a production source file at one exact Git revision."""
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{revision}:{path}"]
    )


def _sha(data):
    """Return the SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def _map_ambient_line(line):
    """Map the old fungal source coordinates after deleting three lines."""
    assert abs(line) not in REMOVED_AMBIENT_LINES
    direction = -1 if line < 0 else 1
    value = abs(line)
    return direction * (value - 3 if value > 6137 else value)


def _map_ambient_arc(arc):
    """Map an old branch arc, excluding the removed fallback edge."""
    if any(abs(part) in REMOVED_AMBIENT_LINES for part in arc):
        return None
    return tuple(_map_ambient_line(part) for part in arc)


def _verify(git=False, revision="HEAD"):
    """Check every frozen source and exact missing-minus-executed union."""
    receipt = json.loads(_read("receipt.json", git=git))
    old = json.loads(gzip.decompress(_read("hosted-eleven.json.gz", git=git)))
    focused = json.loads(gzip.decompress(_read("focused-eleven.json.gz", git=git)))
    assert receipt["hosted_source_sha"] == old["source_sha"]
    assert receipt["target_source_sha"] == focused["target_source_sha"]
    assert set(old["files"]) == set(focused["files"]) == set(receipt["modules"])
    assert len(old["files"]) == 11

    old_ambient = _source(receipt["hosted_source_sha"], "spacr/qt/widgets/ambient.py")
    current_ambient = _source(revision, "spacr/qt/widgets/ambient.py")
    old_fallback = (
        b"                                  default=None)\n"
        b"                    if replace is None:\n"
        b"                        break\n"
    )
    new_fallback = b"                                  key=lambda i: (following[i][6], following[i][8]))\n"
    old_key = b"                                  key=lambda i: (following[i][6], following[i][8]),\n"
    assert old_ambient.count(old_key + old_fallback) == 1
    assert old_ambient.replace(old_key + old_fallback, new_fallback) == current_ambient

    old_deep = _source(receipt["hosted_source_sha"], "spacr/deep_spacr.py")
    current_deep = _source(revision, "spacr/deep_spacr.py")
    old_checkpoint = b"best_model_path = os.path.abspath(resume_checkpoint)"
    new_checkpoint = b"best_model_path = os.path.abspath(initialization_path)"
    assert old_deep.count(old_checkpoint) == 1
    assert old_deep.replace(old_checkpoint, new_checkpoint) == current_deep

    for path, expected in receipt["modules"].items():
        old_source = _source(receipt["hosted_source_sha"], path)
        current_source = _source(revision, path)
        assert _sha(old_source) == expected["hosted_source_sha256"], path
        assert _sha(current_source) == expected["current_source_sha256"], path
        if path not in {"spacr/deep_spacr.py", "spacr/qt/widgets/ambient.py"}:
            assert old_source == current_source, path

        old_row = old["files"][path]
        focused_row = focused["files"][path]
        focused_revision = receipt["focused_input_source_revisions"][focused_row["group"]]
        measured_source = _source(focused_revision, path)
        assert _sha(measured_source) == focused_row["measured_source_sha256"], path
        assert measured_source == current_source, path
        old_lines = set(old_row["missing_lines"])
        old_arcs = {tuple(arc) for arc in old_row["missing_branches"]}
        if path == "spacr/qt/widgets/ambient.py":
            assert 6137 in old_lines and (6136, 6137) in old_arcs
            old_lines = {_map_ambient_line(line) for line in old_lines
                         if line not in REMOVED_AMBIENT_LINES}
            old_arcs = {mapped for arc in old_arcs
                        if (mapped := _map_ambient_arc(arc)) is not None}
        covered_lines = set(focused_row["executed_lines"])
        covered_arcs = {tuple(arc) for arc in focused_row["executed_branches"]}
        remaining_lines = sorted(old_lines - covered_lines)
        remaining_arcs = sorted(old_arcs - covered_arcs)
        assert remaining_lines == expected["remaining_lines"], path
        assert [list(arc) for arc in remaining_arcs] == expected["remaining_branches"], path
        baseline = expected["original_allowance"]
        assert len(remaining_lines) <= baseline["uncovered_statements"], path
        assert len(remaining_arcs) <= baseline["uncovered_branches"], path
        assert old_row["summary"]["excluded_lines"] <= baseline["excluded_lines"]
        assert current_source.count(b"pragma: no cover") <= baseline["pragma_no_cover"]
        if path == "spacr/deep_spacr.py":
            assert 2738 in covered_lines
        if path == "spacr/qt/widgets/ambient.py":
            assert {6132, 6135, 6136} <= covered_lines
        print(path, len(remaining_lines), len(remaining_arcs), "within unchanged allowance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--git", action="store_true")
    parser.add_argument("--revision", default="HEAD")
    arguments = parser.parse_args()
    _verify(git=arguments.git, revision=arguments.revision)
