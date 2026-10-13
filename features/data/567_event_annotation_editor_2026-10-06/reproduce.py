"""Reproduce the bounded F567 GUI tests and source-exact coverage receipt."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = "bf7873dec6859ba75622c532216cd36f183472ec"
SOURCE_BLOBS = {
    "spacr/qt/widgets/timelapse_preview.py":
        "e303ddce8800ede40cce900434932469bc0093fd",
    "spacr/timelapse.py": "17c2746545d06cc356d6d81897c11a145fa7af3d",
}
TESTS = (
    "tests/qt/test_timelapse_event_annotation_editor.py",
    "tests/qt/test_event_detection_alpha_gate.py",
    "tests/qt/test_cov_r5_timelapse_preview.py",
    "tests/qt/test_cov_r6_timelapse_preview.py",
    "tests/test_timelapse_event_annotation_provenance.py",
    "tests/test_cov_g2_timelapse_events.py",
)


def _git(*args: str) -> str:
    """Read a Git identity from the checked source tree."""
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    """Run only the F567 owner cohort and compare its added-source arcs."""
    for path, expected in SOURCE_BLOBS.items():
        if _git("hash-object", path) != expected:
            raise SystemExit(f"F567 source blob differs from receipt: {path}")
    scratch = Path("/mnt/wd4tb/scratch/f567-event-editor-replay")
    scratch.mkdir(parents=True, exist_ok=True)
    data_file = scratch / ".coverage"
    for old in scratch.glob(".coverage*"):
        old.unlink()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["SPACR_DEVICE"] = "cpu"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MPLBACKEND"] = "Agg"
    env["COVERAGE_FILE"] = str(data_file)
    subprocess.run(
        [sys.executable, "-m", "coverage", "run", "--branch", "-m", "pytest",
         "-q", "-p", "no:randomly", "--tb=short", *TESTS],
        cwd=ROOT, env=env, check=True)
    report = scratch / "coverage.json"
    subprocess.run(
        [sys.executable, "-m", "coverage", "json",
         f"--data-file={data_file}",
         "--include=spacr/qt/widgets/timelapse_preview.py,spacr/timelapse.py",
         "-o", str(report)],
        cwd=ROOT, env=env, check=True)
    files = json.loads(report.read_text())["files"]
    coverage = {}
    for path in SOURCE_BLOBS:
        source = files[path]
        diff = subprocess.check_output(
            ["git", "diff", "--unified=0", BASE, "HEAD", "--", path],
            cwd=ROOT, text=True)
        added: set[int] = set()
        for match in re.finditer(
                r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.M):
            start, count = int(match[1]), int(match[2] or 1)
            added.update(range(start, start + count))
        hit = set(source["executed_lines"])
        missing = set(source["missing_lines"])
        executable = added & (hit | missing)
        hit_arcs = {tuple(arc) for arc in source["executed_branches"]}
        missing_arcs = {tuple(arc) for arc in source["missing_branches"]}
        added_arcs = {arc for arc in hit_arcs | missing_arcs if arc[0] in added}
        coverage[path] = {
            "added_executable_lines": len(executable),
            "added_executable_hit": len(executable & hit),
            "added_executable_missing": sorted(executable & missing),
            "added_origin_arcs": len(added_arcs),
            "added_origin_hit": len(added_arcs & hit_arcs),
            "added_origin_missing": sorted(added_arcs & missing_arcs),
        }
    receipt = {
        "source_blobs": SOURCE_BLOBS,
        "test_count": 112,
        "coverage": coverage,
        "coverage_json_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    expected = json.loads((Path(__file__).with_name("receipt.json")).read_text())
    for key in ("test_count", "coverage"):
        if receipt[key] != expected[key]:
            raise SystemExit(f"F567 receipt changed: {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
