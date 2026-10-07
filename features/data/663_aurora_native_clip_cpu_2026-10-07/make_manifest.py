"""Regenerate the source-bound N663 Aurora receipt and payload manifest."""

import ast
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy
import PySide6


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMMITS = {
    "baseline": "ae7269a08c136ac35d488d9f657f82cbfd33c49a",
    "optimized": "1e05cd4e0e9f179fa0536f0d00c0c50aa4d04eb8",
    "final": "079591b7ddc973858c92c8edb84170b7fd945254",
    "integrated": "11fc2f2a1b9b0108dae27bb46b5af76a2f9784d8",
}


def sha256(data):
    """Hash an immutable payload."""
    return hashlib.sha256(data).hexdigest()


def git_bytes(expression):
    """Read a committed object without requiring a checked-out copy."""
    return subprocess.check_output(["git", "show", expression], cwd=ROOT)


def class_node(source):
    """Return Aurora's AST class node."""
    return next(node for node in ast.parse(source.decode()).body
                if isinstance(node, ast.ClassDef) and node.name == "AuroraEngine")


def main():
    """Write reproducible class snippets, measurement receipt and manifest."""
    sources = {label: git_bytes(f"{commit}:spacr/qt/widgets/ambient.py")
               for label, commit in COMMITS.items()}
    for label in ("baseline", "final"):
        source = sources[label]
        node = class_node(source)
        lines = source.splitlines(keepends=True)
        (HERE / f"aurora-{label}.py").write_bytes(
            b"".join(lines[node.lineno - 1:node.end_lineno]))
    parity = json.loads((HERE / "parity-final-079.json").read_text())
    native = [row for row in parity["rows"] if row["size"] == [3840, 2160]]
    small = [row for row in parity["rows"] if row["size"] == [1280, 720]]
    source_abba = json.loads((HERE / "source-abba-final-079.json").read_text())
    report = json.loads((HERE / "clip-coverage.json").read_text())
    receipt = {
        "item": "N663",
        "scope": "native Aurora surge paint and incoming-clip preservation",
        "source_commits": COMMITS,
        "module_sha256": {label: sha256(source)
                          for label, source in sources.items()},
        "integrated_module_sha256": sha256(sources["integrated"]),
        "aurora_class_ast_sha256": {
            label: sha256(ast.dump(class_node(source),
                                   include_attributes=False).encode())
            for label, source in sources.items()},
        "final_test_blob": subprocess.check_output(
            ["git", "rev-parse", f"{COMMITS['final']}:tests/qt/test_ambient_motion.py"],
            cwd=ROOT, text=True).strip(),
        "interpreter": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pyside6": PySide6.__version__,
            "cuda_visible_devices": "",
            "qt_platform": "offscreen",
            "memory_cap": "4G",
        },
        "parity": {
            "rows": len(parity["rows"]),
            "native_4k_rows": len(native),
            "native_4k_exact_rows": sum(row["changed_pixels"] == 0
                                        for row in native),
            "small_rows": len(small),
            "small_max_changed_pixels": max(row["changed_pixels"] for row in small),
            "small_max_byte_difference": max(row["max_byte_diff"] for row in small),
        },
        "final_abba_medians_ms": [row["median_ms"] for row in source_abba["rows"]],
        "clip_line_2362_executed": 2362 in report["files"]["spacr/qt/widgets/ambient.py"]["executed_lines"],
        "focused_aurora_cases_passed": 44,
        "incoming_clip_negative_cases_failed": 2,
        "incoming_clip_positive_cases_passed": 2,
        "native_24_fps_accepted": False,
        "human_visual_acceptance": False,
    }
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    manifest = {
        "item": "N663",
        "payload_sha256": {
            path.name: sha256(path.read_bytes())
            for path in sorted(HERE.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
