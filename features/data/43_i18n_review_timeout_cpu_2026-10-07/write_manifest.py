"""Write or verify the frozen test-source and payload bindings."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEST_COMMIT = "08b497f9c8445fb5f1dad0738c2e3b3100008a84"
PUBLISHED_COMMIT = "28449ef0c84c4be92cf34be18d7372e3f0d9e2c8"
SOURCE_PATHS = (
    "tests/test_i18n_coverage_audit.py",
    "tools/build_i18n_catalogs.py",
    "tools/build_documentation_i18n.py",
    "tools/audit_i18n_coverage.py",
    "docs/i18n/REVIEW_SCOPE_2026-09-04.md",
    "docs/i18n/reviewed/runtime",
    "docs/i18n/reviewed/api",
    "docs/source/_static/i18n/api",
    "packaging/i18n",
    "spacr",
)
PAYLOADS = (
    "README.md",
    "pytest-output.txt",
    "coverage-param.data.gz",
    "hosted-coverage3.log.gz",
    "independent-review.json",
)


def git_object(commit: str, path: str) -> str:
    return subprocess.check_output(
        ("git", "rev-parse", f"{commit}:{path}"),
        cwd=ROOT, text=True,
    ).strip()


def manifest() -> dict[str, object]:
    return {
        "schema": 1,
        "test_commit": TEST_COMMIT,
        "published_comparison_commit": PUBLISHED_COMMIT,
        "test_exit_code": 0,
        "passed": 9,
        "elapsed_seconds": 526.07,
        "artifact_sha256": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in PAYLOADS
        },
        "git_objects": {
            commit: {path: git_object(commit, path) for path in SOURCE_PATHS}
            for commit in (TEST_COMMIT, PUBLISHED_COMMIT)
        },
    }


if __name__ == "__main__":
    path = HERE / "manifest.json"
    rendered = json.dumps(manifest(), indent=2, sort_keys=True) + "\n"
    if sys.argv[1:] == ["--check"]:
        if path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("archive manifest differs from payload or Git objects")
    elif not sys.argv[1:]:
        path.write_text(rendered, encoding="utf-8")
    else:
        raise SystemExit("usage: python write_manifest.py [--check]")
