"""The 100% gate follows the package and cannot be weakened by omission."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "verify_module_coverage.py"
RUNNER_SCRIPT = ROOT / "tools" / "run_coverage_batches.py"

_SPEC = importlib.util.spec_from_file_location("verify_module_coverage", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
ratchet = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ratchet
_SPEC.loader.exec_module(ratchet)

RESOURCE_GENERATORS = {
    "spacr/resources/home/versions/_generators/common.py",
    "spacr/resources/home/versions/_generators/parts.py",
    "spacr/resources/home/versions/_generators/render.py",
    "spacr/resources/home/versions/_generators/variants.py",
    "spacr/resources/icons/backup_icons/_generators/"
    "group_trellis_gate_feature_napari.py",
}


def _load_coverage_runner():
    """Load the coverage-only helper when a test actually exercises it.

    The hardware and marker-specific CI jobs intentionally omit coverage.py.
    Pytest still imports every test module before marker deselection, so doing
    this at module scope made those otherwise independent jobs fail during
    collection.  The coverage shards install the dependency and remain the
    required execution path for these two helper contracts.
    """
    pytest.importorskip(
        "coverage",
        reason="the coverage batch runner requires the coverage.py extra",
    )
    runner_spec = importlib.util.spec_from_file_location(
        "run_coverage_batches", RUNNER_SCRIPT,
    )
    assert runner_spec is not None and runner_spec.loader is not None
    runner = importlib.util.module_from_spec(runner_spec)
    sys.modules[runner_spec.name] = runner
    runner_spec.loader.exec_module(runner)
    return runner


def _project(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True)
    (tmp_path / "setup.py").write_text(
        textwrap.dedent(
            """
            from setuptools import find_packages, setup

            setup(packages=find_packages(exclude=["tests.*", "tests"]))
            """
        ),
        encoding="utf-8",
    )
    package = tmp_path / "demo"
    package.mkdir()
    (package / "__init__.py").write_text("\"\"\"Demo.\"\"\"\n", encoding="utf-8")
    (package / "logic.py").write_text(
        "def choose(value):\n"
        "    if value:\n"
        "        return 1\n"
        "    return 0\n",
        encoding="utf-8",
    )
    return tmp_path


def _row(
    *,
    statements: int = 1,
    covered: int | None = None,
    missing_lines: list[int] | None = None,
    branches: int = 0,
    covered_branches: int | None = None,
    missing_branches: list[list[int]] | None = None,
    excluded_lines: list[int] | None = None,
) -> dict:
    covered = statements if covered is None else covered
    covered_branches = branches if covered_branches is None else covered_branches
    missing_lines = [] if missing_lines is None else missing_lines
    missing_branches = [] if missing_branches is None else missing_branches
    excluded_lines = [] if excluded_lines is None else excluded_lines
    return {
        "executed_lines": [],
        "missing_lines": missing_lines,
        "excluded_lines": excluded_lines,
        "executed_branches": [],
        "missing_branches": missing_branches,
        "summary": {
            "covered_lines": covered,
            "num_statements": statements,
            "excluded_lines": len(excluded_lines),
            "covered_branches": covered_branches,
            "num_branches": branches,
        },
    }


def _coverage(files: dict[str, dict]) -> dict:
    return {
        "meta": {"branch_coverage": True, "version": "test"},
        "files": files,
        "totals": {},
    }


def _run_cli(
    project: Path,
    tmp_path: Path,
    coverage_data: dict,
    expected: int,
) -> tuple[subprocess.CompletedProcess[str], dict, str]:
    coverage_json = tmp_path / "coverage.json"
    json_out = tmp_path / "ratchet.json"
    text_out = tmp_path / "ratchet.txt"
    coverage_json.write_text(json.dumps(coverage_data), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--coverage-json",
            str(coverage_json),
            "--root",
            str(project),
            "--expected-file-count",
            str(expected),
            "--json-out",
            str(json_out),
            "--text-out",
            str(text_out),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        result,
        json.loads(json_out.read_text(encoding="utf-8")),
        text_out.read_text(encoding="utf-8"),
    )


def test_current_packaging_denominator_is_526_not_asset_generators():
    """The ratchet follows all 526 shipped modules, not asset generators.

    Since the previous 506-module pin, the product added the public
    accelerator resolver, plaque analysis, settings-pack support, and the
    model-zoo picker, plus module-specific status-bar hints and the fractal
    region catalog and GPU orbit renderer.  All seven are installed Python
    and belong in coverage. Annotation-dataset generation, GUI-thread GC
    policy, the example-screen data picker and its headless data manifest add
    four more installed modules. The generated setting-to-API target map adds
    one more; all five must have coverage rows too.

    526 since 2026-09-02: the pin had drifted to six modules behind the
    package before `spacr-download` was written, and that command adds two
    more -- `spacr/cli_download.py` and the Qt-free `spacr/example_archives.py`
    it reads its repositories from. Both are installed Python and both need
    coverage rows.
    """
    shipped = set(ratchet.discover_shipped_python_files(ROOT))
    every_spacr_python = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "spacr").rglob("*.py")
    }

    assert len(shipped) == 526
    assert every_spacr_python - shipped == RESOURCE_GENERATORS
    assert not RESOURCE_GENERATORS & shipped


def test_cli_passes_only_at_exact_statement_and_branch_coverage(tmp_path):
    project = _project(tmp_path / "project")
    result, report, text = _run_cli(
        project,
        tmp_path,
        _coverage({
            "demo/__init__.py": _row(),
            "demo/logic.py": _row(statements=4, branches=2),
        }),
        expected=2,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert report["schema"] == "spacr.module-coverage-ratchet/v1"
    assert report["status"] == "pass"
    assert report["summary"] == {
        "failed_modules": 0,
        "global_issue_count": 0,
        "passed_modules": 2,
    }
    assert text.startswith("spaCR shipped-module coverage ratchet: PASS\n")
    assert result.stdout == text


def test_new_shipped_module_is_not_hidden_by_an_old_coverage_file(tmp_path):
    project = _project(tmp_path / "project")
    coverage_data = _coverage({
        "demo/__init__.py": _row(),
        "demo/logic.py": _row(statements=4, branches=2),
    })
    (project / "demo" / "new_feature.py").write_text(
        "ENABLED = True\n", encoding="utf-8"
    )

    result, report, text = _run_cli(
        project, tmp_path, coverage_data, expected=3,
    )

    assert result.returncode == 1
    new_module = next(
        module for module in report["modules"]
        if module["path"] == "demo/new_feature.py"
    )
    assert new_module["issues"] == ["missing coverage row"]
    assert "demo/new_feature.py: missing coverage row" in text


def test_uncovered_line_and_branch_have_actionable_diagnostics(tmp_path):
    project = _project(tmp_path / "project")
    result, report, text = _run_cli(
        project,
        tmp_path,
        _coverage({
            "demo/__init__.py": _row(),
            "demo/logic.py": _row(
                statements=4,
                covered=3,
                missing_lines=[4],
                branches=2,
                covered_branches=1,
                missing_branches=[[2, 4]],
            ),
        }),
        expected=2,
    )

    assert result.returncode == 1
    logic = next(
        module for module in report["modules"]
        if module["path"] == "demo/logic.py"
    )
    assert logic["issues"] == [
        "uncovered statements: 4",
        "uncovered branches: 2->4",
    ]
    assert "demo/logic.py: uncovered statements: 4" in text
    assert "demo/logic.py: uncovered branches: 2->4" in text


def test_excluded_line_and_real_no_cover_comment_both_fail(tmp_path):
    project = _project(tmp_path / "project")
    logic = project / "demo" / "logic.py"
    logic.write_text(
        logic.read_text(encoding="utf-8")
        + "NEVER = 0  # pragma: no cover\n",
        encoding="utf-8",
    )
    result, report, text = _run_cli(
        project,
        tmp_path,
        _coverage({
            "demo/__init__.py": _row(),
            "demo/logic.py": _row(
                statements=4,
                branches=2,
                excluded_lines=[5],
            ),
        }),
        expected=2,
    )

    assert result.returncode == 1
    logic_report = next(
        module for module in report["modules"]
        if module["path"] == "demo/logic.py"
    )
    assert logic_report["issues"] == [
        "coverage-excluded lines: 5",
        "pragma: no cover comments: 5",
    ]
    assert "demo/logic.py: coverage-excluded lines: 5" in text
    assert "demo/logic.py: pragma: no cover comments: 5" in text


def test_branchless_coverage_input_is_rejected_even_when_counts_are_full(tmp_path):
    project = _project(tmp_path / "project")
    coverage_data = _coverage({
        "demo/__init__.py": _row(),
        "demo/logic.py": _row(statements=4),
    })
    coverage_data["meta"]["branch_coverage"] = False

    result, report, text = _run_cli(
        project, tmp_path, coverage_data, expected=2,
    )

    assert result.returncode == 1
    assert report["global_issues"] == [
        "coverage JSON was not produced with branch coverage enabled"
    ]
    assert "not produced with branch coverage enabled" in text


def test_coverage_batches_use_unique_data_files_and_argument_lists(
    tmp_path, monkeypatch,
):
    coverage_runner = _load_coverage_runner()
    project = tmp_path / "project"
    tests = project / "tests"
    tests.mkdir(parents=True)
    (tests / "test_one.py").write_text("def test_one(): pass\n", encoding="utf-8")
    (tests / "test_two.py").write_text("def test_two(): pass\n", encoding="utf-8")
    (project / ".coveragerc").write_text("[run]\nbranch=True\n", encoding="utf-8")
    data_dir = tmp_path / "coverage-data"
    calls = []

    def fake_run(command, *, env, check):
        calls.append((command, env["COVERAGE_FILE"], check))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.chdir(project)
    monkeypatch.setattr(coverage_runner.subprocess, "run", fake_run)

    status = coverage_runner.main([
        "tests",
        "--marker",
        "not gui",
        "--shard-index",
        "0",
        "--shard-count",
        "1",
        "--batch-size",
        "1",
        "--workers",
        "1",
        "--data-dir",
        str(data_dir),
    ])

    assert status == 0
    assert len(calls) == 2
    assert calls[0][1].endswith(".coverage.shard-00.batch-001")
    assert calls[1][1].endswith(".coverage.shard-00.batch-002")
    assert calls[0][1] != calls[1][1]
    assert all(isinstance(command, list) and check is False
               for command, _data, check in calls)


def test_coverage_batches_discard_an_incomplete_child_database(
    tmp_path, monkeypatch, capsys,
):
    coverage_runner = _load_coverage_runner()
    project = tmp_path / "project"
    tests = project / "tests"
    tests.mkdir(parents=True)
    (tests / "test_one.py").write_text(
        "def test_one(): pass\n", encoding="utf-8",
    )
    data_dir = tmp_path / "coverage-data"
    written: dict[str, Path] = {}

    def fake_run(command, *, env, check):
        basename = Path(env["COVERAGE_FILE"])
        readable = basename.with_name(f"{basename.name}.readable")
        data = coverage_runner.CoverageData(basename=str(readable))
        data.add_lines({"spacr/example.py": {1}})
        data.write()
        incomplete = basename.with_name(f"{basename.name}.incomplete")
        incomplete.write_bytes(b"SQLite format 3\x00")
        written.update(readable=readable, incomplete=incomplete)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.chdir(project)
    monkeypatch.setattr(coverage_runner.subprocess, "run", fake_run)

    status = coverage_runner.main([
        "tests",
        "--marker",
        "not gui",
        "--shard-index",
        "0",
        "--shard-count",
        "1",
        "--workers",
        "1",
        "--data-dir",
        str(data_dir),
    ])

    assert status == 0
    assert written["readable"].is_file()
    assert not written["incomplete"].exists()
    output = capsys.readouterr().out
    assert "discarding unreadable coverage process data" in output
    assert written["incomplete"].name in output


def test_coverage_workflow_is_sharded_artifact_safe_and_blocking():
    workflow_path = ROOT / ".github" / "workflows" / "tests.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    shards = jobs["coverage-shards"]
    combine = jobs["coverage-combine"]

    assert shards["strategy"]["matrix"]["shard"] == list(range(12))
    upload = next(
        step for step in shards["steps"]
        if step.get("uses") == "actions/upload-artifact@v7"
    )
    assert "${{ matrix.shard }}" in upload["with"]["name"]
    assert upload["with"]["include-hidden-files"] is True

    shard_script = "\n".join(
        step.get("run", "") for step in shards["steps"]
    )
    assert "tools/run_coverage_batches.py" in shard_script
    assert '--data-dir "$SPACR_COVERAGE_DATA_DIR"' in shard_script

    combine_script = "\n".join(
        step.get("run", "") for step in combine["steps"]
    )
    assert "coverage combine --keep" in combine_script
    assert "coverage json --pretty-print" in combine_script
    assert "--expected-file-count 526" in combine_script
    assert "module-coverage-ratchet.json" in combine_script
    assert "module-coverage-ratchet.txt" in combine_script
    assert "coverage-combine" in jobs["release-gate"]["needs"]


def test_coverage_config_has_branch_and_subprocess_data_without_exclusions():
    config = (ROOT / ".coveragerc").read_text(encoding="utf-8")

    assert "branch = True" in config
    assert "relative_files = True" in config
    assert "patch = subprocess" in config
    assert "source = spacr" in config
    assert "omit" not in config
    assert "exclude_lines =\n" in config
    assert "partial_branches =\n" in config
