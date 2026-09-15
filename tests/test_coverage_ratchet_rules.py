"""No shipped module loses coverage; no new module arrives below 100%.

Item 288, maintainer decision 2026-09-15: the release gate is a per-module
coverage ratchet against ``tools/coverage_baseline.json``, and 100% per
module stays the goal.  Every rule of ``tools/verify_module_coverage.py`` is
exercised here on SYNTHETIC coverage data -- a two-module fake package and
hand-built coverage.py rows -- so none of these tests depends on how well the
real suite covers spaCR.

The negative cases are the point: a module losing one statement fails, a new
module with one uncovered line fails, a module that vanishes fails, a
missing data file fails, a hand-edited baseline is refused, and a green run
never rewrites the baseline.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.test_module_coverage_ratchet import (
    ROOT,
    SCRIPT,
    _coverage,
    _project,
    _row,
    ratchet,
)

BASELINE = ROOT / "tools" / "coverage_baseline.json"
WORKFLOWS = ROOT / ".github" / "workflows"
STAMP = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")


def _gate(project, tmp_path, coverage_data, *extra, baseline=None):
    """Run the CLI; report and text are None when it could not run."""
    coverage_json = tmp_path / "coverage.json"
    json_out = tmp_path / "ratchet.json"
    text_out = tmp_path / "ratchet.txt"
    json_out.unlink(missing_ok=True)
    text_out.unlink(missing_ok=True)
    if coverage_data is not None:
        coverage_json.write_text(json.dumps(coverage_data), encoding="utf-8")
    command = [
        sys.executable, str(SCRIPT),
        "--coverage-json", str(coverage_json),
        "--root", str(project),
        "--json-out", str(json_out),
        "--text-out", str(text_out),
    ]
    if baseline is not None:
        command += ["--baseline", str(baseline)]
    command += list(extra)
    result = subprocess.run(
        command, cwd=project, capture_output=True, text=True, check=False,
    )
    report = (
        json.loads(json_out.read_text(encoding="utf-8"))
        if json_out.exists() else None
    )
    text = text_out.read_text(encoding="utf-8") if text_out.exists() else None
    return result, report, text


def _seed(project, tmp_path, coverage_data):
    baseline = tmp_path / "baseline.json"
    result, _report, _text = _gate(
        project, tmp_path, coverage_data,
        "--reset-baseline", "--reason", "seed for the test", "--commit", "c0ffee",
        baseline=baseline,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return baseline


def _module(report, path):
    return next(module for module in report["modules"] if module["path"] == path)


def _logic(uncovered_statements=0, uncovered_branches=0, excluded=0):
    """Coverage data for the fake package with ``demo/logic.py`` gaps."""
    return _coverage({
        "demo/__init__.py": _row(),
        "demo/logic.py": _row(
            statements=4,
            covered=4 - uncovered_statements,
            missing_lines=list(range(1, uncovered_statements + 1)),
            branches=2,
            covered_branches=2 - uncovered_branches,
            missing_branches=[[2, 3 + i] for i in range(uncovered_branches)],
            excluded_lines=list(range(10, 10 + excluded)),
        ),
    })


def _combine_job():
    workflow = yaml.safe_load(
        (WORKFLOWS / "tests.yml").read_text(encoding="utf-8")
    )
    return workflow["jobs"]


# -- (a) a count rises ------------------------------------------------------


def test_a_module_losing_one_statement_fails(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=1))

    result, report, text = _gate(
        project, tmp_path, _logic(uncovered_statements=2), baseline=baseline,
    )

    assert result.returncode == 1
    assert report["status"] == "fail"
    assert _module(report, "demo/logic.py")["failures"] == [
        "uncovered statements rose from 1 to 2"
    ]
    assert "ERROR: demo/logic.py: uncovered statements rose from 1 to 2" in text
    assert report["summary"]["failed_modules"] == 1


@pytest.mark.parametrize(
    ("before", "after", "failure"),
    [
        (
            {"uncovered_branches": 1},
            {"uncovered_branches": 2},
            "uncovered branches rose from 1 to 2",
        ),
        (
            {"uncovered_statements": 1},
            {"uncovered_statements": 1, "excluded": 1},
            "coverage-excluded lines rose from 0 to 1",
        ),
    ],
)
def test_a_module_losing_a_branch_or_hiding_a_line_fails(
    tmp_path, before, after, failure,
):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(**before))

    result, report, _text = _gate(
        project, tmp_path, _logic(**after), baseline=baseline,
    )

    assert result.returncode == 1
    assert _module(report, "demo/logic.py")["failures"] == [failure]


def test_a_new_no_cover_pragma_fails(tmp_path):
    project = _project(tmp_path / "project")
    logic = project / "demo" / "logic.py"
    logic.write_text(
        logic.read_text(encoding="utf-8") + "A = 1  # pragma: no cover\n",
        encoding="utf-8",
    )
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=1))
    logic.write_text(
        logic.read_text(encoding="utf-8") + "B = 2  # pragma: no cover\n",
        encoding="utf-8",
    )

    result, report, _text = _gate(
        project, tmp_path, _logic(uncovered_statements=1), baseline=baseline,
    )

    assert result.returncode == 1
    assert _module(report, "demo/logic.py")["failures"] == [
        "pragma: no cover comments rose from 1 to 2"
    ]


def test_one_count_rising_fails_even_when_another_falls(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(
        project, tmp_path, _logic(uncovered_statements=2, uncovered_branches=1),
    )

    result, report, _text = _gate(
        project, tmp_path,
        _logic(uncovered_statements=1, uncovered_branches=2),
        baseline=baseline,
    )

    assert result.returncode == 1
    logic = _module(report, "demo/logic.py")
    assert logic["failures"] == ["uncovered branches rose from 1 to 2"]
    assert logic["improvements"] == ["uncovered statements fell from 2 to 1"]


# -- (b) a module at 100% drops ---------------------------------------------


def test_a_module_at_100_percent_that_drops_below_fails(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic())

    result, report, text = _gate(
        project, tmp_path, _logic(uncovered_branches=1), baseline=baseline,
    )

    assert result.returncode == 1
    message = (
        "was at 100% in the baseline and now has 0 uncovered statements, "
        "1 uncovered branches, 0 pragma: no cover comments, "
        "0 coverage-excluded lines"
    )
    assert _module(report, "demo/logic.py")["failures"] == [message]
    assert f"ERROR: demo/logic.py: {message}" in text


# -- (c) no new uncovered module --------------------------------------------


def test_a_new_module_with_one_uncovered_line_fails(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic())
    (project / "demo" / "new_feature.py").write_text(
        "ENABLED = True\nDISABLED = False\n", encoding="utf-8",
    )
    data = _logic()
    data["files"]["demo/new_feature.py"] = _row(
        statements=2, covered=1, missing_lines=[2],
    )

    result, report, text = _gate(project, tmp_path, data, baseline=baseline)

    assert result.returncode == 1
    assert _module(report, "demo/new_feature.py")["failures"] == [
        "new module is not at 100% and is not in the baseline: "
        "1 uncovered statements, 0 uncovered branches, "
        "0 pragma: no cover comments, 0 coverage-excluded lines"
    ]
    assert "GAP: demo/new_feature.py:" in text
    assert "(no baseline entry)" in text


def test_a_new_module_at_100_percent_passes(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=1))
    (project / "demo" / "new_feature.py").write_text(
        "ENABLED = True\n", encoding="utf-8",
    )
    data = _logic(uncovered_statements=1)
    data["files"]["demo/new_feature.py"] = _row()

    result, report, _text = _gate(project, tmp_path, data, baseline=baseline)

    assert result.returncode == 0, result.stdout
    assert _module(report, "demo/new_feature.py")["at_100_percent"] is True


# -- improvement passes, and nothing is written ------------------------------


def test_a_module_improving_passes_with_a_notice_and_changes_nothing(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=2))
    before = baseline.read_bytes()

    result, report, text = _gate(
        project, tmp_path, _logic(uncovered_statements=1), baseline=baseline,
    )

    assert result.returncode == 0, result.stdout
    assert report["status"] == "pass"
    assert _module(report, "demo/logic.py")["improvements"] == [
        "uncovered statements fell from 2 to 1"
    ]
    assert "IMPROVED: demo/logic.py: uncovered statements fell from 2 to 1" in text
    assert report["summary"]["improved_modules"] == 1
    # The goal stays visible: the module is still not at 100%.
    assert report["summary"]["modules_below_100_percent"] == 1
    assert "Modules not yet at 100% (100% per module is still the goal): 1" in text
    # A green run NEVER tightens by itself; that would hide a slow slide.
    assert baseline.read_bytes() == before


# -- (d) the module set is compared both ways --------------------------------


def _with_old_module(project, data, *, name="old.py"):
    (project / "demo" / name).write_text("X = 1\n", encoding="utf-8")
    data["files"][f"demo/{name}"] = _row(statements=1, covered=0, missing_lines=[1])
    return data


def test_a_vanished_module_fails_until_the_baseline_is_trimmed_deliberately(
    tmp_path,
):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _with_old_module(project, _logic()))
    (project / "demo" / "old.py").unlink()

    result, report, text = _gate(project, tmp_path, _logic(), baseline=baseline)

    assert result.returncode == 1
    assert report["stale_baseline_entries"] == ["demo/old.py"]
    assert report["summary"]["stale_baseline_entries"] == 1
    assert "ERROR: demo/old.py: in the baseline but no longer shipped" in text

    result, _report, text = _gate(
        project, tmp_path, _logic(),
        "--update-baseline", "--reason", "old.py was deleted", "--commit", "d00d",
        baseline=baseline,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "BASELINE: TRIMMED: demo/old.py: no longer shipped" in text
    document = ratchet.load_baseline(baseline)
    assert "demo/old.py" not in document["modules"]
    assert [entry["mode"] for entry in document["history"]] == ["reset", "update"]
    assert document["history"][-1]["reason"] == "old.py was deleted"
    assert document["history"][-1]["commit"] == "d00d"


def test_renaming_a_module_does_not_launder_its_gaps(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _with_old_module(project, _logic()))
    (project / "demo" / "old.py").unlink()
    renamed = _with_old_module(project, _logic(), name="renamed.py")

    result, report, _text = _gate(project, tmp_path, renamed, baseline=baseline)

    assert result.returncode == 1
    assert report["stale_baseline_entries"] == ["demo/old.py"]
    assert _module(report, "demo/renamed.py")["failures"][0].startswith(
        "new module is not at 100%"
    )

    result, _report, text = _gate(
        project, tmp_path, renamed,
        "--update-baseline", "--reason", "rename", "--commit", "d00d",
        baseline=baseline,
    )

    assert result.returncode == 1
    assert "BASELINE: NOT ADMITTED: demo/renamed.py" in text
    document = ratchet.load_baseline(baseline)
    assert "demo/old.py" not in document["modules"]
    assert "demo/renamed.py" not in document["modules"]


# -- the gate fails when it cannot measure ----------------------------------


def test_a_missing_coverage_data_file_fails(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic())
    (tmp_path / "coverage.json").unlink()

    result, report, text = _gate(project, tmp_path, None, baseline=baseline)

    assert result.returncode == 2
    assert report is None and text is None
    assert "coverage ratchet could not run" in result.stderr


def test_a_missing_baseline_file_fails(tmp_path):
    project = _project(tmp_path / "project")

    result, report, _text = _gate(
        project, tmp_path, _logic(), baseline=tmp_path / "absent.json",
    )

    assert result.returncode == 2
    assert report is None
    assert "coverage ratchet could not run" in result.stderr


def test_a_module_missing_from_the_data_fails_even_with_an_allowance(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=2))
    data = _logic(uncovered_statements=2)
    del data["files"]["demo/logic.py"]

    result, report, text = _gate(project, tmp_path, data, baseline=baseline)

    assert result.returncode == 1
    assert report["summary"]["modules_checked"] == 1
    assert (
        "checked 1 of 2 shipped modules; the gate never passes on a partial "
        "measurement"
    ) in report["global_issues"]
    assert "Modules checked: 1 of 2" in text
    assert _module(report, "demo/logic.py")["failures"] == ["missing coverage row"]


def test_a_missing_shard_turns_the_combine_step_red(tmp_path):
    if shutil.which("bash") is None:
        pytest.fail("the combine step is bash; this runner has no bash")
    combine = _combine_job()["coverage-combine"]
    step = next(
        step for step in combine["steps"]
        if "SPACR_COVERAGE_SHARD_COUNT" in step.get("env", {})
    )
    script = step["run"]
    end_marker = 'test "$missing" -eq 0'
    check = script[script.index("missing=0"):script.index(end_marker) + len(end_marker)]
    data = tmp_path / "input"
    data.mkdir()
    count = int(step["env"]["SPACR_COVERAGE_SHARD_COUNT"])
    for shard in range(count):
        if shard != 7:
            (data / f".coverage.shard-{shard:02d}.batch-001.host.1.abc").write_text("")
    environment = {
        "PATH": "/usr/bin:/bin",
        "SPACR_COVERAGE_INPUT": str(data),
        "SPACR_COVERAGE_SHARD_COUNT": str(count),
    }

    missing = subprocess.run(
        ["bash", "-e", "-c", check],
        env=environment, capture_output=True, text=True, check=False,
    )
    (data / ".coverage.shard-07.batch-001.host.1.abc").write_text("")
    complete = subprocess.run(
        ["bash", "-e", "-c", check],
        env=environment, capture_output=True, text=True, check=False,
    )

    assert missing.returncode != 0
    assert "coverage shard 7 produced no coverage data" in missing.stdout
    assert complete.returncode == 0, complete.stdout + complete.stderr


# -- the baseline is an output: dated, explicit, tool-written ---------------


def test_a_hand_edited_baseline_is_refused(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=2))
    document = json.loads(baseline.read_text(encoding="utf-8"))
    # Loosen one allowance by hand, keeping the totals consistent with it.
    document["modules"]["demo/logic.py"]["uncovered_statements"] = 3
    document["totals"]["uncovered_statements"] = 3
    baseline.write_text(json.dumps(document, indent=2), encoding="utf-8")
    edited = baseline.read_bytes()

    gate, report, _text = _gate(
        project, tmp_path, _logic(uncovered_statements=3), baseline=baseline,
    )
    update, _report, _text = _gate(
        project, tmp_path, _logic(uncovered_statements=1),
        "--update-baseline", "--reason", "tighten", "--commit", "d00d",
        baseline=baseline,
    )

    assert gate.returncode == 2
    assert report is None
    assert "checksum does not match its body" in gate.stderr
    assert update.returncode == 2
    assert "checksum does not match its body" in update.stderr
    assert baseline.read_bytes() == edited

    # The one way back is a deliberate, reasoned regeneration.
    reset, _report, text = _gate(
        project, tmp_path, _logic(uncovered_statements=2),
        "--reset-baseline", "--reason", "replace a hand-edited file",
        "--commit", "d00d",
        baseline=baseline,
    )
    assert reset.returncode == 0, reset.stdout + reset.stderr
    assert "previous baseline not carried forward" in text
    assert len(ratchet.load_baseline(baseline)["history"]) == 1


def test_reformatting_the_baseline_is_not_an_edit(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=1))
    document = json.loads(baseline.read_text(encoding="utf-8"))
    baseline.write_text(json.dumps(document), encoding="utf-8")

    result, _report, _text = _gate(
        project, tmp_path, _logic(uncovered_statements=1), baseline=baseline,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_update_baseline_tightens_never_loosens_and_is_dated(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(
        project, tmp_path, _logic(uncovered_statements=2, uncovered_branches=1),
    )

    result, _report, text = _gate(
        project, tmp_path,
        _logic(uncovered_statements=1, uncovered_branches=2),
        "--update-baseline", "--reason", "covered line 2", "--commit", "beef",
        baseline=baseline,
    )

    # The branch regression still fails the run that tightened statements.
    assert result.returncode == 1
    assert (
        "BASELINE: KEPT: demo/logic.py: uncovered branches rose from 1 to 2; "
        "--update-baseline never loosens"
    ) in text
    assert (
        "BASELINE: TIGHTENED: demo/logic.py: uncovered statements 2 -> 1"
    ) in text
    document = ratchet.load_baseline(baseline)
    last = document["history"][-1]
    assert last["mode"] == "update"
    assert last["commit"] == "beef"
    assert last["reason"] == "covered line 2"
    assert STAMP.match(last["written_at"])
    assert document["modules"]["demo/logic.py"] == {
        "uncovered_statements": 1,
        "uncovered_branches": 1,
        "pragma_no_cover": 0,
        "excluded_lines": 0,
        "since": last["written_at"],
    }


def test_update_with_nothing_to_tighten_writes_nothing(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=1))
    before = baseline.read_bytes()

    result, _report, text = _gate(
        project, tmp_path, _logic(uncovered_statements=1),
        "--update-baseline", "--reason", "nothing to do", "--commit", "beef",
        baseline=baseline,
    )

    assert result.returncode == 0
    assert "nothing to tighten, nothing written" in text
    assert baseline.read_bytes() == before


@pytest.mark.parametrize("write", ["--update-baseline", "--reset-baseline"])
def test_a_baseline_write_needs_a_reason_and_a_complete_measurement(
    tmp_path, write,
):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=2))
    before = baseline.read_bytes()
    incomplete = _logic(uncovered_statements=1)
    del incomplete["files"]["demo/__init__.py"]

    no_reason, _report, _text = _gate(
        project, tmp_path, _logic(uncovered_statements=1),
        write, "--commit", "beef",
        baseline=baseline,
    )
    partial, _report, _text = _gate(
        project, tmp_path, incomplete,
        write, "--reason", "from a partial run", "--commit", "beef",
        baseline=baseline,
    )

    assert no_reason.returncode == 2
    assert "needs a non-empty --reason" in no_reason.stderr
    assert partial.returncode == 2
    assert "refused: the measurement is incomplete" in partial.stderr
    assert baseline.read_bytes() == before


def test_reset_names_every_loosening(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=1))

    result, _report, text = _gate(
        project, tmp_path, _logic(uncovered_statements=2),
        "--reset-baseline", "--reason", "accepted after review", "--commit", "beef",
        baseline=baseline,
    )

    assert result.returncode == 0
    assert "BASELINE: LOOSENED: demo/logic.py: uncovered statements 1 -> 2" in text
    history = ratchet.load_baseline(baseline)["history"]
    assert [entry["mode"] for entry in history] == ["reset", "reset"]


def test_no_workflow_ever_writes_the_baseline():
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        content = workflow.read_text(encoding="utf-8")
        assert "--update-baseline" not in content, workflow.name
        assert "--reset-baseline" not in content, workflow.name


# -- the committed baseline and its siblings --------------------------------


def test_the_committed_baseline_was_written_by_the_tool():
    document = ratchet.load_baseline(BASELINE)  # checksum, totals, history

    assert document["schema"] == ratchet.BASELINE_SCHEMA
    assert document["totals"] == ratchet._baseline_totals(document["modules"])
    assert document["modules"]
    assert all(
        path.startswith("spacr/") and path.endswith(".py")
        for path in document["modules"]
    )
    for entry in document["history"]:
        assert STAMP.match(entry["written_at"])
        assert entry["commit"].strip() and entry["reason"].strip()


def test_the_coverage_job_gates_on_the_committed_baseline_and_says_so():
    jobs = _combine_job()
    combine = jobs["coverage-combine"]
    script = "\n".join(step.get("run", "") for step in combine["steps"])

    assert combine["name"] == "Coverage / no module loses coverage"
    relative = BASELINE.relative_to(ROOT).as_posix()
    assert f"--baseline {relative}" in script
    assert "coverage-combine" in jobs["release-gate"]["needs"]


def test_the_shard_count_has_one_value_in_three_places():
    jobs = _combine_job()
    shards = jobs["coverage-shards"]
    shard_script = "\n".join(step.get("run", "") for step in shards["steps"])
    declared = re.search(r"--shard-count (\d+)", shard_script)
    checked = next(
        step["env"]["SPACR_COVERAGE_SHARD_COUNT"]
        for step in jobs["coverage-combine"]["steps"]
        if "SPACR_COVERAGE_SHARD_COUNT" in step.get("env", {})
    )

    assert declared is not None
    assert (
        len(shards["strategy"]["matrix"]["shard"])
        == int(declared.group(1))
        == int(checked)
    )


def test_without_a_baseline_every_module_must_be_at_100_percent(tmp_path):
    project = _project(tmp_path / "project")

    result, report, text = _gate(
        project, tmp_path, _logic(uncovered_statements=1),
    )

    assert result.returncode == 1
    assert "Baseline: none (every module must be at 100%)" in text
    assert _module(report, "demo/logic.py")["failures"][0].startswith(
        "new module is not at 100%"
    )


def test_counts_not_line_numbers_so_moved_code_does_not_trip_the_gate(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=1))
    moved = _logic(uncovered_statements=1)
    moved["files"]["demo/logic.py"]["missing_lines"] = [3]

    result, _report, _text = _gate(project, tmp_path, moved, baseline=baseline)

    assert result.returncode == 0, result.stdout


def test_the_tool_path_is_the_one_the_workflow_runs():
    script = "\n".join(
        step.get("run", "")
        for step in _combine_job()["coverage-combine"]["steps"]
    )
    assert Path(SCRIPT).relative_to(ROOT).as_posix() in script
