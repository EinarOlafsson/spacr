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

import fnmatch
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
    _load_coverage_runner,
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


def _retire(project, baseline, *extra):
    """Run --retire-module: no coverage data, no report files."""
    command = [
        sys.executable, str(SCRIPT),
        "--root", str(project),
        "--baseline", str(baseline),
        *extra,
    ]
    return subprocess.run(
        command, cwd=project, capture_output=True, text=True, check=False,
    )


def test_a_module_deleted_on_purpose_is_retired_without_a_coverage_run(
    tmp_path,
):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _with_old_module(project, _logic()))
    (project / "demo" / "old.py").unlink()

    result = _retire(
        project, baseline,
        "--retire-module", "demo/old.py",
        "--reason", "old.py deleted by decision", "--commit", "d00d",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "BASELINE: RETIRED: demo/old.py: deleted on purpose" in result.stdout
    document = ratchet.load_baseline(baseline)  # the tool's own checksum
    assert set(document["modules"]) == {"demo/__init__.py", "demo/logic.py"}
    assert [entry["mode"] for entry in document["history"]] == ["reset", "retire"]
    last = document["history"][-1]
    assert last["reason"] == "old.py deleted by decision"
    assert last["commit"] == "d00d"
    assert last["retired"] == "demo/old.py"
    assert STAMP.match(last["written_at"])

    gate, report, _text = _gate(project, tmp_path, _logic(), baseline=baseline)

    assert gate.returncode == 0, gate.stdout + gate.stderr
    assert report["stale_baseline_entries"] == []


def test_retiring_a_module_that_still_ships_or_is_unknown_is_refused(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _with_old_module(project, _logic()))
    before = baseline.read_bytes()

    shipped = _retire(
        project, baseline,
        "--retire-module", "demo/old.py",
        "--reason", "it was not deleted", "--commit", "d00d",
    )
    unknown = _retire(
        project, baseline,
        "--retire-module", "demo/never.py",
        "--reason", "no such entry", "--commit", "d00d",
    )

    assert shipped.returncode == 2
    assert "demo/old.py still ships" in shipped.stderr
    assert unknown.returncode == 2
    assert "demo/never.py is not in the baseline" in unknown.stderr
    assert baseline.read_bytes() == before


def test_retiring_a_module_needs_a_reason(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _with_old_module(project, _logic()))
    (project / "demo" / "old.py").unlink()
    before = baseline.read_bytes()

    result = _retire(
        project, baseline, "--retire-module", "demo/old.py", "--commit", "d00d",
    )

    assert result.returncode == 2
    assert "needs a non-empty --reason" in result.stderr
    assert baseline.read_bytes() == before


def test_a_module_deleted_without_retirement_still_fails_the_gate(tmp_path):
    """The negative control: deleting a file never clears rule (d) by itself."""
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _with_old_module(project, _logic()))
    (project / "demo" / "old.py").unlink()

    result, report, text = _gate(project, tmp_path, _logic(), baseline=baseline)

    assert result.returncode == 1
    assert report["stale_baseline_entries"] == ["demo/old.py"]
    assert "--retire-module <path> --reason" in text
    assert "demo/old.py" in ratchet.load_baseline(baseline)["modules"]


def test_a_gate_run_still_requires_its_coverage_data_and_reports(tmp_path):
    project = _project(tmp_path / "project")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(project)],
        cwd=project, capture_output=True, text=True, check=False,
    )

    assert result.returncode == 2
    for flag in ("--coverage-json", "--json-out", "--text-out"):
        assert flag in result.stderr


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
        assert "--retire-module" not in content, workflow.name


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


def test_the_shard_count_has_one_value_in_four_places():
    jobs = _combine_job()
    shards = jobs["coverage-shards"]
    shard_script = "\n".join(step.get("run", "") for step in shards["steps"])
    declared = re.search(r"--shard-count (\d+)", shard_script)
    checked = next(
        step["env"]["SPACR_COVERAGE_SHARD_COUNT"]
        for step in jobs["coverage-combine"]["steps"]
        if "SPACR_COVERAGE_SHARD_COUNT" in step.get("env", {})
    )
    gate_script = "\n".join(
        step.get("run", "") for step in jobs["coverage-combine"]["steps"]
    )
    # The gate's integrity check counts the records it expects; a shard
    # count lower than the matrix would let a whole shard's loss go unseen.
    gated = re.search(r"--shard-count (\d+)", gate_script)

    assert declared is not None
    assert gated is not None
    assert (
        len(shards["strategy"]["matrix"]["shard"])
        == int(declared.group(1))
        == int(checked)
        == int(gated.group(1))
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


# -- (f) a crashed worker is an INCOMPLETE MEASUREMENT, not a regression -----
#
# A coverage worker killed by a signal never writes its data (measured on a
# toy package, 2026-09-15; see tests/test_a_crashed_coverage_worker_cannot_
# hide_lost_coverage.py for the real segfault). tools/run_coverage_batches.py
# writes one integrity record per shard; these records are hand-built here in
# that shape so every rule is exercised without a crash.


CRASHED = "tests/test_crashes.py"


def _batch(number=1, *, lost=(), recovered=(), unrecovered=(), discarded=()):
    """One batch entry as tools/run_coverage_batches.py writes it."""
    reasons = [{
        "worker": "gw0",
        "error": "Not properly terminated",
        "running": [f"{lost[0]}::test_crash"],
        "files": list(lost),
    }] if lost else []
    return {
        "batch": number,
        "exit_code": 1 if lost else 0,
        "ledger": f"spacr-xdist-ledger.shard-00.batch-{number:03d}.json",
        "lost": reasons,
        "lost_files": list(lost),
        "recovered_files": [
            {"file": name, "recovered": True, "attempts": []}
            for name in recovered
        ],
        "unrecovered_files": [
            {"file": name, "recovered": False, "attempts": []}
            for name in unrecovered
        ],
        "discarded_data_files": list(discarded),
    }


def _shard_record(directory, shard=0, *, shard_count=1, batches=None, total=None):
    """Write one shard's integrity record; return the gate's arguments."""
    batches = [_batch()] if batches is None else batches
    document = {
        "schema": ratchet.INTEGRITY_SCHEMA,
        "shard_index": shard,
        "shard_count": shard_count,
        "batches_total": len(batches) if total is None else total,
        "batches_finished": len(batches),
        "batches": batches,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"spacr-coverage-integrity.shard-{shard:02d}.json").write_text(
        json.dumps(document), encoding="utf-8",
    )
    return ("--shard-integrity", str(directory), "--shard-count", str(shard_count))


def _unrecovered_crash(tmp_path):
    return _shard_record(
        tmp_path / "shards",
        batches=[_batch(lost=[CRASHED], unrecovered=[CRASHED])],
    )


def test_negative_control_a_real_drop_with_every_shard_complete_is_a_regression(
    tmp_path,
):
    """The crash handling must not soften the ratchet in front of it."""
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=1))
    shards = _shard_record(tmp_path / "shards")

    result, report, text = _gate(
        project, tmp_path, _logic(uncovered_statements=2), *shards,
        baseline=baseline,
    )

    assert result.returncode == 1
    assert report["status"] == "fail"
    assert report["measurement_integrity"]["status"] == "complete"
    logic = _module(report, "demo/logic.py")
    assert logic["failures"] == ["uncovered statements rose from 1 to 2"]
    assert logic["unconfirmed"] == []
    assert "ERROR: demo/logic.py: uncovered statements rose from 1 to 2" in text
    assert "Shard integrity: complete (1 of 1 shards finished" in text
    assert "UNCONFIRMED" not in text
    assert "INCOMPLETE MEASUREMENT" not in text


def test_a_recovered_crash_is_named_and_a_real_drop_beside_it_is_a_regression(
    tmp_path,
):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=1))
    shards = _shard_record(
        tmp_path / "shards",
        batches=[_batch(lost=[CRASHED], recovered=[CRASHED])],
    )

    result, report, text = _gate(
        project, tmp_path, _logic(uncovered_statements=2), *shards,
        baseline=baseline,
    )

    assert result.returncode == 1
    assert report["status"] == "fail"
    assert _module(report, "demo/logic.py")["failures"] == [
        "uncovered statements rose from 1 to 2"
    ]
    assert (
        "RECOVERED: coverage shard 0 batch 1: worker gw0: Not properly "
        f"terminated while running {CRASHED}::test_crash; 1 test file(s) "
        f"re-run serially and their coverage recovered: {CRASHED}"
    ) in text


def test_an_unrecovered_crash_is_an_incomplete_measurement_not_a_regression(
    tmp_path,
):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=1))
    shards = _unrecovered_crash(tmp_path)

    result, report, text = _gate(
        project, tmp_path,
        _logic(uncovered_statements=2, uncovered_branches=1), *shards,
        baseline=baseline,
    )

    assert result.returncode == ratchet.INCOMPLETE_STATUS == 3
    assert report["status"] == "incomplete"
    logic = _module(report, "demo/logic.py")
    assert logic["status"] == "unconfirmed"
    assert logic["failures"] == []
    assert logic["unconfirmed"] == [
        "uncovered statements rose from 1 to 2",
        "uncovered branches rose from 0 to 1",
    ]
    assert report["summary"]["failed_modules"] == 0
    assert report["summary"]["unconfirmed_modules"] == 1
    assert text.startswith(
        "spaCR shipped-module coverage ratchet: INCOMPLETE MEASUREMENT\n"
    )
    assert (
        "INCOMPLETE MEASUREMENT: coverage shard 0 batch 1: worker gw0: Not "
        f"properly terminated while running {CRASHED}::test_crash; the "
        f"coverage of 1 test file(s) could not be recovered: {CRASHED}"
    ) in text
    assert (
        "UNCONFIRMED: demo/logic.py: uncovered statements rose from 1 to 2"
    ) in text
    assert "ERROR: demo/logic.py" not in text


@pytest.mark.parametrize(
    ("kind", "words"),
    [
        ("segfault", "was killed by SIGSEGV (a segfault: the crash family of item 43)"),
        (
            "memory-guard",
            "was ended by the test memory guard (exit 3: its RSS passed "
            "SPACR_TEST_MEMORY_GB, default 6 GB)",
        ),
    ],
)
def test_the_report_names_how_the_worker_ended(tmp_path, kind, words):
    """A segfault points at the crash family, a guard exit at a test that
    needs more memory; xdist logs both as "Not properly terminated"."""
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=1))
    batch = _batch(lost=[CRASHED], unrecovered=[CRASHED])
    batch["lost"][0].update(exit_kind=kind, exit=words, exit_status=0)
    shards = _shard_record(tmp_path / "shards", batches=[batch])

    result, report, text = _gate(
        project, tmp_path, _logic(uncovered_statements=1), *shards,
        baseline=baseline,
    )

    assert result.returncode == 3
    assert report["measurement_integrity"]["loss_kinds"] == {kind: 1}
    assert (
        f"INCOMPLETE MEASUREMENT: coverage shard 0 batch 1: worker gw0 {words} "
        f"[Not properly terminated] while running {CRASHED}::test_crash"
    ) in text
    assert f"lost processes by how they ended: {kind} 1" in text


def test_an_unrecovered_crash_never_passes_even_when_every_count_held(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=1))

    result, report, _text = _gate(
        project, tmp_path, _logic(uncovered_statements=1),
        *_unrecovered_crash(tmp_path), baseline=baseline,
    )

    assert result.returncode == 3
    assert report["status"] == "incomplete"


@pytest.mark.parametrize(
    ("damage", "issue"),
    [
        ("missing", "coverage shard 1 left no integrity record"),
        ("unfinished", "coverage shard 1 finished 1 of 3 batches"),
        ("unreadable", "unreadable shard integrity record"),
        ("other_shard_count", "coverage shard 1 ran as one of 3 shards, not 2"),
    ],
)
def test_a_shard_that_cannot_vouch_for_its_data_makes_the_run_incomplete(
    tmp_path, damage, issue,
):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=1))
    directory = tmp_path / "shards"
    arguments = _shard_record(directory, 0, shard_count=2)
    if damage == "unfinished":
        _shard_record(directory, 1, shard_count=2, total=3)
    elif damage == "unreadable":
        (directory / "spacr-coverage-integrity.shard-01.json").write_text(
            "{not json", encoding="utf-8",
        )
    elif damage == "other_shard_count":
        _shard_record(directory, 1, shard_count=3)

    result, report, text = _gate(
        project, tmp_path, _logic(uncovered_statements=1), *arguments,
        baseline=baseline,
    )

    assert result.returncode == 3
    assert report["status"] == "incomplete"
    assert any(
        issue in entry for entry in report["measurement_integrity"]["issues"]
    ), report["measurement_integrity"]["issues"]
    assert "INCOMPLETE MEASUREMENT: " in text


def test_a_new_pragma_is_still_an_error_when_the_measurement_is_incomplete(
    tmp_path,
):
    """Only the counts a lost worker can raise are downgraded."""
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

    result, report, text = _gate(
        project, tmp_path, _logic(uncovered_statements=2),
        *_unrecovered_crash(tmp_path), baseline=baseline,
    )

    assert result.returncode == 1
    assert report["status"] == "fail"
    module = _module(report, "demo/logic.py")
    assert module["failures"] == ["pragma: no cover comments rose from 1 to 2"]
    assert module["unconfirmed"] == ["uncovered statements rose from 1 to 2"]
    assert (
        "ERROR: demo/logic.py: pragma: no cover comments rose from 1 to 2"
    ) in text
    assert "INCOMPLETE MEASUREMENT: coverage shard 0 batch 1" in text


def test_discarded_data_no_lost_worker_explains_makes_the_run_incomplete(
    tmp_path,
):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=1))
    shell = ".coverage.shard-00.batch-001.host.1.abc"
    shards = _shard_record(
        tmp_path / "shards", batches=[_batch(discarded=[shell])],
    )

    result, _report, text = _gate(
        project, tmp_path, _logic(uncovered_statements=1), *shards,
        baseline=baseline,
    )

    assert result.returncode == 3
    assert (
        f"unreadable coverage data was discarded ({shell}) and no lost "
        "worker explains it"
    ) in text


def test_a_baseline_write_refuses_an_incomplete_measurement(tmp_path):
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _logic(uncovered_statements=2))
    before = baseline.read_bytes()

    result, _report, _text = _gate(
        project, tmp_path, _logic(uncovered_statements=1),
        *_unrecovered_crash(tmp_path),
        "--update-baseline", "--reason", "covered a line", "--commit", "beef",
        baseline=baseline,
    )

    assert result.returncode == 2
    assert "refused: the measurement is incomplete" in result.stderr
    assert "could not be recovered" in result.stderr
    assert baseline.read_bytes() == before


def test_shard_integrity_is_never_checked_without_a_shard_count(tmp_path):
    project = _project(tmp_path / "project")

    result, report, _text = _gate(
        project, tmp_path, _logic(), "--shard-integrity", str(tmp_path),
    )

    assert result.returncode == 2
    assert report is None
    assert "--shard-integrity and --shard-count are given together" in result.stderr


def test_the_gate_reads_the_records_where_the_combine_job_downloads_them():
    jobs = _combine_job()
    combine = jobs["coverage-combine"]
    download = next(
        step for step in combine["steps"]
        if str(step.get("uses", "")).startswith("actions/download-artifact")
    )
    gate = next(
        step for step in combine["steps"]
        if "tools/verify_module_coverage.py" in step.get("run", "")
    )
    shards = jobs["coverage-shards"]
    runs = next(
        step for step in shards["steps"]
        if "tools/run_coverage_batches.py" in step.get("run", "")
    )
    upload = next(
        step for step in shards["steps"]
        if step.get("uses") == "actions/upload-artifact@v7"
    )

    assert '--shard-integrity "$SPACR_COVERAGE_INPUT"' in gate["run"]
    assert gate["env"]["SPACR_COVERAGE_INPUT"] == download["with"]["path"]
    # The runner writes its records into --data-dir, which is what is uploaded.
    assert (
        runs["env"]["SPACR_COVERAGE_DATA_DIR"].rstrip("/")
        == upload["with"]["path"].rstrip("/")
    )


def test_the_runner_writes_the_records_the_gate_reads():
    runner = _load_coverage_runner()

    assert runner.INTEGRITY_SCHEMA == ratchet.INTEGRITY_SCHEMA
    assert fnmatch.fnmatch(runner.integrity_record_name(11), ratchet.INTEGRITY_GLOB)
    # `coverage combine` reads every `.coverage.*` file in the input directory.
    assert not runner.integrity_record_name(0).startswith(".coverage")


# -- 372's --retire-module and 288's --shard-integrity coexist ---------------


def test_retiring_needs_no_integrity_records_and_a_gate_still_demands_them(
    tmp_path,
):
    """A retire run reads no coverage data, so it must not ask for shard
    integrity; a gate run given --shard-integrity must still fail a missing
    shard record as an INCOMPLETE MEASUREMENT, retired module or not."""
    project = _project(tmp_path / "project")
    baseline = _seed(project, tmp_path, _with_old_module(project, _logic()))
    (project / "demo" / "old.py").unlink()
    empty = tmp_path / "no-integrity-records"
    empty.mkdir()

    retired = _retire(
        project, baseline,
        "--retire-module", "demo/old.py",
        "--reason", "old.py deleted by decision", "--commit", "d00d",
        # Pointed at a directory with no records: a retire must ignore it.
        "--shard-integrity", str(empty), "--shard-count", "2",
    )

    assert retired.returncode == 0, retired.stdout + retired.stderr
    assert "BASELINE: RETIRED: demo/old.py" in retired.stdout
    assert "INCOMPLETE" not in retired.stdout + retired.stderr
    assert "demo/old.py" not in ratchet.load_baseline(baseline)["modules"]

    shards = tmp_path / "shards"
    arguments = _shard_record(shards, 0, shard_count=2)  # shard 1 left nothing

    missing, report, text = _gate(
        project, tmp_path, _logic(), *arguments, baseline=baseline,
    )

    assert missing.returncode == 3, missing.stdout + missing.stderr
    assert report["status"] == "incomplete"
    assert report["stale_baseline_entries"] == []  # the retirement held
    assert (
        "INCOMPLETE MEASUREMENT: coverage shard 1 left no integrity record"
    ) in text
    assert "ERROR:" not in text

    # Control: the same run with shard 1's record present passes, so the
    # exit 3 above came from the missing record and not from the retirement.
    _shard_record(shards, 1, shard_count=2)
    complete, report, _text = _gate(
        project, tmp_path, _logic(), *arguments, baseline=baseline,
    )

    assert complete.returncode == 0, complete.stdout + complete.stderr
    assert report["measurement_integrity"]["status"] == "complete"
