"""Retained coverage cannot become a different branch, source, or measurement."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tools.verify_codecov_reupload import verify_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def report(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    coverage = {
        "meta": {"version": "7.16.2", "branch_coverage": True},
        "totals": {"percent_covered": 75.0},
        "files": {"spacr/example.py": {
            "executed_lines": [1, 2], "missing_lines": [3],
            "executed_branches": [[1, 2]], "missing_branches": [[1, 3]],
        }},
    }
    gate = {"status": "fail", "measurement_integrity": {
        "checked": True, "status": "complete", "issues": [],
        "shard_count": 12, "shards_with_records": 12,
    }}
    (original / "coverage.json").write_text(json.dumps(coverage))
    (original / "module-coverage-ratchet.json").write_text(json.dumps(gate))
    return original, coverage, gate


def test_complete_measurement_can_be_uploaded_without_overriding_ratchet_failure(report):
    original, coverage, _ = report
    regenerated = original.parent / "regenerated.json"
    coverage["meta"]["timestamp"] = "a later rendering of the same measurement"
    regenerated.write_text(json.dumps(coverage))
    assert verify_report(original, regenerated) == "7.16.2"
    assert json.loads((original / "module-coverage-ratchet.json").read_text(encoding="utf-8"))["status"] == "fail"


@pytest.mark.parametrize("damage", ["missing_shard", "missing_issues", "unchecked"])
def test_missing_original_integrity_is_refused(report, damage):
    original, _, gate = report
    integrity = gate["measurement_integrity"]
    if damage == "missing_shard":
        integrity["shards_with_records"] = 11
    elif damage == "missing_issues":
        del integrity["issues"]
    else:
        integrity["checked"] = False
    (original / "module-coverage-ratchet.json").write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="integrity record is missing"):
        verify_report(original)


def test_existing_successful_combine_policy_discloses_incomplete_measurement(report, capsys):
    original, _, gate = report
    gate["status"] = "incomplete"
    gate["measurement_integrity"]["status"] = "incomplete"
    gate["measurement_integrity"]["issues"] = ["a coverage file was discarded"]
    (original / "module-coverage-ratchet.json").write_text(json.dumps(gate))
    assert verify_report(original) == "7.16.2"
    output = capsys.readouterr().out
    assert "Original measurement integrity: incomplete" in output
    assert "Original coverage gate: incomplete" in output
    assert "a coverage file was discarded" in output


@pytest.mark.parametrize("damage", ["branch", "line", "total", "version"])
def test_replay_must_preserve_all_measurements(report, damage):
    original, coverage, _ = report
    if damage == "branch":
        coverage["files"]["spacr/example.py"]["missing_branches"] = []
    elif damage == "line":
        coverage["files"]["spacr/example.py"]["executed_lines"] = [1, 2, 3]
    elif damage == "total":
        coverage["totals"]["percent_covered"] = 100.0
    else:
        coverage["meta"]["version"] = "7.17.0"
    regenerated = original.parent / "regenerated.json"
    regenerated.write_text(json.dumps(coverage))
    with pytest.raises(ValueError, match="differs from the original"):
        verify_report(original, regenerated)


def test_upload_mode_is_isolated_and_codecov_identity_is_original():
    docs = yaml.safe_load((ROOT / ".github/workflows/docs.yml").read_text(encoding="utf-8"))
    upload = docs["jobs"]["upload-coverage"]
    assert upload["if"] == "github.event_name == 'workflow_dispatch' && inputs.mode == 'upload-coverage'"
    assert docs["jobs"]["resolve"]["if"] == "github.event_name != 'workflow_dispatch' || inputs.mode == 'publish'"
    assert docs["jobs"]["build"]["needs"] == "resolve"
    assert upload["with"]["source_run_id"] == "${{ inputs.coverage_run_id }}"
    workflow = yaml.safe_load((ROOT / ".github/workflows/coverage-reupload.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["upload"]["steps"]
    codecov = next(step for step in steps if step.get("uses") == "codecov/codecov-action@v7")
    assert codecov["with"]["override_branch"] == "main"
    assert codecov["with"]["override_commit"] == "${{ steps.source.outputs.sha }}"
    assert codecov["with"]["fail_ci_if_error"] is False
    assert codecov["with"]["disable_search"] is True
    checkout = next(step for step in steps if step.get("name") == "Check out the measured source")
    assert checkout["with"]["ref"] == "${{ steps.source.outputs.sha }}"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required to exercise the Actions resolver")
@pytest.mark.parametrize("damage", [None, "nightly", "foreign", "pr", "unfinished", "diverged", "combine", "expired"])
def test_actual_actions_resolver_refuses_wrong_source_and_missing_measurement(damage):
    workflow = yaml.safe_load((ROOT / ".github/workflows/coverage-reupload.yml").read_text(encoding="utf-8"))
    script = workflow["jobs"]["upload"]["steps"][0]["with"]["script"]
    run = {
        "repository": {"full_name": "owner/repo"},
        "head_repository": {"full_name": "owner/repo"},
        "event": "push", "head_branch": "main", "status": "completed",
        "path": ".github/workflows/tests.yml", "head_sha": "a" * 40,
        "run_attempt": 1, "html_url": "https://github.com/owner/repo/actions/runs/42",
    }
    comparison = "ahead"
    conclusion = "success"
    shards = 12
    if damage == "nightly":
        run["head_branch"] = "nightly"
    elif damage == "foreign":
        run["head_repository"]["full_name"] = "someone/fork"
    elif damage == "pr":
        run["event"] = "pull_request"
    elif damage == "unfinished":
        run["status"] = "in_progress"
    elif damage == "diverged":
        comparison = "diverged"
    elif damage == "combine":
        conclusion = "failure"
    elif damage == "expired":
        shards = 11
    setup = f"""
const run = {json.dumps(run)};
const outputs = {{}};
const context = {{repo: {{owner: 'owner', repo: 'repo'}}}};
process.env.SOURCE_RUN_ID = '42';
const core = {{setOutput: (name, value) => outputs[name] = value}};
const github = {{rest: {{actions: {{
  getWorkflowRun: async () => ({{data: run}}),
  listJobsForWorkflowRun: 'jobs', listWorkflowRunArtifacts: 'artifacts'
}}, repos: {{compareCommitsWithBasehead: async () => ({{data: {{status: {json.dumps(comparison)}}}}})}}}},
paginate: async endpoint => endpoint === 'jobs' ? [{{
  name: 'Coverage / every module at 90% and none loses coverage',
  steps: [{{name: 'Combine process data and write coverage.py JSON', conclusion: {json.dumps(conclusion)}}}]
}}] : [{{name: 'spacr-module-coverage-report-42-1', expired: false}},
  ...Array.from({{length: {shards}}}, (_, i) => ({{name: `spacr-coverage-data-42-1-${{i}}`, expired: false}}))]
}};
"""
    command = setup + "(async () => {\n" + script + "\nconsole.log(JSON.stringify(outputs));\n})().catch(error => {console.error(error.message); process.exit(1);});"
    result = subprocess.run(["node", "-e", command], capture_output=True, text=True, timeout=10)
    if damage is None:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["sha"] == "a" * 40
    else:
        assert result.returncode == 1
        assert result.stdout == ""
