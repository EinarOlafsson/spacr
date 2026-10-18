"""Item 709's open-issue block follows GitHub and the tool is idempotent."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "sync_open_issues_item.py"
WORKFLOW = REPO / ".github" / "workflows" / "issues-item.yml"

pytestmark = pytest.mark.skipif(not TOOL.exists(), reason="run from a source checkout")

ITEM = """\
================================================================================
RESOLVE ALL OPEN GITHUB ISSUES
================================================================================

Status:    OPEN — 9 open GitHub issues
Requested: maintainer, 2026-10-18
Owner:     Claude Home

WHAT TO BUILD
Prose that must survive untouched, including -- dashes -- and #12 references.

--- OPEN ISSUES (auto) ---
#1 stale (someone, 2026-01-01)
--- END OPEN ISSUES ---

2026-10-18 — A dated note after the block.
"""


def _tool():
    spec = importlib.util.spec_from_file_location("_sync_open_issues", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_sync_open_issues"] = module
    spec.loader.exec_module(module)
    return module


GH = [
    {"number": 150, "title": "Float  diameters", "author": {"login": "jak18015"},
     "createdAt": "2026-10-15T14:20:38Z"},
    {"number": 146, "title": "Zero diameter", "author": {"login": "jak18015"},
     "createdAt": "2026-10-13T15:47:53Z"},
]

REST = [
    {"number": 152, "title": "annotate app suggestions", "state": "open",
     "user": {"login": "EinarOlafsson"}, "created_at": "2026-10-16T19:55:35Z"},
    {"number": 151, "title": "a pull request", "state": "open",
     "user": {"login": "bot"}, "created_at": "2026-10-16T00:00:00Z",
     "pull_request": {"url": "https://api.github.com/x"}},
]


def _outside_block(text):
    tool = _tool()
    start, end = text.index(tool.BEGIN), text.index(tool.END)
    lines = (text[:start] + text[end:]).splitlines()
    return [line for line in lines if not line.startswith("Status:")]


def test_block_and_status_follow_open_issues():
    tool = _tool()
    out = tool.rewrite(ITEM, tool.normalise(GH))
    assert "Status:    OPEN — 2 open GitHub issues\n" in out
    assert ("--- OPEN ISSUES (auto) ---\n"
            "#146 Zero diameter (jak18015, 2026-10-13)\n"
            "#150 Float diameters (jak18015, 2026-10-15)\n"
            "--- END OPEN ISSUES ---\n") in out
    assert "#1 stale" not in out


def test_one_issue_is_singular():
    tool = _tool()
    out = tool.rewrite(ITEM, tool.normalise(GH[:1]))
    assert "Status:    OPEN — 1 open GitHub issue\n" in out


def test_no_issues_is_done_and_reopens_on_a_new_issue():
    tool = _tool()
    done = tool.rewrite(ITEM, [])
    assert "Status:    DONE — no open GitHub issues (reopens automatically)\n" in done
    assert "--- OPEN ISSUES (auto) ---\n(none)\n--- END OPEN ISSUES ---" in done
    reopened = tool.rewrite(done, tool.normalise(REST))
    assert "Status:    OPEN — 1 open GitHub issue\n" in reopened
    assert "#152 annotate app suggestions (EinarOlafsson, 2026-10-16)" in reopened


def test_pull_requests_and_closed_issues_are_excluded():
    tool = _tool()
    closed = {"number": 9, "title": "closed", "state": "closed",
              "user": {"login": "x"}, "created_at": "2026-01-01T00:00:00Z"}
    issues = tool.normalise(REST + [closed])
    assert [number for number, *_ in issues] == [152]


def test_rewrite_is_idempotent_and_keeps_the_rest_of_the_file():
    tool = _tool()
    once = tool.rewrite(ITEM, tool.normalise(GH + REST))
    assert tool.rewrite(once, tool.normalise(GH + REST)) == once
    assert _outside_block(once) == _outside_block(ITEM)
    assert once.endswith("2026-10-18 — A dated note after the block.\n")


def test_missing_markers_are_refused():
    tool = _tool()
    with pytest.raises(ValueError):
        tool.rewrite(ITEM.replace("--- END OPEN ISSUES ---", ""), [])
    with pytest.raises(ValueError):
        tool.rewrite(ITEM.replace("Status:", "State:"), [])


def test_cli_writes_only_when_changed(tmp_path, capsys):
    tool = _tool()
    item = tmp_path / "709.txt"
    item.write_text(ITEM, encoding="utf-8")
    data = tmp_path / "issues.json"
    data.write_text(json.dumps(GH), encoding="utf-8")
    assert tool.main(["--json", str(data), "--item", str(item)]) == 0
    assert "updated" in capsys.readouterr().out
    first = item.read_text(encoding="utf-8")
    assert tool.main(["--json", str(data), "--item", str(item)]) == 0
    assert "unchanged" in capsys.readouterr().out
    assert item.read_text(encoding="utf-8") == first


def test_the_committed_item_has_both_markers():
    tool = _tool()
    text = tool.ITEM.read_text(encoding="utf-8")
    assert text.count(tool.BEGIN) == 1 and text.count(tool.END) == 1
    assert tool._STATUS.search(text)


def test_the_workflow_syncs_only_the_item_on_issue_events():
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    triggers = document.get("on", document.get(True))
    assert set(triggers["issues"]["types"]) == {
        "opened", "reopened", "closed", "deleted", "transferred"}
    assert "workflow_dispatch" in triggers
    assert document["permissions"] == {"contents": "write", "issues": "read"}
    assert document["concurrency"] == {"group": "issues-item",
                                       "cancel-in-progress": False}
    steps = document["jobs"]["sync"]["steps"]
    assert steps[0]["uses"] == "actions/checkout@v7"
    assert steps[0]["with"]["ref"] == "nightly"
    script = steps[-1]["run"]
    assert "tools/sync_open_issues_item.py" in script
    assert 'git add -- "$ITEM"' in script
    assert "709: sync open issues" in script
    assert "[skip ci]" not in script
    assert "github-actions[bot]" in script
    assert "git push origin HEAD:nightly" in script
