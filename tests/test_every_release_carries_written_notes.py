"""Every release publishes its hand-written summary of changes (2026-11-05)."""
from pathlib import Path

import yaml

from tools import require_release_notes as notes

ROOT = Path(__file__).resolve().parents[1]


def test_the_published_releases_have_notes_that_pass():
    for version in ("1.5.1.5", "1.5.1.6"):
        assert notes.problems(version) == []


def test_missing_short_or_unsectioned_notes_are_refused(tmp_path):
    notes_dir = tmp_path / "docs" / "notes"
    notes_dir.mkdir(parents=True)
    assert "does not exist" in notes.problems("9.9.9", notes_dir)[0]
    notes.notes_path("9.9.9", notes_dir).write_text("## spaCR 9.9.9\n\nSmall.\n")
    found = notes.problems("9.9.9", notes_dir)
    assert any("### New" in problem for problem in found)
    assert any("characters" in problem for problem in found)
    body = "## spaCR 9.9.9\n\n### New\n\n" + "- A change.\n" * 200
    notes.notes_path("9.9.9", notes_dir).write_text(body)
    assert notes.problems("9.9.9", notes_dir) == []
    assert notes.main(["9.9.8"]) == 1


def test_the_release_workflow_checks_and_publishes_the_notes_file():
    jobs = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]
    validate = next(step for step in jobs["bump"]["steps"]
                    if step.get("name") == "Validate release version")
    assert 'python tools/require_release_notes.py "$version"' in validate["run"]
    publish = next(step for step in jobs["github-release"]["steps"]
                   if step.get("name") == "Publish release notes and downloadable installers")
    assert "--generate-notes" not in publish["run"]
    assert '--notes-file "$notes"' in publish["run"]
