"""A bump redraws the README images after it restamps the deck.

The README cover is drawn from the deck's title slide. A bump that
restamped the slide and left the README alone shipped a stale cover, and
the release CI's README check failed on it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tests.test_release_metadata_sync import CITATION, _tree

ROOT = Path(__file__).resolve().parents[1]


def _release():
    spec = importlib.util.spec_from_file_location(
        "spacr_release_helper_images", ROOT / "packaging" / "release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_steps_restamp_the_deck_then_redraw_the_readme():
    release = _release()
    assert release.BUMP_ARTEFACT_STEPS == (release.restamp_info_deck,
                                           release.rebuild_readme_images)


def test_a_bump_runs_every_step_with_the_new_version(tmp_path, monkeypatch):
    release = _release()
    root = _tree(tmp_path, recipe=False)
    citation = root / "CITATION.cff"
    citation.write_text(CITATION.replace("spaCR 1.5.0.9.", "spaCR 1.5.1.0."),
                        encoding="utf-8")
    ran = []
    monkeypatch.setattr(release, "BUMP_ARTEFACT_STEPS", (
        lambda path, version: ran.append(("deck", path, version)),
        lambda path, version: ran.append(("images", path, version))))
    release.bump_release(root / "setup.py", citation, "1.5.1.1",
                         release_date="2026-09-30")
    assert ran == [("deck", root, "1.5.1.1"), ("images", root, "1.5.1.1")]


def test_the_images_step_runs_the_tool_in_the_tree(tmp_path, monkeypatch):
    release = _release()
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "build_readme_images.py").write_text(
        "from pathlib import Path\nPath('built').write_text('yes')\n", encoding="utf-8")
    (tmp_path / "README.rst").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(release, "README_IMAGE_MODULES", ())
    assert release.rebuild_readme_images(tmp_path, "1.5.1.1") is True
    assert (tmp_path / "built").read_text(encoding="utf-8") == "yes"


def test_the_images_step_skips_without_its_modules_or_tool(tmp_path,
                                                          monkeypatch, capsys):
    release = _release()
    assert release.rebuild_readme_images(tmp_path, "1.5.1.1") is False
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "build_readme_images.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
    (tmp_path / "README.rst").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(release, "README_IMAGE_MODULES", ("no_such_module_x",))
    assert release.rebuild_readme_images(tmp_path, "1.5.1.1") is False
    assert "no_such_module_x" in capsys.readouterr().err
    monkeypatch.setattr(release, "README_IMAGE_MODULES", ())
    assert release.rebuild_readme_images(tmp_path, "1.5.1.1") is False
    assert "failed (3)" in capsys.readouterr().err


def test_preparation_refreshes_existing_news_before_freezing_the_release_source():
    import shlex
    import yaml

    jobs = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]
    steps = jobs["bump"]["steps"]
    refresh = [index for index, step in enumerate(steps)
               if step.get("name") == "Refresh notes for releases already published"]
    assert len(refresh) == 1
    version = next(index for index, step in enumerate(steps) if step.get("id") == "version")
    commit = next(index for index, step in enumerate(steps) if step.get("id") == "commit")
    finalize = next(index for index, step in enumerate(steps) if step.get("id") == "finalize")
    assert refresh[0] < version < commit < finalize
    assert steps[refresh[0]]["env"]["GH_TOKEN"] == "${{ github.token }}"
    staging = next(line for line in steps[commit]["run"].splitlines()
                   if line.strip().startswith("git add "))
    assert "spacr/resources/release_notes.json" in shlex.split(staging)
    after_publication = jobs["release-notes"]
    assert "github-release" in after_publication["needs"]
    assert "needs.github-release.result == 'success'" in after_publication["if"]
    assert sum("python tools/build_release_notes.py" in step.get("run", "")
               for step in after_publication["steps"]) == 1


def test_preparation_news_producer_keeps_published_history_without_announcing_drafts(
        tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "release_preparation_news", ROOT / "tools/build_release_notes.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    target = tmp_path / "release_notes.json"
    monkeypatch.setattr(tool, "TARGET", target)
    published = {"tag": "v1.5.1.5", "name": "spaCR 1.5.1.5",
                 "published": "2026-10-01T00:00:00Z", "url": "https://example.invalid/1.5.1.5",
                 "body": "Published note https://example.invalid/details", "prerelease": False}
    monkeypatch.setattr(tool, "fetch", lambda: [dict(published, tag="v1.5.1.6", draft=True),
                                               published])
    assert tool.main([]) == 0
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result["releases"] == [{
        "tag": "v1.5.1.5", "name": "spaCR 1.5.1.5", "published": "2026-10-01",
        "url": published["url"], "body": published["body"],
        "links": ["https://example.invalid/details"], "prerelease": False}]
    before = target.read_bytes()
    monkeypatch.setattr(tool, "fetch", lambda: [dict(published, body="Updated published note")])
    assert tool.main(["--check"]) == 1
    assert target.read_bytes() == before
    def unavailable():
        raise RuntimeError("release endpoint unavailable")
    monkeypatch.setattr(tool, "fetch", unavailable)
    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        tool.main([])
    assert target.read_bytes() == before


@pytest.mark.parametrize("event,expected", [
    ("workflow_dispatch", ["tools/build_release_notes.py"]),
    ("push", ["tools/build_release_notes.py", "--check"]),
])
@pytest.mark.parametrize("result", [0, 3])
def test_news_preparation_refreshes_only_when_the_result_will_be_committed(
        tmp_path, event, expected, result):
    import os
    import subprocess
    import yaml

    steps = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]["bump"]["steps"]
    step = next(step for step in steps
                if step.get("name") == "Refresh notes for releases already published")
    executable = tmp_path / "python"
    executable.write_text("""#!/bin/sh
printf '%s\\n' "$@" > "$NEWS_CALL"
exit "$NEWS_RESULT"
""")
    executable.chmod(0o755)
    call = tmp_path / "call"
    environment = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
                       NEWS_CALL=str(call), NEWS_RESULT=str(result))
    completed = subprocess.run(["bash", "-e", "-c", step["run"].replace(
        "${{ github.event_name }}", event)], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=10)
    assert completed.returncode == result
    assert call.read_text().splitlines() == expected
