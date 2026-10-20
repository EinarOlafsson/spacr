"""A bump redraws the README images after it restamps the deck.

The README cover is drawn from the deck's title slide. A bump that
restamped the slide and left the README alone shipped a stale cover, and
the release CI's README check failed on it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
