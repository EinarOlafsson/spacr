"""The index header must name the folders it reads and the files it skips.

`instructions/` held files in three places -- `open/`, `done/` and its own top
level -- and the rename to `features/` carried all three across. The generator
reads two of the three, and until now the index it writes told a reader it was
"generated from the files themselves", which was true of two thirds of them.
Item 398 is the record; this is the part of it that needed no decision from
anybody: whatever becomes of those files, the index must not describe a scan
it does not perform.

WHY THE LIST IS GLOBBED AND NOT PINNED. The count of these files is already
written down twice -- "eleven stale ledger files" in 398's own title, and
again in `test_a_ledger_file_does_not_contradict_itself.py::_ledger_files` --
and a third test, `test_a_ledger_file_is_not_shadowed_by_a_staler_twin.py`,
pins the eleven NAMES one by one. Prose does not move when a folder does. A
name printed from a live glob is right on the day it is read, and shrinks by
itself the day somebody resolves one. It also puts the top level
under `--check` for the first time: adding a file there now makes the
committed index stale, where before "nothing counts these eleven" was the
complaint 398 was filed to make.

WHAT THIS DOES NOT DO. It does not delete, move or approve anything at the
top level. Two of those files are not duplicates at all, and the decision is
recorded as the maintainer's.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "build_instruction_index.py"
FEATURES = REPO / "features"
INDEX = FEATURES / "00_INDEX.txt"

pytestmark = pytest.mark.skipif(
    not TOOL.exists() or not FEATURES.is_dir(),
    reason="run from a source checkout")


def _tool():
    spec = importlib.util.spec_from_file_location("_instr_index_header", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_instr_index_header"] = module
    spec.loader.exec_module(module)
    return module


def _header(text: str) -> str:
    """Everything above the first section rule."""
    return text.split("-" * 80)[0]


def _fake_features(tool, monkeypatch, tmp_path, strays):
    """A features/ tree with both scanned folders and the given top-level files.

    Hermetic on purpose: the point of the assertions below is that the list is
    read off the filesystem on every run, and that cannot be shown by reading
    the one filesystem the repository happens to have.
    """
    root = tmp_path / "features"
    for folder in tool.SCANNED:
        (root / folder).mkdir(parents=True)
        (root / folder / f"900_{folder}_item.txt").write_text(
            "=" * 80 + f"\nA {folder.upper()} ITEM\n" + "=" * 80 + "\n")
    for name in strays:
        (root / name).write_text("a ledger file nothing indexes\n")
    monkeypatch.setattr(tool, "INSTRUCTIONS", root)
    monkeypatch.setattr(tool, "INDEX", root / "00_INDEX.txt")
    return root


def test_the_header_names_every_folder_it_is_generated_from():
    """"From the files themselves" is not a scan; two globs are."""
    tool = _tool()
    # Both the generator and the artifact it has written: a header that says
    # it only in the renderer leaves the committed index still lying, and one
    # that says it only in the committed file is a hand-edit away from gone.
    for header in (_header(tool.render(today="2026-09-14")),
                   _header(INDEX.read_text())):
        for folder in tool.SCANNED:
            assert f"features/{folder}/*.txt" in header, (
                f"the index is generated from features/{folder}/ and does "
                "not say so; a reader is left to assume it covers the whole "
                "folder")


def test_the_header_lists_every_top_level_file_it_does_not_scan():
    """The files in no list are named in the index, or they are invisible."""
    tool = _tool()
    unscanned = {path.name for path in FEATURES.glob("*.txt")} - {INDEX.name}
    assert unscanned, "no top-level .txt files, so this proves nothing"
    assert tool._unscanned_top_level() == sorted(unscanned, key=str.lower)
    for header in (_header(tool.render(today="2026-09-14")),
                   _header(INDEX.read_text())):
        missing = sorted(name for name in unscanned if name not in header)
        assert missing == [], (
            f"at features/ top level and named nowhere in the index: "
            f"{missing}. These are the files item 398 is about; the index "
            "must at least admit they exist.")


def test_the_index_does_not_list_itself_as_unscanned():
    """A file cannot be an input to its own generation."""
    tool = _tool()
    assert INDEX.name not in tool._unscanned_top_level()
    assert INDEX.name not in _header(tool.render(today="2026-09-14"))
    assert INDEX.name not in _header(INDEX.read_text())


def test_the_list_is_globbed_on_every_run_and_not_pinned(monkeypatch,
                                                         tmp_path):
    """A pinned list is the defect 398 records; a glob cannot go stale.

    Rendered twice over the same tree with one file added in between: the new
    name has to appear without anybody editing the generator.
    """
    tool = _tool()
    root = _fake_features(tool, monkeypatch, tmp_path,
                          ["07_known_bugs_not_fixed.txt", "TEMPLATE.txt"])

    before = _header(tool.render(today="2026-09-14"))
    assert "07_known_bugs_not_fixed.txt" in before
    assert "a_tenth_file_nobody_indexed.txt" not in before

    (root / "a_tenth_file_nobody_indexed.txt").write_text("filed today\n")
    after = _header(tool.render(today="2026-09-14"))
    assert "a_tenth_file_nobody_indexed.txt" in after, (
        "a file added to features/ top level is not in the header, so the "
        "list was pinned rather than globbed")
    assert "07_known_bugs_not_fixed.txt" in after


def test_a_new_top_level_file_now_makes_the_committed_index_stale(monkeypatch,
                                                                  tmp_path):
    """The gap 398 was filed about: nothing counted these files.

    `--check` compares the committed index against a fresh render, ignoring
    only the date stamp. Because the names are printed, a file appearing at
    the top level changes that comparison -- which is the difference between
    a trap that grows silently and one CI reports.
    """
    tool = _tool()
    root = _fake_features(tool, monkeypatch, tmp_path,
                          ["05_version_bump_and_github_actions.txt"])
    body = lambda text: "\n".join(
        line for line in text.splitlines()
        if not line.startswith("Regenerated "))

    committed = tool.render(today="2026-09-14")
    assert body(tool.render(today="2099-01-01")) == body(committed)

    (root / "a_file_left_behind_by_a_rename.txt").write_text("stale copy\n")
    assert body(tool.render(today="2026-09-14")) != body(committed), (
        "a top-level file came and went without the index noticing, which is "
        "the state item 398 recorded: 'nothing counts these eleven'")
