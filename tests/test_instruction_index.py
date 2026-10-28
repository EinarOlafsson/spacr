"""The instruction index cannot go stale without CI saying so.

It was hand-written on 2026-08-05 and was nine days out of date by
2026-08-14: it listed work that had shipped and omitted work that had been
filed. An index that disagrees with the folder is worse than none, because it
is believed -- and this repository has already lost time to eight instruction
files that were wrong about their own state.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "build_instruction_index.py"
INSTRUCTIONS = REPO / "features"

pytestmark = pytest.mark.skipif(
    not TOOL.exists() or not INSTRUCTIONS.is_dir(),
    reason="run from a source checkout")


def _tool():
    spec = importlib.util.spec_from_file_location("_instr_index", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_instr_index"] = module
    spec.loader.exec_module(module)
    return module


def test_the_committed_index_matches_the_instruction_files():
    """If this fails, run tools/build_instruction_index.py."""
    result = subprocess.run(
        [sys.executable, str(TOOL), "--check"], cwd=str(REPO),
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_lifecycle_record_appears_in_its_own_section():
    tool = _tool()
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text(encoding="utf-8")
    future_block = text.split("FUTURE FEATURES")[1].split("NEW FEATURES")[0]
    new_block = text.split("NEW FEATURES")[1]
    for category, section, other in (("future", future_block, new_block),
                                     ("new", new_block, future_block)):
        rows = tool._entries(category)
        assert rows, category
        for _number, _title, name in rows:
            assert name in section
            assert name not in other


def test_the_counts_are_the_real_counts():
    """The index's own tally must be the folders, counted.

    RE-POINTED 2026-09-12, not relaxed. The single instructions list became
    two feature lists, so the sentence the index prints changed with it --
    the assertion still compares the printed numbers against a fresh count
    of both folders, and still fails if either drifts by one.
    """
    tool = _tool()
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text(encoding="utf-8")
    n_future = len(tool._entries("future"))
    n_new = len(tool._entries("new"))
    assert (f"{n_new} Lifecycle: new, {n_future} Lifecycle: future"
            in text)
    # The maintainer explicitly broadened release 1.5.1.5 on 2026-10-14.
    # Preserve the exact scope and the distinction from historical headers.
    assert "BOTH LISTS ARE REQUIRED FOR RELEASE 1.5.1.5" in text
    assert "Historical DONE headers alone" in text
    assert "NEITHER LIST BLOCKS A RELEASE" not in text


def test_duplicate_instruction_numbers_are_ordered_by_filename():
    """A duplicate numeric id must not inherit filesystem iteration order.

    READS `new` AND `future`, THE FOLDERS THAT EXIST. This asked for "done"
    until 2026-09-13, which was the folder's name before the rename. Nothing
    errored: `_entries` globs a missing directory and gets nothing back, so
    the assertion compared an empty list against its own sort and passed for
    that reason. `new` carries 390 rows and eight duplicate ids -- 58, 82, 83,
    84 among them -- which is precisely the case this test is named for, and
    none of it was being read.
    """
    tool = _tool()
    for folder in ("new", "future"):
        rows = tool._entries(folder)
        assert rows, f"{folder} is empty, so this test proves nothing"
        assert rows == sorted(rows, key=lambda row: (int(row[0]), row[2]))


def test_titles_are_read_from_both_instruction_formats():
    """Recent concise records must not become blank index rows."""
    tool = _tool()
    assert tool._instruction_title(
        ["=" * 80, "A STRUCTURED TITLE", "=" * 80], "305", "fallback"
    ) == "A STRUCTURED TITLE"
    assert tool._instruction_title(
        ["304 — Release 1.5.0.5, archived on Zenodo", "", "Asked today"],
        "304", "fallback"
    ) == "Release 1.5.0.5, archived on Zenodo"


def test_generated_titles_brand_the_product_without_rewriting_package_names():
    tool = _tool()
    title = "spaCR".upper() + " UI and spacr.settings"
    assert tool._instruction_title(["=" * 5, title, "=" * 5], "43", "fallback") == "spaCR UI and spacr.settings"
    assert tool._instruction_title(["43 — " + title], "43", "fallback") == "spaCR UI and spacr.settings"


def test_rendered_index_has_no_trailing_whitespace():
    tool = _tool()
    assert not [line for line in tool.render().splitlines()
                if line != line.rstrip()]


def test_codex_owned_open_files_are_marked_do_not_touch():
    """Two sessions editing one file is how work gets lost."""
    tool = _tool()
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text(encoding="utf-8")
    open_numbers = {
        number for number, _title, _name in tool._entries("future")
    }
    # An owner whose instruction has since been DONE is the normal end state.
    # Asserting every owner is still open made the index fail for work being
    # finished, which is the opposite of what this guard is for.
    stale = {n for n in tool.OWNERS if n not in open_numbers}
    assert not stale, (
        f"OWNERS names {sorted(stale)}, which are no longer open; remove "
        f"them so the marking tracks the folder")
    for number in tool.OWNERS:
        paths = [INSTRUCTIONS / "feature_list" / name for n, _title, name
                 in tool._entries("future") if n == number]
        assert len(paths) == 1, number
        block = [b for b in text.split("\n\n") if paths[0].name in b]
        assert block, f"instruction {number} is not in the index"
        assert "DO NOT TOUCH" in block[0], number


def test_the_date_alone_does_not_make_it_stale():
    """An index that failed CI because a day passed is one nobody keeps."""
    tool = _tool()
    fresh = tool.render(today="2099-01-01")
    other = tool.render(today="1999-01-01")
    body = lambda t: "\n".join(l for l in t.splitlines()
                               if not l.startswith("Regenerated "))
    assert body(fresh) == body(other)


def test_the_index_points_at_the_handoff():
    """A new session reads the index; the traps are in the handoff."""
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text(encoding="utf-8")
    assert "spacr/resources/AI/HANDOFF.md" in text
    assert (INSTRUCTIONS.parent / "spacr/resources/AI/HANDOFF.md").exists()


def test_an_open_status_line_does_not_contradict_its_own_body():
    """A Status of "not started" over a body recording the work landing.

    TWICE IN ONE DAY on 2026-09-08/09. Instruction 383's header read "not
    started" while three of its four items were done and recorded below it;
    382's read the same while the body carried the commit that closed it.
    Both were found by a person reading the file, which is the check that
    does not scale -- the header is what a session reads to decide what to
    work on, so a stale one costs a whole session or duplicates work
    another one has already finished.

    The rule is narrow on purpose: it fires only when the header claims
    NOTHING has happened and the body says otherwise in the form this
    ledger actually uses -- a dated entry announcing the work as done. It
    does not police percentages, partial progress, or the many honest ways
    a status can lag its body by a little.
    """
    import re

    stale = []
    for _number, _title, name in _tool()._entries("future"):
        path = INSTRUCTIONS / "feature_list" / name
        text = path.read_text(encoding="utf-8")
        header = re.search(r"^Status:\s*(.+?)(?=^\w+:|\Z)", text,
                           re.M | re.S)
        if not header:
            continue
        claim = " ".join(header.group(1).split()).lower()
        if not claim.startswith("not started"):
            continue
        # A dated section announcing completion, which is how this ledger
        # records it: "2026-09-09 -- ... DONE" or "... IS DONE".
        landed = re.search(r"^\d{4}-\d{2}-\d{2}[^\n]*\b(IS DONE|DONE)\b",
                           text, re.M)
        if landed:
            stale.append(f"{path.name}: says 'not started', body says "
                         f"{landed.group(0)[:60]!r}")
    assert not stale, (
        "an open instruction's Status contradicts its own body:\n  "
        + "\n  ".join(stale))


def test_lifecycle_metadata_cannot_become_a_title():
    tool = _tool()
    for lines in (["", "Lifecycle: new", "A title", "Status: OPEN"],
                  ["Lifecycle: future", "402 — A title", "Status: DONE"]):
        assert tool._instruction_title(lines, "402", "fallback") == "A title"


def test_canonical_list_keeps_every_filename_and_disambiguates_shared_ids():
    tool = _tool()
    text = (INSTRUCTIONS / "FEATURE_LIST.md").read_text(encoding="utf-8")
    all_items = text.split("## All feature items", 1)[1]
    paths = tool._item_paths()
    assert len(paths) == len(tool._entries("new")) + len(tool._entries("future"))
    for path in paths:
        assert all_items.count(f"](feature_list/{path.name})") == 1
        assert tool._recorded_status(path.read_text(encoding="utf-8")).replace("|", "\\|") in all_items
    for number in tool._duplicate_numbers():
        assert f"| {number} (shared ID) |" in all_items
    assert "not a verified unfinished count" in text
    assert "Neither is a verification result" in text


def test_new_record_in_old_folder_is_rejected_instead_of_disappearing(tmp_path, monkeypatch):
    tool = _tool()
    monkeypatch.setattr(tool, "INSTRUCTIONS", tmp_path)
    (tmp_path / "new").mkdir()
    (tmp_path / "new" / "901_lost.txt").write_text("901 — Lost\nLifecycle: new\n")
    with pytest.raises(ValueError, match="outside features/feature_list"):
        tool._item_paths()


@pytest.mark.parametrize("metadata", ["", "Lifecycle: alpha", "Lifecycle: new\nLifecycle: future"])
def test_item_category_is_mandatory_and_unambiguous(tmp_path, monkeypatch, metadata):
    tool = _tool()
    monkeypatch.setattr(tool, "INSTRUCTIONS", tmp_path)
    folder = tmp_path / "feature_list"
    folder.mkdir()
    (folder / "901_test.txt").write_text(f"901 — Test\n{metadata}\nStatus: OPEN\n")
    with pytest.raises(ValueError, match="exactly one Lifecycle"):
        tool.render_feature_list()


def test_recorded_header_status_does_not_reopen_an_archived_status_example():
    tool = _tool()
    # Item288 retains its original not-started record after its current OPEN header.
    source = "288 — Current item\nLifecycle: future\nStatus: OPEN today\n\nArchived original:\nStatus: not started years ago\n"
    assert tool._recorded_status(source) == "OPEN today"
    assert tool._recorded_status("No authoritative status declaration\n") == "Unspecified"
