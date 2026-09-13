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


def test_every_open_instruction_appears():
    tool = _tool()
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text()
    for path in (INSTRUCTIONS / "future").glob("*.txt"):
        assert path.name in text, f"{path.name} is missing from the index"


def test_a_done_instruction_is_not_listed_as_open():
    """A CLOSED item never appears under OPEN -- unless it was REOPENED.

    An item number can come back. 59, 82 and 83 were each closed and then
    opened again with fresh content -- "the maintainer reordered the
    release ahead of the rewrite" is the commit that did it for 82 -- so
    the same filename legitimately exists in both folders, the done copy
    being the earlier closure and the open copy the live item.

    THE GUARD USED TO READ THAT AS A STALE INDEX and failed. What it is
    actually for is catching an index that still advertises work already
    finished, which is a file present ONLY in `done/`. A file present in
    both is a reopening, and the honest index entry for it is the OPEN one
    -- which is what `test_a_reopened_instruction_is_listed_once_and_open`
    below now asserts, so the pair is no weaker than the single rule was.
    """
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text()
    open_block = text.split("OPEN")[1].split("DONE")[0]
    for path in (INSTRUCTIONS / "new").glob("*.txt"):
        if (INSTRUCTIONS / "future" / path.name).exists():
            continue
        assert path.name not in open_block, (
            f"{path.name} is done but listed under OPEN")


def test_a_reopened_instruction_is_listed_once_and_open():
    """The other half: a reopened item is OPEN in the index and not DONE.

    Listing it in both blocks would make the count wrong and would let a
    reader take the closure for the current state -- which is the failure
    the rule above exists to prevent, in the one case that rule now skips.
    """
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text()
    # RE-POINTED 2026-09-12 at the section names the index now prints. The
    # two blocks are still the two lists and the assertions below are
    # unchanged; only the words that separate them moved.
    future_block = text.split("FUTURE FEATURES")[1].split("NEW FEATURES")[0]
    new_block = text.split("NEW FEATURES")[1]
    reopened = [path.name for path in (INSTRUCTIONS / "new").glob("*.txt")
                if (INSTRUCTIONS / "future" / path.name).exists()]
    for name in reopened:
        assert name in future_block, (
            f"{name} is on the future list but is not printed under it")
        assert name not in new_block, (
            f"{name} is printed under both lists; a reader cannot tell "
            "which of the two describes the current state")


def test_the_counts_are_the_real_counts():
    """The index's own tally must be the folders, counted.

    RE-POINTED 2026-09-12, not relaxed. The single instructions list became
    two feature lists, so the sentence the index prints changed with it --
    the assertion still compares the printed numbers against a fresh count
    of both folders, and still fails if either drifts by one.
    """
    tool = _tool()
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text()
    n_future = len(list((INSTRUCTIONS / "future").glob("*.txt")))
    n_new = len(list((INSTRUCTIONS / "new").glob("*.txt")))
    assert (f"{n_new} in features/new, {n_future} in features/future"
            in text)
    # The lists stopped being a gate on the same day they were split, and
    # the index is where a reader finds that out.
    assert "NEITHER LIST BLOCKS A RELEASE" in text


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


def test_rendered_index_has_no_trailing_whitespace():
    tool = _tool()
    assert not [line for line in tool.render().splitlines()
                if line != line.rstrip()]


def test_codex_owned_open_files_are_marked_do_not_touch():
    """Two sessions editing one file is how work gets lost."""
    tool = _tool()
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text()
    open_numbers = {
        path.name.split("_", 1)[0]
        for path in (INSTRUCTIONS / "future").glob("*.txt")
    }
    # An owner whose instruction has since been DONE is the normal end state.
    # Asserting every owner is still open made the index fail for work being
    # finished, which is the opposite of what this guard is for.
    stale = {n for n in tool.OWNERS if n not in open_numbers}
    assert not stale, (
        f"OWNERS names {sorted(stale)}, which are no longer open; remove "
        f"them so the marking tracks the folder")
    for number in tool.OWNERS:
        paths = list((INSTRUCTIONS / "future").glob(f"{number}_*.txt"))
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
    text = (INSTRUCTIONS / "00_INDEX.txt").read_text()
    assert "HANDOFF.md" in text
    assert (INSTRUCTIONS / "HANDOFF.md").exists()


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
    for path in sorted((INSTRUCTIONS / "future").glob("*.txt")):
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
