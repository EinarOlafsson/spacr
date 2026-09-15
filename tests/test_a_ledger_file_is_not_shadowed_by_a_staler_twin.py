"""A ledger file at `features/` top level with a twin in `new/` is a trap.

`instructions/` held files in three places -- `open/`, `done/` and its own top
level -- and the rename to `features/` carried all three across. The index
generator scans `new/` and `future/` only, so the files left at the top level
appear in NO index, and the index header's claim to be generated from the
files themselves is true of two folders out of three.

WHAT MAKES IT A TRAP RATHER THAN UNTIDINESS. Nine of them have a same-named
counterpart in `features/new/`, and in every case the top-level copy is the
STALER one -- while being one directory SHALLOWER, so it is the one a reader
reaches first. `03_five_order_dependent_qt_failures.txt` tells that reader the
five Qt failures are "NOT DIAGNOSED"; its counterpart says "DONE, 2026-08-29 --
the exact combined-order reproducer is green" and carries the whole diagnosis
in the 238 lines the shallow copy does not have.

  THIS TEST DOES NOT DELETE OR MOVE ANYTHING. Item 398 records that deciding
  the nine is the maintainer's call, and it still is. What was missing was any
  way to COUNT them: the index cannot see a top-level file, so nothing goes
  red when another one appears. They were found by grepping for a path that no
  longer exists, which is not a method.

So this pins the known set and fails on a TENTH. A file added to the top level
tomorrow, or a new twin created for one that is currently unique, is a fresh
instance of a trap somebody already paid for.
"""
from __future__ import annotations

import pathlib

FEATURES = pathlib.Path(__file__).resolve().parent.parent / "features"

#: Top-level ledger files that are NOT twins -- real, single copies that
#: happen to live here. They are pinned so that growing a twin for one of them
#: is a failure rather than a silent addition to the trap.
KNOWN_UNIQUE = {
    "segmentation_qc_field_browser.txt",
    "update_tutorials_README.txt",
}

#: Generated or structural, not ledger items.
NOT_LEDGER = {"00_INDEX.txt", "TEMPLATE.txt"}

#: The nine shadowed files as measured 2026-09-13, every one staler than its
#: counterpart in `features/new/`. Pinned, not approved: the decision to
#: delete or move them is recorded in item 398 as the maintainer's.
KNOWN_SHADOWED = frozenset()  # 2026-09-15: the nine were deleted by the maintainer's decision (item 398)


def _top_level():
    return {p.name for p in FEATURES.glob("*.txt")} - NOT_LEDGER


def test_there_are_top_level_ledger_files_to_check():
    """Guards the guard: an empty folder satisfies every assertion below."""
    # 2026-09-15: the floor was 10 while nine stale twins sat at the top
    # level; the maintainer had them deleted (item 398), leaving the two
    # files with no counterpart. The guard still refuses an empty folder.
    assert len(_top_level()) >= 2, sorted(_top_level())


def test_no_new_top_level_ledger_file_appears():
    """A tenth file here joins a trap that already cost somebody a sweep."""
    unexpected = _top_level() - KNOWN_SHADOWED - KNOWN_UNIQUE
    assert unexpected == set(), (
        "new ledger files at `features/` top level, where no index can see "
        f"them: {sorted(unexpected)}. Put them in `features/new/` or "
        "`features/future/`, which is where the index generator looks."
    )


def test_the_known_unique_files_have_not_grown_a_twin():
    """Two copies of one name is the defect; one copy anywhere is not."""
    grew = {name for name in KNOWN_UNIQUE if (FEATURES / "new" / name).exists()}
    assert grew == set(), (
        f"these had no counterpart and now do: {sorted(grew)}. Two files with "
        "one name disagree with each other, and the shallower one is the one a "
        "reader opens first."
    )


def test_every_shadowed_file_really_is_shadowed():
    """If a twin disappears, this list is stale and should shrink with it."""
    missing = {
        name for name in KNOWN_SHADOWED
        if not (FEATURES / "new" / name).exists()
    }
    assert missing == set(), (
        f"pinned as shadowed but no longer has a counterpart: {sorted(missing)}. "
        "If the duplicate was resolved, take it off KNOWN_SHADOWED."
    )
