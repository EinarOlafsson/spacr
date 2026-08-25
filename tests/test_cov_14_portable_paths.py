"""Re-rooting declines the inputs it cannot place, and counts what it moved.

A recorded crop path is re-rooted by matching its tail against a candidate
root. The cases below are the ones where there is nothing to match:

* a path with fewer than two components has no tail to try;
* an empty or blank path is not a path at all;
* a component list that does not reproduce the recorded text (a doubled
  separator) must be skipped rather than used to slice the string, because a
  wrong slice produces a prefix that would then be applied to every other row.

``RerootReport`` also stands in for its own ``moved`` count in boolean and
integer contexts, which is what lets a caller write ``if report:`` without
knowing it is holding a record.
"""
from __future__ import annotations

import os

import pytest

from spacr import portable_paths


def test_a_single_component_path_has_no_tail_to_try():
    """One component means no suffix, so there is nothing to re-root by."""
    assert portable_paths._suffixes("measurements.db") == []


def test_a_two_component_path_offers_its_tail():
    """Two components do give a suffix, so the guard is not over-eager."""
    suffixes = portable_paths._suffixes("plate1/measurements.db")

    assert suffixes
    assert suffixes[0] == ("plate1", "measurements.db")


def test_a_blank_path_is_returned_untouched(tmp_path):
    """Whitespace is not a path; it comes back unchanged with no prefix."""
    assert portable_paths._reroot_with_prefix("   ", str(tmp_path)) == \
        ("   ", None)


def test_a_non_string_path_is_returned_untouched(tmp_path):
    """A NaN or None in a path column is handed straight back."""
    assert portable_paths._reroot_with_prefix(None, str(tmp_path)) == \
        (None, None)


def test_a_suffix_the_recorded_path_does_not_end_with_is_skipped(tmp_path):
    """A tail that is not actually the end of the recorded text is passed over.

    A recorded path with a doubled separator produces a component list that
    does not reassemble into the original string. Slicing the string by such a
    tail yields a prefix pair that is then applied to EVERY remaining row of
    the column, turning one malformed path into a whole broken column -- so
    the tail is skipped and the next one tried.
    """
    root = tmp_path / "plate1" / "data"
    root.mkdir(parents=True)
    (root / "crop.png").write_bytes(b"")

    recorded = "/gone//plate1/data/crop.png"
    tails = ["/".join(s) for s in portable_paths._suffixes(recorded)]

    # The tail that is tried first DOES end the recorded text but is not on
    # disk; the next one does not end it at all and is the one that must be
    # skipped rather than sliced.
    assert tails[0] == "data/crop.png"
    assert not recorded.endswith(tails[1])

    mapped, prefixes = portable_paths._reroot_with_prefix(
        recorded, str(tmp_path))

    assert mapped == str(root / "crop.png")
    assert os.path.exists(mapped)
    head, new_head = prefixes
    assert recorded.startswith(head)
    assert mapped.startswith(new_head)


def test_a_report_counts_as_its_moved_number():
    """``bool`` and ``int`` on the record answer about ``moved``."""
    moved = portable_paths.RerootReport(column="png_path", moved=3,
                                        unresolved=1, root="/x")
    nothing = portable_paths.RerootReport(column="png_path", moved=0,
                                          unresolved=4, root="/x")

    assert bool(moved) is True
    assert int(moved) == 3
    assert bool(nothing) is False
    assert int(nothing) == 0


def test_no_database_path_has_no_source_root():
    """An empty database path yields an empty root, not the process cwd.

    ``os.path.abspath("")`` is the working directory, which would send the
    re-rooting search off through whatever folder the GUI happened to start
    in.
    """
    assert portable_paths.source_root_for_database("") == ""


def test_a_database_path_yields_the_folder_above_measurements(tmp_path):
    """The plate folder is two levels above ``measurements/measurements.db``."""
    db = tmp_path / "plate1" / "measurements" / "measurements.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")

    assert portable_paths.source_root_for_database(str(db)) == \
        str(tmp_path / "plate1")
