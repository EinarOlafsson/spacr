"""Re-rooting declines the inputs it cannot place, and counts what it moved.

A recorded crop path is re-rooted by matching its tail against a candidate
root. The cases below are the ones where there is nothing to match:

* an empty or blank path is not a path at all;
* a path already on this machine is a successful no-op.

``RerootReport`` also stands in for its own ``moved`` count in boolean and
integer contexts, which is what lets a caller write ``if report:`` without
knowing it is holding a record.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr import portable_paths


def test_a_blank_path_is_returned_untouched(tmp_path):
    """Whitespace is not a path; it comes back unchanged with no prefix."""
    assert portable_paths._reroot_with_prefix("   ", str(tmp_path)) == \
        ("   ", None)


def test_a_non_string_path_is_returned_untouched(tmp_path):
    """A NaN or None in a path column is handed straight back."""
    assert portable_paths._reroot_with_prefix(None, str(tmp_path)) == \
        (None, None)


def test_a_report_counts_as_its_moved_number():
    """``bool`` and ``int`` on the record answer about ``moved``."""
    moved = portable_paths.RerootReport(column="png_path", moved=3,
                                        unresolved=1,
                                        first_unresolved="/gone/a.png",
                                        root="/x")
    nothing = portable_paths.RerootReport(column="png_path", moved=0,
                                          unresolved=4, root="/x")

    for name in ("column", "moved", "unresolved", "first_unresolved", "root"):
        assert f":param {name}:" in (portable_paths.RerootReport.__doc__ or "")
    assert moved.first_unresolved == "/gone/a.png"
    assert "/gone/a.png" in moved.describe()
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


def test_an_existing_crop_path_is_neither_moved_nor_reported_missing(tmp_path):
    """A path already on this machine is a successful no-op.

    This is the case where re-rooting returns no prefix because it had no work
    to do.  It must not fall through to the unresolved-path bookkeeping merely
    because the returned path equals the recorded one.
    """
    crops = [tmp_path / "already_here.png", tmp_path / "also_here.png"]
    for crop in crops:
        crop.write_bytes(b"crop")
    recorded = [str(crop) for crop in crops]
    frame = pd.DataFrame({"png_path": recorded})

    report = portable_paths.reroot_column(frame, "png_path", str(tmp_path))

    assert frame["png_path"].tolist() == recorded
    assert report.moved == 0
    assert report.unresolved == 0
    assert report.first_unresolved == ""
