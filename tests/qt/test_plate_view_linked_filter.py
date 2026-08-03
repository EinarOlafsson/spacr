"""The plate heatmap honours the shared Local Data Filter.

This is the first view wired to :mod:`spacr.qt.linked_selection`, so these
tests are as much about the contract every later view has to keep as about
this one screen: narrow in the panel, and the heatmap narrows too, says so,
and never re-reads the database to do it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.linked_selection import linked_selection
from spacr.qt.screens.plate_view import PlateViewScreen
from spacr.selection import CategoryFilter, DataFilter


@pytest.fixture(autouse=True)
def _clean_shared_filter():
    """The link is process-wide; leaving a filter on it poisons other tests."""
    linked_selection().clear_filter()
    yield
    linked_selection().clear_filter()


def _plate_frame() -> pd.DataFrame:
    """Two plates, a full 2x3 grid each, with a value to colour by."""
    rows = []
    for plate in ("p1", "p2"):
        for r in ("A", "B"):
            for c in (1, 2, 3):
                rows.append({
                    "plateID": plate,
                    "rowID": r,
                    "columnID": str(c),
                    "object_count": 50,
                    "value": 1.0 if plate == "p1" else 9.0,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def screen(qtbot):
    s = PlateViewScreen(threaded=False)
    qtbot.addWidget(s)
    s._frame = _plate_frame()
    s._frame_key = ("", "", "value")
    s._refresh_plate_combo(s._frame)
    return s


def test_with_no_filter_every_plate_is_available(screen):
    screen.recompute()
    assert screen._filter_note == ""


def test_a_shared_filter_narrows_the_heatmap(screen):
    """The whole point: narrow in one place, narrow everywhere."""
    screen.recompute()
    unfiltered = len(screen._layout_df)

    # Filter WITHIN the plate: `plate_layout` already scopes to the plate the
    # combo selects, so a plateID clause narrows nothing and would make this
    # test pass for the wrong reason.
    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("columnID", ("1", "2"))))
    screen.recompute()

    assert len(screen._layout_df) < unfiltered, \
        "the heatmap ignored the shared filter"


def test_the_status_line_says_it_is_filtered(screen):
    """A filtered heatmap that does not say so is how an edge-effect verdict
    gets read as covering the whole plate when it covers a third of it."""
    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("plateID", ("p1",))))
    screen.recompute()
    assert "filtered" in screen._filter_note
    assert "plateID" in screen._filter_note


def test_clearing_the_filter_restores_the_full_plate(screen):
    screen.recompute()
    full = len(screen._layout_df)

    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("plateID", ("p1",))))
    screen.recompute()
    linked_selection().clear_filter()
    screen.recompute()

    assert len(screen._layout_df) == full
    assert screen._filter_note == ""


def test_a_filter_change_redraws_without_re_reading_the_database(screen):
    """The frame is cached across renders; a filter must not cost a re-read.

    Proved by never giving the screen a database at all — `_db_path` is empty,
    so any attempt to load would fail — and checking the redraw still happens.
    """
    assert screen._db_path == ""
    screen.recompute()
    before = len(screen._layout_df)

    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("columnID", ("1",))))

    assert len(screen._layout_df) < before, \
        "the filter_changed signal did not trigger a redraw"


def test_a_filter_naming_an_absent_column_draws_everything_and_says_so(screen):
    """Carried over from another table.

    An empty heatmap is a worse answer than a complete one, so the view
    degrades to unfiltered — but it must not pretend the filter applied.
    """
    from spacr.selection import RangeFilter

    linked_selection().set_filter(
        DataFilter().add(RangeFilter("no_such_column", low=1.0)))
    screen.recompute()

    assert screen._layout_df is not None and len(screen._layout_df) > 0
    assert "ignored" in screen._filter_note


def test_a_filter_change_before_anything_is_loaded_is_silent(qtbot):
    """Not a reason to show an error on a screen with no database yet."""
    s = PlateViewScreen(threaded=False)
    qtbot.addWidget(s)
    assert s._frame is None
    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("plateID", ("p1",))))
    assert s._layout_df is None


def test_closing_the_screen_stops_it_listening(screen, qtbot):
    """The link outlives every screen; a stale receiver is a leak.

    A destroyed screen still connected would be called on the next filter
    change, which is the crash this codebase has hit before with a
    process-wide registry and a deferred callback.
    """
    screen.recompute()
    assert screen.is_linked, "the screen was never listening to begin with"
    note_before = screen._filter_note
    screen.close()
    assert not screen.is_linked, "closeEvent did not disconnect the screen"

    # Nothing should still be routed at the closed screen: not only must this
    # not raise, it must not reach the screen at all. Comparing the note either
    # side is what says so — "must not raise" alone passed just as well with
    # the screen still subscribed and quietly recomputing on every change.
    linked_selection().set_filter(
        DataFilter().add(CategoryFilter("plateID", ("p2",))))
    assert screen._filter_note == note_before
