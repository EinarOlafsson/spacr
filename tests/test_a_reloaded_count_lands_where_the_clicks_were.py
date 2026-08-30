"""Reloading an exported count, and the unit check that guards it.

``load_frame`` places markers by WORLD coordinate, which is what makes a reload
land where the clicks were even on a differently-scaled view. The unit check is
the reason that works: a count made in micrometres placed into a pixel session
would put every marker "somewhere plausible and wrong", which is the phrase the
code itself uses -- and plausible-and-wrong is the failure nobody notices.

The uncovered arc is the case where a frame carries NO units column, which is
what a hand-built or older export looks like.
"""
from __future__ import annotations

import pandas as pd
import pytest


def _session(classes=("a", "b")):
    from spacr.counting import CountingSession
    from spacr.layers import LayerStack

    return CountingSession(LayerStack(), classes=list(classes))


def test_a_count_round_trips_through_its_own_frame():
    """The baseline: export and reload place the same markers."""
    session = _session()
    session.add({"y": 1.0, "x": 2.0}, "a")
    session.add({"y": 3.0, "x": 4.0}, "b")
    exported = session.to_frame()

    fresh = _session()
    placed = fresh.load_frame(exported)

    assert placed == 2
    reloaded = fresh.to_frame()
    assert reloaded[["class", "y", "x"]].values.tolist() == \
        exported[["class", "y", "x"]].values.tolist()


def test_a_frame_without_a_units_column_is_accepted():
    """Arc 426 -> 433: the unit check is skipped entirely.

    A hand-built frame, or one exported before units were recorded, has no
    units column. Refusing it would make every older count unreloadable, and
    the coordinates are still in the session's own units by assumption --
    which is exactly what the column exists to stop being an assumption.
    """
    frame = pd.DataFrame({"class": ["a", "a"], "y": [1.0, 5.0],
                          "x": [2.0, 6.0]})

    session = _session()
    assert session.load_frame(frame) == 2


def test_an_empty_frame_with_units_is_accepted():
    """The ``and len(frame)`` half of the same guard.

    An empty export carries a units column and no rows to disagree about, so
    comparing the empty set against the session's units would refuse a frame
    that says nothing at all.
    """
    frame = pd.DataFrame({"class": [], "y": [], "x": [], "units": []})

    session = _session()
    assert session.load_frame(frame) == 0


def test_a_count_made_in_other_units_is_refused():
    """The taken side, and the reason the guard is worth having."""
    from spacr.layers import LayerError

    frame = pd.DataFrame({"class": ["a"], "y": [1.0], "x": [2.0],
                          "units": ["um"]})

    session = _session()
    with pytest.raises(LayerError) as excinfo:
        session.load_frame(frame)

    assert "somewhere plausible and wrong" in str(excinfo.value)


def test_a_frame_missing_a_coordinate_is_refused_by_name():
    """The guard above both, which names what it needed and what it got."""
    from spacr.layers import LayerError

    frame = pd.DataFrame({"class": ["a"], "y": [1.0]})

    session = _session()
    with pytest.raises(LayerError) as excinfo:
        session.load_frame(frame)

    assert "'x'" in str(excinfo.value)


def test_a_class_the_session_does_not_have_is_added_as_it_goes():
    """The documented behaviour that makes someone else's count reloadable."""
    frame = pd.DataFrame({"class": ["unseen"], "y": [1.0], "x": [2.0]})

    session = _session(classes=("a",))
    assert session.load_frame(frame) == 1
    assert "unseen" in session.to_frame()["class"].tolist()
