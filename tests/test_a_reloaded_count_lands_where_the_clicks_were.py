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


@pytest.mark.parametrize('columns, expected', [
    pytest.param(
        {'class': ['a', 'a'], 'y': [1.0, 5.0], 'x': [2.0, 6.0]}, 2,
        id='test_a_frame_without_a_units_column_is_accepted',
    ),
    pytest.param(
        {'class': [], 'y': [], 'x': [], 'units': []}, 0,
        id='test_an_empty_frame_with_units_is_accepted',
    ),
])
def test_original_count_frame_layouts_keep_their_returned_marker_counts(
        columns, expected):
    """Preserve each original frame layout and its returned marker count."""
    frame = pd.DataFrame(columns)
    session = _session()
    assert session.load_frame(frame) == expected


@pytest.mark.parametrize('columns, fragment', [
    pytest.param(
        {'class': ['a'], 'y': [1.0], 'x': [2.0], 'units': ['um']},
        'somewhere plausible and wrong',
        id='test_a_count_made_in_other_units_is_refused',
    ),
    pytest.param(
        {'class': ['a'], 'y': [1.0]}, "'x'",
        id='test_a_frame_missing_a_coordinate_is_refused_by_name',
    ),
])
def test_original_count_frame_refusals_keep_their_error_explanations(
        columns, fragment):
    """Preserve each original frame refusal and its required explanation fragment."""
    from spacr.layers import LayerError

    frame = pd.DataFrame(columns)
    session = _session()
    with pytest.raises(LayerError) as excinfo:
        session.load_frame(frame)
    assert fragment in str(excinfo.value)


def test_a_class_the_session_does_not_have_is_added_as_it_goes():
    """The documented behaviour that makes someone else's count reloadable."""
    frame = pd.DataFrame({"class": ["unseen"], "y": [1.0], "x": [2.0]})

    session = _session(classes=("a",))
    assert session.load_frame(frame) == 1
    assert "unseen" in session.to_frame()["class"].tolist()
