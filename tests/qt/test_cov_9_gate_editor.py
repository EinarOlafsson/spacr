"""The Gate Editor's multi-database working set: labels, removal and reload.

Several measurement databases gated as one frame is the case where a wrong
answer is invisible: the points all plot, the gates all apply, and nothing on
screen says the numbers were computed over two experiments. So the chips have
to name the sources, dropping one has to re-merge the rest, and the audit line
that records what was decided must never be the reason the screen fails.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.gate_editor import GateEditorScreen

pytestmark = pytest.mark.qt


def _objects(plate, n=8):
    return pd.DataFrame({
        "plateID": [plate] * n,
        "rowID": ["A"] * n,
        "columnID": ["1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": list(range(1, n + 1)),
        "area": np.linspace(10.0, 80.0, n),
        "intensity": np.linspace(100.0, 800.0, n),
    })


def _database(path, plate, n=8):
    with sqlite3.connect(str(path)) as db:
        _objects(plate, n).to_sql("cell", db, index=False)
    return str(path)


@pytest.fixture
def screen(qtbot):
    widget = GateEditorScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def two_databases(tmp_path):
    """Two databases holding different plates, so the merge is not refused."""
    return [_database(tmp_path / "runA.db", "plate1"),
            _database(tmp_path / "runB.db", "plate2")]


# ---------------------------------------------------------------------------
# The audit line
# ---------------------------------------------------------------------------

def test_a_decision_with_no_plan_still_records_which_files_were_involved(
        screen, two_databases, monkeypatch):
    """A merge refused before a plan exists still has a decision to record.

    The record is what makes the choice answerable months later, when the
    surviving frame can no longer say which plate1 it held; without the
    plan the file list and the outcome are all there is, and they still
    have to be written down.
    """
    import spacr.multi_database as multi_database

    recorded = []
    monkeypatch.setattr(multi_database, "record_decision", recorded.append)

    screen._record_merge(None, "refused", "the mount went away",
                         paths=two_databases, table="cell")

    assert len(recorded) == 1
    assert recorded[0].sources == tuple(two_databases)
    assert recorded[0].outcome == "refused"
    assert screen._merge_decision is recorded[0]


def test_an_audit_line_that_cannot_be_written_does_not_stop_the_screen(
        screen, two_databases, monkeypatch):
    """Recording the decision is bookkeeping, and bookkeeping may fail.

    A read-only registry must not take down a screen whose actual job -- the
    merge the user asked for -- has already succeeded.
    """
    import spacr.multi_database as multi_database

    def _explode(_decision):
        raise OSError("the decision log is read-only")

    monkeypatch.setattr(multi_database, "record_decision", _explode)

    screen._record_merge(None, "merged", "merged two databases",
                         paths=two_databases, table="cell")

    assert screen._merge_decision is not None, (
        "the screen still knows what it decided, even unwritten")
    assert screen._merge_decision.outcome == "merged"


# ---------------------------------------------------------------------------
# Naming the sources
# ---------------------------------------------------------------------------

def test_a_session_with_no_databases_has_no_labels(screen):
    """No sources means no chips, and no chips means an empty label list.

    A single made-up label would put a chip on screen for a database that is
    not in the working set.
    """
    assert screen.database_labels() == []


def test_labels_fall_back_to_the_file_names(screen, two_databases, monkeypatch):
    """A source always gets a name, even when the labeller cannot run.

    A chip with no text is a chip the user cannot act on, and the file name
    is the one label that is always available.
    """
    import spacr.multi_database as multi_database

    def _explode(_paths):
        raise RuntimeError("the labeller is unavailable")

    monkeypatch.setattr(multi_database, "source_labels", _explode)
    screen._paths = list(two_databases)

    assert screen.database_labels() == ["runA", "runB"]


# ---------------------------------------------------------------------------
# Dropping one database
# ---------------------------------------------------------------------------

def test_a_database_can_be_dropped_by_its_path(screen, two_databases):
    """A caller usually holds the path while the chip shows the label.

    Both have to name the same source, or the screen's own chip would drop a
    database and a programmatic caller would drop nothing.
    """
    screen.load_paths(two_databases, "cell")
    assert len(screen._paths) == 2

    screen.remove_database(two_databases[0])

    assert screen._paths == [two_databases[1]]


def test_the_last_database_cannot_be_dropped(screen, two_databases):
    """Removing the only source would leave the screen gating nothing.

    The chip strip is not even shown for one database, so this is reached
    only by a caller; it still has to be a no-op rather than an empty frame.
    """
    screen.load_paths(two_databases, "cell")
    screen.remove_database(two_databases[0])
    remaining = list(screen._paths)

    screen.remove_database(remaining[0])
    screen.remove_database("no such database")

    assert screen._paths == remaining


# ---------------------------------------------------------------------------
# Reloading a merged working set
# ---------------------------------------------------------------------------

def test_reloading_a_merged_session_reads_every_database_again(
        screen, two_databases):
    """A reload of a multi-database session must not read only the first one.

    Reading one and calling it the working set is silent data loss: the plot
    still draws, with half the objects and no sign that the rest are gone.
    """
    screen.load_paths(two_databases, "cell")

    screen._reload_working_set()

    frame = screen._frame
    assert len(frame) == 16
    assert set(frame["plateID"]) == {"plate1", "plate2"}


def test_a_cross_database_read_downsamples_to_the_point_cap(
        screen, two_databases):
    """The cap bounds what is plotted, evenly across the merged frame.

    Taking the first N rows instead would show one database and none of the
    other, because a merge concatenates its sources in order.
    """
    labels = ["runA", "runB"]
    policy = screen._merge_policy()

    frame = GateEditorScreen._read_across_databases(
        two_databases, labels, ["cell"], 6, policy)

    assert len(frame) == 6
    assert len(set(frame["plateID"])) == 2, "the cap kept both databases"
