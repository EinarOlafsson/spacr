"""The Runs tab over tables it did not write, and gestures aimed at nothing.

Every row in this panel is read back through a composed frame, and the frame
is assembled from two halves that need not agree: a sweep's CSV, written by
an earlier session or by hand, and this session's own recorded runs. The
paths worth driving are the ones where the CSV does NOT look like the panel
expects -- it already names its own sources, or it predates the ``status``
column -- and the ones where a gesture lands on no run at all: a Delete key
with nothing selected, a right-click on a row that is already picked.

Each of those must leave the table saying something TRUE. A summary that
invents failures out of a missing column, a menu offering "Load this run"
over four rows, or a Delete key that swallows itself over an empty selection
are all the same bug in different clothes: the panel answering a question it
was not asked.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

pytestmark = pytest.mark.qt

from PySide6.QtCore import QEvent, Qt                            # noqa: E402
from PySide6.QtGui import QKeyEvent                              # noqa: E402

from spacr.qt.widgets.sweep_runs import (                        # noqa: E402
    SOURCE_SWEEP, STATUS_RUNNING, SweepRunsPanel,
)


@pytest.fixture
def panel(qtbot):
    """A Runs tab with nothing in it, the way a freshly opened screen has."""
    made = SweepRunsPanel()
    qtbot.addWidget(made)
    return made


def _entries(menu):
    """``{verb: action}`` for the entries of a built run menu that carry one."""
    return {action.data(): action for action in menu.actions()
            if action.data()}


def _delete_key():
    """The keystroke the runs table watches for on its selection."""
    return QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)


# ---------------------------------------------------------------------------
# a sweep table that does not look like the sweep's own
# ---------------------------------------------------------------------------

def test_a_table_that_already_names_its_sources_keeps_them(panel, qtbot):
    """A CSV loaded beside the sweep's own may already say where each row
    came from -- runs exported from an earlier session, say, marked "on
    disk". Stamping "sweep trial" over that column would relabel every one of
    those rows as something the sweep produced, and the source column is
    exactly what a user reads to tell a hand-made comparison table apart from
    a generated one. Rows with no source of their own still get the sweep's.
    """
    named = pd.DataFrame({"trial_id": [1, 2],
                          "source": ["on disk", "on disk"],
                          "r_squared": [0.41, 0.52]})

    assert panel.set_frame(named, source="/x/named.csv") is True
    assert panel._frame["source"].tolist() == ["on disk", "on disk"]
    # The trial numbers still become the run names -- the column that was
    # already there is the only thing left alone.
    assert panel._frame["run"].tolist() == ["trial 1", "trial 2"]

    anonymous = pd.DataFrame({"trial_id": [7], "r_squared": [0.3]})
    assert panel.set_frame(anonymous, source="/x/plain.csv") is True
    assert panel._frame["source"].tolist() == [SOURCE_SWEEP]
    assert panel._frame["run"].tolist() == ["trial 7"]


def test_a_table_with_no_status_column_reports_no_failures(panel):
    """An older sweep's CSV carries no ``status`` column at all. The sentence
    over the table counts the runs that "did not produce a regression" off
    that column, and reading a missing column as a failure would tell a user
    every trial in their archive had crashed. The same panel, given a table
    that DOES carry the column, must still count the failure in it -- the
    silence has to come from the column being absent, not from the counting
    being broken.
    """
    quiet = pd.DataFrame({"trial_id": [1, 2], "r_squared": [0.4, 0.5]})
    panel.set_frame(quiet, source="/x/old.csv")

    said = panel._status.text()
    assert said.startswith("2 runs")
    assert "did not produce a regression" not in said, said
    assert "still going" not in said, said

    loud = pd.DataFrame({"trial_id": [1, 2, 3],
                         "status": ["ok", "failed", STATUS_RUNNING],
                         "r_squared": [0.4, 0.5, 0.6]})
    panel.set_frame(loud, source="/x/new.csv")

    said = panel._status.text()
    assert said.startswith("3 runs")
    assert "1 of which did not produce a regression" in said, said
    assert "1 still going" in said, said


def test_the_summary_of_a_frame_with_no_source_column_just_counts(panel):
    """The sentence over the table is also asked for a frame that is not the
    composed one -- a caller summarising a table it holds itself. Splitting
    "N from this session, M from the sweep" reads the ``source`` column, and
    doing that unguarded would raise a KeyError out of the status line rather
    than describing the rows. With the column present the split is what the
    user gets, so the plain count below is the absence of the column and not
    the absence of the feature.
    """
    plain = panel._describe(pd.DataFrame({"run": ["a", "b", "c"]}))
    assert plain.startswith("3 runs")
    assert "from this session" not in plain, plain

    mixed = panel._describe(pd.DataFrame({
        "run": ["a", "b", "c"],
        "source": ["run", SOURCE_SWEEP, SOURCE_SWEEP]}))
    assert "1 from this session, 2 from the sweep" in mixed, mixed


# ---------------------------------------------------------------------------
# the menu over rows that cannot all be acted on
# ---------------------------------------------------------------------------

def test_several_rows_including_a_running_one_offer_no_load(panel):
    """Loading is a single-run act: it replaces the results, the figures and
    the summary on the screen, and there is no such thing as showing four
    runs at once there. So a menu over several rows must not carry a Load
    entry at all -- and the greying that a still-running run applies to the
    other entries has to cope with the Load entry being absent rather than
    reaching for it. One row deep in that same state DOES get the entry,
    disabled and saying why, which is what makes the multi-row silence a
    decision instead of a hole.
    """
    several = [{"run": "ols_1", "folder": "/x/ols_1", "status": STATUS_RUNNING},
               {"run": "ols_2", "folder": "/x/ols_2", "status": "ok"}]

    menu = panel._build_run_menu(several)
    verbs = _entries(menu)

    assert "load" not in verbs, sorted(verbs)
    assert verbs["remove"].text() == "Remove 2 runs from the list"
    assert verbs["remove"].isEnabled() is False
    assert verbs["delete"].isEnabled() is False
    assert verbs["remove"].toolTip().startswith("ols_1 is still going")
    # The reason is spelled out as its own disabled entry, so it is readable
    # without hovering a greyed line.
    assert menu.actions()[-1].text().startswith("ols_1 is still going")
    assert menu.actions()[-1].isEnabled() is False

    alone = panel._build_run_menu([several[0]])
    load = _entries(alone)["load"]
    assert load.text() == "Load this run"
    assert load.isEnabled() is False
    assert "no results to show" in load.toolTip()


def test_a_right_click_on_an_already_selected_row_acts_on_that_row(
        panel, monkeypatch):
    """Right-clicking a row that is already highlighted must act on the
    selection as it stands. If the panel insisted on re-deriving the row from
    the click position it would act on nothing whenever the click landed a
    pixel outside an item, and the entry the user picked would silently do
    nothing. Driven through a stub menu because the real one enters a C++
    event loop that a test cannot leave.
    """
    panel.record_run("ols_1", folder="/x/ols_1")
    panel.table.table.selectRow(0)
    assert [row["run"] for row in panel.selected_runs()] == ["ols_1"]

    class _Chosen:
        def data(self):
            return "remove"

    class _StubMenu:
        def __init__(self):
            self.shown_at = None

        def exec(self, position):
            self.shown_at = position
            return _Chosen()

    stub = _StubMenu()
    monkeypatch.setattr(panel, "_build_run_menu", lambda records: stub)
    removed = []
    panel.runs_removed.connect(removed.append)

    item = panel.table.table.item(0, 0)
    panel._run_menu(panel.table.table.visualItemRect(item).center())

    assert stub.shown_at is not None
    assert panel.table.table.rowCount() == 0
    assert [row["run"] for row in removed[0]] == ["ols_1"]
    assert panel._status.text().startswith("Removed 1 run from the list")


# ---------------------------------------------------------------------------
# the Delete key with nothing under it
# ---------------------------------------------------------------------------

def test_delete_with_nothing_selected_is_left_to_qt(panel):
    """The Delete key over the runs table removes the selected rows from the
    list. With no selection there is nothing to remove, and the panel must
    hand the keystroke back rather than eat it: a filter box and the table's
    own editing both sit under the same event filter, and a key swallowed
    here is a key that never reaches them. The selected case in the same test
    is what proves the handler is wired at all.
    """
    handle = panel.record_run("ols_1", folder="/x/ols_1")
    # Finished, so nothing but the empty selection is standing in the way --
    # a run still going is refused for its own reason, further up.
    panel.update_run(handle, status="ok")
    panel.table.table.clearSelection()
    assert panel.selected_runs() == []

    handled = panel.eventFilter(panel.table.table, _delete_key())

    assert handled is False, "the key was swallowed over an empty selection"
    assert panel.table.table.rowCount() == 1

    panel.table.table.selectRow(0)
    assert panel.eventFilter(panel.table.table, _delete_key()) is True
    assert panel.table.table.rowCount() == 0
    assert panel._status.text().startswith("Removed 1 run from the list")
