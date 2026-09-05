"""Step 1 of the Measurements tab reads its databases off the GUI thread.

THE FREEZE, 2026-09-04. `DatabaseMergePanel.__init__` ends in `refresh()`,
and `refresh()` went:

    refresh
      -> _fill_table -> _source_info
           -> AttachedDatabase.present  -> os.path.exists(<user path>)
           -> mergeable_tables(path)    -> sqlite3.connect(<user path>)
           -> describe_merge(paths, ..) -> sqlite3.connect, once per path
      -> _offer_tables -> joinable_tables(paths)   -> every path again
      -> describe -> _plan_lines -> describe_merge, _table_notes,
                                    _column_kinds -> every path again

every one of those on the GUI thread, once per plate row, with paths the
user chose. On the maintainer's machine one of them was under `/nas_mnt` --
an `autofs` mount whose share was asleep -- and a single `os.path.exists`
on it had not returned after TWENTY SECONDS. So opening the regression
module froze the whole application with no traceback, because a stalled
event loop is not a crash. It was reported as "opening map barcodes crashes
spacr", plus hover flicker and glimpses of other screens; see
`spacr/qt/path_probe.py` for the full list of symptoms one freeze produced.

WHAT IS ASSERTED HERE is the property the freeze violated, not the
mechanism: the panel answers in well under a second however long the
databases take, every plate row is still listed while they are being read,
and the row corrects itself when the answer lands. The last one matters as
much as the first -- off the GUI thread is only right if the answer still
arrives.
"""
from __future__ import annotations

import os
import sqlite3
import time

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

#: Longer than any human would call responsive, far shorter than the twenty
#: seconds actually measured. A test that waited the real duration would be
#: a test nobody runs.
SLOW_S = 8.0

#: What the panel is allowed to take. Generous on purpose: what is being
#: pinned is the difference between a bounded wait and a filesystem's, not a
#: particular number of milliseconds.
RESPONSIVE_S = 1.0


def _database(directory, plate):
    """One plate's measurements.db, in the shape spaCR writes.

    :param directory: the plate folder; created if it is not there.
    :param plate: the plate id written into every row.
    :returns: the path of the database.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(str(directory), "measurements.db")
    identity = {"rowID": "r1", "columnID": "c1", "fieldID": "f1"}
    cell = pd.DataFrame({
        "plateID": [plate] * 3, **{k: [v] * 3 for k, v in identity.items()},
        "object_label": [1, 2, 3],
        "area": [100.0, 200.0, 300.0],
    })
    nucleus = pd.DataFrame({
        "plateID": [plate] * 3, **{k: [v] * 3 for k, v in identity.items()},
        "cell_id": [1, 2, 3], "object_label": [1, 2, 3],
        "nucleus_area": [5.0, 5.0, 5.0],
    })
    with sqlite3.connect(path) as db:
        cell.to_sql("cell", db, index=False)
        nucleus.to_sql("nucleus", db, index=False)
    return path


def _rows(paths):
    """Input-table rows, in the shape the paired table emits.

    :param paths: one measurements database per plate.
    :returns: a list of ``{"plate", "score", "count", "database"}`` dicts.
    """
    return [{"plate": f"plate{index + 1}",
             "score": f"plate{index + 1}_scores.csv",
             "count": f"plate{index + 1}_counts.csv",
             "database": path}
            for index, path in enumerate(paths)]


@pytest.fixture()
def plates(tmp_path):
    """Two plates, each with its own real measurements database."""
    return [_database(tmp_path / "plate1", "plate1"),
            _database(tmp_path / "plate2", "plate2")]


@pytest.fixture()
def sleeping_databases(monkeypatch):
    """Make every database read take :data:`SLOW_S`, as a sleeping mount does.

    Patched on the panel's own module rather than on `spacr.merge_tables`,
    because that is the name the panel calls -- and `joinable_tables`, which
    lives in the panel's module too, then goes slow with it.

    :returns: a list the patched reads append to, one entry per call, so a
        test can count how many times the databases were opened.
    """
    from spacr.qt.widgets import measurement_scan_panel as panel_module

    opened = []

    def crawl(path, *_args, **_kwargs):
        """Stand in for a read of a database on a share that is asleep.

        :param path: the database asked for; recorded, then slept on.
        :returns: never; the sleep outlasts the test.
        """
        opened.append(path)
        time.sleep(SLOW_S)
        raise AssertionError("the GUI thread waited for a database")

    monkeypatch.setattr(panel_module, "mergeable_tables", crawl)
    monkeypatch.setattr(panel_module, "describe_merge", crawl)
    monkeypatch.setattr(panel_module, "column_kinds", crawl)
    return opened


def _panel(qtbot, rows):
    """A merge panel over ``rows``, registered for teardown.

    :param qtbot: the pytest-qt bot.
    :param rows: what the database provider returns.
    :returns: the built :class:`DatabaseMergePanel`.
    """
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    panel = DatabaseMergePanel(lambda: rows)
    qtbot.addWidget(panel)
    return panel


# --------------------------------------------------------------------------- #
#  The freeze itself
# --------------------------------------------------------------------------- #

def test_building_the_panel_returns_before_the_databases_answer(
        qtbot, plates, sleeping_databases):
    """The property the freeze violated: opening the tab does not wait.

    `__init__` ends in `refresh()`, so the constructor is the freeze.
    """
    started = time.monotonic()
    panel = _panel(qtbot, _rows(plates))
    elapsed = time.monotonic() - started

    assert elapsed < RESPONSIVE_S, (
        f"building the panel took {elapsed:.1f}s -- it is reading the "
        f"databases on the GUI thread again, which is the freeze")
    assert panel.table.rowCount() == 2


def test_a_later_refresh_does_not_wait_either(qtbot, plates,
                                              sleeping_databases):
    """`refresh` is called again on every new run and before every merge."""
    panel = _panel(qtbot, _rows(plates))

    started = time.monotonic()
    panel.refresh()
    elapsed = time.monotonic() - started

    assert elapsed < RESPONSIVE_S, (
        f"refresh() took {elapsed:.1f}s reading the databases inline")


def test_the_plan_does_not_wait_for_the_databases(qtbot, plates,
                                                  sleeping_databases):
    """`plan_text` is what a click on a table checkbox ends in."""
    panel = _panel(qtbot, _rows(plates))

    started = time.monotonic()
    text = panel.plan_text()
    elapsed = time.monotonic() - started

    assert elapsed < RESPONSIVE_S, (
        f"plan_text() took {elapsed:.1f}s -- `describe_merge` is still being "
        f"called on the GUI thread")
    assert "Anchor: cell" in text, (
        "what the panel already knows must still be said while it reads")


def test_a_sleeping_stat_does_not_hold_up_a_plate_row(qtbot, plates,
                                                      monkeypatch):
    """The measured call: `os.path.exists` on a path under a sleeping mount.

    `AttachedDatabase.present` asks it once per plate row, and `paths`,
    `screens` and `describe` each ask again. It goes through `path_probe`
    now, which answers from a cache and stats on its own bounded worker.
    """
    from spacr.qt import path_probe

    real_exists = os.path.exists

    def never(path):
        """A stat that triggers an automount on a share that is asleep.

        Slow for THESE databases only: `path_probe.os` is the os module
        itself, so a blanket patch would put every other stat in the
        process -- Qt's, pandas', pytest's -- behind the same sleep.

        :param path: the path being stat-ed.
        :returns: True, eventually, for one of the plate databases;
            immediately, and honestly, for anything else.
        """
        if str(path).endswith("measurements.db"):
            time.sleep(SLOW_S)
        return real_exists(path)

    monkeypatch.setattr(path_probe.os.path, "exists", never)
    path_probe.forget()

    started = time.monotonic()
    panel = _panel(qtbot, _rows(plates))
    elapsed = time.monotonic() - started
    path_probe.forget()

    assert elapsed < RESPONSIVE_S, (
        f"building the panel took {elapsed:.1f}s -- a plate row is stat-ing "
        f"its database on the GUI thread")
    assert panel.table.rowCount() == 2, (
        "a row whose database has not answered is still the user's plate")


# --------------------------------------------------------------------------- #
#  What the user sees instead, and that it is put right
# --------------------------------------------------------------------------- #

def test_every_plate_is_still_listed_while_its_database_is_read(
        qtbot, plates, sleeping_databases):
    """Nothing may vanish. A plate left out of this list is a plate the user
    believes is not in the merge, which is the failure step 1 exists to
    prevent -- so only the three columns that come from INSIDE the file wait
    for it."""
    from spacr.qt.widgets.measurement_scan_panel import READING_TEXT

    panel = _panel(qtbot, _rows(plates))

    assert panel.table.rowCount() == 2
    assert panel.table.item(0, 0).text() == "plate1"
    assert panel.table.item(0, 1).text().endswith("measurements.db")
    assert panel.table.item(0, 3).text() == READING_TEXT, (
        "a column read out of the database should say it is being read")
    assert "2 measurement database(s) attached" in panel.heading.text()


def test_the_row_is_corrected_when_the_read_lands(qtbot, plates, monkeypatch):
    """Off the GUI thread is only correct if the answer still arrives.

    The read here is slower than the paint budget and then succeeds, which
    is the sleeping mount that eventually wakes up.
    """
    from spacr.merge_tables import mergeable_tables as real_tables
    from spacr.qt.widgets import measurement_scan_panel as panel_module
    from spacr.qt.widgets.measurement_scan_panel import READING_TEXT

    def late(path, *args, **kwargs):
        """`mergeable_tables`, but slower than one paint's budget.

        :param path: the database to read.
        :returns: what the real function returns.
        """
        time.sleep(0.6)
        return real_tables(path, *args, **kwargs)

    monkeypatch.setattr(panel_module, "mergeable_tables", late)

    panel = _panel(qtbot, _rows(plates))
    assert panel.table.item(0, 3).text() == READING_TEXT

    qtbot.waitUntil(lambda: panel.table.item(0, 3).text() != READING_TEXT,
                    timeout=15000)
    assert "cell" in panel.table.item(0, 3).text()

    # The chooser reads every database once more, so it lands after the row
    # does. It has to land: a chooser left empty says "no object table is
    # shared by every database", which is a different claim from "not read
    # yet" and is the one that would stop the merge.
    qtbot.waitUntil(lambda: "cell" in panel.selected_tables(), timeout=15000)


def test_repainting_while_a_read_is_out_does_not_re_open_the_databases(
        qtbot, plates, sleeping_databases):
    """Coalesced, not queued.

    Every keystroke and every tick redraws this panel, and a redraw that
    started its own read of a sleeping share would leave one parked thread
    per redraw.
    """
    panel = _panel(qtbot, _rows(plates))
    after_build = len(sleeping_databases)

    for _ in range(20):
        panel._repaint()

    assert len(sleeping_databases) == after_build, (
        f"{len(sleeping_databases) - after_build} extra database reads were "
        f"started by redraws that should have waited for the one in flight")


# --------------------------------------------------------------------------- #
#  The sibling site: the Aggregation rules button
# --------------------------------------------------------------------------- #
#
# Everything above is metadata -- table names, plate ids, row counts. The
# rules button reads ROWS: `read_merged` opens every attached database and
# pulls two hundred rows out of each, and it ran inside the button's own
# click handler. Same mount, same freeze, on a button instead of a tab.


@pytest.fixture()
def sleeping_preview(monkeypatch):
    """Make the rules preview read take :data:`SLOW_S`.

    Only `read_merged`; the metadata reads stay fast, so what a test using
    this fixture measures is the preview and nothing else.

    :returns: a list the patched read appends one entry to per call.
    """
    from spacr.qt.widgets import measurement_scan_panel as panel_module

    opened = []

    def crawl(paths, *_args, **_kwargs):
        """Stand in for a row read from a share that is asleep.

        :param paths: the databases asked for; recorded, then slept on.
        :returns: never; the sleep outlasts the test.
        """
        opened.append(tuple(paths))
        time.sleep(SLOW_S)
        raise AssertionError("the GUI thread waited for the rules preview")

    monkeypatch.setattr(panel_module, "read_merged", crawl)
    return opened


def _ready(qtbot, panel):
    """Wait until the panel has read its databases and chosen a table.

    :param qtbot: the pytest-qt bot.
    :param panel: the merge panel.
    """
    qtbot.waitUntil(lambda: bool(panel.selected_tables()), timeout=15000)


def test_the_rules_button_does_not_wait_for_the_preview(
        qtbot, plates, sleeping_preview):
    """The click returns; it does not sit on `read_merged`."""
    from spacr.qt.widgets.measurement_scan_panel import RULES_READING_LABEL

    panel = _panel(qtbot, _rows(plates))
    _ready(qtbot, panel)

    started = time.monotonic()
    panel.show_aggregation_rules()
    elapsed = time.monotonic() - started

    assert elapsed < RESPONSIVE_S, (
        f"the rules button took {elapsed:.1f}s -- `read_merged` is still "
        f"being called on the GUI thread")
    assert sleeping_preview, "the preview was never asked for at all"
    assert panel.rules_button.text() == RULES_READING_LABEL, (
        "a button that has gone quiet has to say why")
    assert not panel.rules_button.isEnabled()


def test_the_rules_dialog_still_opens_when_the_preview_lands(qtbot, plates,
                                                             monkeypatch):
    """Off the GUI thread is only right if the dialog still arrives."""
    from spacr.multi_database import read_merged as real_read
    from spacr.qt.widgets import measurement_scan_panel as panel_module
    from spacr.qt.widgets.measurement_scan_panel import RULES_LABEL

    def late(paths, *args, **kwargs):
        """`read_merged`, but slower than one click's budget.

        :param paths: the databases to read.
        :returns: what the real function returns.
        """
        time.sleep(0.6)
        return real_read(paths, *args, **kwargs)

    monkeypatch.setattr(panel_module, "read_merged", late)

    panel = _panel(qtbot, _rows(plates))
    _ready(qtbot, panel)
    panel.show_aggregation_rules()
    assert panel._rules_dialog is None

    qtbot.waitUntil(lambda: panel._rules_dialog is not None, timeout=15000)
    dialog = panel._rules_dialog
    qtbot.addWidget(dialog)
    assert dialog.isVisible()
    assert panel.rules_button.text() == RULES_LABEL
    assert panel.rules_button.isEnabled()


def test_a_preview_that_fails_puts_the_button_back(qtbot, plates, monkeypatch):
    """`JobRunner._on_settled` calls `on_done` only on success, and the same
    trap is here: a button left saying "reading" after a read that raised is
    a button that never works again."""
    from spacr.qt.widgets import measurement_scan_panel as panel_module
    from spacr.qt.widgets.measurement_scan_panel import RULES_LABEL

    said = []

    def late_and_broken(*_args, **_kwargs):
        """A preview read that takes longer than the budget and then fails.

        :raises OSError: what a share that woke up refusing looks like.
        """
        time.sleep(0.6)
        raise OSError("the share went away")

    monkeypatch.setattr(panel_module, "read_merged", late_and_broken)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.information",
        staticmethod(lambda *args, **kwargs: said.append(args[2])))

    panel = _panel(qtbot, _rows(plates))
    _ready(qtbot, panel)
    panel.show_aggregation_rules()

    qtbot.waitUntil(lambda: bool(said), timeout=15000)
    assert "the share went away" in said[0]
    assert panel.rules_button.text() == RULES_LABEL, (
        "a failed read must give the button back")
    assert panel.rules_button.isEnabled()


def test_a_second_click_does_not_start_a_second_preview(qtbot, plates,
                                                        sleeping_preview):
    """Coalesced. One parked read per click on a sleeping share is how a
    panel ends up with a thread per impatient click."""
    panel = _panel(qtbot, _rows(plates))
    _ready(qtbot, panel)

    panel.show_aggregation_rules()
    panel.show_aggregation_rules()
    panel.show_aggregation_rules()

    assert len(sleeping_preview) == 1, (
        f"{len(sleeping_preview)} preview reads were started by three clicks")


# --------------------------------------------------------------------------- #
#  What a click may not cost
# --------------------------------------------------------------------------- #

def test_a_checkbox_does_not_re_read_every_database(qtbot, plates,
                                                    sleeping_databases):
    """`describe` is what a checkbox, an anchor change and a repaint all end
    in. A fresh read there asks the same question once per keystroke."""
    panel = _panel(qtbot, _rows(plates))
    after_build = len(sleeping_databases)

    for _ in range(20):
        panel.describe()
        panel.plan_summary()
        panel.plan_evidence()

    assert len(sleeping_databases) == after_build, (
        f"{len(sleeping_databases) - after_build} extra database reads were "
        f"started by clicks that changed nothing about what was asked")


def test_a_late_read_does_not_paint_over_a_newer_one(qtbot, plates,
                                                     monkeypatch):
    """Two reads of one question overlap whenever `refresh` runs while the
    first is still out, and the older one can be the slower -- it is the one
    that went to the mount that was asleep. It must not put its answer back
    once the newer answer has landed."""
    from spacr.qt.widgets import measurement_scan_panel as panel_module

    panel = _panel(qtbot, _rows(plates))
    question = ("tables", plates[0])

    # Two answers to one question, filed out of order: the newer generation
    # first, then the older one landing late behind it.
    panel._file_read((7, question), ("newer",))
    panel._file_read((6, question), ("older",))

    assert panel._shown[question] == (7, ("newer",)), (
        "a read that landed late put the previous generation's answer back")
    assert panel_module  # the module is what was patched-free here


def test_closing_the_panel_leaves_no_probe_subscription(qtbot, plates):
    """`path_probe.probes` is process-wide and outlives every panel."""
    from spacr.qt import path_probe

    panel = _panel(qtbot, _rows(plates))
    assert panel._probe_redraw is not None
    panel.close()
    assert panel._probe_redraw is None
    # Idempotent: closeEvent can arrive twice.
    panel.close()
    assert path_probe.probes is not None
