"""Six screens, one uncovered decision each.

Four are driven. The other two are the same loop written twice -- the
one that turns a worker traceback into a single inline line, in
Agreement and in Power -- and both are pinned, because the ``.strip()``
one line above already guarantees the last line is not blank.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# agreement.py / power.py -- the last NON-BLANK line of a traceback
# ---------------------------------------------------------------------------

TRACEBACK_WITH_TRAILING_BLANKS = (
    "Traceback (most recent call last):\n"
    "  File \"worker.py\", line 12, in run\n"
    "    guide_support(frame)\n"
    "ValueError: no tested rows\n"
    "\n"
    "   \n"
    "\n"
)


class TestOneInlineLineFromAWorkerTraceback:

    def test_agreement_reports_the_last_meaningful_line(self, qtbot):
        """A traceback that ends in whitespace is ordinary -- a
        subprocess that flushed a newline, a formatter that appended
        one. Taking the literal last line would put an empty error on
        the status bar, which reads as "it failed and we cannot say
        why". Neither screen may raise a dialog for a failed job, so
        that one line is the whole error report.
        """
        from spacr.qt.screens.agreement import AgreementScreen

        screen = AgreementScreen()
        qtbot.addWidget(screen)

        screen._on_worker_error_text(TRACEBACK_WITH_TRAILING_BLANKS)

        assert "ValueError: no tested rows" in screen._status.text()

    def test_power_reports_the_last_meaningful_line(self, qtbot):
        from spacr.qt.screens.power import PowerScreen

        screen = PowerScreen()
        qtbot.addWidget(screen)

        screen._on_worker_error_text(TRACEBACK_WITH_TRAILING_BLANKS)

        assert "ValueError: no tested rows" in screen._status.text()

    def test_power_says_something_even_for_a_traceback_of_only_blanks(self,
                                                                      qtbot):
        """Where the two screens differ: Power names the empty case."""
        from spacr.qt.screens.power import PowerScreen

        screen = PowerScreen()
        qtbot.addWidget(screen)

        screen._on_worker_error_text("\n   \n\n")

        assert "unknown error" in screen._status.text()

    def test_the_skip_cannot_fire_because_the_text_is_stripped_first(self):
        """THE PIN, for both screens.

        The loop walks backwards looking for a non-blank line, but the
        text was already ``.strip()``ed -- so the LAST element of
        ``splitlines()`` is never blank and the first candidate always
        breaks. A wholly blank traceback splits to nothing and the loop
        body never runs at all, which is the empty case above.

        Removing the strip is what makes the skip live, and that is what
        this checks, in both copies of the loop.
        """
        import inspect

        from spacr.qt.screens.agreement import AgreementScreen
        from spacr.qt.screens.power import PowerScreen

        for owner in (AgreementScreen, PowerScreen):
            source = inspect.getsource(owner._on_worker_error_text)
            assert "str(tb).strip().splitlines()" in source, (
                f"{owner.__name__} no longer strips before splitting, so a "
                f"trailing blank line can now reach the loop")

        for text in (TRACEBACK_WITH_TRAILING_BLANKS,
                     "one line\n\n\n",
                     "  padded  \n \t \n"):
            lines = str(text).strip().splitlines()
            assert not lines or lines[-1].strip(), (
                "a stripped traceback ended in a blank line")


# ---------------------------------------------------------------------------
# queue.py -- a running item with no elapsed time yet
# ---------------------------------------------------------------------------

class TestTheElapsedColumn:

    def _screen(self, qtbot, items):
        from spacr.qt.plate_queue import PlateQueue
        from spacr.qt.screens.queue import QueueScreen

        queue = PlateQueue()
        for item in items:
            queue._items.append(item)
        screen = QueueScreen(queue)
        qtbot.addWidget(screen)
        screen._refresh_table()
        return screen

    def _item(self, status, start_ts=None):
        from spacr.qt.plate_queue import QueueItem

        item = QueueItem.build("measure", {"src": "/data/plate1"})
        item.status = status
        item.start_ts = start_ts
        return item

    def test_a_running_item_with_a_start_time_shows_its_elapsed_seconds(
            self, qtbot):
        import time

        from spacr.qt.plate_queue import Status

        screen = self._screen(qtbot, [self._item(Status.RUNNING,
                                                 start_ts=time.time() - 3)])
        screen._refresh_elapsed_only()

        cell = screen._table.item(0, 4)
        assert cell is not None and cell.text().endswith(" s")

    def test_a_running_item_that_has_not_started_the_clock_is_left_blank(
            self, qtbot):
        """THE UNCOVERED ARC.

        `elapsed_s` is None until `start_ts` is set, and the status flips
        to RUNNING before the worker stamps it. Formatting None as
        seconds is a TypeError inside a once-a-second timer, which is the
        worst place for one -- it fires again immediately.
        """
        from spacr.qt.plate_queue import Status

        screen = self._screen(qtbot, [self._item(Status.RUNNING,
                                                 start_ts=None)])
        before = screen._table.item(0, 4)
        before_text = before.text() if before is not None else None

        screen._refresh_elapsed_only()      # must not raise

        after = screen._table.item(0, 4)
        assert (after.text() if after is not None else None) == before_text


# ---------------------------------------------------------------------------
# graph_builder.py -- a CSV has one table, so there is nothing to list
# ---------------------------------------------------------------------------

class TestListingTheTablesInASource:

    def test_a_csv_skips_the_table_listing_entirely(self, qtbot, tmp_path):
        """THE UNCOVERED ARC.

        `table_names` is a sqlite_master query. Running it against a CSV
        raises, and the screen would report "could not read plate.csv"
        for a file it can read perfectly well.
        """
        from spacr.qt.screens.graph_builder import GraphBuilderScreen

        csv = tmp_path / "plate.csv"
        csv.write_text("a,b\n1,2\n")

        screen = GraphBuilderScreen(threaded=False)
        qtbot.addWidget(screen)
        screen.load_path(str(csv))

        assert screen._table_picker.isVisibleTo(screen) is False
        assert "could not read" not in screen._source.text()

    def test_a_database_lists_its_tables_in_the_picker(self, qtbot, tmp_path):
        import sqlite3

        from spacr.qt.screens.graph_builder import GraphBuilderScreen

        db = tmp_path / "measurements.db"
        with sqlite3.connect(db) as connection:
            connection.execute("CREATE TABLE cell (a REAL)")
            connection.execute("INSERT INTO cell VALUES (1.0)")
            connection.execute("CREATE TABLE nucleus (a REAL)")
            connection.execute("INSERT INTO nucleus VALUES (2.0)")

        screen = GraphBuilderScreen(threaded=False)
        qtbot.addWidget(screen)
        screen.load_path(str(db))

        names = [screen._table_picker.itemText(i)
                 for i in range(screen._table_picker.count())]
        assert "cell" in names and "nucleus" in names


# ---------------------------------------------------------------------------
# convert.py -- the destination is only proposed when there is none
# ---------------------------------------------------------------------------

class TestProposingADestination:

    def _screen(self, qtbot):
        from spacr.qt.screens.convert import ConvertScreen

        screen = ConvertScreen()
        qtbot.addWidget(screen)
        return screen

    def test_a_source_with_no_destination_yet_proposes_one(self, qtbot):
        screen = self._screen(qtbot)

        screen.set_source("/data/plate1")

        assert screen._dst_edit.text().endswith("plate1_yokogawa")

    def test_a_destination_already_typed_is_not_overwritten(self, qtbot):
        """THE UNCOVERED ARC.

        The proposal is a convenience, not a policy. Overwriting a
        destination the user typed would silently redirect the output of
        a conversion they had already aimed somewhere.
        """
        screen = self._screen(qtbot)
        screen._dst_edit.setText("/somewhere/i/chose")

        screen.set_source("/data/plate1")

        assert screen._dst_edit.text() == "/somewhere/i/chose"

    def test_clearing_the_source_proposes_nothing(self, qtbot):
        screen = self._screen(qtbot)

        screen.set_source("")

        assert screen._dst_edit.text() == ""
        assert screen.source_path() == ""


# ---------------------------------------------------------------------------
# image_umap.py -- a screen with no measurements database behind it
# ---------------------------------------------------------------------------

class TestBuildingTheFoldedPcaView:

    def test_a_host_with_no_source_still_gets_the_view(self, qtbot):
        """THE UNCOVERED ARC.

        The fold is built when the strip is installed, which happens
        before any folder has been chosen. `load_path("")` would report
        a missing file for a path the user has not typed yet, so the
        view arrives empty and waits.
        """
        from PySide6.QtWidgets import QWidget

        from spacr.qt.screens.image_umap import _build_pca, source_path

        host = QWidget()
        qtbot.addWidget(host)
        assert source_path(host) == ""

        view = _build_pca(None, host)
        qtbot.addWidget(view)

        assert view is not None
        assert view._frame is None, "an empty source loaded something anyway"
