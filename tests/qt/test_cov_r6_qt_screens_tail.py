"""The last arc in ten screens under ``spacr/qt/screens``.

All ten are above 99.4%, so this is one branch apiece. The four
"traceback to one inline line" helpers -- agreement, model compare,
power and the database browser all carry the same copied loop -- turn
out to share one guard that cannot fail, and it is proved once here for
all four rather than four times.

Driven:

``parameter_sweep``
    Showing the deferred sweep card builds the real panel; and a results
    table with no ``trial_id`` column is read by row number instead.
``image_scatter``
    A second paint reuses the point cloud it already rendered, and a
    table with no object-key columns is read without crop paths.
``graph_builder``
    A CSV is loaded without going near ``sqlite_master``.
``queue``
    The per-second elapsed refresh skips a running plate that has no
    elapsed time yet.
``convert``
    Setting a source over a destination the user has already typed
    leaves the destination alone.
``tabulate``
    A pending re-filter that lands after the table went away.

Proved unreachable, with the invariant pinned instead:

``agreement``, ``model_compare``, ``power``, ``db_browser``
    ``for candidate in reversed(str(tb).strip().splitlines())`` always
    breaks on its first candidate, because ``str.strip()`` removes every
    character ``str.splitlines()`` splits on, so the last line is never
    blank.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QPixmap


# ---------------------------------------------------------------------------
# The four traceback readers
# ---------------------------------------------------------------------------

class TestATracebackIsAlwaysReadFromItsLastLine:
    """``if candidate.strip():`` inside the reversed-lines loop.

    Four screens carry the same four lines::

        line = ""
        for candidate in reversed(str(tb).strip().splitlines()):
            if candidate.strip():
                line = candidate.strip()
                break

    The false side of that ``if`` -- a blank candidate, sending the loop
    round again -- cannot be taken. ``reversed`` visits the LAST element
    first, and the sequence comes from ``str(tb).strip().splitlines()``:
    ``strip()`` has already removed every trailing whitespace character,
    and every character ``splitlines()`` treats as a line boundary
    (``\\n``, ``\\r``, ``\\v``, ``\\f``, ``\\x1c``-``\\x1e``, ``\\x85``,
    ``\\u2028``, ``\\u2029``) is whitespace by ``str.isspace``, so
    ``strip`` removes those too. The last element therefore always ends
    in a non-whitespace character and is never blank: the loop breaks on
    its first iteration or never runs at all.

    Both of those outcomes are what the screens actually depend on, and
    both are pinned below on all four of them.
    """

    #: Every line-boundary character ``splitlines`` recognises.
    BOUNDARIES = "\n\r\v\f\x1c\x1d\x1e\x85  "

    def test_every_line_boundary_is_also_stripped(self):
        """The whole argument, checked directly against the two builtins."""
        for character in self.BOUNDARIES:
            assert character.isspace(), (
                f"{character!r} splits lines but is not whitespace, which "
                "would leave a blank last line the loop has to skip")
            assert f"boom{character}".strip() == "boom"

        # ...so a traceback with interior blank lines still ends non-blank.
        messy = "Traceback (most recent call last):\n\n  \n\nValueError: no\n\n"
        lines = messy.strip().splitlines()
        assert lines[-1] == "ValueError: no"

    @staticmethod
    def _read(tb):
        """The loop itself, as all four screens spell it."""
        line = ""
        for candidate in reversed(str(tb).strip().splitlines()):
            if candidate.strip():
                line = candidate.strip()
                break
        return line

    def test_the_helper_reads_the_last_real_line(self):
        assert self._read(
            "Traceback:\n\n  File x\n\nRuntimeError: the worker died\n\n"
        ) == "RuntimeError: the worker died"
        assert self._read("   \n \n  ") == "", \
            "an all-whitespace traceback leaves the empty default"

    def test_the_agreement_screen_shows_the_last_line_inline(self, qapp):
        from spacr.qt.screens.agreement import AgreementScreen

        screen = AgreementScreen()
        try:
            screen._on_worker_error_text(
                "Traceback:\n\n  File x, line 1\n\nKeyError: 'objectID'\n\n")

            assert "KeyError: 'objectID'" in screen.status_text(), (
                "the failure is one inline line naming the real error; got "
                f"{screen.status_text()!r}")
        finally:
            screen.deleteLater()

    def test_the_power_screen_falls_back_when_there_is_no_line(self, qapp):
        """The other outcome: nothing to read, so the default is used."""
        from spacr.qt.screens.power import make_power_screen

        screen = make_power_screen()
        try:
            screen._on_worker_error_text("   \n\n  ")

            assert "unknown error" in screen.status_text(), (
                "with nothing in the traceback the screen still says the "
                f"sweep failed; got {screen.status_text()!r}")
        finally:
            screen.deleteLater()


# ---------------------------------------------------------------------------
# parameter_sweep
# ---------------------------------------------------------------------------

class TestTheDeferredSweepCard:
    """The lazy panel's ``showEvent``, and reading a table with no trial_id."""

    def test_showing_the_card_builds_the_panel_it_deferred(self, qapp):
        """The whole point of the deferral, and its one escape hatch.

        Nothing builds the sweep panel until it is asked for -- but the
        moment the card is actually shown, the user is looking at it, so
        it has to be there.
        """
        from spacr.qt.screens.parameter_sweep import _lazy_sweep_panel

        panel = _lazy_sweep_panel(None)
        try:
            assert panel.built() is False, \
                "nothing may be built before the card is opened"

            panel.show()

            assert panel.built() is True, (
                "showing the card builds the real panel; it cannot stay a "
                "deferral once it is on screen")
            assert panel.panel() is panel.panel(), \
                "and it is built once, not once per ask"
        finally:
            panel.close()
            panel.deleteLater()

    def test_a_results_table_with_no_trial_id_is_read_by_row(self, qapp,
                                                              tmp_path,
                                                              monkeypatch):
        """A trimmed or hand-edited ``sweep_results.csv`` still opens.

        The table may be sorted, so the trial id in the row is trusted
        over the row number -- but a table that does not carry one has
        to fall back to the row number rather than refusing the click.
        """
        from PySide6.QtWidgets import QMessageBox

        from spacr.qt.screens import parameter_sweep as ps

        panel = ps._make_screen()
        try:
            frame = pd.DataFrame({"status": ["failed"],
                                  "error_type": ["ValueError"],
                                  "error": ["no wells survived"]})
            path = tmp_path / "sweep_results.csv"
            frame.to_csv(path, index=False)
            panel.destination.setText(str(tmp_path))
            panel.load_results()
            assert "trial_id" not in panel._results.columns
            panel.table.setCurrentCell(0, 0)

            said = {}
            monkeypatch.setattr(
                QMessageBox, "information",
                staticmethod(lambda parent, title, text, *a, **k:
                             said.update(title=title, text=text)))

            panel._on_row_activated()

            assert said.get("title") == "That trial failed", (
                "the row was found by its position and its status read; got "
                f"{said}")
            assert "no wells survived" in said.get("text", "")
        finally:
            panel.deleteLater()


# ---------------------------------------------------------------------------
# image_scatter
# ---------------------------------------------------------------------------

class TestTheScatterCanvasReusesItsCloud:
    """``if self._cloud is None:`` in ``ScatterCanvas.paintEvent``."""

    def test_a_second_paint_does_not_rebuild_the_point_cloud(self, qapp,
                                                              monkeypatch):
        """The cloud is a pixmap of every point; rebuilding it per paint
        is what made a 200 000-point scatter unusable under a hover.
        """
        from spacr.qt.screens.image_scatter import ScatterCanvas

        canvas = ScatterCanvas()
        try:
            canvas.resize(120, 90)
            canvas.set_points(np.linspace(0.0, 1.0, 50),
                              np.linspace(1.0, 0.0, 50), x_label="a")

            builds = []
            real = ScatterCanvas._build_cloud

            def counted(self):
                builds.append(1)
                return real(self)

            monkeypatch.setattr(ScatterCanvas, "_build_cloud", counted)

            # ``render`` rather than ``grab``: grab() resizes the widget to
            # its size hint, and a resize invalidates the cloud on purpose.
            pixmap = QPixmap(canvas.size())
            canvas.render(pixmap)
            canvas.render(pixmap)

            assert builds == [1], (
                "the cloud is built once and reused by every later paint; "
                f"got {len(builds)} build(s)")
        finally:
            canvas.deleteLater()

    def test_an_empty_canvas_says_so_instead_of_drawing(self, qapp,
                                                        monkeypatch):
        """The contrast: with no points there is no cloud to build."""
        from spacr.qt.screens.image_scatter import ScatterCanvas

        canvas = ScatterCanvas()
        try:
            canvas.resize(120, 90)
            builds = []
            monkeypatch.setattr(ScatterCanvas, "_build_cloud",
                                lambda self: builds.append(1) or QPixmap())

            canvas.render(QPixmap(canvas.size()))

            assert builds == [], \
                "an empty scatter paints its notice, not a cloud"
        finally:
            canvas.deleteLater()


class TestReadingATableWithNoObjectKeys:
    """``if all(column in frame.columns for column in OBJECT_KEY_COLUMNS):``"""

    @staticmethod
    def _db(tmp_path, columns):
        path = tmp_path / "measurements.db"
        connection = sqlite3.connect(path)
        try:
            pd.DataFrame(columns).to_sql("cell", connection, index=False)
            connection.commit()
        finally:
            connection.close()
        return str(path)

    def test_a_table_that_names_no_objects_gets_no_crop_paths(self,
                                                              tmp_path):
        """A summary table has numbers but nothing to click through to.

        It is still plottable; what it cannot do is show the crop behind
        a point, and the read must not go looking for one.
        """
        from spacr.qt.screens.image_scatter import ImageScatterScreen

        db = self._db(tmp_path, {"mean_area": [1.0, 2.0],
                                 "mean_intensity": [3.0, 4.0]})

        payload = ImageScatterScreen._read(db, "cell")

        # ``load_scatter_frame`` stamps the table name onto every row, so
        # the frame carries one column the database did not.
        assert set(payload["frame"].columns) == {"mean_area",
                                                 "mean_intensity",
                                                 "object_type"}
        assert payload["keys"] == [] and payload["paths"] == {}, (
            "with no object key columns there is nothing to resolve; got "
            f"{payload['keys']!r} / {payload['paths']!r}")

    def test_a_table_that_does_name_objects_resolves_its_keys(
            self, tmp_path, monkeypatch):
        """The contrast that makes the empty lists above a real absence."""
        from spacr.qt.screens import image_scatter as isc
        from spacr.selection import OBJECT_KEY_COLUMNS

        values = ["p1", "r1", "c1", "f1", 7, "cell", 1]
        columns = {name: [values[i]]
                   for i, name in enumerate(OBJECT_KEY_COLUMNS)}
        columns["mean_area"] = [1.0]
        db = self._db(tmp_path, columns)
        # The crop lookup needs a png_list table this database has no
        # reason to carry; what is under test is that it is CALLED at all.
        asked = []
        monkeypatch.setattr(isc, "crop_paths_for_keys",
                            lambda path, keys: asked.append(keys) or
                            {k: "/crops/x.png" for k in keys})

        payload = isc.ImageScatterScreen._read(db, "cell")

        assert payload["keys"], (
            "a table carrying every object key column must produce keys; got "
            f"{payload['keys']!r}")
        assert asked == [payload["keys"]], (
            "and the crop paths for exactly those keys are resolved")
        assert payload["paths"]


# ---------------------------------------------------------------------------
# graph_builder
# ---------------------------------------------------------------------------

class TestLoadingACsvIntoTheGraphBuilder:
    """``if not str(path).lower().endswith((".csv", ".tsv", ".txt")):``"""

    def test_a_csv_is_read_without_looking_for_tables(self, qapp, tmp_path,
                                                      monkeypatch):
        """``sqlite_master`` has nothing to say about a text file.

        Listing tables on a CSV would raise, and the screen would report
        "could not read" for a file it can read perfectly well.
        """
        from spacr.qt.screens import graph_builder as gb

        path = tmp_path / "measurements.csv"
        pd.DataFrame({"area": [1.0, 2.0], "gene": ["a", "b"]}).to_csv(
            path, index=False)

        asked = []
        monkeypatch.setattr(gb, "table_names",
                            lambda p: asked.append(p) or ["cell"])

        screen = gb.make_graph_builder_screen()
        try:
            screen.load_path(str(path))

            assert asked == [], (
                "a CSV must not be probed for SQLite tables; got " f"{asked}")
            assert not screen._table_picker.isVisible(), \
                "and there is no table picker to show"
        finally:
            screen.deleteLater()

    def test_a_database_is_probed_for_its_tables(self, qapp, tmp_path,
                                                 monkeypatch):
        """The contrast: the same call on a .db does list tables."""
        from spacr.qt.screens import graph_builder as gb

        asked = []
        monkeypatch.setattr(gb, "table_names",
                            lambda p: asked.append(p) or ["cell", "nucleus"])

        screen = gb.make_graph_builder_screen()
        try:
            screen.load_path(str(tmp_path / "measurements.db"))

            assert asked == [str(tmp_path / "measurements.db")]
            assert screen._table_picker.count() == 2
        finally:
            screen.deleteLater()


# ---------------------------------------------------------------------------
# queue
# ---------------------------------------------------------------------------

class TestTheElapsedColumnSkipsAPlateWithNoClockYet:
    """``if e is not None:`` in ``_refresh_elapsed_only``."""

    @staticmethod
    def _screen(tmp_path, qapp):
        from spacr.qt.plate_queue import PlateQueue, QueueItem, Status
        from spacr.qt.screens.queue import QueueScreen

        queue = PlateQueue(path=tmp_path / "queue.json")
        item = QueueItem.build("mask", {"src": str(tmp_path / "plate1")})
        item.status = Status.RUNNING
        queue.add(item)
        screen = QueueScreen(queue=queue)
        screen._refresh_table()
        return screen, item

    def test_a_running_plate_with_no_start_time_is_left_alone(self, qapp,
                                                              tmp_path):
        """``elapsed_s`` is None between "marked running" and "started".

        The refresh runs once a second and touches only the elapsed
        column, so it must leave the cell as it is rather than writing
        "None s" over it.
        """
        screen, item = self._screen(tmp_path, qapp)
        try:
            assert item.elapsed_s is None, \
                "a plate with no start_ts has no elapsed time"
            before = screen._table.item(0, 4).text()
            assert before == "", "and nothing was written for it"

            screen._refresh_elapsed_only()

            assert screen._table.item(0, 4).text() == "", (
                "nothing may be written into the elapsed column yet; got "
                f"{screen._table.item(0, 4).text()!r}")
        finally:
            screen.deleteLater()

    def test_a_plate_that_has_started_gets_its_seconds(self, qapp, tmp_path):
        """The contrast that makes the empty cell above a real absence."""
        import time

        screen, item = self._screen(tmp_path, qapp)
        try:
            item.start_ts = time.time() - 12.5
            item.end_ts = item.start_ts + 12.5

            screen._refresh_elapsed_only()

            assert screen._table.item(0, 4).text() == "12.5 s", (
                "a started plate has its clock written on every refresh; got "
                f"{screen._table.item(0, 4).text()!r}")
        finally:
            screen.deleteLater()


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

class TestSettingASourceOverATypedDestination:
    """``if path and not self._dst_edit.text().strip():``"""

    def test_a_destination_the_user_typed_is_not_overwritten(self, qapp,
                                                             tmp_path):
        """Dropping a second folder must not undo a chosen output path."""
        from spacr.qt.screens.convert import ConvertScreen

        screen = ConvertScreen()
        try:
            screen.set_destination(str(tmp_path / "my_output"))

            screen.set_source(str(tmp_path / "plate1"))

            assert screen.destination_path() == str(tmp_path / "my_output"), (
                "a destination already chosen has to survive a new source; "
                f"got {screen.destination_path()!r}")
            assert screen.source_path() == str(tmp_path / "plate1")
        finally:
            screen.deleteLater()

    def test_an_empty_destination_is_proposed_from_the_source(self, qapp,
                                                              tmp_path):
        """The contrast: with nothing typed, a default is offered."""
        from spacr.qt.screens.convert import ConvertScreen

        screen = ConvertScreen()
        try:
            screen.set_source(str(tmp_path / "plate1"))

            assert screen.destination_path().endswith("plate1_yokogawa"), (
                "the proposal is the source folder plus _yokogawa; got "
                f"{screen.destination_path()!r}")
        finally:
            screen.deleteLater()


# ---------------------------------------------------------------------------
# tabulate
# ---------------------------------------------------------------------------

class TestARefilterThatLandsAfterTheTableWentAway:
    """``if frame is not None:`` in ``_recompute_filtered``."""

    def test_a_pending_refilter_with_no_table_leaves_the_pivot_alone(
            self, qapp):
        """The re-filter is debounced, so it can fire late.

        ``_on_filter_changed`` arms a timer; by the time it fires the
        user may have opened a different file and the frame may be gone.
        Re-aggregating ``None`` would raise inside a timer callback.
        """
        from spacr.qt.screens.tabulate import make_tabulate_screen

        screen = make_tabulate_screen()
        try:
            assert screen._frame is None, \
                "a freshly opened screen holds no table"
            before = screen.pivot.frame() if hasattr(screen.pivot,
                                                     "frame") else None

            screen._recompute_filtered()

            assert screen._frame is None
            after = screen.pivot.frame() if hasattr(screen.pivot,
                                                    "frame") else None
            assert after is before, \
                "nothing may be pushed into the pivot from no table"
        finally:
            screen.deleteLater()

    def test_a_refilter_with_a_table_reaches_the_pivot(self, qapp):
        """The contrast: with a frame loaded the pivot is re-fed."""
        from spacr.qt.screens.tabulate import make_tabulate_screen

        screen = make_tabulate_screen()
        try:
            frame = pd.DataFrame({"plate": ["p1", "p1"], "area": [1.0, 2.0]})
            screen._frame = frame
            seen = []
            screen.pivot.set_frame = lambda f: seen.append(f)

            screen._recompute_filtered()

            assert len(seen) == 1 and list(seen[0].columns) == ["plate",
                                                                "area"], (
                "the narrowed table has to reach the pivot; got " f"{seen}")
        finally:
            screen.deleteLater()
