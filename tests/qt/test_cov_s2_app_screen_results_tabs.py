"""The results half of the regression screen when a panel will not cooperate.

Every method here is wired to a signal from a panel the screen does not own:
the Runs tab, the figure queue, the coefficient table, the compare panel. Each
of those is created late, destroyed on navigation, and replaced when a
different run is loaded, so "the panel is not there" and "the panel raised" are
both ordinary states rather than bugs.

What matters is which of the two answers the screen gives. Going quiet is right
when there was nothing to do; it is wrong when the user made a gesture, because
a gesture that produces nothing visible reads as a broken button. Both cases
appear below, and the difference between them is the point.
"""
from __future__ import annotations

import os
import types

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    widget._console = types.SimpleNamespace(
        said=[], append_notice=None, append_stdout=None)
    lines = []
    widget._console.said = lines
    widget._console.append_notice = lines.append
    widget._console.append_stdout = lines.append
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


def _boom(*_args, **_kwargs):
    raise RuntimeError("the panel is half built")


class TestTheRunsTabRow:

    def test_a_runs_tab_that_will_not_record_gives_back_no_handle(
            self, screen, monkeypatch):
        """A row is bookkeeping; the run itself still has to start.

        Returning a handle that was never recorded would make every later
        update write to a row that does not exist.
        """
        monkeypatch.setattr(screen, "_sweep_runs",
                            types.SimpleNamespace(record_run=_boom))

        assert screen._record_run_in_runs_tab("run 1", "src", {}) is None

    def test_a_screen_with_no_runs_tab_records_nothing(self, screen,
                                                        monkeypatch):
        monkeypatch.setattr(screen, "_sweep_runs", None)

        assert screen._record_run_in_runs_tab("run 1", "src", {}) is None

    def test_an_update_that_cannot_land_says_it_did_not(self, screen,
                                                         monkeypatch):
        """The caller uses the answer to decide whether to say so elsewhere."""
        monkeypatch.setattr(screen, "_sweep_runs",
                            types.SimpleNamespace(update_run=_boom))
        monkeypatch.setattr(screen, "_run_handle", object(), raising=False)

        assert screen._update_run_in_runs_tab(status="ok") is False

    def test_an_update_with_no_row_to_update_says_so(self, screen,
                                                     monkeypatch):
        monkeypatch.setattr(screen, "_sweep_runs",
                            types.SimpleNamespace(update_run=lambda *a, **k: True))
        monkeypatch.setattr(screen, "_run_handle", None, raising=False)

        assert screen._update_run_in_runs_tab(status="ok") is False

    def test_a_column_fit_outcome_writes_to_its_own_row(self, screen,
                                                         monkeypatch):
        """A queue puts several rows up at once.

        Using the screen's last handle would move every outcome onto one row
        and leave the other eleven saying "running" for ever.
        """
        written = {}
        monkeypatch.setattr(screen, "_sweep_runs", types.SimpleNamespace(
            update_run=lambda handle, **fields: written.update(
                handle=handle, **fields)))
        screen._column_run_handles["response_a"] = "row-7"

        screen._on_column_fit_finished(
            "response_a", {"ok": True, "folder": "/data/run", "n_results": 12})

        assert written["handle"] == "row-7"
        assert written["status"] == "ok"
        assert written["n_results"] == 12

    def test_a_column_fit_row_that_will_not_take_the_outcome_is_survivable(
            self, screen, monkeypatch):
        updates = []

        def refusing_update(handle, **fields):
            updates.append((handle, fields))
            raise RuntimeError("the row has gone")

        monkeypatch.setattr(screen, "_sweep_runs",
                            types.SimpleNamespace(update_run=refusing_update))
        screen._column_run_handles["response_a"] = "row-7"

        screen._on_column_fit_finished("response_a", {"ok": False})

        assert updates == [("row-7", {
            "status": "failed", "folder": None, "n_results": None,
            "error_type": "did not fit"})]
        assert "response_a" not in screen._column_run_handles

    def test_a_column_fit_with_no_row_is_left_alone(self, screen,
                                                    monkeypatch):
        monkeypatch.setattr(screen, "_sweep_runs", types.SimpleNamespace(
            update_run=lambda *a, **k: pytest.fail("no row to update")))

        screen._on_column_fit_finished("never_queued", {"ok": True})

    def test_a_run_that_would_not_open_hands_the_mark_back(self, screen,
                                                            monkeypatch):
        """The mark is a consequence of the run being SHOWN.

        The tab moves it and asks; without the other half of that
        conversation a failed load leaves a mark pointing at nothing.
        """
        told = []
        monkeypatch.setattr(screen, "_sweep_runs", types.SimpleNamespace(
            the_load_failed=told.append))

        screen._the_run_did_not_open("the table is empty")

        assert told == ["the table is empty"]

    def test_a_runs_tab_that_will_not_take_the_mark_back_is_survivable(
            self, screen, monkeypatch):
        refusals = []

        def refusing_mark(why):
            refusals.append(why)
            raise RuntimeError("the Runs tab has gone")

        monkeypatch.setattr(screen, "_sweep_runs",
                            types.SimpleNamespace(
                                the_load_failed=refusing_mark))

        screen._the_run_did_not_open("the table is empty")

        assert refusals == ["the table is empty"]

    def test_no_runs_tab_means_no_mark_to_hand_back(self, screen,
                                                    monkeypatch):
        monkeypatch.setattr(screen, "_sweep_runs", None)

        screen._the_run_did_not_open("the table is empty")

        assert screen._sweep_runs is None
        assert screen._console.said == []


class TestRebuildingTheFigureGrid:

    def test_a_screen_with_no_grid_rebuilds_nothing(self, screen,
                                                     monkeypatch):
        pin_attempts = []
        monkeypatch.setattr(screen, "_figure_grid", None)
        monkeypatch.setattr(screen, "_pin_regression_graph",
                            lambda: pin_attempts.append(True))

        screen._refresh_figure_grid()

        assert pin_attempts == []

    def test_a_grid_that_refuses_the_rebuild_still_gets_its_live_tile(
            self, screen, monkeypatch):
        """The live tile is the only route back to the interactive volcano.

        Losing it because the saved figures could not be laid out would leave
        a view the user can leave and not return to.
        """
        pinned = []
        monkeypatch.setattr(screen, "_figure_grid",
                            types.SimpleNamespace(set_figures=_boom))
        monkeypatch.setattr(screen, "_pin_regression_graph",
                            lambda: pinned.append(True))

        screen._refresh_figure_grid()

        assert pinned == [True]

    def test_nothing_is_pinned_without_a_grid_or_a_results_panel(
            self, screen, monkeypatch):
        frame_reads = []
        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(
            results_frame=lambda: frame_reads.append(True)))
        monkeypatch.setattr(screen, "_figure_grid", None)
        screen._pin_regression_graph()

        monkeypatch.setattr(screen, "_figure_grid", object())
        monkeypatch.setattr(screen, "_results_panel", None)
        screen._pin_regression_graph()

        assert frame_reads == []

    def test_a_results_panel_that_will_not_be_photographed_is_survivable(
            self, screen, monkeypatch):
        """The tile is a photograph of a live widget; a run has other output."""
        frame_reads = []
        grid_updates = []

        def refusing_frame():
            frame_reads.append(True)
            raise RuntimeError("the results panel has gone")

        monkeypatch.setattr(screen, "_figure_grid", types.SimpleNamespace(
            set_figures=lambda *args, **kwargs:
            grid_updates.append((args, kwargs))))
        monkeypatch.setattr(screen, "_results_panel",
                            types.SimpleNamespace(results_frame=refusing_frame))

        screen._pin_regression_graph()

        assert frame_reads == [True]
        assert grid_updates == []

    def test_a_tile_menu_that_will_not_open_leaves_the_grid_alone(
            self, screen, monkeypatch):
        refreshed = []
        monkeypatch.setattr(screen, "_figure_queue", types.SimpleNamespace(
            show_figure_menu=_boom))
        monkeypatch.setattr(screen, "_refresh_figure_grid",
                            lambda: refreshed.append(True))

        screen._figure_grid_menu(0, None)

        assert refreshed == [], (
            "the grid was rebuilt after a menu that never opened")

    def test_a_tile_menu_that_restyled_a_figure_rebuilds_the_grid(
            self, screen, monkeypatch):
        """The grid is built from pictures, so a restyle has to be retaken."""
        refreshed = []
        monkeypatch.setattr(screen, "_figure_queue", types.SimpleNamespace(
            show_figure_menu=lambda *a, **k: None))
        monkeypatch.setattr(screen, "_refresh_figure_grid",
                            lambda: refreshed.append(True))

        screen._figure_grid_menu(0, None)

        assert refreshed == [True]

    def test_a_tile_size_that_cannot_be_remembered_still_redraws_now(
            self, screen, monkeypatch):
        """Reading preference, not run state: the redraw is the visible half."""
        from spacr.qt import preferences

        widths = []
        monkeypatch.setattr(screen, "_figure_grid", types.SimpleNamespace(
            set_target_cell_width=widths.append))
        monkeypatch.setattr(preferences, "set_figure_grid_size", _boom)

        screen._on_figure_size(320)

        assert widths == [320]

    def test_a_tile_that_will_not_open_leaves_the_page_where_it_was(
            self, screen, monkeypatch):
        stack = types.SimpleNamespace(
            setCurrentWidget=lambda _w: pytest.fail("the page moved"))
        monkeypatch.setattr(screen, "_figure_queue",
                            types.SimpleNamespace(show_index=_boom))
        monkeypatch.setattr(screen, "_figures_stack", stack)

        screen._open_figure_from_grid(3)

    def test_a_trial_with_no_grid_to_load_into_reports_no_figures(
            self, screen, monkeypatch, tmp_path):
        monkeypatch.setattr(screen, "_figure_grid", None)

        assert screen._load_trial_figures(str(tmp_path)) == 0


class TestTheLiveTiles:

    def test_a_tile_with_no_tab_in_this_run_says_so(self, screen,
                                                    monkeypatch):
        """The one gesture that otherwise ends in nothing visible happening."""
        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(
            show_panel=lambda _key: False))

        screen._open_live_tile("qq")

        assert any("no tab in this run" in line for line in screen._console.said)

    def test_a_tile_pressed_with_no_results_panel_is_silent(self, screen,
                                                             monkeypatch):
        """Nothing was asked of a panel, so there is nothing to report."""
        monkeypatch.setattr(screen, "_results_panel", None)

        screen._open_live_tile("qq")
        screen._pinned_menu(None)

        assert screen._console.said == []

    def test_a_key_with_no_live_widget_has_no_menu(self, screen,
                                                    monkeypatch):
        """Absent from the table means the tile opens on a left click.

        That is the honest state rather than a bug, so the right-click does
        nothing and says nothing.
        """
        pinned = []
        monkeypatch.setattr(screen, "_pin_regression_graph",
                            lambda: pinned.append(True))
        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace())

        screen._live_tile_menu("no_such_tile", None)

        assert pinned == []

    def test_a_panel_missing_the_widget_has_no_menu_either(self, screen,
                                                            monkeypatch):
        pinned = []
        monkeypatch.setattr(screen, "_pin_regression_graph",
                            lambda: pinned.append(True))
        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace())

        screen._live_tile_menu("qq", None)

        assert pinned == []

    def test_a_menu_that_restyled_the_graph_retakes_its_photograph(
            self, screen, monkeypatch):
        """The tile is a photograph, so a restyle behind it has to be retaken."""
        pinned = []
        opened = []
        monkeypatch.setattr(screen, "_pin_regression_graph",
                            lambda: pinned.append(True))
        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(
            qq=types.SimpleNamespace(build_style_menu=lambda: types.SimpleNamespace(
                exec=opened.append))))

        screen._live_tile_menu("qq", "at-the-pointer")

        assert opened == ["at-the-pointer"]
        assert pinned == [True]

    def test_a_menu_that_will_not_build_still_retakes_the_photograph(
            self, screen, monkeypatch):
        pinned = []
        monkeypatch.setattr(screen, "_pin_regression_graph",
                            lambda: pinned.append(True))
        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(
            qq=types.SimpleNamespace(build_style_menu=_boom)))

        screen._live_tile_menu("qq", None)

        assert pinned == [True]


class TestThePublicationSheet:

    def test_with_no_table_loaded_it_says_what_to_do(self, screen,
                                                     monkeypatch):
        """The sheet answers "what did this run find"; nothing was fitted."""
        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(
            results_frame=lambda: None))

        screen._show_publication_sheet()

        assert any("Open a finished run first" in line
                   for line in screen._console.said)

    def test_a_sheet_that_will_not_draw_names_the_reason(self, screen,
                                                          monkeypatch):
        """A figure that failed silently is a button that looks broken."""
        import spacr.figures as figures

        frame = pd.DataFrame({"coefficient": [0.5], "p_value": [0.01]})
        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(
            results_frame=lambda: frame))
        monkeypatch.setattr(figures, "build_sheet", lambda *a, **k: (
            _ for _ in ()).throw(ValueError("no effect column")))

        screen._show_publication_sheet()

        assert any("Could not draw the publication figure" in line
                   and "no effect column" in line
                   for line in screen._console.said)


class TestARunRemovedFromTheRunsTab:

    def test_nothing_is_forgotten_without_a_panel_or_a_record(self, screen,
                                                               monkeypatch):
        monkeypatch.setattr(screen, "_results_panel", None)
        screen._on_runs_removed([{"folder": "/data/run"}])

        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(
            forget_run=lambda _f: pytest.fail("nothing was removed")))
        screen._on_runs_removed([])

    def test_a_row_with_no_folder_is_skipped(self, screen, monkeypatch):
        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(
            forget_run=lambda _f: pytest.fail("there was no folder")))

        screen._on_runs_removed([{"folder": ""}, None])

    def test_a_panel_that_will_not_forget_still_lets_the_figures_go(
            self, screen, monkeypatch):
        """A grid still showing a deleted run's tiles is the same stale answer.

        The two are separate stores, so failing to drop the plot state must
        not leave the pictures behind as well.
        """
        forgotten = []
        monkeypatch.setattr(screen, "_results_panel",
                            types.SimpleNamespace(forget_run=_boom))
        monkeypatch.setattr(screen, "_compare_panel", None, raising=False)
        monkeypatch.setattr(screen, "_figure_queue", types.SimpleNamespace(
            forget_run=lambda label: forgotten.append(label) or True))
        monkeypatch.setattr(screen, "_queue_figure_grid_refresh",
                            lambda: None)

        screen._on_runs_removed([{"folder": "/data/run", "run": "run 3"}])

        assert forgotten == ["run 3"]

    def test_a_queue_that_will_not_forget_is_survivable(self, screen,
                                                         monkeypatch):
        panel_forgets = []
        queue_forgets = []
        folder = "/data/run"
        screen._run_photographs[os.path.abspath(folder)] = object()

        def refusing_queue(label):
            queue_forgets.append(label)
            raise RuntimeError("the figure queue has gone")

        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(
            forget_run=panel_forgets.append))
        monkeypatch.setattr(screen, "_compare_panel", None, raising=False)
        monkeypatch.setattr(screen, "_figure_queue",
                            types.SimpleNamespace(forget_run=refusing_queue))

        screen._on_runs_removed([{"folder": folder, "run": "run 3"}])

        assert panel_forgets == [folder]
        assert queue_forgets == ["run 3"]
        assert os.path.abspath(folder) not in screen._run_photographs


class TestOpeningASecondRunBeside:

    def test_a_row_with_no_folder_on_disk_says_why(self, screen):
        """The gesture came from a menu, so it has to answer."""
        assert screen.open_run_beside({"folder": ""}) is False
        assert any("nothing to open beside" in line
                   for line in screen._console.said)

    def test_the_run_already_on_screen_is_not_opened_twice(self, screen,
                                                            monkeypatch,
                                                            tmp_path):
        """Two views of one run is not the comparison that was asked for."""
        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(
            run_folder=lambda: str(tmp_path)))

        assert screen.open_run_beside({"folder": str(tmp_path)}) is False
        assert any("already on screen" in line
                   for line in screen._console.said)

    def test_a_screen_with_no_split_cannot_open_one_beside(self, screen,
                                                            monkeypatch):
        monkeypatch.setattr(screen, "_results_panel", None)

        assert screen.open_run_beside({"folder": "/data/run"}) is False
        assert screen._console.said == []

    def test_a_compare_panel_that_will_not_be_photographed_still_closes(
            self, screen, monkeypatch, qtbot):
        """The still is a convenience; the panel closing is not."""
        from PySide6.QtWidgets import QWidget

        panel = QWidget()
        qtbot.addWidget(panel)
        panel.run_folder = lambda: "/data/run"
        panel.volcano = types.SimpleNamespace(grab=_boom)
        monkeypatch.setattr(screen, "_compare_panel", panel, raising=False)

        screen.close_run_beside()

        assert screen._compare_panel is None


class TestOpeningAResultsTab:

    def test_a_screen_with_no_tabs_raises_nothing(self, screen, monkeypatch):
        page = object()
        monkeypatch.setattr(screen, "_results_tabs", None)
        monkeypatch.setattr(screen, "_results_page", page, raising=False)

        screen._raise_the_results_tab()

        assert screen._results_tabs is None
        assert screen._results_page is page

    def test_opening_the_cells_tab_re_reads_it(self, screen, monkeypatch):
        """Nothing signals a database being attached while the tab is behind."""
        refreshed = []
        monkeypatch.setattr(screen, "_cell_montage", types.SimpleNamespace(
            refresh=lambda: refreshed.append(True),
            shutdown=lambda: None))
        monkeypatch.setattr(screen, "_results_tabs", types.SimpleNamespace(
            widget=lambda _i: screen._cell_montage))

        screen._on_results_tab_changed(0)

        assert refreshed == [True]

    def test_a_cells_tab_that_will_not_refresh_is_survivable(self, screen,
                                                              monkeypatch):
        refreshes = []

        def refusing_refresh():
            refreshes.append(True)
            raise RuntimeError("the cells tab has gone")

        monkeypatch.setattr(screen, "_cell_montage", types.SimpleNamespace(
            refresh=refusing_refresh, shutdown=lambda: None))
        monkeypatch.setattr(screen, "_results_tabs", types.SimpleNamespace(
            widget=lambda _i: screen._cell_montage))

        screen._on_results_tab_changed(0)

        assert refreshes == [True]

    def test_an_older_measurements_tab_with_no_refresh_is_left_alone(
            self, screen, monkeypatch):
        """The capability is asked for, not assumed."""
        older_panel = types.SimpleNamespace()
        monkeypatch.setattr(screen, "_cell_montage", None)
        monkeypatch.setattr(screen, "_scan_panel", older_panel,
                            raising=False)
        monkeypatch.setattr(screen, "_results_tabs", types.SimpleNamespace(
            widget=lambda _i: screen._scan_panel))

        screen._on_results_tab_changed(0)

        assert screen._scan_panel is older_panel

    def test_a_measurements_tab_that_will_not_refresh_is_survivable(
            self, screen, monkeypatch):
        refreshes = []

        def refusing_refresh():
            refreshes.append(True)
            raise RuntimeError("the measurements tab has gone")

        monkeypatch.setattr(screen, "_cell_montage", None)
        monkeypatch.setattr(screen, "_scan_panel", types.SimpleNamespace(
            refresh_databases=refusing_refresh), raising=False)
        monkeypatch.setattr(screen, "_results_tabs", types.SimpleNamespace(
            widget=lambda _i: screen._scan_panel))

        screen._on_results_tab_changed(0)

        assert refreshes == [True]

    def test_a_screen_with_no_results_tabs_changes_nothing(self, screen,
                                                            monkeypatch):
        refreshes = []
        monkeypatch.setattr(screen, "_results_tabs", None)
        monkeypatch.setattr(screen, "_cell_montage", types.SimpleNamespace(
            refresh=lambda: refreshes.append("cells"),
            shutdown=lambda: None))
        monkeypatch.setattr(screen, "_scan_panel", types.SimpleNamespace(
            refresh_databases=lambda: refreshes.append("measurements")),
                            raising=False)

        screen._on_results_tab_changed(0)

        assert refreshes == []


def test_a_refit_with_no_settings_starts_nothing(screen):
    """The dialog was cancelled; there is no model to fit."""
    assert screen._on_refit(None) is False


def test_a_guide_click_with_no_gene_tile_still_raises_the_graph(screen,
                                                                monkeypatch):
    """Drawing a ring on a view nobody is looking at draws nothing."""
    raised = []
    monkeypatch.setattr(screen, "_show_regression_graph",
                        lambda: raised.append(True))
    monkeypatch.setattr(screen, "_gene_split", None, raising=False)

    screen._on_guide_selected("123_1")

    assert raised == [True]


def test_a_folder_that_arrives_as_several_values_matches_no_run():
    """A run folder read off a concatenated frame is not always one path.

    An array of them is ambiguous the moment it is tested for emptiness, and
    the comparison has to answer "not the same run" rather than raising into
    whichever signal handler asked.
    """
    import numpy as np

    assert AppScreen._same_run_folder(
        np.array(["/data/run", "/data/other"]), "/data/run") is False


def test_a_finished_run_whose_results_will_not_open_still_finishes(
        screen, monkeypatch):
    """The run produced a table on disk; failing to show it is not failing.

    Letting it out would leave the Run button disabled and the progress bar
    up, because the rest of ``_on_finished`` never runs.
    """
    monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(),
                        raising=False)
    monkeypatch.setattr(screen, "_results_loaded_in_memory", False,
                        raising=False)
    monkeypatch.setattr(screen, "_load_regression_results", _boom)

    screen._on_finished(True)

    assert screen._btn_run.isEnabled()
    assert not screen._progress.isVisible()


class TestFindingTheTableTheRunJustWrote:

    def _arm(self, screen, monkeypatch, tables_by_folder, loads=True):
        from spacr.qt.widgets import regression_results

        monkeypatch.setattr(
            regression_results, "find_results_tables",
            lambda folder: tables_by_folder.get(str(folder), []))
        loaded = []

        def load(folder):
            loaded.append(str(folder))
            return loads

        monkeypatch.setattr(screen, "_results_panel", types.SimpleNamespace(
            load=load, say=lambda _text: screen._console.said.append(_text)))
        monkeypatch.setattr(screen, "_show_figure_grid", lambda: None)
        return loaded

    def test_a_screen_with_no_results_panel_loads_nothing(self, screen,
                                                           monkeypatch):
        monkeypatch.setattr(screen, "_results_panel", None)

        assert screen._load_regression_results() is False

    def test_the_newest_table_across_every_root_wins(self, screen,
                                                      monkeypatch, tmp_path):
        """``src`` and the count folder are different places.

        Both can hold a table, so "the first root that loads" can be last
        month's run.
        """
        old = tmp_path / "old"
        new = tmp_path / "new"
        for folder in (old, new):
            folder.mkdir()
            (folder / "coefficients.csv").write_text("a\n", encoding="utf-8")
        os.utime(old / "coefficients.csv", (1_000_000, 1_000_000))
        os.utime(new / "coefficients.csv", (2_000_000, 2_000_000))
        monkeypatch.setattr(screen._settings_model, "collect", lambda: {
            "src": str(old), "count_data": str(new / "counts.csv")})
        loaded = self._arm(screen, monkeypatch, {
            str(old): [str(old / "coefficients.csv")],
            str(new): [str(new / "coefficients.csv")]})

        assert screen._load_regression_results() is True
        assert loaded[0] == str(new)

    def test_a_list_valued_count_setting_names_its_folder_too(self, screen,
                                                              monkeypatch,
                                                              tmp_path):
        (tmp_path / "coefficients.csv").write_text("a\n", encoding="utf-8")
        monkeypatch.setattr(screen._settings_model, "collect", lambda: {
            "src": "", "count_data": [str(tmp_path / "counts.csv")]})
        loaded = self._arm(screen, monkeypatch, {
            str(tmp_path): [str(tmp_path / "coefficients.csv")]})

        assert screen._load_regression_results() is True
        assert loaded == [str(tmp_path)]

    def test_a_table_that_vanished_between_listing_and_stat_is_skipped(
            self, screen, monkeypatch, tmp_path):
        """A run folder on a share can go away mid-search."""
        monkeypatch.setattr(screen._settings_model, "collect",
                            lambda: {"src": str(tmp_path)})
        self._arm(screen, monkeypatch, {
            str(tmp_path): [str(tmp_path / "never-written.csv")]})

        assert screen._load_regression_results() is False

    def test_a_table_that_will_not_load_leaves_its_own_reason_on_screen(
            self, screen, monkeypatch, tmp_path):
        """The panel already said why; saying it again would be two answers."""
        (tmp_path / "coefficients.csv").write_text("a\n", encoding="utf-8")
        monkeypatch.setattr(screen._settings_model, "collect",
                            lambda: {"src": str(tmp_path)})
        self._arm(screen, monkeypatch, {
            str(tmp_path): [str(tmp_path / "coefficients.csv")]}, loads=False)

        assert screen._load_regression_results() is False
        assert screen._console.said == []

    def test_settings_that_name_no_folder_at_all_say_where_to_point(
            self, screen, monkeypatch):
        """A run that finished with the panel empty reads as producing nothing."""
        monkeypatch.setattr(screen._settings_model, "collect", lambda: {})
        self._arm(screen, monkeypatch, {})

        assert screen._load_regression_results() is False
        assert any("Load results" in line for line in screen._console.said)


class TestHandingAFinishedRunToTheResultsPanel:

    @staticmethod
    def _payload(frame):
        return {"results": frame, "res_folder": "/data/plate1/results/run_1",
                "settings": {"regression_type": "ols"}, "model": object(),
                "regression_type": "ols"}

    def _panel(self, monkeypatch, screen, **overrides):
        handed = {"frame": None}

        def set_frame(frame, source=""):
            handed["frame"] = frame
            handed["source"] = source
            return True

        panel = types.SimpleNamespace(
            set_frame=set_frame,
            set_run_settings=lambda s: handed.update(settings=s),
            set_diagnostics=lambda model, regression_type=None: handed.update(
                model=model),
            set_summary=lambda model, regression_type=None: handed.update(
                summary=model))
        for name, value in overrides.items():
            setattr(panel, name, value)
        monkeypatch.setattr(screen, "_results_panel", panel)
        monkeypatch.setattr(screen, "_update_run_in_runs_tab",
                            lambda **fields: True)
        monkeypatch.setattr(screen, "_say_the_qc_verdict", lambda _p: "")
        monkeypatch.setattr(screen, "_show_figure_grid", lambda: None)
        return handed

    def test_the_run_is_shown_from_memory_rather_than_read_back(self, screen,
                                                                 monkeypatch):
        """No path to guess, no newest-run heuristic, no other run's table."""
        frame = pd.DataFrame({"coefficient": [0.5, -0.2]})
        handed = self._panel(monkeypatch, screen)

        screen._on_pipeline_result(self._payload(frame))

        assert handed["frame"] is frame
        assert handed["settings"] == {"regression_type": "ols"}
        assert screen._results_loaded_in_memory is True
        assert screen._last_run_folder == "/data/plate1/results/run_1"
        assert any("coefficients loaded from the run itself" in line
                   for line in screen._console.said)

    def test_a_run_that_produced_no_rows_is_not_handed_over(self, screen,
                                                             monkeypatch):
        handed = self._panel(monkeypatch, screen)

        screen._on_pipeline_result(self._payload(pd.DataFrame()))
        screen._on_pipeline_result(self._payload(None))

        assert handed["frame"] is None

    def test_settings_the_panel_will_not_take_do_not_cost_the_table(
            self, screen, monkeypatch):
        """The coefficients are the run; the settings seed a later re-fit."""
        frame = pd.DataFrame({"coefficient": [0.5]})
        handed = self._panel(monkeypatch, screen, set_run_settings=_boom)

        screen._on_pipeline_result(self._payload(frame))

        assert handed["frame"] is frame
        assert screen._results_loaded_in_memory is True

    def test_a_model_the_panel_will_not_take_does_not_cost_the_table_either(
            self, screen, monkeypatch):
        """The residual and influence tabs go; the coefficients stay."""
        frame = pd.DataFrame({"coefficient": [0.5]})
        handed = self._panel(monkeypatch, screen, set_diagnostics=_boom)

        screen._on_pipeline_result(self._payload(frame))

        assert handed["frame"] is frame
        assert screen._results_loaded_in_memory is True

    def test_a_panel_that_refuses_the_frame_leaves_the_run_on_disk(
            self, screen, monkeypatch):
        """The files are written either way; only the view is lost."""
        frame = pd.DataFrame({"coefficient": [0.5]})
        self._panel(monkeypatch, screen, set_frame=_boom)

        screen._on_pipeline_result(self._payload(frame))

        assert screen._last_run_folder == "/data/plate1/results/run_1"
