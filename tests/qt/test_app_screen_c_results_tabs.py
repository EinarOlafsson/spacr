"""The results half of the regression screen when a panel will not cooperate.

Every method here is wired to a signal from a panel the screen does not own:
the Runs tab, the figure queue and grid, the coefficient table, the cells and
measurements tabs. Each is created late, destroyed on navigation and replaced
when another run loads, so "the panel is not there" and "the panel raised" are
ordinary states. What matters is the answer the screen gives: quiet when there
was nothing to do, a sentence when the user made a gesture that produced
nothing visible.

One screen is built per test and reused across its cases; every case's
overrides go through ``monkeypatch.context()`` and the fake console is cleared,
so no case can see another's state.
"""
from __future__ import annotations

import os
import types

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt.screens import app_screen as aps                   # noqa: E402
from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt

_ANY = object()


def _boom(*_args, **_kwargs):
    raise RuntimeError("the panel is half built")


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    lines = []
    widget._console = types.SimpleNamespace(
        said=lines, append_notice=lines.append, append_stdout=lines.append)
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


def _drive(screen, monkeypatch, subtests, cases):
    """Run each ``(name, attrs, act, expect)`` case on the shared screen.

    ``attrs(log)`` gives the screen attributes to stand in for the case (the
    fakes write what they were asked into ``log``); ``act(screen)`` is the
    gesture. ``expect`` may name ``returns``, the exact ``log``, ``said`` (a
    substring one console line must hold, or ``[]`` for silence) and
    ``after`` -- a ``(read, value)`` pair checked once the gesture is done.
    """
    for name, attrs, act, expect in cases:
        with subtests.test(case=name), monkeypatch.context() as m:
            log = []
            screen._console.said.clear()
            for attr, value in attrs(log).items():
                m.setattr(screen, attr, value, raising=False)
            got = act(screen)
            if expect.get("returns", _ANY) is not _ANY:
                assert got == expect["returns"]
            assert log == expect.get("log", [])
            said = expect.get("said", _ANY)
            if said == []:
                assert screen._console.said == []
            elif said is not _ANY:
                assert any(said in line for line in screen._console.said), (
                    screen._console.said)
            if "after" in expect:
                read, value = expect["after"]
                assert read(screen) == value


def _runs_tab(log, refuse=False):
    """A Runs tab that writes every call into ``log`` and may refuse it."""
    def call(name):
        def handler(*args, **fields):
            log.append((name, args, fields))
            if refuse:
                raise RuntimeError("the Runs tab has gone")
            return True
        return handler
    return types.SimpleNamespace(record_run=call("record_run"),
                                 update_run=call("update_run"),
                                 the_load_failed=call("the_load_failed"))


def test_runs_tab_bookkeeping_survives_a_missing_or_refusing_tab(
        screen, monkeypatch, subtests):
    """A row is bookkeeping, so its failure never stops the run.

    Invariant: with no Runs tab or a tab that raises, recording gives back no
    handle and an update answers False; a column fit writes to ITS OWN row
    (never the last handle) and forgets that row's handle whatever the tab
    did; a run that would not open hands the mark back exactly once.
    """
    ok_fields = {"status": "ok", "folder": "/data/run", "n_results": 12,
                 "error_type": None}
    failed_fields = {"status": "failed", "folder": None, "n_results": None,
                     "error_type": "did not fit"}
    record = (lambda s: s._record_run_in_runs_tab("run 1", "src", {}))
    update = (lambda s: s._update_run_in_runs_tab(status="ok"))
    handles = (lambda s: dict(s._column_run_handles))
    cases = [
        ("record: tab refuses",
         lambda log: {"_sweep_runs": _runs_tab(log, refuse=True)}, record,
         {"returns": None,
          "log": [("record_run", ("run 1", "src", {}), {})]}),
        ("record: no tab", lambda log: {"_sweep_runs": None}, record,
         {"returns": None}),
        ("update: tab refuses",
         lambda log: {"_sweep_runs": _runs_tab(log, refuse=True),
                      "_run_handle": "row-1"}, update,
         {"returns": False,
          "log": [("update_run", ("row-1",), {"status": "ok"})]}),
        ("update: no row",
         lambda log: {"_sweep_runs": _runs_tab(log), "_run_handle": None},
         update, {"returns": False}),
        ("column fit lands on its own row",
         lambda log: {"_sweep_runs": _runs_tab(log),
                      "_run_handle": "the-last-row",
                      "_column_run_handles": {"response_a": "row-7"}},
         lambda s: s._on_column_fit_finished(
             "response_a", {"ok": True, "folder": "/data/run",
                            "n_results": 12}),
         {"log": [("update_run", ("row-7",), ok_fields)],
          "after": (handles, {})}),
        ("column fit row refuses the outcome",
         lambda log: {"_sweep_runs": _runs_tab(log, refuse=True),
                      "_column_run_handles": {"response_a": "row-7"}},
         lambda s: s._on_column_fit_finished("response_a", {"ok": False}),
         {"log": [("update_run", ("row-7",), failed_fields)],
          "after": (handles, {})}),
        ("column fit never queued",
         lambda log: {"_sweep_runs": _runs_tab(log),
                      "_column_run_handles": {"other": "row-2"}},
         lambda s: s._on_column_fit_finished("never_queued", {"ok": True}),
         {"log": [], "after": (handles, {"other": "row-2"})}),
        ("failed load hands the mark back",
         lambda log: {"_sweep_runs": _runs_tab(log)},
         lambda s: s._the_run_did_not_open("the table is empty"),
         {"log": [("the_load_failed", ("the table is empty",), {})]}),
        ("tab refuses the mark",
         lambda log: {"_sweep_runs": _runs_tab(log, refuse=True)},
         lambda s: s._the_run_did_not_open("the table is empty"),
         {"log": [("the_load_failed", ("the table is empty",), {})]}),
        ("no tab, no mark", lambda log: {"_sweep_runs": None},
         lambda s: s._the_run_did_not_open("the table is empty"),
         {"said": [], "after": (lambda s: s._sweep_runs, None)}),
    ]
    _drive(screen, monkeypatch, subtests, cases)


def test_figure_grid_and_live_tiles_keep_their_route_back(
        screen, monkeypatch, subtests):
    """The grid is pictures; the live tile is the only way back to a widget.

    Invariant: a grid, queue, panel or menu that is missing or raises costs
    only its own step -- the live tile is still re-pinned after a refused
    rebuild or restyle, nothing is photographed without both a grid and a
    panel, an empty table unpins the tile, a tile size that cannot be saved still redraws, a tile that will
    not open leaves the page alone, and the one silent-gesture case (a tile
    with no tab in this run) says so.
    """
    def pin(log):
        return lambda: log.append("pin")

    def frame(log, refuse=False):
        def results_frame():
            log.append("frame")
            if refuse:
                raise RuntimeError("the results panel has gone")
        return results_frame

    def style_menu(log):
        return lambda: types.SimpleNamespace(
            exec=lambda at: log.append(("menu", at)))

    def refusing_preference(monkeypatch_):
        from spacr.qt import preferences
        monkeypatch_.setattr(preferences, "set_figure_grid_size", _boom)

    cases = [
        ("refresh with no grid",
         lambda log: {"_figure_grid": None, "_pin_regression_graph": pin(log)},
         lambda s: s._refresh_figure_grid(), {"log": []}),
        ("grid refuses the rebuild, tile still pinned",
         lambda log: {"_figure_grid": types.SimpleNamespace(set_figures=_boom),
                      "_pin_regression_graph": pin(log)},
         lambda s: s._refresh_figure_grid(), {"log": ["pin"]}),
        ("pin with no grid",
         lambda log: {"_figure_grid": None, "_results_panel":
                      types.SimpleNamespace(results_frame=frame(log))},
         lambda s: s._pin_regression_graph(), {"log": []}),
        ("pin with no results panel",
         lambda log: {"_figure_grid": object(), "_results_panel": None},
         lambda s: s._pin_regression_graph(), {"log": []}),
        ("panel refuses its photograph",
         lambda log: {"_figure_grid": types.SimpleNamespace(
             set_figures=lambda *a, **k: log.append("grid")),
             "_results_panel": types.SimpleNamespace(
                 results_frame=frame(log, refuse=True))},
         lambda s: s._pin_regression_graph(), {"log": ["frame"]}),
        ("an empty table unpins the live tile",
         lambda log: {"_figure_grid": types.SimpleNamespace(
             set_pinned=lambda *a: log.append(("pinned",) + a)),
             "_results_panel": types.SimpleNamespace(
                 results_frame=lambda: pd.DataFrame())},
         lambda s: s._pin_regression_graph(), {"log": [("pinned", None, "")]}),
        ("tile menu will not open",
         lambda log: {"_figure_queue": types.SimpleNamespace(
             show_figure_menu=_boom),
             "_refresh_figure_grid": lambda: log.append("refresh")},
         lambda s: s._figure_grid_menu(0, None), {"log": []}),
        ("tile menu restyled a figure",
         lambda log: {"_figure_queue": types.SimpleNamespace(
             show_figure_menu=lambda *a, **k: None),
             "_refresh_figure_grid": lambda: log.append("refresh")},
         lambda s: s._figure_grid_menu(0, None), {"log": ["refresh"]}),
        ("tile size cannot be remembered",
         lambda log: {"_figure_grid": types.SimpleNamespace(
             set_target_cell_width=lambda w: log.append(("width", w)))},
         lambda s: (refusing_preference(monkeypatch), s._on_figure_size(320)),
         {"log": [("width", 320)]}),
        ("tile will not open",
         lambda log: {"_figure_queue": types.SimpleNamespace(show_index=_boom),
                      "_figures_stack": types.SimpleNamespace(
                          setCurrentWidget=lambda w: log.append("moved"))},
         lambda s: s._open_figure_from_grid(3), {"log": []}),
        ("trial figures with no grid", lambda log: {"_figure_grid": None},
         lambda s: s._load_trial_figures(os.getcwd()), {"returns": 0}),
        ("live tile with no tab in this run",
         lambda log: {"_results_panel": types.SimpleNamespace(
             show_panel=lambda key: log.append(("show", key)) or False)},
         lambda s: s._open_live_tile("qq"),
         {"log": [("show", "qq")], "said": "no tab in this run"}),
        ("live tile and pinned menu with no panel",
         lambda log: {"_results_panel": None},
         lambda s: (s._open_live_tile("qq"), s._pinned_menu(None)),
         {"said": []}),
        ("menu for a key with no live widget",
         lambda log: {"_results_panel": types.SimpleNamespace(),
                      "_pin_regression_graph": pin(log)},
         lambda s: s._live_tile_menu("no_such_tile", None), {"log": []}),
        ("menu for a panel missing the widget",
         lambda log: {"_results_panel": types.SimpleNamespace(),
                      "_pin_regression_graph": pin(log)},
         lambda s: s._live_tile_menu("qq", None), {"log": []}),
        ("menu restyled the graph",
         lambda log: {"_results_panel": types.SimpleNamespace(
             qq=types.SimpleNamespace(build_style_menu=style_menu(log))),
             "_pin_regression_graph": pin(log)},
         lambda s: s._live_tile_menu("qq", "at-the-pointer"),
         {"log": [("menu", "at-the-pointer"), "pin"]}),
        ("menu will not build",
         lambda log: {"_results_panel": types.SimpleNamespace(
             qq=types.SimpleNamespace(build_style_menu=_boom)),
             "_pin_regression_graph": pin(log)},
         lambda s: s._live_tile_menu("qq", None), {"log": ["pin"]}),
    ]
    _drive(screen, monkeypatch, subtests, cases)


def test_switching_results_tabs_refreshes_only_what_is_there(
        screen, monkeypatch, subtests):
    """A tab re-reads when it is opened, and only if it can.

    Invariant: the opened tab's refresh is asked once (cells: ``refresh``,
    measurements: ``refresh_databases``, runs: ``load`` of the sweep
    destination); a tab without the capability or that raises is left alone;
    with no tab bar nothing refreshes; raising the Results tab still shows
    and opens the figures card when the tab bar refuses; a loaded-run change
    asks the measurements tab even after the cells tab raised; a guide click
    always raises the graph and opens a collapsed gene tile.
    """
    def recorder(log, tag, refuse=False):
        def call(*_args):
            log.append(tag)
            if refuse:
                raise RuntimeError(f"the {tag} tab has gone")
        return call

    def tabs_showing(attr):
        return types.SimpleNamespace(
            widget=lambda _i: getattr(screen, attr))

    class GeneSplit:
        def __init__(self, log):
            self.log = log

        def sizes(self):
            return [100, 0]

        def is_collapsed(self, name):
            return True

        def set_collapsed(self, name, collapsed, by_user):
            self.log.append((name, collapsed, by_user))

        def setSizes(self, sizes):
            self.log.append(("sizes", tuple(sizes)))

    def refusing_tab_bar(_page):
        raise TypeError("not a page of this tab bar")

    cases = [
        ("cells tab re-reads",
         lambda log: {"_cell_montage": types.SimpleNamespace(
             refresh=recorder(log, "cells"), shutdown=lambda: None),
             "_results_tabs": tabs_showing("_cell_montage")},
         lambda s: s._on_results_tab_changed(0), {"log": ["cells"]}),
        ("cells tab refuses",
         lambda log: {"_cell_montage": types.SimpleNamespace(
             refresh=recorder(log, "cells", refuse=True),
             shutdown=lambda: None),
             "_results_tabs": tabs_showing("_cell_montage")},
         lambda s: s._on_results_tab_changed(0), {"log": ["cells"]}),
        ("older measurements tab has no refresh",
         lambda log: {"_cell_montage": None,
                      "_scan_panel": types.SimpleNamespace(),
                      "_results_tabs": tabs_showing("_scan_panel")},
         lambda s: s._on_results_tab_changed(0),
         {"log": [], "after": (lambda s: vars(s._scan_panel), {})}),
        ("measurements tab refuses",
         lambda log: {"_cell_montage": None,
                      "_scan_panel": types.SimpleNamespace(
                          refresh_databases=recorder(log, "measurements",
                                                     refuse=True)),
                      "_results_tabs": tabs_showing("_scan_panel")},
         lambda s: s._on_results_tab_changed(0), {"log": ["measurements"]}),
        ("runs tab reloads the sweep destination",
         lambda log: {"_sweep_runs": types.SimpleNamespace(
             load=lambda folder: log.append(("load", folder))),
             "_sweep_destination": lambda: "/runs", "_cell_montage": None,
             "_scan_panel": None, "_results_tabs": tabs_showing("_sweep_runs")},
         lambda s: s._on_results_tab_changed(3), {"log": [("load", "/runs")]}),
        ("no tab bar refreshes nothing",
         lambda log: {"_results_tabs": None,
                      "_cell_montage": types.SimpleNamespace(
                          refresh=recorder(log, "cells"),
                          shutdown=lambda: None),
                      "_scan_panel": types.SimpleNamespace(
                          refresh_databases=recorder(log, "measurements"))},
         lambda s: s._on_results_tab_changed(0), {"log": []}),
        ("raise with no tab bar",
         lambda log: {"_results_tabs": None, "_results_page": "page"},
         lambda s: s._raise_the_results_tab(),
         {"after": (lambda s: (s._results_tabs, s._results_page),
                    (None, "page"))}),
        ("raise survives a refusing tab bar",
         lambda log: {"_results_tabs": types.SimpleNamespace(
             setCurrentWidget=refusing_tab_bar),
             "_results_page": object(),
             "_figures_card": types.SimpleNamespace(
                 show=lambda: log.append("show"),
                 folder=types.SimpleNamespace(
                     set_shut=lambda *a, **k: log.append("open")))},
         lambda s: s._raise_the_results_tab(), {"log": ["show", "open"]}),
        ("loaded run change survives tabs that raise",
         lambda log: {"_cell_montage": types.SimpleNamespace(
             clear=recorder(log, "cells", refuse=True), refresh=None),
             "_scan_panel": types.SimpleNamespace(
                 refresh_databases=recorder(log, "measurements",
                                            refuse=True))},
         lambda s: s._on_loaded_run_changed_refresh_tabs(),
         {"returns": None, "log": ["cells", "measurements"]}),
        ("guide click with no gene tile",
         lambda log: {"_show_regression_graph": lambda: log.append("graph"),
                      "_gene_split": None},
         lambda s: s._on_guide_selected("123_1"), {"log": ["graph"]}),
        ("guide click opens a collapsed gene tile",
         lambda log: {"_show_regression_graph": lambda: log.append("graph"),
                      "_gene_split": GeneSplit(log), "_gene_opened": False},
         lambda s: s._on_guide_selected("guide"),
         {"log": ["graph", ("Gene", False, False), ("sizes", (60, 40))],
          "after": (lambda s: s._gene_opened, True)}),
        ("refit with no settings", lambda log: {},
         lambda s: s._on_refit(None), {"returns": False}),
    ]
    _drive(screen, monkeypatch, subtests, cases)


def test_the_publication_sheet_always_answers(screen, monkeypatch, subtests):
    """The sheet answers "what did this run find"; it never fails silently.

    Invariant: no table says to open a finished run; a sheet that will not
    draw names its reason; a module whose package cannot be resolved still
    finds the sheet builder through the absolute import and queues the
    figure and shows it.
    """
    import spacr.figures as figures
    from matplotlib.figure import Figure
    from PySide6.QtWidgets import QApplication

    def refuse_to_draw(*_a, **_k):
        raise ValueError("no effect column")

    with subtests.test(case="no table loaded"), monkeypatch.context() as m:
        screen._console.said.clear()
        m.setattr(screen, "_results_panel", types.SimpleNamespace(
            results_frame=lambda: None))
        screen._show_publication_sheet()
        assert any("Open a finished run first" in line
                   for line in screen._console.said)

    with subtests.test(case="sheet will not draw"), monkeypatch.context() as m:
        screen._console.said.clear()
        frame = pd.DataFrame({"coefficient": [0.5], "p_value": [0.01]})
        m.setattr(screen, "_results_panel", types.SimpleNamespace(
            results_frame=lambda: frame))
        m.setattr(figures, "build_sheet", refuse_to_draw)
        screen._show_publication_sheet()
        assert any("Could not draw the publication figure" in line
                   and "no effect column" in line
                   for line in screen._console.said)

    with subtests.test(case="package cannot be resolved"), \
            monkeypatch.context() as m:
        figure = Figure()
        sheet = types.SimpleNamespace(
            figure=figure, skipped=(),
            legend=lambda: "Panels a-c: coefficients.")
        m.setattr(screen, "_results_panel", types.SimpleNamespace(
            results_frame=lambda: pd.DataFrame({"gene": ["A"],
                                                "coefficient": [1.0]})))
        m.setattr(figures, "build_sheet", lambda frame, width="double": sheet)
        shown = []
        m.setattr(screen, "_figure_queue", types.SimpleNamespace(
            add_figure=shown.append, count=lambda: 1,
            show_index=shown.append))
        m.setattr(screen, "_figures_stack", None)
        # Only the sheet call gets the broken package; the layout timers it
        # arms run afterwards with the real one, as in the module simulated.
        with monkeypatch.context() as broken:
            broken.setattr(aps, "__package__", "not_a_package.qt.screens")
            broken.setattr(aps, "__spec__", None)
            screen._show_publication_sheet()
        QApplication.processEvents()
        assert shown == [figure, 0]
