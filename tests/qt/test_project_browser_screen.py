"""``N4`` — the Project Browser screen: what it lists, and what it never claims.

Everything about *stages, staleness and sizes* is asserted against the engine
in ``tests/test_projects.py``; this file only asserts that the screen shows
that result and no other. So the numbers here are compared with
:mod:`spacr.projects`' own — the screen is allowed to be wrong about layout
and never about the facts.

The screen is built with ``threaded=False`` so the walk runs inline.
:class:`spacr.qt.job_runner.JobRunner` emits the same signals in the same
order either way, which is the point of the flag: the test drives the real
code path rather than a synchronous stand-in. One test does run it threaded,
because "the scan does not block the GUI" is the reason the runner is there
at all, and a job that never retires is the specific failure its docstring
warns about.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr import artifacts, ports, projects
from spacr.qt.screens.project_browser import (
    APP_KEY, COLUMNS, MAX_PROJECTS, ProjectBrowserScreen,
    make_project_browser_screen, register,
)

pytestmark = pytest.mark.qt


def _plate(root, name="plate1", *, merged=2, db=True, raw=False):
    """A project folder built at the paths ``spacr.ports`` declares."""
    plate = os.path.join(str(root), name)
    os.makedirs(plate, exist_ok=True)
    if merged:
        os.makedirs(os.path.join(plate, "merged"), exist_ok=True)
        for index in range(merged):
            np.save(os.path.join(plate, "merged", f"f{index}.npy"),
                    np.zeros((4, 4, 3), dtype=np.uint16))
    if db:
        os.makedirs(os.path.join(plate, "measurements"), exist_ok=True)
        connection = sqlite3.connect(
            os.path.join(plate, "measurements", "measurements.db"))
        connection.execute("CREATE TABLE IF NOT EXISTS cell (id INTEGER)")
        connection.execute("INSERT INTO cell VALUES (1)")
        connection.commit()
        connection.close()
    if raw:
        with open(os.path.join(plate, f"{name}_A01_T0001F001L01A01Z01C01.tif"),
                  "wb") as handle:
            handle.write(b"tif")
    return plate


@pytest.fixture
def screen(qtbot, tmp_path):
    """A browser pointed at ``tmp_path``, running its jobs inline."""
    widget = ProjectBrowserScreen(threaded=False, roots=(str(tmp_path),))
    qtbot.addWidget(widget)
    return widget


def _row_for(widget, name):
    for row in range(widget._table.rowCount()):
        if widget._table.item(row, 0).text() == name:
            return row
    raise AssertionError(f"{name!r} is not in the table")


def _cell(widget, name, column):
    return widget._table.item(_row_for(widget, name),
                              COLUMNS.index(column)).text()


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def test_it_lists_the_projects_under_the_chosen_folder(qtbot, tmp_path):
    _plate(tmp_path, name="plate_a", merged=1)
    _plate(tmp_path, name="plate_b", merged=1)
    widget = ProjectBrowserScreen(threaded=False, roots=(str(tmp_path),))
    qtbot.addWidget(widget)
    assert widget._table.rowCount() == 2
    assert {_cell(widget, n, "Project") for n in ("plate_a", "plate_b")} == {
        "plate_a", "plate_b"}


def test_every_cell_is_the_engines_own_answer(screen, tmp_path):
    """The screen renders :mod:`spacr.projects`; it does not recompute."""
    plate = _plate(tmp_path, name="plate_a", merged=2)
    screen.rescan()
    summary = projects.scan(plate, with_next_steps=False)
    from spacr.data_manager import human_bytes

    assert _cell(screen, "plate_a", "Stage") == summary.stage_label
    assert _cell(screen, "plate_a", "Size") == human_bytes(summary.size_bytes)
    assert _cell(screen, "plate_a", "State") == summary.staleness_note()
    assert _cell(screen, "plate_a", "Note") == summary.note()


def test_an_empty_folder_says_so_rather_than_showing_an_empty_table(screen):
    screen.rescan()
    assert screen._table.rowCount() == 0
    assert "No projects found" in screen._status.text()


def test_the_scanned_signal_carries_how_many_were_listed(qtbot, tmp_path):
    _plate(tmp_path, name="one", merged=1)
    widget = ProjectBrowserScreen(threaded=False)
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.scanned, timeout=5000) as caught:
        widget.add_root(str(tmp_path))
    assert caught.args == [1]


# ---------------------------------------------------------------------------
# The project the registry has never seen
# ---------------------------------------------------------------------------

def test_an_unrecorded_project_is_listed_and_marked_as_unrecorded(screen, tmp_path):
    """The case the browser exists for.

    A folder copied in this morning appears with its stage and its size, and
    the state column says *unknown* — not "current", which is what an empty
    stale count would read as.
    """
    _plate(tmp_path, name="copied_in", merged=2)
    screen.rescan()
    assert _cell(screen, "copied_in", "Stage") == "mask"
    assert _cell(screen, "copied_in", "State") == "unknown — nothing recorded"
    assert "not in the registry" in _cell(screen, "copied_in", "Note")
    assert _cell(screen, "copied_in", "Last run") != "never"
    assert "not in the registry" in screen._status.text()


def test_a_raw_plate_nobody_has_run_anything_on_is_listed_too(screen, tmp_path):
    _plate(tmp_path, name="fresh", merged=0, db=False, raw=True)
    screen.rescan()
    assert _cell(screen, "fresh", "Stage") == "nothing run"
    assert _cell(screen, "fresh", "Last run") == "never"


def test_the_detail_pane_explains_why_nothing_can_be_checked(screen, tmp_path):
    plate = _plate(tmp_path, name="copied_in", merged=1)
    screen.rescan()
    text = screen.show_detail(plate)
    assert "Nothing here has a run record" in text
    assert "Stages" in text and "mask: complete" in text


# ---------------------------------------------------------------------------
# The recorded project
# ---------------------------------------------------------------------------

def test_a_recorded_project_shows_the_registry_verdict(screen, tmp_path):
    plate = _plate(tmp_path, name="recorded", merged=2)
    registry = artifacts.open_registry(plate)
    registry.register(project=plate, kind=ports.MERGED_ARRAYS, role="merged",
                      path=os.path.join(plate, "merged"), module="mask",
                      settings={"src": plate})
    screen.rescan()
    assert _cell(screen, "recorded", "State") == "current"
    note = _cell(screen, "recorded", "Note")
    assert "not in the registry" not in note and "out of date" not in note


def test_the_detail_pane_names_the_stale_reason_and_the_next_step(screen, tmp_path):
    import time

    plate = _plate(tmp_path, name="drifted", merged=2)
    registry = artifacts.open_registry(plate)
    merged = registry.register(
        project=plate, kind=ports.MERGED_ARRAYS, role="merged",
        path=os.path.join(plate, "merged"), module="mask",
        settings={"src": plate, "cell_diameter": 30})
    time.sleep(0.01)
    registry.register(
        project=plate, kind=ports.MEASUREMENTS_DB, role="db",
        path=os.path.join(plate, "measurements", "measurements.db"),
        module="measure", settings={"src": plate},
        inputs=[merged.artifact_id])
    time.sleep(0.01)
    registry.register(
        project=plate, kind=ports.MERGED_ARRAYS, role="merged",
        path=os.path.join(plate, "merged"), module="mask",
        settings={"src": plate, "cell_diameter": 45})

    screen.rescan()
    assert "stale" in _cell(screen, "drifted", "State")
    text = screen.show_detail(plate)
    assert "Out of date" in text
    assert "measurements-db from measure" in text
    assert "What could run next" in text


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_selecting_a_row_draws_that_projects_detail(screen, tmp_path):
    plate = _plate(tmp_path, name="picked", merged=1)
    screen.rescan()
    screen._table.selectRow(_row_for(screen, "picked"))
    assert screen.selected_root() == plate
    assert plate in screen._detail.toPlainText()


def test_the_root_travels_with_the_row_so_sorting_cannot_mis_select(screen, tmp_path):
    """A re-sorted table must still select the project the user clicked."""
    _plate(tmp_path, name="zeta", merged=1)
    _plate(tmp_path, name="alpha", merged=1)
    screen.rescan()
    screen._table.sortItems(0)
    for name in ("alpha", "zeta"):
        screen._table.selectRow(_row_for(screen, name))
        assert os.path.basename(screen.selected_root()) == name


def test_double_clicking_publishes_the_project_root(qtbot, screen, tmp_path):
    plate = _plate(tmp_path, name="opened", merged=1)
    screen.rescan()
    screen._table.selectRow(_row_for(screen, "opened"))
    with qtbot.waitSignal(screen.project_chosen, timeout=2000) as caught:
        screen._on_double_clicked(screen._table.item(0, 0))
    assert caught.args == [plate]


def test_asking_for_the_detail_of_something_not_listed_is_empty_not_an_error(screen):
    assert screen.show_detail("/nowhere/at/all") == ""
    assert screen.summary_for("/nowhere/at/all") is None


# ---------------------------------------------------------------------------
# The search folders
# ---------------------------------------------------------------------------

def test_adding_and_removing_a_folder_rescans(qtbot, tmp_path):
    _plate(tmp_path, name="one", merged=1)
    widget = ProjectBrowserScreen(threaded=False)
    qtbot.addWidget(widget)
    assert widget.roots() == ()
    assert widget.add_root(str(tmp_path)) is True
    assert widget._table.rowCount() == 1
    # A folder already searched is not added twice.
    assert widget.add_root(str(tmp_path)) is False
    widget._root_list.setCurrentRow(0)
    widget.forget_selected_root()
    assert widget.roots() == ()
    assert widget._table.rowCount() == 0


def test_the_depth_control_reaches_the_walk(qtbot, tmp_path):
    nested = tmp_path / "experiment"
    nested.mkdir()
    _plate(nested, name="deep", merged=1)
    widget = ProjectBrowserScreen(threaded=False, roots=(str(tmp_path),))
    qtbot.addWidget(widget)
    assert widget._table.rowCount() == 1
    widget._depth.setValue(1)
    widget.rescan()
    assert widget._table.rowCount() == 0


def test_removing_nothing_selected_does_not_raise(screen):
    screen._root_list.setCurrentRow(-1)
    screen.forget_selected_root()
    assert screen.roots()


# ---------------------------------------------------------------------------
# Off the GUI thread
# ---------------------------------------------------------------------------

def test_a_threaded_scan_delivers_and_retires_its_job(qtbot, tmp_path):
    """The failure ``job_runner``'s docstring warns about, asserted here.

    A completion handler wired as a closure never retires the job, so
    ``active_jobs()`` never returns to zero and the screen is permanently
    busy. Both halves are checked: the result arrived, and the bookkeeping
    came back.
    """
    _plate(tmp_path, name="threaded", merged=1)
    widget = ProjectBrowserScreen(threaded=True)
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.scanned, timeout=20000):
        widget.add_root(str(tmp_path))
    assert widget._table.rowCount() == 1
    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=20000)
    assert widget.is_busy() is False
    widget.close()


def test_a_failing_scan_says_so_without_a_modal(qtbot, screen, monkeypatch):
    """A stale mount must not open a dialog nobody is there to dismiss."""
    from spacr.qt.screens import project_browser as module

    monkeypatch.setattr(module, "_browse", lambda roots, depth: (_ for _ in ())
                        .throw(RuntimeError("the mount went away")))
    with qtbot.waitSignal(screen.failed, timeout=5000) as caught:
        screen.rescan()
    assert "the mount went away" in caught.args[0]
    assert "the mount went away" in screen._status.text()
    assert screen._rescan.isEnabled()


def test_closing_the_screen_abandons_work_in_flight(qtbot, tmp_path):
    _plate(tmp_path, name="closing", merged=1)
    widget = ProjectBrowserScreen(threaded=True, roots=(str(tmp_path),))
    qtbot.addWidget(widget)
    widget.close()
    assert widget.active_jobs() == 0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_the_factory_builds_a_screen():
    widget = make_project_browser_screen()
    try:
        assert isinstance(widget, ProjectBrowserScreen)
    finally:
        widget.close()
        widget.deleteLater()


def test_register_is_idempotent_and_fans_the_strings_out():
    from spacr.qt import app as app_mod

    already = any(row[0] == APP_KEY for row in app_mod.APPS)
    if not already:
        assert register() is True
    assert register() is False
    try:
        row = next(r for r in app_mod.APPS if r[0] == APP_KEY)
        assert row[3] == app_mod.SECTION_DATA
        meta = app_mod.APP_META[APP_KEY]
        assert meta["intro"] and meta["cli_note"]
        assert meta["api_module"] == "qt/screens/project_browser"
        assert len(meta["translations"]) == 9
        assert app_mod.APP_FACTORIES[APP_KEY] is make_project_browser_screen
    finally:
        if not already:
            app_mod.unregister_app(APP_KEY)


def test_one_row_in_self_registering_modules_turns_the_screen_on():
    from spacr.qt import SELF_REGISTERING_MODULES

    assert "spacr.qt.screens.project_browser" in SELF_REGISTERING_MODULES
    # Before `maturity`, which can only reassess apps already registered.
    assert (SELF_REGISTERING_MODULES.index("spacr.qt.screens.project_browser")
            < SELF_REGISTERING_MODULES.index("spacr.qt.maturity"))


def test_the_project_cap_is_declared_rather_than_unbounded():
    """A browser pointed at a home directory must return a table."""
    assert 0 < MAX_PROJECTS <= 1000
