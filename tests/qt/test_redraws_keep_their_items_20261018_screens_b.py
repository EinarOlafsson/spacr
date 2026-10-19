"""Screens redraws keep their items (stale-wrapper sweep, group screens_b).

Database browser, Distributed Jobs, Dose-response, Embeddings, Experiment
design and Foreign used to delete every Python-made item in a redraw a click
can trigger; with accessibility on that segfaulted the tutorial host. Each
redraw now rewrites the items it has and parks the surplus.
"""
from __future__ import annotations

import sqlite3
from collections import deque

import numpy as np
import pandas as pd
import pytest
import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.qt


@pytest.fixture
def accessible():
    was = QAccessible.isActive()
    QAccessible.setActive(True)
    try:
        yield
    finally:
        QAccessible.setActive(was)


def _walk(view):
    interface = QAccessible.queryAccessibleInterface(view)
    if interface is None:
        return
    for index in range(min(interface.childCount(), 60)):
        child = interface.child(index)
        if child is not None:
            child.text(QAccessible.Name)


def _list_items(listing):
    return [listing.item(r) for r in range(listing.count())]


def _cells(table):
    return [table.item(r, c) for r in range(table.rowCount())
            for c in range(table.columnCount()) if table.item(r, c)]


def _make_db(path, names):
    con = sqlite3.connect(path)
    for name in names:
        con.execute(f'CREATE TABLE "{name}" (a INTEGER)')
        con.execute(f'INSERT INTO "{name}" VALUES (1)')
    con.commit()
    con.close()
    return str(path)


def test_db_browser_table_list(qtbot, qt_theme_applied, accessible, tmp_path):
    from spacr.qt.screens.db_browser import DbBrowserScreen
    screen = DbBrowserScreen(threaded=False)
    qtbot.addWidget(screen)
    seen = []
    plans = [["a", "b", "c"], ["x"], ["p", "q", "r", "s"], []]
    for step, names in enumerate(plans * 2):
        path = _make_db(tmp_path / f"d{step}.db", names)
        screen.set_database(path)
        _walk(screen._table_list)
        QApplication.processEvents()
        seen.extend(_list_items(screen._table_list))
        assert all(shiboken6.isValid(i) for i in seen)
        assert [i.text() for i in _list_items(screen._table_list)] \
            == screen.tables() == sorted(names) or \
            [i.text() for i in _list_items(screen._table_list)] == names
        if names:
            assert screen._table_list.currentRow() == 0
            screen.select_table(names[-1]) if hasattr(screen, "select_table") \
                else None
        else:
            assert screen._table_list.currentItem() is None
    assert screen._table_list.findItems("x", Qt.MatchExactly) is not None


def test_distributed_jobs_table(qtbot, qt_theme_applied, accessible, tmp_path):
    from spacr.qt.screens.distributed_jobs import DistributedJobsScreen
    from spacr.remote_execution import (
        CommandResult, ExecutionProfile, JobStore, ProfileStore,
        RemoteJobManager,
    )
    results = deque(CommandResult(0, f"cloud-{i}\n") for i in range(4))

    def runner(argv, **kwargs):
        return results.popleft()

    profiles = ProfileStore(tmp_path / "profiles.json")
    profiles.save(ExecutionProfile(
        "cloud", "command", submit_command="cloud-submit {module} {settings}",
        status_command="cloud-status {external_id}",
        cancel_command="cloud-cancel {external_id}",
        log_command="cloud-logs {external_id}"))
    manager = RemoteJobManager(profiles, JobStore(tmp_path / "jobs.json"),
                               runner)
    screen = DistributedJobsScreen(manager=manager, threaded=False,
                                   auto_poll=False)
    qtbot.addWidget(screen)
    seen = []
    for n in range(3):
        screen.configure_submission("mask", {"src": "/d", "random_state": n})
        qtbot.mouseClick(screen._submit, Qt.LeftButton)
        _walk(screen._table)
        seen.extend(_cells(screen._table))
        assert screen._table.rowCount() == n + 1
        assert all(shiboken6.isValid(i) for i in seen)
    jobs = list(screen._jobs)
    for keep in (1, 3, 2, 0):
        screen._render_jobs(jobs[:keep])
        _walk(screen._table)
        QApplication.processEvents()
        seen.extend(_cells(screen._table))
        assert all(shiboken6.isValid(i) for i in seen)
        assert screen._table.rowCount() == keep
        for r in range(keep):
            assert screen._table.item(r, 0).data(Qt.UserRole) \
                == jobs[r].job_id
            assert not screen._table.item(r, 0).flags() & Qt.ItemIsEditable
        if keep:
            assert screen._selected_job() is not None


def test_dose_response_table(qtbot, qt_theme_applied, accessible):
    from spacr.qt.screens.dose_response import DoseResponseScreen
    doses = 27.0 / 3.0 ** np.arange(10)
    rng = np.random.default_rng(0)

    def frame(genes):
        parts = []
        for gene in genes:
            d = np.repeat(doses, 3)
            parts.append(pd.DataFrame({
                "gene": gene, "conc_uM": d,
                "signal": 100 / (1 + (1.0 / d) ** -1.0)
                + rng.normal(0, 1, d.size)}))
        return pd.concat(parts, ignore_index=True)

    screen = DoseResponseScreen(threaded=False)
    qtbot.addWidget(screen)
    seen = []
    for genes in (["a", "b", "c"], ["a"], ["a", "b"], ["a", "b", "c", "d"]):
        screen.set_frame(frame(genes), label="x")
        assert screen.table.rowCount() == 0
        screen.concentration_picker.setCurrentText("conc_uM")
        screen.response_picker.setCurrentText("signal")
        screen.group_picker.setCurrentText("gene")
        screen.fit()
        _walk(screen.table)
        QApplication.processEvents()
        seen.extend(_cells(screen.table))
        assert all(shiboken6.isValid(i) for i in seen)
        assert screen.table.rowCount() == len(genes)
        assert [screen.table.item(r, 0).data(Qt.UserRole)
                for r in range(len(genes))] == list(range(len(genes)))
        assert len(screen.table.selectedItems()) > 0


def test_embeddings_preview(qtbot, accessible):
    import spacr.qt.screens.embeddings as em
    screen = em.EmbeddingsScreen(threaded=False)
    qtbot.addWidget(screen)
    seen = []
    for rows, cols in ((5, 4), (2, 2), (7, 6), (0, 3)):
        frame = pd.DataFrame(
            np.arange(rows * cols, dtype=float).reshape(rows, cols),
            columns=[f"d{i}" for i in range(cols)])
        screen._fill_preview(frame)
        _walk(screen._table)
        QApplication.processEvents()
        seen.extend(_cells(screen._table))
        assert all(shiboken6.isValid(i) for i in seen)
        assert screen._table.rowCount() == rows
        assert screen._table.columnCount() == min(cols, em.PREVIEW_DIMENSIONS)
        for r in range(rows):
            assert screen._table.item(r, 0).text() == f"{float(r * cols):.4f}"
    screen._fill_preview(pd.DataFrame(
        {"d0": [3.0, 1.0, 2.0]}))
    screen._table.sortItems(0, Qt.AscendingOrder)
    assert [screen._table.item(r, 0).text() for r in range(3)] \
        == ["1.0000", "2.0000", "3.0000"]


def test_experiment_design_conditions(qtbot, qt_theme_applied, accessible):
    from spacr.qt.screens.experiment_design import (
        Condition, ExperimentDesignScreen, ROLE_TREATMENT,
    )
    screen = ExperimentDesignScreen()
    qtbot.addWidget(screen)
    table = screen._table
    seen = []
    for n in (3, 1, 4, 2):
        screen._set_conditions(
            [Condition(f"c{i}", i + 1, ROLE_TREATMENT) for i in range(n)])
        _walk(table)
        seen.extend(_cells(table))
        assert table.rowCount() == n
        assert [table.item(r, 0).text() for r in range(n)] \
            == [f"c{i}" for i in range(n)]
        assert [table.item(r, 1).text() for r in range(n)] \
            == [str(i + 1) for i in range(n)]
        assert all(shiboken6.isValid(i) for i in seen)
    screen._add_row()
    seen.extend(_cells(table))
    table.selectRow(0)
    screen._remove_row()
    QApplication.processEvents()
    assert table.rowCount() == 2
    assert all(shiboken6.isValid(i) for i in seen)
    screen._add_row()
    screen._set_conditions([Condition("z", 5, ROLE_TREATMENT)])
    assert all(shiboken6.isValid(i) for i in seen)
    assert table.item(0, 0).text() == "z"


def test_foreign_mask_list(qtbot, qt_theme_applied, accessible, tmp_path):
    from spacr.qt.screens.foreign import ForeignScreen, OBJECT_CHOICES
    screen = ForeignScreen(threaded=False)
    qtbot.addWidget(screen)
    seen = []
    names = list(OBJECT_CHOICES)[:3]
    for step in range(3):
        for name in names:
            folder = tmp_path / f"{name}{step}"
            folder.mkdir()
            screen.add_mask_folder(name, str(folder))
            _walk(screen._mask_list)
            seen.extend(_list_items(screen._mask_list))
            assert all(shiboken6.isValid(i) for i in seen)
        listing = screen._mask_list
        assert [i.data(Qt.UserRole) for i in _list_items(listing)] \
            == [n for n in OBJECT_CHOICES if n in screen.mask_folders()]
        screen.remove_mask_folder(names[0])
        assert listing.currentItem() is None
        assert listing.count() == len(names) - 1
        assert all(shiboken6.isValid(i) for i in seen)
        screen.remove_mask_folder(names[1])
        screen.remove_mask_folder(names[2])
        assert listing.count() == 0
