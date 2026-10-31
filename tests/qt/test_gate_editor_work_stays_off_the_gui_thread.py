"""Save graph and the cluster walk run off the GUI thread in the Gate Editor.

Lesson 64's retake froze the window for seconds on both: Save graph imported
the print renderer and drew the raster inside the click, and the cluster walk
fitted DBSCAN once per radius there. Both now run on a worker and finish on
the GUI thread; the gates and files are the ones the synchronous path makes.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from spacr.qt.screens.gate_editor import GateEditorScreen
from spacr.qt.widgets.gate_settings import GateEditorSettings

#: Longest GUI-thread stall, in seconds, while the action runs, measured
#: offscreen on the workstation with the 60,000-row table below: Save graph
#: 0.13 s (2.3 s before), cluster walk 0.16 s (8.8 s before). It may be
#: lowered as the finishing step is split, never raised.
STALL_BUDGET_S = 0.6


def _pump(seconds, until=None):
    """Run the event loop; QTest.qWait would starve a worker's imports."""
    app = QApplication.instance()
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.004)
    return until() if until is not None else True


def _table(rows=60000):
    rng = np.random.default_rng(0)
    third = rows // 3
    area = np.concatenate([rng.normal(0, 1, third), rng.normal(6, 1, third),
                           rng.normal(12, 1, rows - 2 * third)])
    intensity = np.concatenate([rng.normal(0, 1, third),
                                rng.normal(6, 1, third),
                                rng.normal(0, 1, rows - 2 * third)])
    return pd.DataFrame({"area": area, "intensity": intensity})


@pytest.fixture
def screen(qapp, monkeypatch):
    boxes = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: boxes.append(a[1])))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: boxes.append(a[1])))
    win = GateEditorScreen()
    win.resize(1200, 800)
    win.show()
    win.set_frame(_table(), label="test")
    _pump(1.0)
    win._y.setCurrentText("intensity")
    _pump(0.6)
    win._boxes = boxes
    yield win
    win.close()


def _gap_while(action, until, limit=60):
    gaps = []
    last = [time.perf_counter()]

    def tick():
        now = time.perf_counter()
        gaps.append(now - last[0])
        last[0] = now

    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(tick)
    timer.start()
    last[0] = time.perf_counter()
    try:
        action()
        _pump(limit, until)
    finally:
        timer.stop()
    return max(gaps)


def _gate_names(win):
    return sorted(win.gates.gates.names)


def test_a_cluster_walk_in_the_background_makes_the_gates_the_walk_makes(
        screen):
    screen.gates.apply_settings(GateEditorSettings(
        cluster_eps=0.5, cluster_min_samples=40, cluster_walk=True,
        cluster_walk_steps=4))
    screen.gates.run_cluster(ask=False)
    expected = _gate_names(screen)
    assert len(expected) >= 2
    fresh = type(screen.gates.gates)()
    screen.gates._gates = fresh
    screen.gates.canvas.set_gates(fresh)
    gap = _gap_while(lambda: screen.gates.run_cluster_in_background(ask=False),
                     lambda: len(_gate_names(screen)) == len(expected))
    assert _gate_names(screen) == expected
    assert gap < STALL_BUDGET_S


def test_a_newer_cluster_request_drops_the_older_answer(screen):
    screen.gates.apply_settings(GateEditorSettings(
        cluster_eps=0.5, cluster_min_samples=40, cluster_walk=False))
    screen.gates.run_cluster_in_background(ask=False)
    screen.gates.run_cluster_in_background(ask=False)
    runner = screen.gates._cluster_runner
    _pump(30, lambda: len(_gate_names(screen)) > 0 and not runner.is_busy())
    first = _gate_names(screen)
    assert first
    _pump(1.0)
    assert _gate_names(screen) == first


def test_an_answer_for_a_table_that_was_replaced_is_dropped(screen):
    screen.gates.apply_settings(GateEditorSettings(
        cluster_eps=0.5, cluster_min_samples=40, cluster_walk=False))
    screen.gates.run_cluster_in_background(ask=False)
    screen.set_frame(_table(30000), label="other")
    runner = screen.gates._cluster_runner
    _pump(30, lambda: not runner.is_busy())
    _pump(1.0)
    assert _gate_names(screen) == []


def test_save_graph_from_the_dialog_writes_the_file_off_the_gui_thread(
        screen, monkeypatch, tmp_path):
    target = tmp_path / "graph.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    gap = _gap_while(lambda: screen.save_graph(),
                     lambda: target.exists() and target.stat().st_size > 1000)
    assert target.stat().st_size > 1000
    assert gap < STALL_BUDGET_S


def test_save_graph_with_a_path_still_writes_before_it_returns(
        screen, tmp_path):
    target = tmp_path / "graph.png"
    assert screen.save_graph(str(target)) == str(target)
    assert target.stat().st_size > 1000


def test_the_displayed_population_is_computed_once_per_filter(screen):
    from spacr.selection import DataFilter, RangeFilter

    canvas = screen.gates.canvas
    first = canvas.population()
    assert canvas.population() is first
    canvas.link.set_filter(DataFilter([RangeFilter("area", -3.0, 3.0)]))
    narrowed = canvas.population()
    assert narrowed is not first
    assert len(narrowed) < len(first)
    assert canvas.population() is narrowed
    displayed = screen._displayed_population()
    assert screen._displayed_population() is displayed
