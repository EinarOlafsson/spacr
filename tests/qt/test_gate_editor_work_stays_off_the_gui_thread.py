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
    gap = _gap_while(lambda: screen.save_graph_in_background(),
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


#: The xD projection: 1.45 s on the GUI thread before, 0.46 s now (150,000
#: rows by 23 columns, offscreen); what is left is drawing the projected
#: table. Lower it as that is split, never raise it.
PROJECTION_STALL_BUDGET_S = 1.0


def _wide(rows=60000, columns=12):
    frame = _table(rows)
    rng = np.random.default_rng(1)
    for k in range(columns):
        frame[f"m{k}"] = rng.normal(size=rows) + frame["area"] * (k % 3)
    return frame


def test_the_background_projection_is_the_one_the_screen_computes(
        qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: 0))
    results = []
    for background in (False, True):
        win = GateEditorScreen()
        win.resize(1200, 800)
        win.show()
        win.set_frame(_wide(), label="test")
        _pump(1.0)
        win._on_projection_requested(True, background=background)
        runner = getattr(win, "_projection_runner", None)
        _pump(60, lambda: runner is None or not runner.is_busy())
        _pump(0.5)
        results.append(win._frame[["PC1", "PC2", "PC3"]].copy())
        assert (win._x.currentText(), win._y.currentText()) == ("PC1", "PC2")
        win.close()
    pd.testing.assert_frame_equal(results[0], results[1])


def test_a_projection_for_a_replaced_table_is_dropped(qapp, monkeypatch):
    win = GateEditorScreen()
    win.resize(1200, 800)
    win.show()
    win.set_frame(_wide(), label="test")
    _pump(1.0)
    win._on_projection_requested(True, background=True)
    win.set_frame(_table(30000), label="other")
    runner = win._projection_runner
    _pump(60, lambda: not runner.is_busy())
    _pump(1.0)
    assert "PC1" not in win._frame.columns
    win.close()


def test_projecting_in_the_background_keeps_the_gui_turning(qapp):
    win = GateEditorScreen()
    win.resize(1200, 800)
    win.show()
    win.set_frame(_wide(), label="test")
    _pump(1.0)
    gap = _gap_while(
        lambda: win._on_projection_requested(True, background=True),
        lambda: win._frame is not None and "PC1" in win._frame.columns)
    assert gap < PROJECTION_STALL_BUDGET_S
    win.close()


def _many_sided(name="ring", sides=120):
    from spacr.qt.widgets.gate_spec import PolygonGate

    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    return PolygonGate(name=name, x_column="area", y_column="intensity",
                       vertices=tuple((6.0 + 3.0 * np.cos(a),
                                       6.0 + 3.0 * np.sin(a))
                                      for a in angles))


def _handles_one_norm_at_a_time(canvas, ax, gate):
    """The anchors as they were placed before the distances were vectorized."""
    from spacr.qt.widgets.gate_spec import EllipseGate, Handle

    flat = canvas._as_flat(gate)
    handles = list(flat.handles(canvas._view(ax)))
    points = canvas._gate_points(ax, flat)
    density = canvas._anchor_density
    if isinstance(flat, EllipseGate):
        positions = [i * len(points) / (8 * density) for i in range(8 * density)]
    else:
        positions = [edge + i / (2 * density) for edge in range(len(points))
                     for i in range(2 * density)]
    for position in positions:
        edge = int(position)
        fraction = position - edge
        point = np.asarray(points[edge]) + fraction * (
            np.asarray(points[(edge + 1) % len(points)]) - points[edge])
        if any(np.linalg.norm(point - (h.x, h.y)) <= 1e-10 for h in handles):
            continue
        handles.append(Handle(float(point[0]), float(point[1]),
                              f"outline:{edge}:{fraction:.17g}"))
    return tuple(handles)


def test_outline_anchors_are_the_ones_one_norm_per_handle_placed(screen):
    from spacr.qt.widgets.gate_spec import (EllipseGate, PolygonGate,
                                            RectGate)

    canvas = screen.gates.canvas
    ax = next(iter(canvas.panel_axes().values()))
    gates = [
        _many_sided(),
        PolygonGate(name="doubled", x_column="area", y_column="intensity",
                    vertices=((0.0, 0.0), (0.0, 0.0), (4.0, 0.0),
                              (4.0, 4.0), (4.0, 4.0), (0.0, 4.0))),
        RectGate(name="box", x_column="area", y_column="intensity",
                 x_low=1.0, x_high=5.0, y_low=-1.0, y_high=2.0),
        EllipseGate(name="oval", x_column="area", y_column="intensity",
                    x_centre=6.0, y_centre=6.0, x_radius=2.0, y_radius=1.0),
    ]
    for density in (1, 2, 5):
        canvas._anchor_density = density
        for gate in gates:
            assert canvas._handles_for(ax, gate) == \
                _handles_one_norm_at_a_time(canvas, ax, gate)


def test_one_render_filters_the_population_once(screen):
    canvas = screen.gates.canvas
    calls = []
    original = canvas.linked_visible

    def counted(frame):
        calls.append(len(frame))
        return original(frame)

    canvas.linked_visible = counted
    canvas.set_frame(canvas._frame.copy())
    assert len(calls) == 1


def test_the_displayed_count_from_the_worker_is_the_synchronous_one(screen):
    screen.set_axis_cutoffs("x", 2.0, 9.0)
    _pump(10, lambda: not screen._count_runner.is_busy())
    _pump(0.3)
    shown = screen._point_count.text()
    worker_rows = screen._displayed_population()
    screen._displayed_cache = None
    pd.testing.assert_frame_equal(screen._displayed_population(), worker_rows)
    assert f"Displayed {len(worker_rows):,} of " in shown
    assert 0 < len(worker_rows) < len(screen.gates.canvas.population())


def test_toggling_the_projection_with_a_many_sided_gate_keeps_the_gui_turning(
        qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: 0))
    win = GateEditorScreen()
    win.resize(1200, 800)
    win.show()
    win.set_frame(_wide(), label="test")
    _pump(1.0)
    win._y.setCurrentText("intensity")
    _pump(0.6)
    from spacr.qt.widgets.gate_spec import GateSet

    win.gates.set_gates(GateSet().add(_many_sided()))
    assert win.gates.canvas._handles_for(
        next(iter(win.gates.canvas.panel_axes().values())), _many_sided())
    _pump(0.5)
    gap = _gap_while(
        lambda: win._on_projection_toggled(True),
        lambda: win._frame is not None and "PC1" in win._frame.columns)
    _pump(1.0)
    assert gap < STALL_BUDGET_S
    win.close()


def test_toggling_the_projection_counts_no_column_on_the_gui_thread(
        qapp, monkeypatch):
    import threading

    from spacr.qt.widgets import data_filter_panel

    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: 0))
    win = GateEditorScreen()
    win.resize(1200, 800)
    win.show()
    win.set_frame(_wide(), label="test")
    _pump(1.0)
    win._y.setCurrentText("intensity")
    _pump(0.6)
    uncached = data_filter_panel._classify_columns_uncached
    on_gui = []

    def counted(frame):
        if threading.current_thread() is threading.main_thread():
            on_gui.append(frame.shape)
        return uncached(frame)

    monkeypatch.setattr(data_filter_panel, "_classify_columns_uncached",
                        counted)
    gap = _gap_while(
        lambda: win._on_projection_toggled(True),
        lambda: win._frame is not None and "PC1" in win._frame.columns)
    _pump(0.5)
    assert on_gui == []
    assert gap < STALL_BUDGET_S
    shown = win.formulas.computed_frame()
    assert data_filter_panel.classify_columns(shown) == uncached(shown)
    displayed = win.gates.canvas._frame
    assert data_filter_panel.classify_columns(displayed) == uncached(shown)
    win.close()
