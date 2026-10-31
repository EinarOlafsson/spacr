"""The model zoo picker opened from a settings row draws its table in steps.

A row's version box costs a stylesheet polish each, so drawing a long list
while the dialog was built held the window for most of a second (the freeze
when a Model cell in Make Masks opens the picker). The picker a settings row
opens now draws a few rows per turn of the event loop and ends up identical
to the one drawn at once.
"""
from __future__ import annotations

import dataclasses

import pytest

from spacr import model_zoo
from spacr.qt.widgets.model_zoo_picker import ModelZooPicker


def _entries(count):
    return [model_zoo.ModelEntry(
        key=f"m{i}", name=f"model_{i:02d}.CP_model", kind="cellpose",
        source="remote", path="", uri=f"https://example.invalid/{i}",
        version="1", sha256="a" * 64, size_bytes=1, trained_on="synthetic")
        for i in range(count)]


def _cells(dialog):
    table = dialog.table
    return [[table.item(r, c).text() if table.item(r, c) else None
             for c in range(4)] + [table.cellWidget(r, 4).currentText()]
            for r in range(table.rowCount())]


@pytest.fixture
def pair(qapp, qtbot, monkeypatch):
    entries = _entries(40)
    monkeypatch.setattr(model_zoo, "catalogue", lambda *a, **k: entries)
    monkeypatch.setattr(model_zoo, "shared_catalogue_is_stale", lambda: False)
    once = ModelZooPicker(kinds=("cellpose",))
    steps = ModelZooPicker(kinds=("cellpose",), stepwise=True)
    qtbot.addWidget(once)
    qtbot.addWidget(steps)
    yield once, steps
    once.close()
    steps.close()


def test_a_stepwise_picker_has_drawn_no_row_when_its_constructor_returns(
        qapp, qtbot, monkeypatch):
    entries = _entries(40)
    monkeypatch.setattr(model_zoo, "catalogue", lambda *a, **k: entries)
    monkeypatch.setattr(model_zoo, "shared_catalogue_is_stale", lambda: False)
    steps = ModelZooPicker(kinds=("cellpose",), stepwise=True)
    qtbot.addWidget(steps)
    assert steps._rebuilding
    assert all(steps.table.cellWidget(r, 4) is None
               for r in range(steps.table.rowCount()))
    qtbot.waitUntil(lambda: not steps._rebuilding, timeout=20000)
    assert all(steps.table.cellWidget(r, 4) is not None
               for r in range(steps.table.rowCount()))


def test_it_ends_with_the_table_the_one_pass_picker_draws(pair, qtbot):
    once, steps = pair
    qtbot.waitUntil(lambda: not steps._rebuilding, timeout=20000)
    assert steps.table.rowCount() == once.table.rowCount()
    assert _cells(steps) == _cells(once)


def test_every_turn_of_the_event_loop_draws_only_a_slice(pair, qtbot,
                                                         monkeypatch):
    once, steps = pair
    qtbot.waitUntil(lambda: not steps._rebuilding, timeout=20000)
    monkeypatch.setattr(ModelZooPicker, "_STEP_BUDGET_S", 0.0)
    drawn = []
    original = ModelZooPicker._draw_row
    monkeypatch.setattr(ModelZooPicker, "_draw_row",
                        lambda self, row: (drawn.append(row),
                                           original(self, row))[1])
    steps._drawn_signature = None
    steps.refresh()
    from PySide6.QtCore import QCoreApplication
    QCoreApplication.sendPostedEvents(None, 0)
    qtbot.wait(5)
    assert 0 < len(drawn) < steps.table.rowCount()
    qtbot.waitUntil(lambda: not steps._rebuilding, timeout=20000)
    assert drawn == list(range(len(steps._groups)))


def test_an_unchanged_catalogue_is_not_drawn_twice(pair, qtbot, monkeypatch):
    once, steps = pair
    qtbot.waitUntil(lambda: not steps._rebuilding, timeout=20000)
    generation = steps._draw_generation
    steps.refresh()
    assert steps._draw_generation == generation
    assert not steps._rebuilding


def test_a_newer_draw_cancels_the_one_in_progress(pair, qtbot, monkeypatch):
    once, steps = pair
    qtbot.waitUntil(lambda: not steps._rebuilding, timeout=20000)
    monkeypatch.setattr(ModelZooPicker, "_STEP_BUDGET_S", 0.0)
    steps._drawn_signature = None
    steps.refresh()
    steps._rebuild(steps._entries)
    qtbot.wait(100)
    assert not steps._rebuilding
    assert _cells(steps) == _cells(once)


#: Opening the picker from Make Masks' Model cell, measured offscreen on the
#: workstation: the longest GUI stall was 0.88-1.13 s when the table was drawn
#: while the dialog was built, and is 0.42-0.74 s now. What is left is Qt
#: polishing the dialog and its widgets under the window sheet when it first
#: shows. The budget may be lowered as that is split, never raised.
OPEN_STALL_BUDGET_S = 1.0


def test_opening_the_picker_from_a_model_cell_keeps_the_gui_turning(
        qapp, qtbot, monkeypatch, tmp_path):
    import time

    from PySide6.QtCore import QTimer
    from spacr.qt import app as app_module
    from spacr.qt.widgets.object_settings_grid import ObjectSettingsGrid

    monkeypatch.setenv("SPACR_PREWARM", "0")
    window = app_module.MainWindow(initial_app="__home__")
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    window._on_nav_selected("mask")
    qtbot.wait(1500)
    grid = window._screens["mask"].findChildren(ObjectSettingsGrid)[0]
    gaps = []
    last = [time.perf_counter()]

    def tick():
        now = time.perf_counter()
        gaps.append(now - last[0])
        last[0] = now

    ticker = QTimer()
    ticker.setInterval(5)
    ticker.timeout.connect(tick)

    def show_and_wait(dialog):
        dialog.show()
        deadline = time.perf_counter() + 30
        while dialog._rebuilding and time.perf_counter() < deadline:
            qtbot.wait(10)
        dialog.close()
        return 0

    monkeypatch.setattr(ModelZooPicker, "exec", show_and_wait)
    ticker.start()
    last[0] = time.perf_counter()
    try:
        grid.choose_model_for("cell")
    finally:
        ticker.stop()
    assert max(gaps) < OPEN_STALL_BUDGET_S
