"""Outlier scans land only on a live table, and only the latest one lands.

Lesson 66 crashed (SIGSEGV) in ``_fill_table`` reached from
``JobRunner._on_settled -> _on_scanned -> _fill_object_table``. A settled
scan could arrive after the screen had closed, an earlier scan could arrive
after a later one, and a fill could be re-entered from a signal the fill
itself emitted. Each case is driven here on real threads or real signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
shiboken6 = pytest.importorskip("shiboken6")

from spacr.qt.screens import outliers as O  # noqa: E402
from spacr.qt.screens.outliers import OutliersScreen  # noqa: E402

pytestmark = pytest.mark.qt


def _plate(seed, per_well=25, extra=0):
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(8):
        factor = 1.4 if well == 3 else 1.0
        values = [factor * rng.lognormal(0.0, 0.2, per_well)
                  for _ in range(2 + extra)]
        for i in range(per_well):
            rows.append(("p1", "r1", f"c{well + 1}", "f1", i,
                         *[v[i] for v in values]))
    features = ["cell_area", "cell_perimeter"] + [
        f"cell_extra_{k}" for k in range(extra)]
    return pd.DataFrame(rows, columns=[
        "plateID", "rowID", "columnID", "fieldID", "object_label", *features])


def _wait_idle(qtbot, screen):
    qtbot.waitUntil(lambda: not screen.is_busy(), timeout=30000)
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=30000)


def test_back_to_back_scans_show_only_the_last(qtbot):
    screen = OutliersScreen(threaded=True)
    qtbot.addWidget(screen)
    first, second = _plate(1, per_well=25), _plate(2, per_well=30, extra=3)
    filled = []
    real = O._fill_table
    monkey = pytest.MonkeyPatch()
    monkey.setattr(O, "_fill_table",
                   lambda table, frame: (filled.append(len(frame)),
                                         real(table, frame)))
    try:
        screen.set_frame(first)
        screen.set_frame(second)
        _wait_idle(qtbot, screen)
        qtbot.wait(50)
    finally:
        monkey.undo()
    assert screen.object_table.rowCount() == min(len(second), O.MAX_TABLE_ROWS)
    assert len(filled) == 2, "the superseded scan was shown as well"
    assert screen.result is not None


def test_a_scan_that_settles_after_the_screen_closed_is_dropped(qtbot):
    screen = OutliersScreen(threaded=True)
    qtbot.addWidget(screen)
    filled = []
    real = O._fill_table
    monkey = pytest.MonkeyPatch()
    monkey.setattr(O, "_fill_table",
                   lambda table, frame: (filled.append(table),
                                         real(table, frame)))
    try:
        screen.set_frame(_plate(3))
        screen.close()
        qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=30000)
        qtbot.wait(50)
    finally:
        monkey.undo()
    assert filled == []


def test_a_result_for_a_deleted_screen_touches_nothing(qtbot):
    screen = OutliersScreen(threaded=False)
    screen.set_frame(_plate(4))
    result = screen.result
    token = screen._scan_token
    table = screen.object_table
    shiboken6.delete(screen)
    assert not shiboken6.isValid(table)
    OutliersScreen._deliver_scan(screen, token, result)
    O._fill_table(table, pd.DataFrame({"a": [1.0]}))


def test_a_refill_during_a_refill_waits_its_turn(qtbot):
    screen = OutliersScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.set_frame(_plate(5))
    later_frame = _plate(6, per_well=12, extra=2)
    later = O.detect_outliers(later_frame, screen.spec())
    nested = []

    def refill_inside(*_args):
        if not nested:
            nested.append(True)
            screen._on_scanned(later)

    model = screen.object_table.model()
    model.dataChanged.connect(refill_inside)
    try:
        screen.set_frame(_plate(7, per_well=40), scan=False)
        screen.scan()
        assert nested, "the nested refill was never attempted"
        screen._frame = later_frame
        qtbot.waitUntil(lambda: screen.result is later, timeout=5000)
        qtbot.waitUntil(lambda: screen.object_table.rowCount()
                        == min(len(later_frame), O.MAX_TABLE_ROWS),
                        timeout=5000)
    finally:
        model.dataChanged.disconnect(refill_inside)
    assert not screen._filling
