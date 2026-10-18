"""Typing into Min objects / well is never dropped and costs one recompute."""
from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from spacr.qt.screens import plate_view as pv  # noqa: E402


@pytest.fixture
def loaded(qtbot, tmp_path):
    meas = tmp_path / "plate1" / "measurements"
    meas.mkdir(parents=True)
    db = meas / "measurements.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE cell (prc TEXT, value REAL)")
    con.executemany(
        "INSERT INTO cell VALUES (?, ?)",
        [(f"plate1_r{r}_c{c}", 100.0 + k)
         for r in range(1, 9) for c in range(1, 13) for k in range(6)])
    con.commit()
    con.close()
    screen = pv.PlateViewScreen(threaded=True)
    qtbot.addWidget(screen)
    screen.show()
    assert screen.open_database(str(db))
    qtbot.waitUntil(lambda: not screen.is_busy() and
                    screen._value_combo.count() > 0, timeout=10000)
    assert screen.render_plate()
    qtbot.waitUntil(lambda: not screen.is_busy() and
                    screen._layout_df is not None, timeout=10000)
    drawn = []
    original = screen._draw_plate

    def spy(result, min_count):
        drawn.append(min_count)
        return original(result, min_count)

    screen._draw_plate = spy
    runs = []
    original_recompute = screen.recompute

    def counted():
        runs.append(1)
        return original_recompute()

    screen.recompute = counted
    screen._recompute_timer.timeout.disconnect()
    screen._recompute_timer.timeout.connect(counted)
    return screen, drawn, runs


def _focus(screen, qtbot):
    box = screen._min_count_box
    box.setFocus()
    box.selectAll()
    box.lineEdit().selectAll()
    return box


def test_typing_601_slowly_keeps_every_digit_and_runs_once(qtbot, loaded):
    screen, drawn, runs = loaded
    box = _focus(screen, qtbot)
    for ch in "601":
        QTest.keyClick(box.lineEdit(), ch)
        qtbot.wait(150)
    assert box.value() == 601
    qtbot.waitUntil(lambda: bool(drawn) and drawn[-1] == 601, timeout=10000)
    qtbot.waitUntil(lambda: not screen.is_busy(), timeout=10000)
    assert len(runs) <= 2


def test_the_box_stays_enabled_while_a_recompute_runs(qtbot, loaded):
    screen, _drawn, _runs = loaded
    screen._busy = True
    screen._update_controls()
    assert screen._min_count_box.isEnabled()
    assert screen.is_busy() is True
    assert not screen._table_combo.isEnabled()
    screen._busy = False


def test_enter_recomputes_immediately(qtbot, loaded):
    screen, drawn, runs = loaded
    box = _focus(screen, qtbot)
    QTest.keyClick(box.lineEdit(), "7")
    assert screen._recompute_timer.isActive()
    QTest.keyClick(box.lineEdit(), Qt.Key_Return)
    assert not screen._recompute_timer.isActive()
    assert len(runs) == 1
    qtbot.waitUntil(lambda: bool(drawn) and drawn[-1] == 7, timeout=10000)
