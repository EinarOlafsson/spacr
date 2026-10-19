"""The batch queue table survives rows and cells it did not draw itself."""
from __future__ import annotations

import pytest

from spacr import batch as bt
from spacr.qt.screens.batch import COLUMNS, BatchScreen

pytestmark = pytest.mark.qt


class _Runner:
    def __call__(self, job, settings_path, log_path):
        return 0


@pytest.fixture()
def screen(qtbot):
    widget = BatchScreen(threaded=False, runner=_Runner())
    qtbot.addWidget(widget)
    return widget


@pytest.fixture()
def settings(tmp_path):
    plate = tmp_path / "plate1"
    plate.mkdir()
    (plate / "plate1_A01_T0001F001L01A01Z01C01.tif").write_bytes(b"")
    path = tmp_path / "mask.csv"
    path.write_text(f"Key,Value\nsrc,{plate}\ncell_channel,0\n", encoding="utf-8")
    return str(path)


def test_empty_rows_below_the_queue_are_dropped_without_error(screen, settings):
    screen.add_job("mask", settings)
    screen.add_job("mask", settings)
    screen._table.setRowCount(5)
    assert screen._table.item(3, 0) is None
    screen._refresh_table()
    assert screen._table.rowCount() == 2
    assert [screen.row_values(r)[0] for r in range(2)] == ["1", "2"]


def test_the_tick_redraws_a_time_cell_the_table_lost(screen, settings):
    screen.add_job("mask", settings)
    job = screen.queue().find("mask-1")
    job.status = bt.STATUS_RUNNING
    job.started = bt._now_iso()
    screen._refresh_table()
    column = COLUMNS.index("Time")
    assert screen._table.takeItem(0, column) is not None
    assert screen._table.item(0, column) is None
    screen._refresh_running_row()
    cell = screen._table.item(0, column)
    assert cell is not None and cell.text() == bt.fmt_duration(job.elapsed_s)
