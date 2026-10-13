"""Outliers-screen paths the well-formed plate never takes.

A table with no well identifiers in it, and the table picker changing while
there is nothing loaded behind it.

``threaded=False`` throughout, so a scan has finished by the time
``set_frame`` returns. Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.outliers import OutliersScreen              # noqa: E402

pytestmark = pytest.mark.qt


def wellless_measurements(seed: int = 7, n: int = 120) -> pd.DataFrame:
    """Objects with no plate/row/column columns — a bare feature table."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "object_label": np.arange(n),
        "cell_area": rng.lognormal(0.0, 0.2, n),
        "cell_perimeter": rng.lognormal(0.0, 0.2, n),
    })


@pytest.fixture()
def screen(qtbot):
    widget = OutliersScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# A table with no wells in it
# ---------------------------------------------------------------------------

def test_a_table_with_no_well_columns_scans_objects_and_reports_no_wells(
        screen):
    """Without well keys the across-well pass produces nothing to show."""
    screen.per_well.setChecked(False)
    screen.set_frame(wellless_measurements())

    result = screen.result
    assert result is not None
    assert not result.has_wells
    assert screen.object_table.rowCount() > 0
    assert screen.well_table.rowCount() == 0
    assert screen.tabs.tabText(1) == "Wells", (
        "with no wells the tab carries no count")


def test_exporting_a_wellless_scan_writes_no_wells_file(
        screen, tmp_path, monkeypatch):
    """The wells file is skipped, and the status line does not claim one."""
    screen.per_well.setChecked(False)
    screen.set_frame(wellless_measurements())
    target = tmp_path / "scan.csv"
    monkeypatch.setattr(
        "spacr.qt.screens.outliers.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "CSV (*.csv)")))

    screen.export_csv()

    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ["scan_flagged.csv", "scan_objects.csv",
                       "scan_report.txt"]
    assert "_wells" not in screen._source.text()
    assert "wrote scan_objects / _flagged" in screen._source.text()


# ---------------------------------------------------------------------------
# The table picker with nothing behind it
# ---------------------------------------------------------------------------

def test_the_table_picker_changing_with_no_file_loaded_reads_nothing(screen):
    """A picker entry appearing before any path is set starts no read."""
    assert screen.frame is None

    # Adding the first entry moves the combo off index -1, which is what
    # emits currentTextChanged.
    screen._table_picker.addItem("object")

    assert screen._table_picker.currentText() == "object"
    assert screen.frame is None
    assert not screen.is_busy()
    assert screen._source.text() == "no table loaded"
