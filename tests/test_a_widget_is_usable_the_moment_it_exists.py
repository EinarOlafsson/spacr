"""Every control works on a widget that has been handed no data yet.

Reported 2026-08-16, as a crash at startup:

    AttributeError: 'ResultsTable' object has no attribute '_significance'
      File "spacr/qt/widgets/fast_plots.py", line 982, in _apply_filter
        if hits_only and self._significance:

`_significance` was created in `set_frame` and nowhere else, while the filter
controls are wired up in `__init__`. So every path that touched a control
before the first frame arrived took the application down -- and `configure()`
is such a path, because it can uncheck "significant only", which emits
`toggled`, which calls `_apply_filter`.

THE CLASS OF BUG, which is why this file is not one test: a constructor that
leaves state to be created later turns into a trap. The widget exists, it is
on screen, its controls are connected and clickable, and it is not actually
usable until some other method has been called. Nothing says so.

So: build every one of these widgets, hand them NOTHING, and drive every
control on them.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


# --------------------------------------------------------------------------- #
#  The exact crash
# --------------------------------------------------------------------------- #

def test_filtering_an_empty_results_table_does_not_raise(qtbot):
    from spacr.qt.widgets.fast_plots import ResultsTable

    table = ResultsTable()
    qtbot.addWidget(table)

    table._apply_filter()               # this is the reported traceback
    assert "0 of 0 rows" in table._count.text()


def test_configure_before_any_frame_does_not_raise(qtbot):
    """The path that actually fired it: configure() unchecks the box, which
    emits toggled, which filters."""
    from spacr.qt.widgets.fast_plots import ResultsTable

    table = ResultsTable()
    qtbot.addWidget(table)
    table._only_hits.setChecked(True)

    table.configure(placeholder="anything", significance_filter=False)

    assert table._only_hits.isChecked() is False


def test_every_control_on_a_bare_table(qtbot):
    from spacr.qt.widgets.fast_plots import ResultsTable

    table = ResultsTable()
    qtbot.addWidget(table)

    table._filter.setText("tgme49")
    table._only_hits.setChecked(True)
    table._only_hits.setChecked(False)
    table._filter.setText("")
    table.copy_visible()
    assert table.select_key("nothing") is False
    assert table.key_for_row(0) is None
    assert table.select_frame_row(0) is False


def test_the_significance_attribute_exists_from_construction(qtbot):
    """Named so that moving it back into set_frame fails here."""
    from spacr.qt.widgets.fast_plots import ResultsTable

    table = ResultsTable()
    qtbot.addWidget(table)
    assert hasattr(table, "_significance")
    assert table._significance is None


def test_a_frame_missing_its_significance_column_still_filters(qtbot):
    """The column is remembered from the last frame. A NEW frame without it
    must not index a column that is not there."""
    import pandas as pd

    from spacr.qt.widgets.fast_plots import ResultsTable

    table = ResultsTable()
    qtbot.addWidget(table)
    table.set_frame(pd.DataFrame({"a": [1, 2], "q_value": [0.01, 0.9]}))
    assert table._significance == "q_value"

    table.set_frame(pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
                    significance_column="q_value")
    table._only_hits.setChecked(True)          # would index a missing column


# --------------------------------------------------------------------------- #
#  The same question asked of every panel built on it
# --------------------------------------------------------------------------- #

def _panels(qtbot):
    from spacr.qt.widgets.fast_plots import (ControlSeparation, PValueHistogram,
                                             QQPlot, ResultsTable, VolcanoPlot)
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel
    from spacr.qt.widgets.regression_results import RegressionResultsPanel
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    built = [ResultsTable(), VolcanoPlot(), PValueHistogram(), QQPlot(),
             ControlSeparation(), SweepRunsPanel(), MeasurementScanPanel(),
             RegressionResultsPanel()]
    for widget in built:
        qtbot.addWidget(widget)
    return built


def test_every_panel_constructs_with_no_data(qtbot):
    assert len(_panels(qtbot)) == 8


def test_every_checkbox_and_combo_can_be_driven_with_no_data(qtbot):
    """A control that is on screen is a control the user can click, whether
    or not anything has been loaded.

    The assertion is the COUNT: this must actually reach controls. A version
    of this test that found none would pass silently and prove nothing, which
    is exactly how it slipped past the assertion-free guard the first time.
    """
    from PySide6.QtWidgets import QCheckBox, QComboBox, QPushButton

    driven = {"checkbox": 0, "combo": 0, "button": 0}
    for widget in _panels(qtbot):
        for box in widget.findChildren(QCheckBox):
            box.setChecked(not box.isChecked())
            box.setChecked(not box.isChecked())
            driven["checkbox"] += 1
        for combo in widget.findChildren(QComboBox):
            for index in range(min(combo.count(), 4)):
                combo.setCurrentIndex(index)
            driven["combo"] += 1
        for button in widget.findChildren(QPushButton):
            text = button.text().lower()
            # Anything that opens a modal or writes a file is not a "control
            # the user clicks by accident"; the rest must survive.
            if any(word in text for word in ("save", "export", "load",
                                             "browse", "…", "...")):
                continue
            button.click()
            driven["button"] += 1

    assert driven["checkbox"] >= 8, driven
    assert driven["button"] >= 4, driven
    assert sum(driven.values()) >= 15, (
        f"only {sum(driven.values())} controls were reached; this test is not "
        f"exercising the widgets it claims to: {driven}")


def test_the_sweep_and_scan_panels_report_rather_than_crash(qtbot):
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    runs = SweepRunsPanel()
    scan = MeasurementScanPanel()
    qtbot.addWidget(runs)
    qtbot.addWidget(scan)

    assert runs.reload() is False
    assert runs.selected_trial() is None
    assert scan.run_scan() is False
    assert scan.result is None
    assert scan._status.text()
