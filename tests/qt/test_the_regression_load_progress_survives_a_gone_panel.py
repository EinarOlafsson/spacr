"""A regression load's progress reports survive a panel that is going away.

Pinned behaviour of :class:`spacr.qt.widgets.regression_results.RegressionResultsPanel`:

* the worker's progress relay does not raise once the panel's C++ half has
  been deleted (a panel closed while a read is still running);
* the step-3 sentence ("building the table...") is still shown when the
  immediate repaint of the header fails.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

import shiboken6  # noqa: E402

from spacr.qt.widgets.regression_results import (  # noqa: E402
    RegressionResultsPanel,
)

pytestmark = pytest.mark.qt


def test_a_progress_relay_after_the_panel_is_deleted_is_dropped(qtbot):
    panel = RegressionResultsPanel()
    shiboken6.delete(panel)
    assert not shiboken6.isValid(panel)

    assert panel._relay_load_progress(1, 2, 1, 3, "results.csv") is None


def test_the_building_step_is_announced_when_the_repaint_fails(
        qtbot, monkeypatch):
    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)

    def _gone():
        raise RuntimeError("Internal C++ object already deleted.")

    monkeypatch.setattr(panel._source, "repaint", _gone)

    panel._say_building(pd.DataFrame({"coef": [0.1, 0.2, 0.3]}))

    assert panel._source.text() == panel._load_stage_text(3, name="3")
    assert "3 rows" in panel._source.text()
