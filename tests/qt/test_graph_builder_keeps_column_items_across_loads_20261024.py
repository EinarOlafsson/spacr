"""Loading table after table never deletes the column well's list items.

Lesson 58 crashed in ``ColumnWell._rebuild_items`` under
``GraphBuilderScreen._on_frame_loaded``: every settled load cleared the list,
freeing items whose wrappers were still referenced while the list was painted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from spacr.qt.screens import graph_builder as screens
from spacr.qt.widgets import graph_builder as gb


def _frame(width, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"feature_{i}": rng.normal(size=12) for i in range(width)})


def _listed(well):
    return [well._list.item(i).data(Qt.UserRole) for i in range(well._list.count())]


def test_rapid_frames_reuse_items_and_never_free_one(qtbot):
    well = gb.ColumnWell()
    qtbot.addWidget(well)
    well.show()
    seen = []
    for step, width in enumerate((6, 2, 9, 0, 4, 9, 1, 6) * 3):
        well.set_frame(_frame(width, step) if width else None)
        seen.extend(well._list.item(i) for i in range(well._list.count()))
        well._list.viewport().repaint()
        QApplication.processEvents()
        assert _listed(well) == list(well.columns())
        assert all(not well._list.item(i).isHidden() for i in range(well._list.count()))
    assert seen and all(isValid(item) for item in seen)
    assert len({id(item) for item in seen}) <= 9


def test_settled_loads_in_a_row_keep_the_well_consistent(qtbot):
    screen = screens.GraphBuilderScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.show()
    well = screen.builder.well
    first = None
    for step, width in enumerate((5, 3, 8, 3, 7, 2, 8)):
        frame = _frame(width, step)
        screen._on_frame_loaded(screens._Loaded(names=[], chosen=None, frame=frame, problem=None))
        QApplication.processEvents()
        assert _listed(well) == list(frame.columns)
        first = well._list.item(0) if first is None else first
        assert well._list.item(0) is first
