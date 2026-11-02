"""Refreshing the metadata wizard's plate list keeps each row's own widgets.

Every rescan used to empty the plate table and give each row a new Apply
button (and Use-subfolders switch): Qt deleted the old ones with
``deleteLater`` while the wizard went on. Run alone,
``test_well_only_regex_groups_channels_and_live_preview_requires_valid_wells``
then died in that deferred delete with a bus error in about 1 run in 15
(2 of 30 before this change, 0 of 30 after). The rows now keep their widgets
and only tell them which plate they stand for.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")

from shiboken6 import isValid

from spacr.qt.widgets import qc_field_browser as browser

pytestmark = pytest.mark.qt


def _plate(path, mixed=False):
    return {"path": str(path), "default_name": path.name, "images": 2, "mixed": mixed}


def test_a_rescan_reuses_the_row_widgets_and_points_them_at_the_new_plates(tmp_path, qtbot, monkeypatch):
    tifffile.imwrite(tmp_path / "A01_C1.tif", np.zeros((5, 7), np.uint16))
    wizard = browser._OverviewMetadataWizard(str(tmp_path))
    qtbot.addWidget(wizard)
    wizard._startup_timer.stop()
    one, two = tmp_path / "plate_one", tmp_path / "plate_two"
    wizard._show_plates({"plates": [_plate(one, mixed=True), _plate(two)]})
    table = wizard.plate_table
    buttons = {table.cellWidget(row, 5).property("platePath"): table.cellWidget(row, 5) for row in range(2)}
    toggle = next(table.cellWidget(row, 3) for row in range(2) if table.cellWidget(row, 3) is not None)

    wizard._show_plates({"plates": [_plate(two), _plate(one, mixed=True)]})

    after = [table.cellWidget(row, 5) for row in range(table.rowCount())]
    assert table.rowCount() == 2
    assert all(isValid(button) for button in after)
    assert set(map(id, after)) == set(map(id, buttons.values()))
    assert {button.property("platePath") for button in after} == {str(one), str(two)}
    assert isValid(toggle)
    applied, switched = [], []
    monkeypatch.setattr(wizard, "_apply_to_plate", lambda path, *_: applied.append(path))
    monkeypatch.setattr(wizard, "_use_plate_subfolders", lambda path, on: switched.append((path, on)))
    for button in after:
        button.click()
    toggle.setChecked(True)
    assert sorted(applied) == sorted([str(one), str(two)])
    assert switched == [(toggle.property("platePath"), True)]
    wizard.reject()


def test_a_rescan_in_the_same_order_keeps_the_very_same_switch(tmp_path, qtbot):
    tifffile.imwrite(tmp_path / "A01_C1.tif", np.zeros((5, 7), np.uint16))
    wizard = browser._OverviewMetadataWizard(str(tmp_path))
    qtbot.addWidget(wizard)
    wizard._startup_timer.stop()
    one, two = tmp_path / "plate_one", tmp_path / "plate_two"
    wizard._show_plates({"plates": [_plate(one, mixed=True), _plate(two)]})
    table = wizard.plate_table
    toggle = table.cellWidget(0, 3)
    assert toggle is not None and table.cellWidget(1, 3) is None
    wizard._use_subfolders.add(str(one))

    wizard._show_plates({"plates": [_plate(one, mixed=True), _plate(two)]})

    assert table.cellWidget(0, 3) is toggle
    assert toggle.property("platePath") == str(one)
    assert toggle.isChecked()
    wizard.reject()
