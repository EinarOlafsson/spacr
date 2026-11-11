"""Issue #183: the Merge tables dialog validates a re-run set in default mode."""
import sqlite3

import pandas as pd
import pytest
from PySide6.QtCore import Qt

from spacr.qt.widgets.merge_tables_dialog import MergeTablesDialog


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "measurements.db")
    ident = dict(plateID=["p1"] * 3, rowID=["r1"] * 3, columnID=["c1"] * 3, fieldID=["f1"] * 3)
    cell = pd.DataFrame({**ident, "object_label": [1, 2, 3], "area": [100., 200., 300.]})
    cytoplasm = cell.assign(area=[60., 150., 210.])
    pathogen = pd.DataFrame({**ident, "object_label": [7, 8, 9], "cell_id": [1, 1, 2],
                             "area": [2., 3., 7.]})
    with sqlite3.connect(path) as connection:
        for suffix in ("", "_1"):
            cell.to_sql("cell" + suffix, connection, index=False)
            cytoplasm.to_sql("cytoplasm" + suffix, connection, index=False)
            pathogen.to_sql("pathogen" + suffix, connection, index=False)
    return path


def _check(dialog, name):
    item = dialog.tables.findItems(name, Qt.MatchExactly)[0]
    item.setCheckState(Qt.Checked)


def test_checking_rerun_tables_follows_their_run_and_previews(qtbot, db):
    dialog = MergeTablesDialog(db, threaded=False)
    qtbot.addWidget(dialog)
    assert {"cell_1", "cytoplasm_1", "pathogen_1"} <= {
        dialog.tables.item(i).text() for i in range(dialog.tables.count())}
    for name in ("cell_1", "cytoplasm_1", "pathogen_1"):
        _check(dialog, name)
    assert dialog.base.currentText() == "cell_1"
    assert sorted(dialog._selected()) == ["cell_1", "cytoplasm_1", "pathogen_1"]
    dialog.validate_preview()
    text = dialog.preview_text.toPlainText()
    assert dialog.create.isEnabled(), text
    assert "3 output rows" in text
    assert "Default output is one row per cell" not in text


def test_mixed_runs_are_refused_with_a_clear_message(qtbot, db):
    dialog = MergeTablesDialog(db, selected=["cell", "pathogen"], threaded=False)
    qtbot.addWidget(dialog)
    dialog.base.setCurrentText("cytoplasm_1")
    dialog.validate_preview()
    assert not dialog.create.isEnabled()
    assert "different Measure runs" in dialog.preview_text.toPlainText()
