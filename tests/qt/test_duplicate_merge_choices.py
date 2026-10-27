"""Duplicate merge choices preserve source data and use displayed column rules."""
import sqlite3

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from tests.test_n41_observation_key_diagnostics import _database
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


def _duplicates(tmp_path):
    path, cells = _database(tmp_path)
    cells['area'] = [300., 500.]
    with sqlite3.connect(path) as db:
        cells.to_sql('cell', db, index=False, if_exists='append')
    return path


def _popup_answer(qtbot, monkeypatch, accept, seen):
    def answer(box):
        box.show()
        assert box.windowTitle() == 'Duplicate objects detected'
        assert box.defaultButton() is box.button(QMessageBox.Abort)
        assert box.escapeButton() is box.button(QMessageBox.Abort)
        assert 'Source tables will remain unchanged' in box.text()
        assert 'Time is included only when present' in box.text()
        button = next(b for b in box.buttons() if b.text() == 'Merge duplicates')
        seen.append(box.detailedText())
        qtbot.mouseClick(button if accept else box.button(QMessageBox.Abort), Qt.LeftButton)
        return box.result()
    monkeypatch.setattr(QMessageBox, 'exec', answer)


@pytest.mark.parametrize('accept', [False, True])
def test_shared_merge_warning_abort_or_explicit_aggregation(qtbot, monkeypatch, tmp_path,
                                                          private_ai_profile, accept):
    from spacr.qt.widgets.merge_tables_dialog import MergeTablesDialog
    from spacr.merge_tables import _duplicate_merge_enabled

    path = _duplicates(tmp_path)
    before = path.read_bytes()
    seen = []
    _popup_answer(qtbot, monkeypatch, accept, seen)
    dialog = MergeTablesDialog(str(path), selected=('cell', 'nucleus'), threaded=False)
    qtbot.addWidget(dialog)
    dialog._overrides = {'cell': {'area': 'mean'}, 'nucleus': {'area': 'max'}}
    dialog._fill_rules()
    rows = {(dialog.rules.item(i, 0).text(), dialog.rules.item(i, 1).text()):
            dialog.rules.item(i, 2).text() for i in range(dialog.rules.rowCount())}
    assert rows[('cell', 'area')] == 'mean'
    assert rows[('nucleus', 'area')] == 'max'
    dialog.validate_preview()
    assert len(seen) == 1
    assert dialog.create.isEnabled() is accept
    if accept:
        assert dialog.definition['merge_duplicates'] is True
        assert dialog.result_frame.cell_area.tolist() == [200., 350.]
        assert dialog.result_frame.nucleus_area.tolist() == [3., 11.]
    else:
        assert dialog.result_frame is None
    assert path.read_bytes() == before
    assert _duplicate_merge_enabled.get() is False
    dialog.close()


@pytest.mark.parametrize('accept', [False, True])
def test_gate_ordinary_working_set_warns_before_duplicate_merging(qtbot, monkeypatch, tmp_path,
                                                               private_ai_profile, accept):
    from spacr.qt.screens.gate_editor import GateEditorScreen
    from spacr.merge_tables import _duplicate_merge_enabled

    path = _duplicates(tmp_path)
    before = path.read_bytes()
    seen = []
    _popup_answer(qtbot, monkeypatch, accept, seen)
    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.load_path(str(path), 'cell')
    previous = screen._frame.copy()
    screen._tables = ['cell', 'nucleus']
    screen._settings = screen._settings.replaced(merge_overrides={'cell.area': 'mean'})
    screen._reload_working_set()
    assert len(seen) == 1
    assert 'cell.area: mean' in seen[0]
    if accept:
        assert screen._frame.cell_area.tolist() == [200., 350.]
        assert screen._frame.nucleus_area.tolist() == [5., 11.]
    else:
        pd.testing.assert_frame_equal(screen._frame, previous)
    assert path.read_bytes() == before
    assert _duplicate_merge_enabled.get() is False
    screen.close()
