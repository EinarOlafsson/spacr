"""Actual condition dialog and Graph persistence use stable source rows."""
import copy
import json
import sqlite3

import pandas as pd
import pytest
from PySide6.QtCore import QItemSelectionModel, QPointF, Qt
from PySide6.QtGui import QDropEvent

from spacr.condition_annotations import new_definition, source_context
from spacr.qt.screens.graph_builder import GraphBuilderScreen
from spacr.qt.widgets.condition_annotation_dialog import ConditionAnnotationDialog
from spacr.qt.widgets.graph_spec import GraphSpec


def definition_for(screen):
    definition = new_definition(screen._annotation_base_frame, screen._condition_source, column='treatment')
    definition['conditions'] = [dict(name='treated', metadata_column='wellID', include='^A', exclude='A2', manual_rows=[])]
    return definition


def dialog_for(qtbot, frame=None):
    if frame is None:
        frame = pd.DataFrame({'wellID': ['A1', 'A2', 'B1', 'B1'], 'value': [3, 1, 2, 2]}, index=[8, 8, 1, 1])
    dialog = ConditionAnnotationDialog(frame, source_context(), threaded=False)
    qtbot.addWidget(dialog)
    return dialog


def test_real_multirow_drag_after_sort_filter_uses_source_identity(qtbot):
    dialog = dialog_for(qtbot)
    dialog.filter.setText('B1')
    dialog.table.sortByColumn(1, Qt.DescendingOrder)
    select = dialog.table.selectionModel()
    for row in range(dialog.proxy.rowCount()):
        select.select(dialog.proxy.index(row, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
    mime = dialog.table.selection_mime()
    assert len(json.loads(bytes(mime.data('application/x-spacr-condition-rows')))['tokens']) == 2
    box = dialog.boxes[0]
    event = QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    box.dropEvent(event)
    assert event.isAccepted()
    dialog.refresh_preview()
    dialog.accept()
    assert dialog.result_frame.condition.fillna('').tolist() == ['', '', 'Condition 1', 'Condition 1']


def test_conflicts_invalid_regex_and_manual_removal(qtbot):
    dialog = dialog_for(qtbot)
    first = dialog.boxes[0]
    first.include.setText('^A')
    second = dialog.add_box()
    second.include.setText('A1')
    dialog.refresh_preview()
    assert not dialog.apply_button.isEnabled()
    assert '1 overlapping' in dialog.status.text()
    second.exclude.setText('A1')
    dialog.refresh_preview()
    assert dialog.apply_button.isEnabled()
    first.include.setText('[')
    dialog.refresh_preview()
    assert not dialog.apply_button.isEnabled()
    assert 'invalid include' in dialog.status.text()
    assert 'condition' not in dialog.frame
    first.include.setText('')
    first.manual_rows = dialog.source_model.tokens[:2]
    first._show_manual_rows()
    first.rows.item(0).setSelected(True)
    first.remove_selected_rows()
    assert first.manual_rows == dialog.source_model.tokens[1:2]
    first.clear_manual_rows()
    assert first.manual_rows == []
    dialog.reject()


def test_large_drop_caps_visible_summaries_but_keeps_assignments(qtbot):
    dialog = dialog_for(qtbot, pd.DataFrame({'wellID': [f'A{i}' for i in range(800)]}))
    box = dialog.boxes[0]
    box.manual_rows = dialog.source_model.tokens[:]
    box._show_manual_rows()
    assert box.rows.count() <= 201
    dialog.refresh_preview()
    dialog.accept()
    assert dialog.result_frame.condition.notna().sum() == 800


@pytest.mark.parametrize('source_kind', ['csv', 'sqlite'])
def test_graph_annotation_roundtrip_export_reopen_and_invalid_edit(qtbot, tmp_path, source_kind):
    frame = pd.DataFrame({'wellID': ['A1', 'A2', 'B1'], 'value': [1., 2., 3.]})
    path = tmp_path / ('source.csv' if source_kind == 'csv' else 'source.db')
    if source_kind == 'csv':
        frame.to_csv(path, index=False)
    else:
        with sqlite3.connect(path) as db:
            frame.to_sql('samples', db, index=False)
            frame.to_sql('other', db, index=False)
    screen = GraphBuilderScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.load_path(str(path), 'samples' if source_kind == 'sqlite' else None)
    definition = definition_for(screen)
    screen.apply_condition_definition(definition)
    assert screen._frame.treatment.fillna('').tolist() == ['treated', '', '']
    screen.builder.set_spec(GraphSpec(kind='box', x='treatment', y='value'))
    assert screen.builder.canvas.figure().axes
    assert 'treatment' in screen.builder.canvas._frame
    bad = copy.deepcopy(definition)
    bad['conditions'][0]['include'] = '['
    with pytest.raises(ValueError, match='invalid include'):
        screen.apply_condition_definition(bad)
    assert screen._frame.treatment.iloc[0] == 'treated'
    saved = tmp_path / 'chart.json'
    screen.save_chart(saved)
    destination = tmp_path / 'working.csv'
    screen.export_table(destination)
    assert pd.read_csv(destination).treatment.iloc[0] == 'treated'
    assert json.loads((tmp_path / 'working.csv.conditions.json').read_text())['condition_annotation'] == definition
    with pytest.raises(ValueError, match='preserve'):
        screen.export_table(path)
    if source_kind == 'sqlite':
        screen.load_path(str(path), 'other')
        assert 'treatment' not in screen._frame
        screen.load_path(str(path), 'samples')
        assert screen._frame.treatment.iloc[0] == 'treated'
        with sqlite3.connect(path) as db:
            assert 'treatment' not in pd.read_sql_query('select * from samples', db)
    screen.load_chart(saved)
    assert screen.builder.spec.x == 'treatment'
    assert screen._frame.treatment.iloc[0] == 'treated'
    dialog = ConditionAnnotationDialog(screen._annotation_base_frame, screen._condition_source,
                                       definition=screen._condition_definition, threaded=False)
    qtbot.addWidget(dialog)
    assert dialog.boxes[0].include.text() == '^A'
    dialog.output_column.setText('new label')
    dialog.refresh_preview()
    dialog.accept()
    screen.apply_condition_definition(dialog.definition)
    assert 'new label' in screen._frame and 'treatment' not in screen._frame
    screen.close()


def test_derived_annotation_roundtrip_and_changed_source_refusal(qtbot, tmp_path):
    from spacr.derived_tables import default_definition
    path = tmp_path / 'derived.db'
    with sqlite3.connect(path) as db:
        pd.DataFrame({'id': [1, 2], 'wellID': ['A1', 'B1']}).to_sql('samples', db, index=False)
        pd.DataFrame({'parent': [1, 2], 'value': [4., 8.]}).to_sql('events', db, index=False)
    screen = GraphBuilderScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.load_path(str(path), 'samples')
    merge = default_definition(str(path), ['samples', 'events'], base='samples', name='Combined')
    merge.update(mode='custom', acknowledged=True, base_keys=['id'])
    merge['joins'][0].update(left_keys=['id'], right_keys=['parent'], how='left')
    screen.use_derived_table(merge)
    definition = new_definition(screen._annotation_base_frame, screen._condition_source)
    metadata = next(c for c in screen._frame if c.endswith('wellID'))
    definition['conditions'] = [dict(name='treated', metadata_column=metadata, include='A', exclude='', manual_rows=[])]
    screen.apply_condition_definition(definition)
    saved = tmp_path / 'derived-chart.json'
    screen.save_chart(saved)
    screen.load_path(str(path), 'samples')
    screen.load_chart(saved)
    assert screen._frame.condition.fillna('').tolist() == ['treated', '']
    before = screen._frame.copy()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE samples SET wellID='C1' WHERE id=1")
    screen.load_chart(saved)
    assert 'changed' in screen._source.text()
    pd.testing.assert_frame_equal(screen._frame, before)
    screen.close()
