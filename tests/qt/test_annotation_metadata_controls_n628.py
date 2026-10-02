"""Author extraction, readable rules and ordered compositions through real controls."""
import copy

import pandas as pd
from PySide6.QtCore import QItemSelectionModel, QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent

from spacr.condition_annotations import source_context
from spacr.qt.widgets import condition_annotation_dialog as module


def make_dialog(qtbot, definition=None):
    frame = pd.DataFrame({
        'original_filename': ['HeLa_WT_rep2_24h_DMSO.tif', 'A549_KO_rep1_48h_drug.tif',
                              'HeLa_WT_rep3_24h_drug.tif', None],
        'filename': ['converted1.tif', 'converted2.tif', 'converted3.tif', 'converted4.tif'],
        'measurement': [9, 7, 5, 3]}, index=[4, 4, 1, 1])
    dialog = module.ConditionAnnotationDialog(frame, source_context(), definition=definition, threaded=False)
    qtbot.addWidget(dialog)
    dialog.show()
    return dialog


def click(qtbot, button):
    qtbot.mouseClick(button, Qt.LeftButton)


def extract(qtbot, dialog):
    dialog.column_kind.setCurrentIndex(dialog.column_kind.findData('extract'))
    assert dialog.extract_source.currentText() == 'original_filename'
    dialog.extract_pattern.setText(r'(?P<cell_type>[^_]+)_(?P<genotype>[^_]+)_(?P<replicate>[^_]+)_(?P<time>[^_]+)_(?P<treatment>[^.]+)')
    click(qtbot, dialog.named_groups_button)
    assert [c['column'] for c in dialog.configuration()['columns']] == ['cell_type', 'genotype', 'replicate', 'time', 'treatment']
    click(qtbot, dialog.preview_button)
    assert dialog.apply_button.isEnabled(), dialog.status.text()


def add_rule(qtbot, dialog):
    dialog.column_selector.setCurrentIndex(4)
    click(qtbot, dialog.add_column)
    dialog.output_column.setText('assignment')
    box = dialog.boxes[0]
    box.name.setText('control')
    row = box.criteria[0]
    row.column.setCurrentText('genotype')
    row.operator.setCurrentIndex(row.operator.findData('equals'))
    row.value.setText('WT')
    click(qtbot, box.add_criterion)
    row = box.criteria[1]
    row.column.setCurrentText('treatment')
    row.operator.setCurrentIndex(row.operator.findData('not_contains'))
    row.value.setText('drug')
    return box


def add_token(qtbot, dialog, name):
    dialog.template_palette.setCurrentItem(dialog.template_palette.findItems(name, Qt.MatchExactly)[0])
    click(qtbot, dialog.template_add_column)


def test_extract_assign_compose_real_controls_and_reopen(qtbot):
    dialog = make_dialog(qtbot)
    original = dialog.frame.copy(deep=True)
    tokens = list(dialog.source_model.tokens)
    assert not dialog.apply_button.isEnabled()  # A blank readable predicate must not label every row.
    extract(qtbot, dialog)
    box = add_rule(qtbot, dialog)
    click(qtbot, dialog.preview_button)
    assert dialog.result_frame.assignment.fillna('').tolist() == ['control', '', '', '']
    box.criteria_match.setCurrentIndex(box.criteria_match.findData('any'))
    click(qtbot, dialog.preview_button)
    assert dialog.result_frame.assignment.fillna('').tolist() == ['control', '', 'control', '']
    click(qtbot, dialog.add_column)
    dialog.output_column.setText('condition')
    dialog.column_kind.setCurrentIndex(dialog.column_kind.findData('combine'))
    for i, column in enumerate(('cell_type', 'genotype', 'replicate', 'time', 'treatment')):
        if i:
            dialog.template_text.setText('_')
            click(qtbot, dialog.template_add_text)
        add_token(qtbot, dialog, column)
    click(qtbot, dialog.preview_button)
    assert dialog.apply_button.isEnabled(), dialog.status.text()
    assert dialog.configuration()['version'] == 3
    assert dialog.result_frame.condition.fillna('').tolist() == [
        'HeLa_WT_rep2_24h_DMSO', 'A549_KO_rep1_48h_drug', 'HeLa_WT_rep3_24h_drug', '']
    assert dialog.template_parts.item(0).text() == '{cell_type}'
    assert dialog.configuration()['columns'][-1]['parts'][0] == {'kind': 'column', 'column': 'cell_type'}
    assert '→' in dialog.example.text()
    assert dialog.source_model.columnCount() == len(original.columns) + 7
    assert dialog.source_model.tokens == tokens
    pd.testing.assert_frame_equal(dialog.frame, original)
    recipe = copy.deepcopy(dialog.configuration())
    reopened = make_dialog(qtbot, recipe)
    assert reopened.configuration() == recipe
    assert reopened.apply_button.isEnabled(), reopened.status.text()
    reopened.column_selector.setCurrentIndex(5)
    assert reopened.boxes[0].criteria[0].column.currentText() == 'genotype'
    assert reopened.boxes[0].criteria[1].column.currentText() == 'treatment'
    assert reopened.configuration() == recipe
    click(qtbot, reopened.preview_button)
    click(qtbot, reopened.apply_button)
    assert reopened.result() == 1


def test_palette_drop_edit_text_and_reorder_invalidates_preview(qtbot):
    dialog = make_dialog(qtbot)
    dialog.column_kind.setCurrentIndex(dialog.column_kind.findData('combine'))
    dialog.template_text.setText('prefix_')
    click(qtbot, dialog.template_add_text)
    mime = QMimeData()
    mime.setData(module._COLUMN_MIME, b'filename')
    event = QDropEvent(QPointF(900, 100), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    dialog.template_parts.dropEvent(event)
    assert event.isAccepted()
    click(qtbot, dialog.preview_button)
    assert dialog.result_frame.condition.iloc[0] == 'prefix_converted1.tif'
    dialog.template_parts.item(0).setText('sample: ')
    assert not dialog.apply_button.isEnabled()
    click(qtbot, dialog.preview_button)
    assert dialog.result_frame.condition.iloc[0] == 'sample: converted1.tif'
    dialog.template_parts.setCurrentRow(0)
    internal = dialog.template_parts._drag_mime()
    for event in (QDragEnterEvent(QPoint(900, 100), Qt.MoveAction, internal, Qt.LeftButton, Qt.NoModifier),
                  QDragMoveEvent(QPoint(900, 100), Qt.MoveAction, internal, Qt.LeftButton, Qt.NoModifier)):
        (dialog.template_parts.dragEnterEvent if isinstance(event, QDragEnterEvent)
         else dialog.template_parts.dragMoveEvent)(event)
        assert event.isAccepted()
    event = QDropEvent(QPointF(900, 100), Qt.MoveAction, internal, Qt.LeftButton, Qt.NoModifier)
    dialog.template_parts.dropEvent(event)
    assert event.isAccepted()
    assert not dialog.apply_button.isEnabled()
    click(qtbot, dialog.preview_button)
    assert dialog.result_frame.condition.iloc[0] == 'converted1.tifsample: '
    # Move it back with the accessible button before the independent signal-path check.
    dialog.template_parts.setCurrentRow(1)
    click(qtbot, dialog.template_up)
    click(qtbot, dialog.preview_button)
    dialog.template_parts.setCurrentRow(1)
    click(qtbot, dialog.template_up)
    assert not dialog.apply_button.isEnabled()
    click(qtbot, dialog.preview_button)
    assert dialog.result_frame.condition.iloc[0] == 'converted1.tifsample: '
    # QListWidget's native move implementation may emit remove/insert instead of rowsMoved.
    item = dialog.template_parts.takeItem(0)
    dialog.template_parts.insertItem(1, item)
    assert not dialog.apply_button.isEnabled()
    click(qtbot, dialog.preview_button)
    assert dialog.result_frame.condition.iloc[0] == 'sample: converted1.tif'
    bad = QMimeData()
    bad.setData(module._COLUMN_MIME, b'future_output')
    event = QDropEvent(QPointF(5, 5), Qt.CopyAction, bad, Qt.LeftButton, Qt.NoModifier)
    dialog.template_parts.dropEvent(event)
    assert not event.isAccepted()


def test_named_groups_collision_invalid_pattern_and_optional_capture(qtbot):
    dialog = make_dialog(qtbot)
    dialog.column_kind.setCurrentIndex(dialog.column_kind.findData('extract'))
    dialog.extract_pattern.setText('(?P<filename>.*)')
    before = copy.deepcopy(dialog.configuration())
    click(qtbot, dialog.named_groups_button)
    assert 'already exist' in dialog.status.text()
    assert dialog.configuration() == before
    dialog.extract_pattern.setText('[')
    click(qtbot, dialog.preview_button)
    assert not dialog.apply_button.isEnabled()
    dialog.extract_pattern.setText(r'(?P<type>HeLa)(?:_(?P<optional>missing))?')
    click(qtbot, dialog.named_groups_button)
    click(qtbot, dialog.preview_button)
    assert dialog.apply_button.isEnabled(), dialog.status.text()
    assert dialog.result_frame.type.fillna('').tolist() == ['HeLa', '', 'HeLa', '']
    assert dialog.result_frame.optional.isna().all()


def test_readable_literal_semantics_manual_identity_and_blank_guard(qtbot):
    dialog = make_dialog(qtbot)
    box = dialog.boxes[0]
    box.name.setText('selected')
    row = box.criteria[0]
    row.column.setCurrentText('original_filename')
    row.value.setText('HeLa.*')
    click(qtbot, dialog.preview_button)
    assert dialog.result_frame.condition.isna().all()  # Contains is literal.
    row.operator.setCurrentIndex(row.operator.findData('regex'))
    click(qtbot, dialog.preview_button)
    assert dialog.result_frame.condition.fillna('').tolist() == ['selected', '', 'selected', '']
    row.value.clear()
    click(qtbot, dialog.preview_button)
    assert not dialog.apply_button.isEnabled()
    row.value.setText('HeLa')
    dialog.filter.setText('converted2')
    selection = dialog.table.selectionModel()
    selection.select(dialog.proxy.index(0, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
    assert box.add_mime_rows(dialog.table.selection_mime())
    click(qtbot, dialog.preview_button)
    assert dialog.result_frame.condition.fillna('').tolist() == ['selected', 'selected', 'selected', '']
    assert box.manual_rows == [dialog.source_model.tokens[1]]
