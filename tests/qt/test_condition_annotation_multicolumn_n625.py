"""Multi-column annotation controls preserve source identity and literal matching."""
import copy
import threading

import pandas as pd
from PySide6.QtCore import QItemSelectionModel, Qt

from spacr.condition_annotations import source_context
from spacr.qt.widgets import condition_annotation_dialog as module


def make_dialog(qtbot, *, threaded=False, definition=None):
    frame = pd.DataFrame({'rowID': ['r1', 'r2', 'r3', 'r4', 'r5'],
                          'columnID': ['c1', 'c2', 'c3', 'c10', 'c1'],
                          'value': [4, 3, 2, 1, 0]}, index=[7, 7, 3, 3, 1])
    dialog = module.ConditionAnnotationDialog(frame, source_context(),
                                               threaded=threaded, definition=definition)
    qtbot.addWidget(dialog)
    return dialog


def exact(box, label, column, values):
    box.name.setText(label)
    box.column.setCurrentText(column)
    box.match_mode.setCurrentIndex(1)
    box.include_values.setPlainText(values)


def click(qtbot, button):
    qtbot.mouseClick(button, Qt.LeftButton)


def populate(qtbot, dialog):
    dialog.output_column.setText('genotype')
    exact(dialog.boxes[0], 'WT', 'columnID', 'c1, c2, c1\n c3 ')
    click(qtbot, dialog.add_column)
    dialog.output_column.setText('replicate')
    exact(dialog.boxes[0], 'replicate 1', 'rowID', 'r1, r2')
    click(qtbot, dialog.add_condition)
    exact(dialog.boxes[1], 'replicate 1', 'rowID', 'r3\nr4')
    click(qtbot, dialog.add_column)
    dialog.output_column.setText('condition')
    dialog.column_kind.setCurrentIndex(1)
    for column in ('genotype', 'replicate'):
        dialog.combine_available.setCurrentText(column)
        click(qtbot, dialog.add_combine_input)
    click(qtbot, dialog.preview_button)


def test_user_creates_genotype_replicate_and_combined_condition(qtbot):
    dialog = make_dialog(qtbot)
    original = dialog.frame.copy(deep=True)
    tokens = list(dialog.source_model.tokens)
    populate(qtbot, dialog)
    assert dialog.apply_button.isEnabled(), dialog.status.text()
    assert dialog.configuration()['version'] == 2
    result = dialog.result_frame
    assert result.genotype.fillna('').tolist() == ['WT', 'WT', 'WT', '', 'WT']
    assert result.replicate.fillna('').tolist() == ['replicate 1'] * 4 + ['']
    assert result.condition.fillna('').tolist() == ['WT_replicate 1'] * 3 + ['', '']
    assert dialog.source_model.columnCount() == len(original.columns) + 3
    assert [dialog.source_model.headerData(i, Qt.Horizontal) for i in range(3, 6)] == ['genotype', 'replicate', 'condition']
    assert dialog.source_model.tokens == tokens
    pd.testing.assert_frame_equal(dialog.frame, original)
    dialog.column_selector.setCurrentIndex(1)
    click(qtbot, dialog.preview_button)
    assert all('2 matching rows' in box.count.text() for box in dialog.boxes)
    saved = copy.deepcopy(dialog.configuration())
    reopened = make_dialog(qtbot, definition=saved)
    assert reopened.configuration() == saved
    assert reopened.apply_button.isEnabled()
    click(qtbot, reopened.apply_button)
    assert reopened.result() == 1


def test_exact_lists_csv_exclusions_same_label_and_conflict(qtbot):
    dialog = make_dialog(qtbot)
    first = dialog.boxes[0]
    exact(first, 'same', 'columnID', 'c1, c2, c1\n c3')
    first.exclude_values.setPlainText('c2')
    click(qtbot, dialog.preview_button)
    assert dialog.configuration()['version'] == 1
    assert dialog.result_frame.condition.fillna('').tolist() == ['same', '', 'same', '', 'same']
    second = dialog.add_box()
    exact(second, 'same', 'columnID', 'c1')
    click(qtbot, dialog.preview_button)
    assert dialog.configuration()['version'] == 2
    assert dialog.apply_button.isEnabled()
    second.name.setText('other')
    click(qtbot, dialog.preview_button)
    assert not dialog.apply_button.isEnabled()
    assert dialog.result_frame is None
    assert module._exact_values('"a,b", x\nx, "literal.*"') == ['a,b', 'x', 'literal.*']
    assert module._exact_values('"a,b",x\n"literal.*"') == ['a,b', 'x', 'literal.*']


def test_reorder_outputs_and_components_and_source_drag_after_all_previews(qtbot):
    dialog = make_dialog(qtbot)
    populate(qtbot, dialog)
    dialog.combine_inputs.setCurrentRow(1)
    click(qtbot, dialog.combine_input_up)
    dialog.combine_separator.setText(' / ')
    click(qtbot, dialog.preview_button)
    assert dialog.result_frame.condition.iloc[0] == 'replicate 1 / WT'
    click(qtbot, dialog.column_up)
    click(qtbot, dialog.preview_button)
    assert not dialog.apply_button.isEnabled()  # replicate now follows its consumer
    click(qtbot, dialog.column_down)
    click(qtbot, dialog.preview_button)
    assert dialog.apply_button.isEnabled()
    dialog.column_selector.setCurrentIndex(0)
    dialog.filter.setText('r5')
    dialog.table.sortByColumn(2, Qt.AscendingOrder)
    selection = dialog.table.selectionModel()
    selection.select(dialog.proxy.index(0, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
    mime = dialog.table.selection_mime()
    assert dialog.boxes[0].add_mime_rows(mime)
    assert dialog.boxes[0].manual_rows == [dialog.source_model.tokens[4]]
    dialog.column_selector.setCurrentIndex(2)
    click(qtbot, dialog.remove_column)
    assert [c['column'] for c in dialog.configuration()['columns']] == ['genotype', 'replicate']


def test_threaded_stale_preview_cannot_restore_changed_multicolumn_draft(qtbot, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = module.preview

    def blocked(frame, definition, source):
        if definition.get('column') == 'slow':
            entered.set()
            assert release.wait(5)
        return original(frame, definition, source)

    monkeypatch.setattr(module, 'preview', blocked)
    dialog = make_dialog(qtbot, threaded=True)
    qtbot.waitUntil(lambda: dialog.apply_button.isEnabled(), timeout=5000)
    dialog.output_column.setText('slow')
    click(qtbot, dialog.preview_button)
    qtbot.waitUntil(entered.is_set, timeout=5000)
    try:
        populate(qtbot, dialog)
        qtbot.waitUntil(lambda: dialog.apply_button.isEnabled(), timeout=5000)
        assert list(dialog.result_frame.columns[-3:]) == ['genotype', 'replicate', 'condition']
    finally:
        release.set()
    qtbot.waitUntil(lambda: dialog._jobs.active_jobs() == 0, timeout=5000)
    qtbot.wait(50)
    assert dialog.configuration()['version'] == 2
    assert 'slow' not in dialog.result_frame
    dialog.reject()


def test_existing_source_components_empty_separator_and_literal_punctuation(qtbot):
    dialog = make_dialog(qtbot)
    exact(dialog.boxes[0], 'literal.*', 'rowID', 'r1')
    click(qtbot, dialog.add_column)
    dialog.output_column.setText('joined')
    dialog.column_kind.setCurrentIndex(1)
    for column in ('rowID', 'condition'):
        dialog.combine_available.setCurrentText(column)
        click(qtbot, dialog.add_combine_input)
    dialog.combine_separator.clear()
    click(qtbot, dialog.preview_button)
    assert dialog.result_frame.joined.fillna('').tolist() == ['r1literal.*', '', '', '', '']
    click(qtbot, dialog.add_column)
    dialog.output_column.setText('literal_match')
    exact(dialog.boxes[0], 'matched', 'condition', 'literal.*')
    click(qtbot, dialog.preview_button)
    assert dialog.result_frame.literal_match.fillna('').tolist() == ['matched', '', '', '', '']
