"""Selected plot points label original source positions without replacing rules."""
import copy

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from spacr.condition_annotations import (
    AnnotationError,
    apply_conditions,
    new_definition,
    source_context,
    table_identity,
)
from spacr.qt.widgets import condition_annotation_dialog as module


def _frame():
    return pd.DataFrame({'well': ['A', 'B', 'C', 'D'], 'value': [4, 3, 2, 1]},
                        index=[8, 8, 1, 1])


def test_selected_original_positions_survive_duplicate_indices_and_cancel(qtbot, monkeypatch):
    frame, source = _frame(), source_context()
    original = frame.copy(deep=True)
    seen = []

    def answer(dialog):
        seen.append(dialog.findChildren(module.QLabel)[0].text())
        dialog.output_column.setEditText('review')
        dialog.label.setEditText('selected')
        qtbot.mouseClick(dialog.apply_button, Qt.LeftButton)
        assert dialog.result() == QDialog.Accepted, dialog.status.text()
        return dialog.result()

    monkeypatch.setattr(module._SelectionAnnotationDialog, 'exec', answer)
    definition = module._annotate_selection(None, frame, source, None, [3, 1, 3])
    result = apply_conditions(frame, definition, source)
    assert result.review.fillna('').tolist() == ['', 'selected', '', 'selected']
    assert seen == ['Assign a label to 2 selected displayed rows only.']
    pd.testing.assert_frame_equal(frame, original)
    saved = copy.deepcopy(definition)
    monkeypatch.setattr(module._SelectionAnnotationDialog, 'exec', lambda self: QDialog.Rejected)
    assert module._annotate_selection(None, frame, source, definition, [0]) is None
    assert definition == saved


def test_reassignment_preserves_other_columns_rules_and_unselected_labels(qtbot):
    frame, source = _frame(), source_context()
    tokens = table_identity(frame)[2]
    definition = new_definition(frame, source)
    definition.pop('column')
    definition.pop('conditions')
    definition.update(version=3, columns=[
        {'column': 'review', 'kind': 'rules', 'conditions': [
            {'name': 'old', 'metadata_column': 'well', 'include': '', 'exclude': '',
             'manual_rows': tokens[:2]},
            {'name': 'regex', 'metadata_column': 'well', 'include': '^C$', 'exclude': '',
             'manual_rows': []}]},
        {'column': 'well_copy', 'kind': 'extract', 'metadata_column': 'well',
         'pattern': '(.)', 'group': 1},
        {'column': 'combined', 'kind': 'template', 'parts': [
            {'kind': 'column', 'column': 'well_copy'}, {'kind': 'text', 'text': ':'},
            {'kind': 'column', 'column': 'review'}]}])
    saved = copy.deepcopy(definition)
    dialog = module._SelectionAnnotationDialog(None, frame, source, definition, [1, 3])
    qtbot.addWidget(dialog)
    dialog.output_column.setEditText('review')
    dialog.label.setEditText('new')
    qtbot.mouseClick(dialog.apply_button, Qt.LeftButton)
    assert dialog.result() == QDialog.Accepted, dialog.status.text()
    output = apply_conditions(frame, dialog.definition, source)
    assert output.review.tolist() == ['old', 'new', 'regex', 'new']
    assert output.combined.tolist() == ['A:old', 'B:new', 'C:regex', 'D:new']
    assert dialog.definition['columns'][1:] == saved['columns'][1:]
    assert dialog.definition['columns'][0]['conditions'][1] == saved['columns'][0]['conditions'][1]
    assert definition == saved


def test_regex_conflicts_and_source_column_overwrite_block_acceptance(qtbot):
    frame, source = _frame(), source_context()
    definition = new_definition(frame, source)
    definition['conditions'] = [{'name': 'pattern', 'metadata_column': 'well',
                                  'include': '^A$', 'exclude': '', 'manual_rows': []}]
    dialog = module._SelectionAnnotationDialog(None, frame, source, definition, [0])
    qtbot.addWidget(dialog)
    dialog.label.setEditText('different')
    qtbot.mouseClick(dialog.apply_button, Qt.LeftButton)
    assert dialog.definition is None and 'overlap' in dialog.status.text()
    dialog.output_column.setEditText('value')
    qtbot.mouseClick(dialog.apply_button, Qt.LeftButton)
    assert dialog.definition is None and 'preserve' in dialog.status.text()
    assert definition['conditions'][0]['include'] == '^A$'


def test_same_label_with_exclusion_adds_manual_rule_without_changing_regex(qtbot):
    frame, source = _frame(), source_context()
    definition = new_definition(frame, source)
    definition['conditions'] = [{'name': 'same', 'metadata_column': 'well',
                                  'include': '^A$', 'exclude': '^B$', 'manual_rows': []}]
    saved = copy.deepcopy(definition)
    dialog = module._SelectionAnnotationDialog(None, frame, source, definition, [1])
    qtbot.addWidget(dialog)
    dialog.label.setEditText('same')
    qtbot.mouseClick(dialog.apply_button, Qt.LeftButton)
    assert dialog.result() == QDialog.Accepted, dialog.status.text()
    assert dialog.definition['version'] == 2
    assert dialog.definition['columns'][0]['conditions'][0] == saved['conditions'][0]
    assert apply_conditions(frame, dialog.definition, source).condition.fillna('').tolist() == ['same', 'same', '', '']
    assert definition == saved


def test_empty_and_invalid_original_positions_do_not_open_a_dialog(qtbot, monkeypatch):
    frame, source = _frame(), source_context()
    monkeypatch.setattr(module._SelectionAnnotationDialog, 'exec', lambda self: pytest.fail('unexpected dialog'))
    assert module._annotate_selection(None, frame, source, None, []) is None
    for positions in ([4], [-1], [True], [1.0]):
        with pytest.raises(AnnotationError, match='original source'):
            module._annotate_selection(None, frame, source, None, positions)
    definition = new_definition(frame, source)
    changed = frame.copy()
    changed.iloc[0, 1] = 999
    with pytest.raises(AnnotationError, match='source rows'):
        module._annotate_selection(None, changed, source, definition, [0])
