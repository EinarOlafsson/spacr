"""Column-aware regex help can be copied without changing annotation rules."""
import re

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QApplication, QComboBox, QLineEdit

from spacr.condition_annotations import source_context
from spacr.qt.widgets.condition_annotation_dialog import ConditionAnnotationDialog, _regex_examples


@pytest.fixture
def dialog(qtbot):
    frame = pd.DataFrame({'wellID': ['A01', 'B02'], 'original_filename': ['drug(A)+01.tif', 'control[2].tif']})
    widget = ConditionAnnotationDialog(frame, source_context(), threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_examples_escape_real_values_and_explain_regex_operators(dialog):
    examples = _regex_examples(dialog.frame, 'original_filename')
    patterns = [row[1] for row in examples]
    first, second = dialog.frame.original_filename
    assert patterns[0] == re.escape(first)
    assert re.fullmatch(patterns[1], first)
    assert re.fullmatch(patterns[1], second)
    assert not re.fullmatch(patterns[1], 'drugA01Xtif')
    assert re.fullmatch(patterns[2], first)
    assert re.fullmatch(patterns[3], first)
    assert re.search(patterns[4], first)
    assert not re.search(patterns[4], first + '.backup')
    assert re.search(patterns[5], first.upper())


def test_guide_above_first_box_follows_column_and_copies_exact_text(dialog, qtbot):
    dialog.boxes[0].match_mode.setCurrentIndex(0)  # Open the legacy regex editor explicitly.
    before = dialog.configuration()
    box = dialog.boxes[0]
    box.column.setCurrentText('original_filename')
    assert dialog._regex_column.currentText() == 'original_filename'
    assert dialog.box_layout.indexOf(box) > 0
    dialog._regex_kind.setCurrentIndex(1)
    shown = dialog._regex_pattern.text()
    assert 'drug' in shown and 'control' in shown
    qtbot.mouseClick(dialog._copy_regex, Qt.LeftButton)
    assert QApplication.clipboard().text() == shown
    assert box.include.text() == '' and box.exclude.text() == ''
    assert before['conditions'][0]['manual_rows'] == box.manual_rows
    dialog._regex_pattern.setText(r'^(one|two).*\.tif$')
    qtbot.mouseClick(dialog._copy_regex, Qt.LeftButton)
    assert QApplication.clipboard().text() == r'^(one|two).*\.tif$'


def test_all_direct_annotation_controls_explain_their_action(dialog):
    controls = [dialog.output_column, dialog.add_condition, dialog.filter,
                dialog.table, dialog.preview_button, dialog.apply_button,
                dialog._regex_column, dialog._regex_kind, dialog._regex_pattern, dialog._copy_regex]
    box = dialog.boxes[0]
    controls += [box.name, box.column, box.include, box.exclude, box.rows]
    controls += box.findChildren(QAbstractButton)
    assert all(control.toolTip() for control in controls)
    assert 'manual' in box.include.toolTip()
    assert 'including manually' in box.exclude.toolTip()
    assert 'not a regex' in dialog.filter.toolTip()
    assert dialog.findChildren(QLineEdit) and dialog.findChildren(QComboBox)


def test_examples_have_safe_placeholders_for_missing_values_and_do_not_scan_whole_column():
    frame = pd.DataFrame({'empty': [None] * 300, 'metadata': ['a.b'] * 256 + ['later|value'] * 44})
    for column in frame:
        examples = _regex_examples(frame, column)
        for _label, pattern, _help in examples:
            re.compile(pattern)
    assert 'sample_value' in _regex_examples(frame, 'empty')[0][1]
    assert 'later' not in _regex_examples(frame, 'metadata')[1][1]
