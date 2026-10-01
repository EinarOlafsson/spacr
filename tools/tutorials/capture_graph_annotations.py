"""Enter the illustrative Graph lesson recipe through actual dialog controls."""
from __future__ import annotations

import time


def fill_annotation_controls(dialog, capture, settle, timeout):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    def reveal(widget):
        if dialog._rule_scroll.isAncestorOf(widget):
            dialog._rule_scroll.ensureWidgetVisible(widget)
        widget.setFocus()
        settle(0.01)

    def fill(widget, value):
        reveal(widget)
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, value)
        QTest.keyClick(widget, Qt.Key_Tab)

    def choose(widget, value, *, data=False):
        index = widget.findData(value) if data else widget.findText(value)
        if index < 0:
            raise RuntimeError(f'Missing annotation choice: {value}')
        reveal(widget)
        QTest.keyClick(widget, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(widget, Qt.Key_Down)
        QTest.keyClick(widget, Qt.Key_Tab)
        if widget.currentIndex() != index:
            raise RuntimeError(f'Annotation choice did not change: {value}')

    def click(widget):
        reveal(widget)
        QTest.mouseClick(widget, Qt.LeftButton)
        settle(0.01)

    def preview(visual):
        click(dialog.preview_button)
        deadline = time.monotonic() + timeout
        # Superseded workers may still retire after the current preview is ready.
        # Wait for the same current-result contract that the Apply button uses.
        while (dialog._timer.isActive() or not dialog.apply_button.isEnabled()
               or dialog.result_frame is None
               or dialog.definition != dialog.configuration()):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f'Annotation preview did not become applicable: {dialog.status.text()}; '
                    f'timer={dialog._timer.isActive()}, apply={dialog.apply_button.isEnabled()}, '
                    f'active={dialog._jobs.active_jobs()}, pending={dialog._jobs.pending_jobs()}')
            settle(0.02)
        capture(visual)

    def rule(box, label, column, values):
        fill(box.name, label)
        choose(box.column, column)
        choose(box.match_mode, 'values', data=True)
        fill(box.include_values, ','.join(values))

    fill(dialog.output_column, 'genotype')
    rule(dialog.boxes[0], 'WildType', 'columnID', ['c1', 'c2', 'c3'])
    click(dialog.add_condition)
    rule(dialog.boxes[-1], 'mutant', 'columnID', ['c7', 'c4', 'c5', 'c6'])
    preview('13_genotype_exact_values')
    click(dialog.add_column)
    fill(dialog.output_column, 'replicate')
    rule(dialog.boxes[0], 'replicate1', 'rowID', ['r1', 'r4', 'r5', 'r6'])
    click(dialog.add_condition)
    rule(dialog.boxes[-1], 'replicate1', 'rowID', ['r7', 'r9', 'r10'])
    preview('14_replicate_rule_boxes')
    click(dialog.add_column)
    fill(dialog.output_column, 'condition')
    choose(dialog.column_kind, 'combine', data=True)
    for column in ('genotype', 'replicate'):
        choose(dialog.combine_available, column)
        click(dialog.add_combine_input)
    fill(dialog.combine_separator, '_')
    preview('15_composed_condition_preview')
    click(dialog.apply_button)


def record_annotations(app, screen, capture, settle, timeout):
    from graph_evidence import check_annotation_recipe
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest

    from spacr.qt.widgets.condition_annotation_dialog import ConditionAnnotationDialog

    base = screen._annotation_base_frame.copy(deep=True)
    errors = []

    def interact():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, ConditionAnnotationDialog):
                raise RuntimeError('Annotate conditions did not open its editor')
            fill_annotation_controls(dialog, capture, settle, timeout)
        except Exception as exc:
            errors.append(exc)
            if dialog is not None:
                dialog.reject()

    QTimer.singleShot(0, interact)
    QTest.mouseClick(screen._conditions_button, Qt.LeftButton)
    if errors:
        raise errors[0]
    if not base.equals(screen._annotation_base_frame):
        raise RuntimeError('Annotation controls changed the base measurements')
    proof = check_annotation_recipe(base, screen._frame, screen._condition_definition)
    proof['actual_dialog_controls'] = True
    capture('16_annotations_applied')
    return proof
