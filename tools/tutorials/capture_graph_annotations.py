"""Enter the illustrative Graph lesson recipe through actual dialog controls."""
from __future__ import annotations

import time


def fill_annotation_controls(dialog, capture, settle, timeout, extras=None):
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
        choose(box.match_mode, 'values', data=True)
        choose(box.column, column)
        fill(box.include_values, ','.join(values))

    def append_column_token(column):
        palette = dialog.template_palette
        items = palette.findItems(column, Qt.MatchExactly)
        if len(items) != 1:
            raise RuntimeError(f'Missing annotation column token: {column}')
        reveal(palette)
        palette.scrollToItem(items[0])
        QTest.mouseClick(palette.viewport(), Qt.LeftButton,
                         pos=palette.visualItemRect(items[0]).center())
        click(dialog.template_add_column)

    if extras is not None:
        # Give the editor room on the 4K recording, centred on the window.
        dialog.resize(2600, 1700)
        parent = dialog.parentWidget().window()
        dialog.move(parent.geometry().center() - dialog.rect().center())
        settle(0.5)
    fill(dialog.output_column, 'genotype')
    rule(dialog.boxes[0], 'WildType', 'columnID', ['c1', 'c2', 'c3'])
    click(dialog.add_condition)
    rule(dialog.boxes[-1], 'mutant', 'columnID', ['c7', 'c4', 'c5', 'c6'])
    preview('13_genotype_exact_values')
    click(dialog.add_column)
    fill(dialog.output_column, 'replicate')
    rule(dialog.boxes[0], 'replicate 1', 'rowID', ['r1', 'r4', 'r5', 'r6'])
    click(dialog.add_condition)
    rule(dialog.boxes[-1], 'replicate 1', 'rowID', ['r7', 'r9', 'r10'])
    preview('14_replicate_rule_boxes')
    click(dialog.add_column)
    fill(dialog.output_column, 'condition')
    choose(dialog.column_kind, 'combine', data=True)
    append_column_token('genotype')
    fill(dialog.template_text, '_')
    click(dialog.template_add_text)
    append_column_token('replicate')
    preview('15_composed_condition_preview')
    if extras:
        # Extract text (628): one output per named group of a regex, read
        # from an existing text column (the example has no filenames).
        click(dialog.add_column)
        fill(dialog.output_column, 'row_number')
        choose(dialog.column_kind, 'extract', data=True)
        choose(dialog.extract_source, 'rowID')
        fill(dialog.extract_pattern, r'r(?P<row_number>\d+)')
        click(dialog.named_groups_button)
        settle(0.5)
        preview('15b_extract_text')
        # Save schema (629): the rules, not the values, to a JSON file.
        schema = extras['schema']
        _file_dialog(dialog.save_schema_button, schema, capture, settle, '15c_save_schema', click)
        deadline = time.monotonic() + timeout
        while not schema.is_file() or not dialog.save_schema_button.isEnabled():
            if time.monotonic() >= deadline:
                raise RuntimeError('Save schema did not finish: ' + dialog.status.text())
            settle(0.05)
        extras['schema_saved_status'] = dialog.status.text()
        # Load schema: validated into a preview, never applied by itself.
        before = dialog.status.text()
        _file_dialog(dialog.load_schema_button, schema, capture, settle, None, click)
        deadline = time.monotonic() + timeout
        settle(0.5)
        while (dialog._jobs.active_jobs() or dialog._timer.isActive()
               or not dialog.apply_button.isEnabled()
               or dialog.definition != dialog.configuration()):
            if time.monotonic() >= deadline:
                raise RuntimeError('Load schema did not reach its preview: ' + dialog.status.text())
            settle(0.05)
        # Show the loaded recipe's last output (the extracted column): the
        # rules editor of the first output currently redraws its rows over
        # each other right after a load, which a recording must not show.
        choose(dialog.column_selector, 'row_number')
        settle(3)
        capture('15d_load_schema')
        extras['schema_loaded_status'] = dialog.status.text()
        # The loaded rules of the first output, as the schema restored them.
        choose(dialog.column_selector, 'genotype')
        settle(3)
        capture('15e_loaded_rules')
    click(dialog.apply_button)


def _file_dialog(button, path, capture, settle, name, click):
    """Press ``button`` and answer its Qt file dialog with ``path``."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QFileDialog, QLineEdit
    done = []

    def answer():
        dialogs = [w for w in QApplication.topLevelWidgets() if isinstance(w, QFileDialog) and w.isVisible()]
        if not dialogs:
            if len(done) < 100:
                done.append(None)
                QTimer.singleShot(200, answer)
            return
        dialog = dialogs[0]
        dialog.resize(1400, 950)
        dialog.setDirectory(str(path.parent))
        settle(0.5)
        edit = dialog.findChild(QLineEdit, 'fileNameEdit')
        edit.setText(path.name)
        settle(0.5)
        if name:
            capture(name)
        done.append(dialog)
        dialog.selectFile(str(path))
        dialog.accept()
    QTimer.singleShot(300, answer)
    click(button)
    if not any(done):
        raise RuntimeError('The file dialog did not open')


def record_annotations(app, screen, capture, settle, timeout, extras=None):
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
            fill_annotation_controls(dialog, capture, settle, timeout, extras)
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
    proof = check_annotation_recipe(base, screen._frame, screen._condition_definition,
                                    extract=bool(extras))
    if extras:
        proof['schema'] = {k: v for k, v in extras.items() if k != 'schema'}
    proof['actual_dialog_controls'] = True
    capture('16_annotations_applied')
    return proof


def record_graph_extras(app, screen, capture, settle, timeout, write_json, captures):
    """Cleared channels, Save annotated table and Merge tables (613, 620, 621).

    These write new tables into the example database, so they run after every
    read-only scene; the original tables are checked unchanged afterwards.
    """
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QDialog, QInputDialog, QPushButton

    proof = {}

    def wait(condition, what):
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() >= deadline:
                raise RuntimeError(what)
            settle(0.1)

    # Clear all channels: the canvas says what is missing.
    clear = [b for b in screen.findChildren(QPushButton)
             if b.isVisible() and b.text().replace('&', '') == 'Clear all channels']
    if len(clear) != 1:
        raise RuntimeError('Expected one visible Clear all channels button')
    QTest.mouseClick(clear[0], Qt.LeftButton)
    settle(1.5)
    proof['cleared_notice'] = screen.builder.canvas.notice()
    capture('10b_channels_cleared')

    # Save annotated table…: a new table, existing tables preserved.
    asked = []

    def answer_name():
        boxes = [w for w in QApplication.topLevelWidgets() if isinstance(w, QInputDialog) and w.isVisible()]
        if not boxes:
            if len(asked) < 100:
                asked.append(None)
                QTimer.singleShot(200, answer_name)
            return
        box = boxes[0]
        settle(0.5)
        proof['save_annotated_prompt'] = [box.labelText(), box.textValue()]
        capture('16b_save_annotated')
        asked.append(box)
        box.accept()
    if not screen._save_annotated_button.isEnabled():
        raise RuntimeError('Save annotated table is not enabled after Apply conditions')
    QTimer.singleShot(300, answer_name)
    QTest.mouseClick(screen._save_annotated_button, Qt.LeftButton)
    if not any(asked):
        raise RuntimeError('Save annotated table did not ask for a name')
    wait(lambda: not screen._jobs.is_busy(), 'Saving the annotated table did not finish')
    settle(1)
    proof['save_annotated_status'] = screen._source.text()

    # Merge tables: cell plus nucleus and pathogen, output level cell.
    picker = screen._table_picker
    if picker.currentText() != 'cell':
        picker.setCurrentIndex(picker.findText('cell'))
        wait(lambda: picker.currentText() == 'cell' and not screen._jobs.is_busy()
             and screen._frame is not None and len(screen._frame) > 0, 'The cell table did not reload')
        settle(1.5)
    merged = []

    def merge():
        dialog = app.activeModalWidget()
        try:
            if dialog is None or dialog.objectName() != 'MergeTablesDialog':
                raise RuntimeError('Merge tables did not open its dialog')
            dialog.resize(1700, 1200)
            for i in range(dialog.tables.count()):
                item = dialog.tables.item(i)
                item.setCheckState(Qt.Checked if item.text() in ('cell', 'nucleus', 'pathogen')
                                   else Qt.Unchecked)
            dialog.base.setCurrentText('cell')
            dialog.name.setText('cell_merged')
            settle(0.5)
            QTest.mouseClick(dialog.preview, Qt.LeftButton)
            wait(lambda: dialog.create.isEnabled(), 'Merge preview did not validate: ' + dialog.state.text())
            settle(1)
            proof['merge_preview'] = dialog.preview_text.toPlainText()[:2000]
            capture('17_merge_tables')
            merged.append(True)
            QTest.mouseClick(dialog.create, Qt.LeftButton)
        except Exception as exc:
            proof['merge_error'] = str(exc)
            if dialog is not None:
                dialog.reject()
    QTimer.singleShot(500, merge)
    QTest.mouseClick(screen._merge_button, Qt.LeftButton)
    if not merged:
        raise RuntimeError(proof.get('merge_error', 'Merge tables did not run'))
    wait(lambda: screen._table_picker.currentText() == 'cell_merged' and not screen._jobs.is_busy(),
         'The merged table was not selected: ' + screen._source.text())
    settle(2)
    proof['merged_rows'] = len(screen._frame)
    proof['merged_columns'] = len(screen._frame.columns)
    capture('18_merged_table')
    write_json(captures / 'graph_extras.json', proof)
    return proof
