"""Record Settings templates (657) and Change shortcuts (656) on Measure.

Runs after Measure's test data has loaded (``--module measure --download
--templates-shortcuts-tour``). Every control is the real one: the Templates
button on the settings strip, its Save, Import (a real settings CSV a run
wrote, copied into the private stage), Rename, then F1's cheat sheet and its
Change shortcuts… editor, closed with Cancel so no key is changed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import time


def record_templates_shortcuts(app, window, screen, stage, captures, capture, settle,
                               write_json, timeout):
    from PySide6.QtCore import Qt, QTimer, QUrl
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QInputDialog,
                                   QLineEdit, QListWidget, QPushButton, QToolButton, QTreeView)

    from spacr.qt import recipes
    from spacr.qt.recipes import RecipeDialog
    from capture_settings import require_unchanged_settings

    csv_path = Path(stage) / 'settings_examples' / 'measure_crop_settings.csv'
    if not csv_path.is_file():
        raise RuntimeError('Copy a real Measure settings CSV into settings_examples/ first')
    csv_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    settings_before = deepcopy(screen._settings_model.collect())
    proof = {'accepted': False, 'csv': str(csv_path), 'keys_changed': False}

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled():
            raise RuntimeError('A required control is unavailable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.5)

    def later(action, delay=700):
        errors = []

        def run():
            try:
                action()
            except Exception as exc:          # recorded, then raised below
                errors.append(exc)
                modal = app.activeModalWidget()
                if modal is not None:
                    modal.reject()
        QTimer.singleShot(delay, run)
        return errors

    def input_dialog(text, frame):
        def fill():
            dialog = next(d for d in app.topLevelWidgets()
                          if isinstance(d, QInputDialog) and d.isVisible())
            edit = dialog.findChild(QLineEdit)
            edit.selectAll()
            QTest.keyClicks(edit, text)
            settle(.3)
            capture(frame)
            dialog.accept()
        return later(fill)

    if recipes.list_recipes('measure'):
        raise RuntimeError('Record on a fresh profile: Measure already has templates')
    button = screen.findChild(QToolButton, recipes.RECIPE_BUTTON_NAME)
    if button is None:
        raise RuntimeError('Measure has no Templates button')
    click(button)
    dialogs = [d for d in app.topLevelWidgets() if isinstance(d, RecipeDialog) and d.isVisible()]
    if len(dialogs) != 1:
        raise RuntimeError('Templates did not open its dialog')
    dialog = dialogs[0]
    dialog.resize(1400, 900)
    settle(.5)
    capture('30_templates_open')
    errors = input_dialog('Measure example, 4 channels', '31_template_save_name')
    click(dialog._btn_save)
    settle(1)
    if errors:
        raise errors[0]

    def pick_csv():
        picker = next(d for d in app.topLevelWidgets() if isinstance(d, QFileDialog) and d.isVisible())
        picker.setDirectory(str(csv_path.parent))
        picker.setSidebarUrls([QUrl.fromLocalFile(str(csv_path.parent))])
        file_type = picker.findChild(QComboBox, 'fileTypeCombo')
        choices = [file_type.itemText(i) for i in range(file_type.count())]
        wanted = next(i for i, text in enumerate(choices) if text == 'Settings CSV (*.csv)')
        file_type.setFocus()
        QTest.keyClick(file_type, Qt.Key_Home)
        for _ in range(wanted):
            QTest.keyClick(file_type, Qt.Key_Down)
        if file_type.currentText() != 'Settings CSV (*.csv)':
            raise RuntimeError('The actual import filter did not select Settings CSV')
        settle(.4)
        view = picker.findChild(QTreeView, 'treeView')
        deadline = time.monotonic() + timeout
        selected = None
        while selected is None:
            root = view.rootIndex()
            for row in range(view.model().rowCount(root)):
                index = view.model().index(row, 0, root)
                if index.data() == csv_path.name:
                    selected = index
                    break
            if selected is None:
                if time.monotonic() >= deadline:
                    raise RuntimeError('The actual CSV view did not list the completed run settings file')
                settle(.1)
        view.scrollTo(selected)
        settle(.3)
        rect = view.visualRect(selected)
        if rect.isEmpty() or not view.viewport().rect().contains(rect.center()):
            raise RuntimeError('The actual completed-run CSV row is not visible')
        QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=rect.center())
        settle(.3)
        if picker.selectedFiles() != [str(csv_path)]:
            raise RuntimeError('The native CSV row did not select the exact completed-run settings file')
        capture('32_template_import_csv')
        open_button = picker.findChild(QDialogButtonBox).button(QDialogButtonBox.Open)
        if not open_button.isEnabled():
            raise RuntimeError('The actual picker refuses the complete run settings CSV')
        QTest.mouseDClick(view.viewport(), Qt.LeftButton, pos=rect.center())
    errors = later(pick_csv, 900)
    previous_directory = Path.cwd()
    try:
        os.chdir(csv_path.parent)
        click(dialog._btn_import)
    finally:
        os.chdir(previous_directory)
    settle(1.2)
    if errors:
        raise errors[0]
    names = [dialog._list.item(i).text() for i in range(dialog._list.count())]
    if len(names) < 2:
        raise RuntimeError(f'The CSV import did not add a template: {names}')
    imported = next(i for i in range(dialog._list.count())
                    if 'measure_crop_settings' in dialog._list.item(i).text())
    dialog._list.setCurrentRow(imported)
    settle(.4)
    capture('33_templates_imported')
    errors = input_dialog('Measure, last run', '34_template_rename')
    click(dialog._btn_rename)
    settle(1)
    if errors:
        raise errors[0]
    renamed = [dialog._list.item(i).text() for i in range(dialog._list.count())]
    if not any('Measure, last run' in n for n in renamed):
        raise RuntimeError(f'Rename did not take: {renamed}')
    capture('35_templates_renamed')
    proof['templates'] = renamed
    dialog.close()
    settle(.5)

    # Help > Keyboard shortcuts is the visible route to the sheet F1 opens.
    bar = window.menuBar()
    help_action = next(a for a in bar.actions() if a.text().replace('&', '') == 'Help')
    menu = help_action.menu()
    entry = [a for a in menu.actions() if a.text().replace('&', '').startswith('Keyboard shortcuts')]
    if len(entry) != 1:
        raise RuntimeError('Help has no unique Keyboard shortcuts entry')
    QTest.mouseClick(bar, Qt.LeftButton, pos=bar.actionGeometry(help_action).center())
    settle(.4)
    QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(entry[0]).center())
    settle(1)
    overlay = getattr(window, '_spacr_shortcut_overlay', None)
    if overlay is None or not overlay.isVisible():
        raise RuntimeError('F1 did not show the cheat sheet')
    capture('36_cheat_sheet')
    edit = overlay.findChild(QPushButton, 'ShortcutOverlayEdit')
    if edit is None:
        raise RuntimeError('The cheat sheet has no Change shortcuts… button')
    click(edit)
    settle(1)
    keymap = [d for d in app.topLevelWidgets() if isinstance(d, QDialog)
              and d.objectName() == 'KeymapDialog' and d.isVisible()]
    if len(keymap) != 1:
        raise RuntimeError('Change shortcuts… did not open its editor')
    keymap[0].resize(1300, 1100)
    settle(.5)
    capture('37_change_shortcuts')
    keymap[0].reject()
    settle(.5)
    require_unchanged_settings(settings_before, screen._settings_model.collect())
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != csv_hash:
        raise RuntimeError('The genuine run settings CSV changed during template import')
    proof.update(csv_sha256=csv_hash, measure_settings_unchanged=True,
                 template_applied=False, imported_csv_unchanged=True)
    proof['accepted'] = True
    write_json(captures / 'templates_shortcuts.json', proof)
