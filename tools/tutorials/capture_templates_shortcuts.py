"""Record Settings templates (657) and Change shortcuts (656) on Measure.

Runs after Measure's test data has loaded (``--module measure --download
--templates-shortcuts-tour``). Every control is the real one: the Templates
button on the settings strip, its Save, Import (a real settings CSV a run
wrote, copied into the private stage), Rename, then F1's cheat sheet and its
Change shortcuts… editor, closed with Cancel so no key is changed.
"""
from __future__ import annotations

from pathlib import Path


def record_templates_shortcuts(app, window, screen, stage, captures, capture, settle,
                               write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QInputDialog,
                                   QLineEdit, QListWidget, QPushButton, QToolButton)

    from spacr.qt import recipes
    from spacr.qt.recipes import RecipeDialog

    csv_path = Path(stage) / 'settings_examples' / 'measure_crop_settings.csv'
    if not csv_path.is_file():
        raise RuntimeError('Copy a real Measure settings CSV into settings_examples/ first')
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
        edit = picker.findChild(QLineEdit, 'fileNameEdit')
        edit.selectAll()
        QTest.keyClicks(edit, str(csv_path))
        settle(.3)
        capture('32_template_import_csv')
        picker.findChild(QDialogButtonBox).button(QDialogButtonBox.Open).click()
    errors = later(pick_csv, 900)
    click(dialog._btn_import)
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
    proof['accepted'] = True
    write_json(captures / 'templates_shortcuts.json', proof)
