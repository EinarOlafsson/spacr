"""First inspect real Data Manager plans; never delete or archive in this pass."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

from manager_data import verify_bind, verify_crop_plan, verify_original


def record_manager(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QFileDialog, QDialogButtonBox, QLineEdit

    inputs = json.loads((Path(stage)/'data_manager_state'/f'{Path(captures).name}.json').read_text())
    source = verify_bind(inputs)
    deadline = time.monotonic() + timeout
    proof = {'lesson': '44_data_manager', 'accepted': False, 'scope': 'Real scan and plans only',
             'inputs': inputs, 'deleted_files': [], 'archive_executed': False,
             'app_source_modified': False, 'analysis_run': False, 'published': False}
    path = Path(captures)/'manager_acceptance.json'
    screen = None

    def tick():
        if time.monotonic() > deadline:
            raise TimeoutError('Bounded Data Manager capture timed out')

    def wait_for(predicate):
        while not predicate():
            tick()
            settle(.05)

    def click(widget):
        tick()
        if not widget.isVisible() or not widget.isEnabled():
            raise ValueError('Requested Data Manager control is unavailable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.15)

    def done():
        wait_for(lambda: not screen._busy and not screen._jobs)
        if not outcomes or not all(outcomes):
            raise ValueError('A genuine Data Manager worker failed')

    def button(text):
        matches = [w for w in screen.findChildren(QAbstractButton) if w.isVisible() and w.text() == text]
        if len(matches) != 1:
            raise ValueError(f'Expected a unique actual button: {text}')
        return matches[0]

    def choose_folder(control, directory, frame):
        accepted, errors = [], []
        timer, watchdog = QTimer(window), QTimer(window)
        timer.setSingleShot(True)
        watchdog.setSingleShot(True)

        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise ValueError('Expected the genuine folder picker')
                dialog.accepted.connect(lambda: accepted.append(True))
                edit = dialog.findChild(QLineEdit, 'fileNameEdit')
                click(edit)
                QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
                QTest.keyClicks(edit, str(directory))
                capture(frame)
                buttons = dialog.findChild(QDialogButtonBox)
                choices = [b for b in buttons.buttons()
                           if buttons.buttonRole(b) == QDialogButtonBox.AcceptRole]
                if len(choices) != 1:
                    raise ValueError('The folder dialog has no unique accept control')
                click(choices[0])
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:
                    dialog.reject()

        def abort():
            errors.append('The genuine folder picker timed out')
            dialog = app.activeModalWidget()
            if dialog is not None:
                dialog.reject()

        timer.timeout.connect(handle)
        watchdog.timeout.connect(abort)
        timer.start(300)
        watchdog.start(15000)
        try:
            click(control)
        finally:
            timer.stop()
            watchdog.stop()
            timer.deleteLater()
            watchdog.deleteLater()
        if errors or not accepted:
            raise ValueError('; '.join(errors) or 'Folder picker did not accept the directory')

    def tab(index):
        bar = screen.tabs.tabBar()
        QTest.mouseClick(bar, Qt.LeftButton, pos=bar.tabRect(index).center())
        settle(.2)
        if screen.tabs.currentIndex() != index:
            raise ValueError('Actual Data Manager tab selection failed')

    outcomes = []
    try:
        action = next(a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help')
        menu = action.menu()
        choices = [a for a in menu.actions() if a.text().replace('&', '') == 'Data manager']
        if len(choices) != 1:
            raise ValueError('No unique Help -> Data manager route')
        QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                         pos=window.menuBar().actionGeometry(action).center())
        settle(.2)
        capture('01_help_data_manager')
        QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(choices[0]).center())
        wait_for(lambda: window._screens.get('data_manager') is not None)
        screen = window._screens['data_manager']
        screen.job_finished.connect(lambda ok: outcomes.append(bool(ok)))
        if not screen._threaded:
            raise ValueError('Expected the normal threaded Data Manager')
        choose_folder(button('Choose project…'), source, '02_choose_private_project')
        done()
        if screen.project != str(source):
            raise ValueError('Data Manager selected a different project')
        proof['usage'] = asdict(screen.usage)
        proof['usage_table'] = [[screen.usage_table.item(r, c).text()
                                for c in range(screen.usage_table.columnCount())]
                               for r in range(screen.usage_table.rowCount())]
        capture('03_actual_usage')
        tab(1)
        capture('04_prune_choices')
        click(screen.plan_button)
        done()
        proof['prune_plan'] = asdict(screen.plan)
        capture('05_real_prune_plan')
        # Resolve and verify targets read-only before any later recording is
        # permitted to demonstrate deletion. This first pass never deletes.
        proof['independent_crop_plan'] = verify_crop_plan(screen.plan, inputs)
        proof['kept_table'] = [[screen.kept_table.item(r, c).text()
                               for c in range(screen.kept_table.columnCount())]
                              for r in range(screen.kept_table.rowCount())]
        capture('06_kept_reasons')
        tab(2)
        choose_folder(button('Choose destination…'), inputs['archive'], '07_choose_archive_destination')
        click(screen.archive_plan_button)
        done()
        proof['archive_plan'] = asdict(screen._archive_plan)
        capture('08_actual_archive_plan')
        proof['original_preservation'] = verify_original(inputs)
        proof['worker_outcomes'] = outcomes
        proof['remaining_workers'] = len(screen._jobs)
        proof['accepted'] = True
    finally:
        write_json(path, proof)
        if screen is not None:
            screen.close()
