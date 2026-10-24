"""Record genuine Data Manager controls only on a verified disposable project."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import capture_ui
from capture_ui_drv_c import Scenes
from manager_data import (verify_bind, verify_crop_plan, verify_original, verify_pruned_files,
                          verify_pruned_registry, registry_rows, verify_archive_plan,
                          verify_archived_files)
from capture_report import snapshot_source, require_unchanged


def record_manager(app, window, stage, captures, capture, settle, write_json, timeout, *, execute=False):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QFileDialog, QDialogButtonBox, QLineEdit, QMessageBox
    from spacr.qt.screens.data_manager import ConfirmDeleteDialog

    scenes = Scenes(capture, settle)
    inputs = json.loads((Path(stage)/'data_manager_state'/f'{Path(captures).name}.json').read_text(encoding="utf-8"))
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
                if not capture_ui.live(capture):
                    capture(frame)      # host: the open clip shows the typing
                buttons = dialog.findChild(QDialogButtonBox)
                choices = [b for b in buttons.buttons()
                           if buttons.buttonRole(b) == QDialogButtonBox.AcceptRole]
                if len(choices) != 1:
                    raise ValueError('The folder dialog has no unique accept control')
                capture_ui.accept_file_dialog(capture, dialog, choices[0])
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
        if capture_ui.live(capture):
            capture_ui.when_modal(capture, 300, handle)
        else:
            timer.start(300)
        watchdog.start(120000 if capture_ui.live(capture) else 15000)
        scenes.begin(frame)
        try:
            click(control)
        finally:
            timer.stop()
            watchdog.stop()
            timer.deleteLater()
            watchdog.deleteLater()
        if errors or not accepted:
            raise ValueError('; '.join(errors) or 'Folder picker did not accept the directory')
        if capture_ui.live(capture):
            scenes.end(frame, control)

    def tab(index):
        capture_ui.select_tab(capture, screen.tabs, index)
        settle(.2)
        if screen.tabs.currentIndex() != index:
            raise ValueError('Actual Data Manager tab selection failed')

    def modal(control, handler):
        errors, handled = [], []
        timer, watchdog = QTimer(window), QTimer(window)
        timer.setSingleShot(True)
        watchdog.setSingleShot(True)

        def handle():
            dialog = app.activeModalWidget()
            try:
                handler(dialog)
                handled.append(True)
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:
                    dialog.reject()

        def abort():
            errors.append('Actual confirmation dialog timed out')
            dialog = app.activeModalWidget()
            if dialog is not None:
                dialog.reject()

        timer.timeout.connect(handle)
        watchdog.timeout.connect(abort)
        if capture_ui.live(capture):
            capture_ui.when_modal(capture, 300, handle)
        else:
            timer.start(300)
        watchdog.start(120000 if capture_ui.live(capture) else 30000)
        try:
            click(control)
        finally:
            timer.stop()
            watchdog.stop()
            timer.deleteLater()
            watchdog.deleteLater()
        if errors or not handled:
            raise ValueError('; '.join(errors) or 'Confirmation was not handled')

    def confirm_delete(dialog, accept):
        if not isinstance(dialog, ConfirmDeleteDialog):
            raise ValueError('Expected the genuine file-list confirmation dialog')
        checked = verify_crop_plan(screen.plan, inputs)
        wait_for(lambda: dialog.acknowledged.isEnabled())
        listing = dialog.listing.toPlainText()
        shown_files = [line.strip() for line in listing.splitlines() if line.strip().startswith(str(source)+'/')]
        if sorted(shown_files) != checked['file_list']:
            raise ValueError('The confirmation does not show the exact complete crop file list')
        delete = dialog.buttons.button(QDialogButtonBox.Ok)
        if dialog.acknowledged.isChecked() or delete.isEnabled():
            raise ValueError('Delete is not initially gated by acknowledgment')
        # Resize the actual resizable dialog; no text or widget policy is patched.
        capture_ui.dialog_geometry(capture, dialog, (3300, 1700),
                                   window.geometry().center() - __import__('PySide6.QtCore', fromlist=['QPoint']).QPoint(1650, 850))
        settle(.3)
        if not window.screen().availableGeometry().contains(dialog.frameGeometry()):
            raise ValueError('The complete confirmation dialog must fit the recorded desktop')
        if not accept:
            scenes.end('09_confirmation_locked', dialog.listing)
            scenes.begin('10_complete_file_list_end')
            scroll = dialog.listing.verticalScrollBar()
            capture_ui.scroll_to_value(capture, dialog.listing, scroll.maximum())
            settle(.2)
            scenes.end('10_complete_file_list_end', dialog.listing)
            scenes.begin('11_acknowledged_not_deleted')
        click(dialog.acknowledged)
        if not delete.isEnabled():
            raise ValueError('Acknowledgment did not enable the real Delete control')
        if not accept:
            scenes.end('11_acknowledged_not_deleted', dialog.acknowledged)
            scenes.begin('12_cancel_preserves_every_file')
            click(dialog.acknowledged)
            if delete.isEnabled():
                raise ValueError('Removing acknowledgment did not disable Delete')
            click(dialog.buttons.button(QDialogButtonBox.Cancel))
        else:
            verify_crop_plan(screen.plan, inputs)
            scenes.end('13_confirm_only_private_crops', dialog.listing)
            scenes.begin('14_actual_cleanup_result')
            click(delete)
        proof.setdefault('delete_dialogs', []).append({'accepted': accept, 'listed_files': len(shown_files),
                                                     'initially_locked': True, 'acknowledgment_enabled_delete': True})

    def confirm_archive(dialog, accept):
        if not isinstance(dialog, QMessageBox) or dialog.windowTitle() != 'Archive this project':
            raise ValueError('Expected the genuine archive question')
        verify_archive_plan(screen._archive_plan, inputs)
        if (str(source) not in dialog.text() or inputs['archive'] not in dialog.text() or
                dialog.defaultButton() != dialog.button(QMessageBox.No)):
            raise ValueError('Archive question has different targets or an unsafe default')
        scenes.end('19_archive_confirm_yes' if accept else '17_archive_default_no', dialog)
        scenes.begin('20_actual_archive_result' if accept else '18_archive_cancel_keeps_project')
        click(dialog.button(QMessageBox.Yes if accept else QMessageBox.No))
        proof.setdefault('archive_dialogs', []).append({'accepted': accept, 'default_no': True})

    outcomes = []
    try:
        action = next(a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help')
        menu = action.menu()
        choices = [a for a in menu.actions()
                   if a.text().replace('&', '').casefold() == 'data manager']
        if len(choices) != 1:
            raise ValueError('No unique Help -> Data manager route')
        scenes.begin('01_help_data_manager')
        QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                         pos=window.menuBar().actionGeometry(action).center())
        settle(.2)
        scenes.end('01_help_data_manager', menu)
        scenes.begin('02_choose_private_project')
        QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(choices[0]).center())
        wait_for(lambda: window._screens.get('data_manager') is not None)
        screen = window._screens['data_manager']
        screen.job_finished.connect(lambda ok: outcomes.append(bool(ok)))
        if not screen._threaded:
            raise ValueError('Expected the normal threaded Data Manager')
        choose_folder(button('Choose project…'), source, '02_choose_private_project')
        scenes.begin('03_actual_usage')
        done()
        if screen.project != str(source):
            raise ValueError('Data Manager selected a different project')
        proof['usage'] = asdict(screen.usage)
        proof['usage_table'] = [[screen.usage_table.item(r, c).text()
                                for c in range(screen.usage_table.columnCount())]
                               for r in range(screen.usage_table.rowCount())]
        scenes.end('03_actual_usage', screen.usage_table)
        scenes.begin('04_prune_choices')
        tab(1)
        scenes.end('04_prune_choices', screen.plan_button)
        scenes.begin('05_real_prune_plan')
        click(screen.plan_button)
        done()
        scenes.end('05_real_prune_plan', screen.plan_button)
        proof['prune_plan'] = asdict(screen.plan)
        # Resolve and verify targets read-only before any later recording is
        # permitted to demonstrate deletion. This first pass never deletes.
        proof['independent_crop_plan'] = verify_crop_plan(screen.plan, inputs)
        proof['kept_table'] = [[screen.kept_table.item(r, c).text()
                               for c in range(screen.kept_table.columnCount())]
                              for r in range(screen.kept_table.rowCount())]
        scenes.begin('06_kept_reasons')
        scenes.end('06_kept_reasons', screen.kept_table)
        scenes.begin('07_choose_archive_destination')
        tab(2)
        choose_folder(button('Choose destination…'), inputs['archive'], '07_choose_archive_destination')
        scenes.begin('08_actual_archive_plan')
        click(screen.archive_plan_button)
        done()
        scenes.end('08_actual_archive_plan', screen.archive_plan_button)
        proof['archive_plan'] = asdict(screen._archive_plan)
        if execute:
            proof['scope'] = 'Real confirmation, cancellation, cleanup and archive of the independently bound private copy only'
            scenes.begin('09_confirmation_locked')
            tab(1)
            before_cancel = snapshot_source(source)
            modal(screen.delete_button, lambda dialog: confirm_delete(dialog, False))
            done()
            scenes.end('12_cancel_preserves_every_file', screen.delete_button)
            require_unchanged(before_cancel, snapshot_source(source))
            proof['cancel_preserves_every_file'] = True
            checked = verify_crop_plan(screen.plan, inputs)
            scenes.begin('13_confirm_only_private_crops')
            modal(screen.delete_button, lambda dialog: confirm_delete(dialog, True))
            done()
            scenes.end('14_actual_cleanup_result', screen.freed_label)
            after_prune = snapshot_source(source)
            proof['cleanup'] = verify_pruned_files(before_cancel, after_prune)
            proof['deleted_files'] = checked['file_list']
            proof['prune_registry_marks'] = verify_pruned_registry(
                inputs['artifact_rows'], registry_rows(source), checked['bytes'])
            proof['freed_label'] = screen.freed_label.text()
            if '104' not in proof['freed_label'] or (source/'data').exists():
                raise ValueError('GUI cleanup outcome differs from the scoped files')
            scenes.begin('15_usage_after_cleanup')
            tab(0)
            scenes.end('15_usage_after_cleanup', screen.usage_table)
            proof['usage_after_prune'] = asdict(screen.usage)
            scenes.begin('16_updated_archive_plan')
            tab(2)
            click(screen.archive_plan_button)
            done()
            scenes.end('16_updated_archive_plan', screen.archive_plan_button)
            before_archive = verify_archive_plan(screen._archive_plan, inputs)
            proof['post_cleanup_archive_plan'] = asdict(screen._archive_plan)
            scenes.begin('17_archive_default_no')
            modal(screen.archive_button, lambda dialog: confirm_archive(dialog, False))
            done()
            scenes.end('18_archive_cancel_keeps_project', screen.archive_button)
            require_unchanged(before_archive, snapshot_source(source))
            if any(Path(inputs['archive']).iterdir()):
                raise ValueError('Cancelled archive wrote destination files')
            proof['archive_cancel_preserves_every_file'] = True
            scenes.begin('19_archive_confirm_yes')
            modal(screen.archive_button, lambda dialog: confirm_archive(dialog, True))
            done()
            scenes.end('20_actual_archive_result', screen.archive_button)
            destination = Path(inputs['archive'])
            proof['archive_files'] = verify_archived_files(before_archive, snapshot_source(destination),
                                                          snapshot_source(source))
            manifest = json.loads((destination/'spacr_archive.json').read_text(encoding="utf-8"))
            ledger = json.loads((source/'spacr_archive_log.json').read_text(encoding="utf-8"))
            plan = proof['post_cleanup_archive_plan']
            keys = ('origin', 'destination', 'whole_project', 'total_bytes', 'total_files')
            expected = {'origin': str(source), **{key: plan[key] for key in keys if key != 'origin'}}
            if (any(manifest.get(k) != v for k, v in expected.items()) or len(ledger) != 1 or
                    any(ledger[0].get(k) != v for k, v in expected.items())):
                raise ValueError('Archive manifest or origin ledger has wrong source, destination or totals')
            proof['archive_records'] = {'manifest': str(destination/'spacr_archive.json'),
                                        'ledger': str(source/'spacr_archive_log.json'), **expected}
            proof['archived_registry_rows'] = registry_rows(destination)
            proof['archive_executed'] = True
            choose_folder(button('Choose project…'), destination, '21_choose_archived_project')
            scenes.begin('22_rescan_archived_project')
            done()
            tab(0)
            proof['archived_usage'] = asdict(screen.usage)
            scenes.end('22_rescan_archived_project', screen.usage_table)
        proof['original_preservation'] = verify_original(inputs)
        proof['worker_outcomes'] = outcomes
        proof['remaining_workers'] = len(screen._jobs)
        proof['accepted'] = True
    finally:
        write_json(path, proof)
        if screen is not None:
            screen.close()
