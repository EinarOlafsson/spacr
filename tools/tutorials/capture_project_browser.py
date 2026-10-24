"""Capture actual Project Browser discovery and sorting without changing data."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import capture_ui
from capture_ui_drv_c import Scenes
from capture_report import snapshot_source, verify_private_sqlite_changes
from manager_data import verify_bind
from project_browser_data import verify_originals, verify_summary


def record_browser(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer, QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QDialogButtonBox, QLineEdit, QSplitter

    scenes = Scenes(capture, settle)
    inputs = json.loads((Path(stage)/'project_browser_state'/f'{Path(captures).name}.json').read_text(encoding="utf-8"))
    registered = verify_bind(inputs)
    conversion = Path(inputs['conversion_copy'])
    deadline = time.monotonic()+timeout
    proof = {'lesson': '45_project_browser', 'accepted': False, 'inputs': inputs,
             'app_source_modified': False, 'analysis_run': False, 'published': False}
    screen = None
    errors, scans, chosen = [], [], []

    def tick():
        if time.monotonic() > deadline:
            raise TimeoutError('Bounded Project Browser capture timed out')

    def wait_for(predicate):
        while not predicate():
            tick()
            settle(.05)

    def click(widget):
        tick()
        if not widget.isVisible() or not widget.isEnabled():
            raise ValueError('Actual Project Browser control is unavailable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.2)

    def done():
        wait_for(lambda: not screen.is_busy() and not screen.active_jobs())
        if errors:
            raise ValueError('; '.join(errors))

    def add_folder(directory, frame):
        accepted, failures = [], []
        timer, watchdog = QTimer(window), QTimer(window)
        timer.setSingleShot(True)
        watchdog.setSingleShot(True)

        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise ValueError('Expected actual Add folder dialog')
                dialog.accepted.connect(lambda: accepted.append(True))
                edit = dialog.findChild(QLineEdit, 'fileNameEdit')
                click(edit)
                QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
                QTest.keyClicks(edit, str(directory))
                if not capture_ui.live(capture):
                    capture(frame)      # host: the open clip shows the typing
                box = dialog.findChild(QDialogButtonBox)
                buttons = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
                if len(buttons) != 1:
                    raise ValueError('Expected exactly one actual folder accept button')
                capture_ui.accept_file_dialog(capture, dialog, buttons[0])
            except Exception as error:
                failures.append(str(error))
                if dialog is not None:
                    dialog.reject()

        def abort():
            failures.append('The actual folder picker timed out')
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
            click(screen._add)
        finally:
            timer.stop()
            watchdog.stop()
            timer.deleteLater()
            watchdog.deleteLater()
        if failures or not accepted:
            raise ValueError('; '.join(failures) or 'Actual folder picker was not accepted')
        done()
        if capture_ui.live(capture):
            scenes.end(frame, screen._add)

    def depth(value):
        click(screen._depth)
        QTest.keyClick(screen._depth, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(screen._depth, str(value))
        QTest.keyClick(screen._depth, Qt.Key_Tab)
        if screen._depth.value() != value:
            raise ValueError('Actual depth control did not change')
        click(screen._rescan)
        done()

    def roots_in_table():
        return [str(screen._table.item(row, 0).data(Qt.UserRole)) for row in range(screen._table.rowCount())]

    def expect_roots(expected):
        actual = roots_in_table()
        if len(actual) != len(expected) or set(actual) != set(map(str, expected)):
            raise ValueError('Displayed discovery roots differ from the scoped actual folders')
        return actual

    def select(root):
        row = roots_in_table().index(str(root))
        item = screen._table.item(row, 0)
        if screen.selected_root() == str(root) and not screen._detail.toPlainText() and screen._table.rowCount() > 1:
            # The real rescan clears details but can retain the selected cell.
            # Selecting another real row and returning emits selectionChanged;
            # do not call show_detail or patch the application's behavior.
            proof.setdefault('rescan_selected_row_empty_detail', []).append(str(root))
            other = screen._table.item((row+1) % screen._table.rowCount(), 0)
            QTest.mouseClick(screen._table.viewport(), Qt.LeftButton,
                             pos=screen._table.visualItemRect(other).center())
            settle(.2)
        QTest.mouseClick(screen._table.viewport(), Qt.LeftButton,
                         pos=screen._table.visualItemRect(item).center())
        settle(.2)
        if screen.selected_root() != str(root) or not screen._detail.toPlainText().startswith(str(root)+'\n'):
            raise ValueError('Selection or detail lost the project identity')
        return item

    def inspect(root, records):
        summary = screen.summary_for(str(root))
        proof.setdefault('summaries', {})[str(root)] = asdict(summary)
        check = verify_summary(summary, root, snapshot_source(root), records)
        detail = screen._detail.toPlainText()
        stamp = (f'from the {summary.last_run_source}' if summary.last_run_utc else 'last run: never')
        if (stamp not in detail or summary.last_run_utc not in detail or 'Stages' not in detail):
            raise ValueError('Details omit timestamp provenance or the stage inventory')
        if not records and 'nothing here can be checked against what produced it' not in detail:
            raise ValueError('Unregistered detail hides its lack of provenance')
        proof.setdefault('details', {})[str(root)] = detail
        proof.setdefault('independent_checks', {})[str(root)] = check
        proof.setdefault('summaries', {})[str(root)] = asdict(summary)

    def sort_size(descending, frame):
        table, header = screen._table, screen._table.horizontalHeader()
        scenes.begin(frame)
        by_root = {s.root: s.size_bytes for s in screen.summaries()}
        expected = sorted(by_root, key=by_root.get, reverse=descending)
        for _ in range(3):
            QTest.mouseClick(header.viewport(), Qt.LeftButton,
                             pos=QPoint(header.sectionViewportPosition(2)+header.sectionSize(2)//2,
                                        header.height()//2))
            settle(.2)
            if roots_in_table() == expected:
                break
        if roots_in_table() != expected:
            raise ValueError('Actual header clicks did not produce numeric byte ordering')
        proof.setdefault('sorts', []).append({'descending': descending, 'roots': expected,
                                             'bytes': [by_root[root] for root in expected]})
        scenes.end(frame, header.viewport())

    try:
        action = next(a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help')
        menu = action.menu()
        matches = [a for a in menu.actions() if a.text().replace('&', '').lower() == 'project browser']
        if len(matches) != 1:
            raise ValueError('No unique Help -> Project browser route')
        scenes.begin('01_help_project_browser')
        QTest.mouseClick(window.menuBar(), Qt.LeftButton, pos=window.menuBar().actionGeometry(action).center())
        settle(.2)
        scenes.end('01_help_project_browser', menu)
        scenes.begin('02_empty_browser')
        QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(matches[0]).center())
        wait_for(lambda: window._screens.get('project_browser') is not None)
        screen = window._screens['project_browser']
        screen.failed.connect(lambda message: errors.append(str(message)))
        screen.scanned.connect(lambda n: scans.append(int(n)))
        screen.project_chosen.connect(lambda path: chosen.append(str(path)))
        done()
        if screen.roots():
            raise ValueError('Fresh isolated Project Browser preferences must start without search roots')
        scenes.end('02_empty_browser', screen._add)
        scenes.begin('03_choose_search_parent')
        depth(0)
        add_folder(inputs['conversion_search'], '03_choose_search_parent')
        scenes.begin('04_depth_zero_no_projects')
        expect_roots([])
        scenes.end('04_depth_zero_no_projects', screen._table.viewport())
        scenes.begin('05_real_unregistered_conversion')
        depth(1)
        expect_roots([conversion])
        select(conversion)
        inspect(conversion, [])
        scenes.end('05_real_unregistered_conversion', screen._detail)
        add_folder(registered, '06_choose_registered_project')
        scenes.begin('07_two_real_projects')
        expect_roots([conversion, registered])
        scenes.end('07_two_real_projects', screen._table.viewport())
        scenes.begin('08_registered_measurements_detail')
        select(registered)
        inspect(registered, inputs['artifact_rows'])
        scenes.end('08_registered_measurements_detail', screen._detail)
        splitter = getattr(screen, '_body_splitter', None) or screen._table.parentWidget()
        if not isinstance(splitter, QSplitter):
            raise ValueError('Expected actual project table/detail splitter')
        scenes.begin('09_enlarged_details')
        handle = splitter.handle(1)
        start = handle.rect().center()
        QTest.mousePress(handle, Qt.LeftButton, pos=start)
        QTest.mouseMove(handle, start-QPoint(400, 0), delay=120)
        QTest.mouseRelease(handle, Qt.LeftButton, pos=start-QPoint(400, 0))
        settle(.3)
        scenes.end('09_enlarged_details', screen._detail)
        scroll = screen._detail.verticalScrollBar()
        scenes.begin('10_stages_and_next_steps')
        capture_ui.scroll_to_value(capture, screen._detail, scroll.maximum())
        settle(.2)
        scenes.end('10_stages_and_next_steps', screen._detail)
        sort_size(False, '11_size_ascending')
        scenes.begin('12_size_descending')
        select(conversion)
        inspect(conversion, [])
        sort_size(True, '12_size_descending')
        scenes.begin('13_double_click_observation')
        item = select(registered)
        QTest.mouseDClick(screen._table.viewport(), Qt.LeftButton,
                         pos=screen._table.visualItemRect(item).center())
        settle(.5)
        proof['double_click_project_chosen'] = chosen
        proof['double_click_browser_still_visible'] = screen.isVisible()
        if chosen != [str(registered)]:
            raise ValueError('Actual double-click did not emit the selected project identity once')
        scenes.end('13_double_click_observation', screen._table.viewport())
        scenes.begin('14_depth_zero_keeps_direct_project')
        depth(0)
        expect_roots([registered])
        scenes.end('14_depth_zero_keeps_direct_project', screen._depth)
        scenes.begin('15_depth_restores_both')
        depth(1)
        expect_roots([conversion, registered])
        scenes.end('15_depth_restores_both', screen._depth)
        # Remove changes only search roots; do not remove a directory.
        scenes.begin('16_remove_search_root_only')
        index = list(screen.roots()).index(str(registered))
        root_item = screen._root_list.item(index)
        QTest.mouseClick(screen._root_list.viewport(), Qt.LeftButton,
                         pos=screen._root_list.visualItemRect(root_item).center())
        click(screen._forget)
        done()
        expect_roots([conversion])
        scenes.end('16_remove_search_root_only', screen._forget)
        add_folder(registered, '17_add_back_unchanged_project')
        scenes.begin('18_restored_project_inventory')
        expect_roots([conversion, registered])
        select(registered)
        inspect(registered, inputs['artifact_rows'])
        scenes.end('18_restored_project_inventory', screen._detail)
        proof['source_preservation'] = verify_originals(inputs)
        proof['private_sqlite_sidecar_changes'] = verify_private_sqlite_changes(
            registered, inputs['clone_files'], snapshot_source(registered))
        proof['scan_counts'] = scans
        proof['remaining_workers'] = screen.active_jobs()
        proof['accepted'] = True
    finally:
        write_json(Path(captures)/'project_browser_acceptance.json', proof)
        if screen is not None:
            screen.close()
