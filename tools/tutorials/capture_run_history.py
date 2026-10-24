"""Inspect genuine existing tutorial journals without deleting or rerunning."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import capture_ui
from capture_ui_drv_c import Scenes
from history_evidence import read_record, snapshot, verify_panels, verify_preserved, verify_visible

RECRUITMENT = '2026-09-09_201318_6b415b85__recruitment'
REGRESSION = '2026-09-09_154327_c79bb2f6__regression'
FAILED = '2026-09-09_210220_6f581c02__job'
BATCH = ('2026-09-10_035702_b2c28a18__convert',
         '2026-09-10_035705_c03d4a59__convert',
         '2026-09-10_035706_e2d6090c__convert')


def record_history(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QSplitter

    scenes = Scenes(capture, settle)
    root = Path(stage) / 'runs'
    actual_root = Path.home() / '.spacr/runs'
    if (root.stat().st_dev, root.stat().st_ino) != (actual_root.stat().st_dev, actual_root.stat().st_ino):
        raise ValueError('History is not bound to the isolated tutorial journal')
    before = snapshot(root)
    if os.environ.get('SPACR_TUTORIAL_HISTORY_MODE') == 'restore_latest':
        return _record_restore_latest(app, window, root, actual_root, before, captures,
                                      capture, settle, write_json, timeout)
    records = {rid: read_record(root, rid) for rid in (*BATCH, RECRUITMENT, REGRESSION, FAILED)}
    proof = {'lesson': '40_run_history', 'accepted': False, 'published': False,
             'app_source_modified': False, 'journal_before': before,
             'analysis_run': False, 'deleted_runs': 0, 'ai_request_sent': False}
    path = Path(captures) / 'history_acceptance.json'
    write_json(path, proof)
    deadline = time.monotonic() + timeout
    screen = None

    def tick():
        if time.monotonic() > deadline:
            raise TimeoutError('Bounded Run History capture timed out')

    def click(widget):
        tick()
        if not widget.isVisible() or not widget.isEnabled():
            raise ValueError('Requested History control is not usable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.rect().center())
        settle(.12)

    def wait_for(predicate):
        while not predicate():
            tick()
            settle(.05)

    def fill(text):
        click(screen._search)
        QTest.keyClick(screen._search, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(screen._search, Qt.Key_Backspace)
        QTest.keyClicks(screen._search, text)
        settle(.2)
        if screen._search.text() != text:
            raise ValueError('Actual search input differs')

    def choose(combo, value):
        for _ in range(300):                     # the journal is read off the GUI thread; the choices fill when it lands
            if combo.findData(value) >= 0:
                break
            settle(.1)
        index = combo.findData(value)
        if index < 0:
            raise ValueError('No such actual filter choice')
        capture_ui.select_combo(capture, combo, index)
        settle(.2)
        if combo.currentData() != value:
            raise ValueError('Native filter selected the wrong value')

    def visible(expected):
        return verify_visible([screen._table.item(i, 0).data(Qt.UserRole)
                               for i in range(screen._table.rowCount())
                               if not screen._table.isRowHidden(i)], expected)    # filtered rows are hidden, not removed

    def panels(rid):
        raw = {'overview': screen._overview.toPlainText(),
               'settings': screen._settings.toPlainText(),
               'files': screen._outputs.toPlainText(),
               'problems': screen._problems.toPlainText(),
               'environment': screen._environment.toPlainText()}
        proof.setdefault('panel_checks', []).append(verify_panels(records[rid], raw))
        return raw

    def tab(index, frame):
        scenes.begin(frame)
        capture_ui.select_tab(capture, screen._tabs, index)
        settle(.2)
        if screen._tabs.currentIndex() != index:
            raise ValueError('The requested native detail tab did not open')
        scenes.end(frame, screen._tabs)

    def scroll_bottom(widget):
        bar = widget.verticalScrollBar()
        capture_ui.scroll_to_value(capture, widget, bar.maximum())
        settle(.15)
        if bar.value() != bar.maximum():
            raise ValueError('The native text scrollbar did not reach the end')

    try:
        help_action = next(a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help')
        menu = help_action.menu()
        choices = [a for a in menu.actions() if a.text().replace('&', '').lower() == 'run history']
        if len(choices) != 1:
            raise ValueError('No unique Help -> Run history route')
        scenes.begin('01_help_history')
        QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                         pos=window.menuBar().actionGeometry(help_action).center())
        settle(.2)
        scenes.end('01_help_history', menu)
        scenes.begin('02_loaded_history')
        QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(choices[0]).center())
        wait_for(lambda: window._screens.get('run_history') is not None)
        screen = window._screens['run_history']
        wait_for(lambda: screen._loaded_once and not screen._busy and not screen._jobs)
        if screen.last_error or not screen._threaded:
            raise ValueError('Normal threaded History refresh did not succeed')
        proof['loaded_records'] = len(screen.records)
        proof['status_choices'] = [screen._status_filter.itemText(i)
                                  for i in range(screen._status_filter.count())]
        scenes.end('02_loaded_history', screen._table)
        scenes.begin('03_batch_search')
        # Move the genuine, user-resizable divider to enlarge the detail panes.
        splitter = screen.findChild(QSplitter)
        handle = splitter.handle(1)
        QTest.mousePress(handle, Qt.LeftButton, pos=handle.rect().center())
        QTest.mouseMove(handle, handle.rect().center() - QPoint(0, 450), 150)
        QTest.mouseRelease(handle, Qt.LeftButton)
        settle(.3)
        proof['splitter_sizes'] = splitter.sizes()
        choose(screen._module, 'convert')
        choose(screen._status_filter, 'success')
        fill('real-channels-u0zjboui')
        proof['batch_filter_ids'] = visible(BATCH)
        scenes.end('03_batch_search', screen._search)
        rid = screen._table.item(screen._table.currentRow(), 0).data(Qt.UserRole)
        panels(rid)
        tab(2, '04_batch_empty_inventory')
        # These real conversions succeeded but skipped file hashing. Empty
        # dictionaries are not evidence that the jobs produced no images.
        if records[rid]['manifest']['input_hashing'] != 'skipped':
            raise ValueError('The selected conversion no longer demonstrates skipped hashing')
        scenes.begin('05_no_match')
        fill('no-such-tutorial-run-7d40a281')
        visible([])
        if any(w.toPlainText() for w in (screen._overview, screen._settings, screen._outputs,
                                         screen._problems, screen._environment)):
            raise ValueError('No-match search left stale detail text')
        if any(w.isEnabled() for w in (screen._copy_path, screen._open_folder, screen._load_settings)):
            raise ValueError('No-match search left selected-run actions enabled')
        scenes.end('05_no_match', screen._search)
        scenes.begin('06_restored_search')
        fill('real-channels-u0zjboui')
        visible(BATCH)
        panels(screen._table.item(screen._table.currentRow(), 0).data(Qt.UserRole))
        scenes.end('06_restored_search', screen._table.viewport())

        scenes.begin('07_recruitment_overview')
        choose(screen._module, 'recruitment')
        fill(RECRUITMENT)
        visible([RECRUITMENT])
        raw = panels(RECRUITMENT)
        tab(0, '07_recruitment_overview')
        tab(1, '08_resolved_settings')
        scenes.begin('09_settings_source')
        scroll_bottom(screen._settings)
        scenes.end('09_settings_source', screen._settings)
        tab(2, '10_recorded_files')
        tab(4, '11_environment')
        scenes.begin('12_hashes_and_seeds')
        scroll_bottom(screen._environment)
        scenes.end('12_hashes_and_seeds', screen._environment)
        scenes.begin('13_copy_run_path')
        click(screen._copy_path)
        copied = app.clipboard().text()
        from spacr.run_journal import runs_root     # the app reports its journal under SPACR_HOME, the same folder as ~/.spacr/runs here
        if copied not in {str(actual_root / RECRUITMENT), str(runs_root() / RECRUITMENT)}:
            raise ValueError('Copy path returned a different journal')
        proof['copied_run_path'] = copied
        scenes.end('13_copy_run_path', screen._copy_path)

        scenes.begin('14_recorded_warning')
        choose(screen._module, 'regression')
        fill(REGRESSION)
        visible([REGRESSION])
        panels(REGRESSION)
        tab(3, '14_recorded_warning')
        scenes.begin('15_actual_failed_job')
        choose(screen._module, '')
        choose(screen._status_filter, 'failed')
        fill(FAILED)
        visible([FAILED])
        panels(FAILED)
        tab(3, '15_actual_failed_job')
        scenes.begin('16_failure_exception')
        scroll_bottom(screen._problems)
        scenes.end('16_failure_exception', screen._problems)

        scenes.begin('17_ready_to_restore')
        choose(screen._status_filter, 'success')
        choose(screen._module, 'recruitment')
        fill(RECRUITMENT)
        visible([RECRUITMENT])
        panels(RECRUITMENT)
        click(screen._refresh)
        wait_for(lambda: not screen._busy and not screen._jobs)
        if screen.last_error:
            raise ValueError('The second real refresh failed')
        visible([RECRUITMENT])
        panels(RECRUITMENT)
        tab(1, '17_ready_to_restore')
        scenes.begin('18_restored_recruitment')
        click(screen._load_settings)
        wait_for(lambda: window._screens.get('recruitment') is not None)
        target = window._screens['recruitment']
        wait_for(lambda: getattr(target, '_settings_model', None) is not None)
        settle(.5)
        collected = target._settings_model.collect()
        original = records[RECRUITMENT]['settings']
        proof['restored_settings'] = collected
        proof['restoration_differences'] = {k: [v, collected.get(k)] for k, v in original.items()
                                             if k in collected and collected[k] != v}
        proof['settings_not_represented'] = sorted(set(original) - set(collected))
        proof['restored_matching_keys'] = sorted(k for k, v in original.items()
                                                if k in collected and collected[k] == v)
        scenes.end('18_restored_recruitment', target)
        if proof['restoration_differences']:
            # Preserve evidence of current limitations; never silently fix an
            # application-owned seed handler to make the recording succeed.
            proof['exact_restoration'] = False
        else:
            proof['exact_restoration'] = not proof['settings_not_represented']
        if target._settings_model.collect().get('src') != original['src']:
            raise ValueError('Load settings did not even restore the source')
        proof['journal_preservation'] = verify_preserved(root, before)
        proof['remaining_refresh_workers'] = len(screen._jobs)
        proof['accepted'] = True
    finally:
        write_json(path, proof)
        if screen is not None:
            screen.close()


def _record_restore_latest(app, window, root, actual_root, before, captures, capture,
                           settle, write_json, timeout):
    """Help -> Run history over Home, then restore the newest recruitment run.

    Uses whatever genuine journals the neutral recording profile holds (for
    example a recruitment run recorded in the same stage), so no run ID or
    maintainer path is baked into the recording.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    scenes = Scenes(capture, settle)
    proof = {'lesson': '40_run_history', 'mode': 'restore_latest', 'accepted': False,
             'published': False, 'app_source_modified': False, 'analysis_run': False,
             'deleted_runs': 0, 'journal_before': before}
    path = Path(captures) / 'history_acceptance.json'
    deadline = time.monotonic() + timeout

    def wait_for(predicate):
        while not predicate():
            if time.monotonic() > deadline:
                raise TimeoutError('Bounded Run History capture timed out')
            settle(.05)

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled():
            raise ValueError('Requested History control is not usable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.rect().center())
        settle(.12)

    screen = None
    try:
        help_action = next(a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help')
        menu = help_action.menu()
        choices = [a for a in menu.actions() if a.text().replace('&', '').lower() == 'run history']
        if len(choices) != 1:
            raise ValueError('No unique Help -> Run history route')
        scenes.begin('01_help_history')
        QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                         pos=window.menuBar().actionGeometry(help_action).center())
        settle(.2)
        scenes.end('01_help_history', menu)
        scenes.begin('02_loaded_history')
        QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(choices[0]).center())
        wait_for(lambda: window._screens.get('run_history') is not None)
        screen = window._screens['run_history']
        wait_for(lambda: screen._loaded_once and not screen._busy and not screen._jobs)
        if screen.last_error:
            raise ValueError('History refresh failed')
        combo = screen._module
        index = combo.findData('recruitment')
        if index < 0:
            raise ValueError('No recruitment run in the neutral journal')
        scenes.begin('17_ready_to_restore')
        capture_ui.select_combo(capture, combo, index)
        settle(.4)
        shown = [i for i in range(screen._table.rowCount()) if not screen._table.isRowHidden(i)]
        rows = len(shown)
        if rows < 1:
            raise ValueError('No recruitment row to restore')
        if capture_ui.live(capture):
            first = screen._table.item(shown[0], 0)
            QTest.mouseClick(screen._table.viewport(), Qt.LeftButton,
                             pos=screen._table.visualItemRect(first).center())
        else:
            screen._table.selectRow(shown[0])
        settle(.4)
        rid = screen._table.item(shown[0], 0).data(Qt.UserRole)
        record = read_record(root, rid)
        capture_ui.select_tab(capture, screen._tabs, 1)
        settle(.3)
        scenes.end('17_ready_to_restore', screen._tabs)
        scenes.begin('18_restored_recruitment')
        click(screen._load_settings)
        wait_for(lambda: window._screens.get('recruitment') is not None)
        target = window._screens['recruitment']
        wait_for(lambda: getattr(target, '_settings_model', None) is not None)
        settle(.8)
        collected = target._settings_model.collect()
        if collected.get('src') != record['settings'].get('src'):
            raise ValueError('Load settings did not restore the source')
        scenes.end('18_restored_recruitment', target)
        proof.update(run_id=rid, restored_src=collected.get('src'),
                     journal_preservation=verify_preserved(root, before), accepted=True)
    finally:
        write_json(path, proof)
        if screen is not None:
            screen.close()
