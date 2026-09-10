"""Inspect genuine existing tutorial journals without deleting or rerunning."""
from __future__ import annotations

import json
from pathlib import Path
import time

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

    root = Path(stage) / 'runs'
    actual_root = Path.home() / '.spacr/runs'
    if (root.stat().st_dev, root.stat().st_ino) != (actual_root.stat().st_dev, actual_root.stat().st_ino):
        raise ValueError('History is not bound to the isolated tutorial journal')
    before = snapshot(root)
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
        index = combo.findData(value)
        if index < 0:
            raise ValueError('No such actual filter choice')
        click(combo)
        view = combo.view()
        QTest.keyClick(view, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(view, Qt.Key_Down)
        QTest.keyClick(view, Qt.Key_Return)
        settle(.2)
        if combo.currentData() != value:
            raise ValueError('Native filter selected the wrong value')

    def visible(expected):
        return verify_visible([screen._table.item(i, 0).data(Qt.UserRole)
                               for i in range(screen._table.rowCount())], expected)

    def panels(rid):
        raw = {'overview': screen._overview.toPlainText(),
               'settings': screen._settings.toPlainText(),
               'files': screen._outputs.toPlainText(),
               'problems': screen._problems.toPlainText(),
               'environment': screen._environment.toPlainText()}
        proof.setdefault('panel_checks', []).append(verify_panels(records[rid], raw))
        return raw

    def tab(index, frame):
        bar = screen._tabs.tabBar()
        QTest.mouseClick(bar, Qt.LeftButton, pos=bar.tabRect(index).center())
        settle(.2)
        if screen._tabs.currentIndex() != index:
            raise ValueError('The requested native detail tab did not open')
        capture(frame)

    def scroll_bottom(widget):
        bar = widget.verticalScrollBar()
        QTest.keyClick(bar, Qt.Key_End)
        settle(.15)
        if bar.value() != bar.maximum():
            raise ValueError('The native text scrollbar did not reach the end')

    try:
        help_action = next(a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help')
        menu = help_action.menu()
        choices = [a for a in menu.actions() if a.text().replace('&', '') == 'Run history']
        if len(choices) != 1:
            raise ValueError('No unique Help -> Run history route')
        QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                         pos=window.menuBar().actionGeometry(help_action).center())
        settle(.2)
        capture('01_help_history')
        QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(choices[0]).center())
        wait_for(lambda: window._screens.get('run_history') is not None)
        screen = window._screens['run_history']
        wait_for(lambda: screen._loaded_once and not screen._busy and not screen._jobs)
        if screen.last_error or not screen._threaded:
            raise ValueError('Normal threaded History refresh did not succeed')
        proof['loaded_records'] = len(screen.records)
        proof['status_choices'] = [screen._status_filter.itemText(i)
                                  for i in range(screen._status_filter.count())]
        capture('02_loaded_history')
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
        capture('03_batch_search')
        rid = screen._table.item(screen._table.currentRow(), 0).data(Qt.UserRole)
        panels(rid)
        tab(2, '04_batch_empty_inventory')
        # These real conversions succeeded but skipped file hashing. Empty
        # dictionaries are not evidence that the jobs produced no images.
        if records[rid]['manifest']['input_hashing'] != 'skipped':
            raise ValueError('The selected conversion no longer demonstrates skipped hashing')
        fill('no-such-tutorial-run-7d40a281')
        visible([])
        if any(w.toPlainText() for w in (screen._overview, screen._settings, screen._outputs,
                                         screen._problems, screen._environment)):
            raise ValueError('No-match search left stale detail text')
        if any(w.isEnabled() for w in (screen._copy_path, screen._open_folder, screen._load_settings)):
            raise ValueError('No-match search left selected-run actions enabled')
        capture('05_no_match')
        fill('real-channels-u0zjboui')
        visible(BATCH)
        panels(screen._table.item(screen._table.currentRow(), 0).data(Qt.UserRole))
        capture('06_restored_search')

        choose(screen._module, 'recruitment')
        fill(RECRUITMENT)
        visible([RECRUITMENT])
        raw = panels(RECRUITMENT)
        tab(0, '07_recruitment_overview')
        tab(1, '08_resolved_settings')
        scroll_bottom(screen._settings)
        capture('09_settings_source')
        tab(2, '10_recorded_files')
        tab(4, '11_environment')
        scroll_bottom(screen._environment)
        capture('12_hashes_and_seeds')
        click(screen._copy_path)
        copied = app.clipboard().text()
        if copied != str(actual_root / RECRUITMENT):
            raise ValueError('Copy path returned a different journal')
        proof['copied_run_path'] = copied
        capture('13_copy_run_path')

        choose(screen._module, 'regression')
        fill(REGRESSION)
        visible([REGRESSION])
        panels(REGRESSION)
        tab(3, '14_recorded_warning')
        choose(screen._module, '')
        choose(screen._status_filter, 'failed')
        fill(FAILED)
        visible([FAILED])
        panels(FAILED)
        tab(3, '15_actual_failed_job')
        scroll_bottom(screen._problems)
        capture('16_failure_exception')

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
        capture('18_restored_recruitment')
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
