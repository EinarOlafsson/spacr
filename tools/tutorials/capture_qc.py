"""Record the real QC reader against existing downloaded and computed evidence.

No report, score, timestamp or verdict is manufactured for this recording.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import time


def record_qc(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QLineEdit, QPushButton, QDialogButtonBox, QScrollArea
    from spacr.qt.screens.qc_dashboard import FOLDED_APPS
    from spacr.qt.widgets.fold_strip import FoldButton

    sources = {
        'downloaded_annotation': stage / 'annotate_fresh/example_data/plate1',
        'computed_subset': stage / 'example_data/plate1/test',
        'downloaded_plate': stage / 'example_data/plate1',
    }
    paths = sorted({path for root in sources.values() for path in
                    [root / 'measurements/measurements.db', *sorted((root / 'qc').glob('*.csv'))]
                    if path.is_file()})
    if not all(root.is_dir() for root in sources.values()):
        raise RuntimeError('The real downloaded examples and completed bounded subset are required')

    def hashes():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    before = hashes()
    snapshots = {}

    def wait_read(root):
        deadline = time.monotonic() + timeout
        while screen.is_busy() or screen.active_jobs():
            if time.monotonic() >= deadline:
                raise TimeoutError('The QC reader did not complete')
            settle(0.1)
        settle(0.6)
        dashboard = screen.dashboard()
        if dashboard is None or dashboard.root != str(root):
            raise RuntimeError('The visible QC reader did not load the selected source')
        if {card.key for card in dashboard.cards} != {'segmentation', 'units', 'leakage', 'plate', 'agreement'}:
            raise RuntimeError('The actual dashboard omitted an expected check')
        if dashboard.blocks_run:
            raise RuntimeError('The QC dashboard is no longer advisory')
        return dashboard

    def choose(name, frame):
        root = sources[name]
        errors, accepted = [], []
        def pick():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise RuntimeError('Browse did not open the actual directory picker')
                watchdog = QTimer(dialog)
                watchdog.setSingleShot(True)
                watchdog.timeout.connect(dialog.reject)
                watchdog.start(12000)
                dialog.accepted.connect(lambda: accepted.append(True))
                edit = dialog.findChild(QLineEdit, 'fileNameEdit')
                edit.setFocus()
                QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
                QTest.keyClicks(edit, str(root))
                QTest.keyClick(edit, Qt.Key_Tab)
                settle(0.2)
                capture(frame + '_picker')
                QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open), Qt.LeftButton)
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:
                    dialog.reject()
        QTimer.singleShot(400, pick)
        buttons = [button for button in screen.findChildren(QPushButton)
                   if button.isVisible() and button.text() == 'Browse...']
        if len(buttons) != 1:
            raise RuntimeError('Expected one actual Browse control')
        QTest.mouseClick(buttons[0], Qt.LeftButton)
        if errors or not accepted:
            raise RuntimeError('; '.join(errors) or 'The source picker was cancelled')
        dashboard = wait_read(root)
        snapshots[name] = asdict(dashboard)
        capture(frame)
        return dashboard

    for key in FOLDED_APPS:
        buttons = [button for button in screen.findChildren(FoldButton)
                   if button.isVisible() and button.app_key == key]
        if len(buttons) != 1:
            raise RuntimeError(f'Expected one visible QC fold: {key}')
        QTest.mouseMove(buttons[0])
        settle(0.8)
        capture('02_fold_' + key)

    missing = choose('downloaded_annotation', '03_missing_reports')
    if missing.card('segmentation').verdict != 'missing' or missing.card('units').verdict != 'ok':
        raise RuntimeError('The actual initial evidence differs from the planned missing/units comparison')
    subset = choose('computed_subset', '04_computed_subset')
    if subset.card('segmentation').verdict != 'warn' or subset.card('units').verdict != 'ok':
        raise RuntimeError('The real bounded-run scorecards do not support the planned warning scene')
    scrolls = [area for area in screen.findChildren(QScrollArea) if area.widget() is screen._cards_panel]
    if len(scrolls) != 1:
        raise RuntimeError('Expected the real dashboard card scroll area')
    bar = scrolls[0].verticalScrollBar()
    bar.setFocus()
    QTest.keyClick(bar, Qt.Key_End)
    settle()
    if bar.value() != bar.maximum():
        raise RuntimeError('The actual scrollbar did not reveal the bottom cards')
    capture('05_subset_remaining_checks')
    plate = choose('downloaded_plate', '06_original_plate')
    if plate.card('segmentation').verdict != 'fail':
        raise RuntimeError('The downloaded full-plate scorecards do not support the planned failure scene')
    bar.setFocus()
    QTest.keyClick(bar, Qt.Key_Home)
    settle()
    capture('07_original_plate_details')
    refresh = [button for button in screen.findChildren(QPushButton)
               if button.isVisible() and button.text() == 'Refresh']
    if len(refresh) != 1:
        raise RuntimeError('Expected one actual Refresh control')
    QTest.mouseClick(refresh[0], Qt.LeftButton)
    after_refresh = wait_read(sources['downloaded_plate'])
    if asdict(after_refresh) != snapshots['downloaded_plate']:
        raise RuntimeError('Refreshing unchanged source files unexpectedly changed the cached evidence')
    if 'Nothing on disk has changed' not in screen.status_text():
        raise RuntimeError('The actual refresh did not report the unchanged source')
    capture('08_unchanged_refresh')
    after = hashes()
    if after != before:
        raise RuntimeError('The QC recording altered original evidence')
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'dashboards': snapshots, 'folds': list(FOLDED_APPS),
        'input_hashes_before': before, 'input_hashes_after': after,
        'source_reports_and_databases_unchanged': True, 'refresh_used_cache': True,
        'verdicts_injected': False, 'new_scoring_performed': False,
        'all_checks_pass': False, 'published': False,
    })
    print('Accepted actual QC reading: missing, warn and fail; original evidence unchanged', flush=True)
