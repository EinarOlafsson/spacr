"""Capture lesson09's bounded Suggest/confirm/reject segment on a private copy.

The input must be a complete, real project (an explicitly documented subset
is allowed). No original database/image is opened for writing. This offscreen
recording does not claim native desktop or published-video acceptance.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    source, stage = args.source.resolve(), args.stage.resolve()
    if stage.exists():
        raise FileExistsError('Use a fresh stage to preserve previous evidence')
    if source == stage or source in stage.parents:
        raise ValueError('Staging must be outside the original project')
    stage.mkdir(parents=True)
    profile = stage / 'profile'
    os.environ.update(QT_QPA_PLATFORM='offscreen', XDG_CONFIG_HOME=str(profile / 'config'),
                      XDG_STATE_HOME=str(profile / 'state'), XDG_CACHE_HOME=str(profile / 'cache'),
                      SPACR_LOG_DIR=str(stage / 'logs'), MPLCONFIGDIR=str(profile / 'mpl'),
                      CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg')
    os.environ.pop('SPACR_CHAINING_PINS', None)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from prepare_annotation_capture import digest, prepare

    original_database = source / 'measurements/measurements.db'
    original_hash = digest(original_database)
    prepared = stage / 'example_data/plate1'
    prepared.parent.mkdir(parents=True)
    copy_receipt = prepare(source, prepared, prepared)
    database = prepared / 'measurements/measurements.db'
    captures = stage / 'captures'
    captures.mkdir()

    from capture_annotate import record_suggest_judgement
    from capture_policy import configure_appearance
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QMenu

    from spacr.qt import preferences
    from spacr.qt.app import MainWindow, _load_bundled_fonts, _use_open_sans
    from spacr.qt.first_run import mark_tour_seen
    from spacr.qt.screens.annotate import _SettingsDialog
    from spacr.suggest import verdict_column

    app = QApplication([])
    store = Path(preferences._settings().fileName()).resolve()
    if not store.is_relative_to(profile):
        raise RuntimeError('Preference sandbox was not established before writes')
    preferences.set_language('en')
    preferences.set_preload_policy('on_demand')
    preferences.set_refresh_news(False)
    configure_appearance('dark', 'blobs')
    mark_tour_seen()
    _load_bundled_fonts()
    _use_open_sans(app)
    preferences.apply_preferences_to_app(app)
    window = MainWindow()
    window.resize(3840, 2160)
    window.show()
    window.open_module('annotate')
    screen = window._screens['annotate']
    annotation = 'tutorial_suggest_610'
    observations = []
    active = {'worker': None, 'cancel': False, 'clicked': False, 'cancelled': False}
    frames = {}

    def write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    def settle(seconds=.15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)

    def wait_for(predicate, message):
        deadline = time.monotonic() + args.timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError(message)
            settle(.05)
        settle(.1)

    def query(sql, parameters=()):
        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
            return db.execute(sql, parameters).fetchall()

    def page_ready():
        n = len(screen._page_paths)
        return (n > 0 and screen._page_worker is None
                and screen._pending_page_load is None
                and len(screen._raw_thumb_images) >= n
                and all(image is not None for image in screen._raw_thumb_images[:n]))

    def capture(name):
        pixmap = window.grab()
        path = captures / (name + '.png')
        if not pixmap.save(str(path)):
            raise RuntimeError('Cannot save the actual widget capture')
        frames[name] = {'path': path.name, 'sha256': digest(path),
                        'width': pixmap.width(), 'height': pixmap.height(),
                        'qpa': app.platformName(), 'status': screen._status_label.text()}

    def saw_cancellation():
        active['cancelled'] = True
        observations.append({'cancelled': True, 'time': time.monotonic()})

    def observe():
        worker = screen._suggest_worker
        if worker is not None and worker is not active['worker']:
            active['worker'] = worker
            worker.progress.connect(lambda n, total, step: observations.append(
                {'step': n, 'total': total, 'name': step, 'time': time.monotonic()}))
            worker.cancelled.connect(saw_cancellation)
        if (active['cancel'] and not active['clicked'] and worker is not None
                and 'step 2 of 5' in screen._status_label.text()):
            active['clicked'] = True
            QTest.mouseClick(screen._btn_suggest_cancel, Qt.LeftButton)

    observer = QTimer()
    observer.timeout.connect(observe)
    observer.start(5)
    errors = []

    def edit_settings():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, _SettingsDialog):
                raise RuntimeError('The actual annotation Settings dialog did not open')
            for widget, value in ((dialog._src_edit, str(prepared)), (dialog._ann_col, annotation)):
                widget.setFocus()
                QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
                QTest.keyClicks(widget, value)
                QTest.keyClick(widget, Qt.Key_Tab)
            QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok), Qt.LeftButton)
        except Exception as error:
            errors.append(str(error))
            if dialog is not None:
                dialog.reject()

    try:
        QTimer.singleShot(200, edit_settings)
        QTest.mouseClick(screen._btn_settings, Qt.LeftButton)
        if errors:
            raise RuntimeError('; '.join(errors))
        wait_for(page_ready, 'Real source crops did not load')
        if len(screen._page_paths) < 30:
            raise RuntimeError('The capture needs at least30crops on the page')
        capture('18_before_suggest')
        started = time.monotonic()
        record_suggest_judgement(app, screen, captures, capture, settle, write_json,
                                 wait_for, page_ready, query, annotation)
        completed_seconds = time.monotonic() - started
        if completed_seconds > args.timeout:
            raise AssertionError('The Suggest segment exceeded its bound')
        QTest.mouseClick(screen._btn_next, Qt.LeftButton)
        wait_for(page_ready, 'Judgements did not save through Next')
        verdict = verdict_column(annotation)
        wait_for(lambda: query(f'SELECT count(*) FROM png_list WHERE "{verdict}" > 0')[0][0] == 1,
                 'Confirm judgement was not persisted')
        wait_for(lambda: query(f'SELECT count(*) FROM png_list WHERE "{verdict}" < 0')[0][0] == 1,
                 'Reject judgement was not persisted')
        human_before = query(f'SELECT png_path,"{annotation}","{verdict}" FROM png_list '
                             f'WHERE "{annotation}" BETWEEN 1 AND 9 OR "{verdict}" IS NOT NULL ORDER BY png_path')
        active.update(cancel=True, clicked=False)
        cancelled_started = time.monotonic()
    
        def choose_cancel_run():
            menu = app.activePopupWidget()
            if not isinstance(menu, QMenu):
                errors.append('Second Suggest menu did not open')
                return
            action = next(a for a in menu.actions() if a.text() == 'Suggest for the images on this page')
            QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(action).center())
    
        QTimer.singleShot(200, choose_cancel_run)
        QTest.mouseClick(screen._btn_suggest, Qt.LeftButton)
        wait_for(lambda: active['clicked'], 'Actual Cancel was not reached during measurement reading')
        wait_for(lambda: screen._suggest_worker is None, 'Cancelled worker did not finish')
        if not active['cancelled']:
            raise AssertionError('Worker did not emit its real cancelled signal')
        wait_for(page_ready, 'The page did not finish reloading after cancellation')
        capture('22_cancelled')
        cancelled_seconds = time.monotonic() - cancelled_started
        assert query(f'SELECT count(*) FROM png_list WHERE "{annotation}" >= 10')[0][0] == 0
        human_after = query(f'SELECT png_path,"{annotation}","{verdict}" FROM png_list '
                            f'WHERE "{annotation}" BETWEEN 1 AND 9 OR "{verdict}" IS NOT NULL ORDER BY png_path')
        assert human_before == human_after
        assert digest(original_database) == original_hash
        assert all(digest(source / name) == h for name, h in copy_receipt['image_sha256'].items())
        receipt = {'accepted': True, 'scope': 'Lesson09 Suggest frames20/21 plus real cancellation; offscreen, not published video',
                   'source_database_sha256': original_hash, 'source_unchanged': True,
                   'real_rows': copy_receipt['rows'], 'real_pngs': copy_receipt['unique_images'],
                   'completion_seconds_including_label_clicks': completed_seconds,
                   'cancel_seconds': cancelled_seconds, 'human_judgements_preserved_after_cancel': True,
                   'confirmed': 1, 'rejected': 1, 'frames': frames, 'observations': observations,
                   'console': screen._console.as_text()}
        write_json(captures / 'receipt.json', receipt)
    finally:
        observer.stop()
        screen.close()
        window.close()
        settle(.2)
    print(json.dumps({'accepted': True, 'rows': copy_receipt['rows'], 'completion_seconds': completed_seconds,
                      'cancel_seconds': cancelled_seconds, 'stage': str(stage)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
