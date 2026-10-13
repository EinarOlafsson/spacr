"""Drive an actual annotation pass on downloaded crops, preserving their labels.

Only visible application controls write data. Read-only SQL and source hashes
verify persistence and preservation; no crop, annotation, or worker is mocked.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path


def require_preserved(before, after):
    """An example label must not alter source pixels or pre-existing columns."""
    if before != after:
        raise RuntimeError('Annotation tour changed original data or labels')


def record_annotation(app, window, screen, stage, captures, capture, settle,
                      write_json, timeout):
    from PySide6.QtCore import Qt, QTimer, QUrl
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QDialogButtonBox, QFileDialog, QLineEdit, QMessageBox
    from spacr.qt.screens.annotate import _SettingsDialog
    from spacr.qt.widgets.fold_strip import FoldButton

    # capture_refresh binds this private dataset onto the downloader's cache.
    # First verify the genuine automatic opening, then show the optional
    # folder picker using this neutral alias of the same private dataset.
    source = Path(stage) / 'example_data/plate1'
    cache = Path.home() / '.cache/spacr/example_data/plate1'
    if not source.is_dir() or not cache.is_dir() or not source.samefile(cache):
        raise RuntimeError('Annotate requires the isolated recording cache')
    database = source / 'measurements/measurements.db'
    if not database.is_file():
        raise RuntimeError('Download the real Annotate example first')

    def query(sql, parameters=()):
        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
            return db.execute(sql, parameters).fetchall()

    columns = [row[1] for row in query('PRAGMA table_info(png_list)')]
    index = 1
    while f'tutorial_demo_{index:02d}' in columns:
        index += 1
    annotation = f'tutorial_demo_{index:02d}'
    original_select = ','.join('"' + c.replace('"', '""') + '"' for c in columns)
    source_images = sorted(source.rglob('*.png'))
    if not source_images:
        raise RuntimeError('The downloaded crop dataset has no PNG images')

    def snapshot():
        rows = query(f'SELECT {original_select} FROM png_list ORDER BY rowid')
        return {
            'original_rows_sha256': hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
            'row_count': len(rows),
            'png_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in source_images},
        }

    before = snapshot()
    write_json(captures / 'annotation_originals.json', before)
    # Preserve the genuine loading transcript before starting the annotation
    # segment. Subsequent label operations emit their own unmodified messages.
    write_json(captures / 'console_before_annotation.json', {
        'text': screen._console.as_text(),
        'reason': 'New annotation segment after the cached-data chooser',
    })
    screen._console.clear()

    def wait_for(predicate, label):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError(label)
            settle(0.15)
        settle(0.4)

    def page_ready():
        n = len(screen._page_paths)
        return (n > 0 and screen._page_worker is None
                and screen._pending_page_load is None
                and len(screen._raw_thumb_images) >= n
                and all(image is not None for image in screen._raw_thumb_images[:n])
                and screen._grid_scroll.isVisible())

    def fill(widget, value):
        if not widget.isVisible():
            raise RuntimeError('The tutorial field is not visible')
        widget.setFocus()
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(value))
        QTest.keyClick(widget, Qt.Key_Tab)

    def modal_action(button, action):
        errors, accepted = [], []

        def drive():
            dialog = app.activeModalWidget()
            try:
                if dialog is None:
                    raise RuntimeError('Expected a real modal dialog')
                dialog.accepted.connect(lambda: accepted.append(True))
                action(dialog)
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:
                    dialog.reject()

        def reject_stalled():
            dialog = app.activeModalWidget()
            if dialog is not None and not accepted:
                errors.append('The real dialog did not accept the requested action')
                dialog.reject()

        QTimer.singleShot(500, drive)
        QTimer.singleShot(12000, reject_stalled)
        QTest.mouseClick(button, Qt.LeftButton)
        settle()
        if errors or not accepted:
            raise RuntimeError('; '.join(errors) or 'Dialog action was cancelled')

    wait_for(page_ready, 'Load test data did not automatically open real crops')
    if not Path(screen._settings.src).samefile(source):
        raise RuntimeError('Automatic opening selected a different dataset')
    if screen._settings.annotation_column != 'infected':
        raise RuntimeError('Automatic opening did not select the published labels')
    write_json(captures / 'automatic_opening.json', {
        'accepted': True, 'trigger': 'Load test data / Load',
        'extra_open_source_action': False, 'source': screen._settings.src,
        'annotation_column': screen._settings.annotation_column,
        'loaded_crops': len(screen._page_paths),
        'original_rows': before['row_count'],
    })
    def choose_source(dialog, *, record_picker=True):
        if not isinstance(dialog, QFileDialog):
            raise RuntimeError('Open source did not open the folder picker')
        dialog.setDirectory(str(source))
        dialog.setSidebarUrls([QUrl.fromLocalFile(str(stage))])
        settle()
        dialog.resize(1300, 950)
        edit = dialog.findChild(QLineEdit, 'fileNameEdit')
        if edit is None:
            raise RuntimeError('Folder picker has no filename field')
        fill(edit, source)
        if record_picker:
            capture('04_open_source_dialog')
        QTest.keyClick(edit, Qt.Key_Return)

    # The automatically opened grid displays the actual account cache path
    # behind the picker. First select the same files through their neutral
    # alias without saving a frame, then demonstrate the optional picker.
    # This changes only the displayed path, not the dataset or opening proof.
    modal_action(screen._btn_open, lambda dialog: choose_source(dialog, record_picker=False))
    wait_for(page_ready, 'The neutral alias did not reopen the same real crops')
    if not Path(screen._settings.src).samefile(source):
        raise RuntimeError('Neutral source preparation changed the dataset')
    write_json(captures / 'neutral_source_preparation.json', {
        'method': 'Actual Open source button and folder picker before recording',
        'same_dataset': True, 'source': screen._settings.src,
        'changes_automatic_opening_evidence': False,
        'preceding_console': screen._console.as_text(),
    })
    screen._console.clear()
    # This is an optional, separately narrated choice of an existing plate,
    # never a workaround required to make Load test data open the grid.
    modal_action(screen._btn_open, choose_source)
    wait_for(page_ready, 'The optional Open source action did not load real crops')
    capture('05_loaded_crops')

    def settings(column=None, primaries=None, frame='06_annotation_settings'):
        def edit(dialog):
            # Deferred destruction of an earlier dialog can clear the screen's
            # retained reference while a new modal is already visible. Drive
            # the actual active editor, not that lifecycle bookkeeping slot.
            if not isinstance(dialog, _SettingsDialog):
                capture('unexpected_settings_dialog')
                raise RuntimeError(f'Settings opened {type(dialog).__name__}: {dialog.windowTitle()}')
            if column is not None:
                fill(dialog._ann_col, column)
            if primaries is not None:
                combo = dialog._display_primaries
                target = combo.findData(primaries)
                if target < 0 or not combo.isVisible():
                    raise RuntimeError('The requested channel-colour mode is unavailable')
                combo.setFocus()
                QTest.keyClick(combo, Qt.Key_Home)
                for _ in range(target):
                    QTest.keyClick(combo, Qt.Key_Down)
                if combo.currentData() != primaries:
                    raise RuntimeError('Channel-colour selection did not change')
            settle()
            capture(frame)
            ok = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
            if not ok.isVisible() or not window.screen().geometry().contains(ok.mapToGlobal(ok.rect().center())):
                raise RuntimeError('Settings confirmation is outside the recording surface')
            QTest.mouseClick(ok, Qt.LeftButton)
        modal_action(screen._btn_settings, edit)
        wait_for(page_ready, 'Changed annotation settings did not load crops')

    settings(column=annotation, primaries='rgb')
    if screen._settings.annotation_column != annotation:
        raise RuntimeError('The new demonstration column was not selected')
    capture('07_new_annotation_column')
    first_path = screen._page_paths[0][0]

    def saved_label():
        rows = query(f'SELECT "{annotation}" FROM png_list WHERE png_path=?', (first_path,))
        if len(rows) != 1:
            raise RuntimeError('Expected one unique persisted crop row')
        return rows[0][0]

    if saved_label() is not None:
        raise RuntimeError('A new demonstration column already contains a label')
    thumb = screen._thumbs[0]
    QTest.mouseClick(thumb, Qt.LeftButton)
    settle()
    if screen._current_value(0) != 1:
        raise RuntimeError('The actual crop click did not assign class 1')
    capture('08_class_one_pending')
    # The current app batches writes when changing page, not on a timer.
    # Demonstrate that real boundary instead of calling its private flush.
    QTest.mouseClick(screen._btn_next, Qt.LeftButton)
    wait_for(page_ready, 'The real Next action did not load the next page')
    wait_for(lambda: saved_label() == 1, 'Class 1 was not saved by the real annotation worker')
    capture('08_next_page_saved')
    QTest.mouseClick(screen._btn_prev, Qt.LeftButton)
    wait_for(page_ready, 'The real Back action did not reload the first page')
    if screen._page_paths[0] != (first_path, 1):
        raise RuntimeError('The saved class did not reload into its original crop')
    capture('08_class_one_saved')

    def show_counts(dialog):
        if not isinstance(dialog, QMessageBox) or 'Class counts' not in dialog.windowTitle():
            raise RuntimeError('Expected the actual Class counts dialog')
        write_json(captures / 'annotation_class_counts.json', {'text': dialog.text()})
        capture('09_class_counts')
        QTest.mouseClick(dialog.button(QMessageBox.Ok), Qt.LeftButton)

    modal_action(screen._btn_count, show_counts)
    QTest.mouseClick(screen._btn_coverage, Qt.LeftButton)
    settle()
    report = screen._reports.get('Annotation coverage')
    if report is None or not report.isVisible():
        raise RuntimeError('Coverage did not open its actual report')
    capture('09_annotation_coverage')
    report.close()
    settle()
    QTest.mouseClick(screen._thumbs[0], Qt.LeftButton)
    QTest.mouseClick(screen._btn_next, Qt.LeftButton)
    wait_for(page_ready, 'Next did not load after clearing the example label')
    wait_for(lambda: saved_label() is None, 'Clicking the same class did not clear the saved label')
    QTest.mouseClick(screen._btn_prev, Qt.LeftButton)
    wait_for(page_ready, 'Back did not reload the cleared example')
    capture('10_class_cleared')

    def pixels():
        from PySide6.QtGui import QImage
        result = []
        for path, pixmap in zip(screen._page_paths, screen._thumb_pixmaps):
            if pixmap is None:
                raise RuntimeError('A displayed crop lost its pixels')
            image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
            result.append({'path': path[0], 'size': [image.width(), image.height()],
                           'sha256': hashlib.sha256(image.bits().tobytes()).hexdigest()})
        return result

    rgb = pixels()
    write_json(captures / 'view_rgb_before.json', {'pixels': rgb, 'settings': vars(screen._settings)})
    settings(primaries='cmy', frame='11_display_colour_settings')
    cmy = pixels()
    write_json(captures / 'view_cmy.json', {'pixels': cmy, 'settings': vars(screen._settings)})
    if not rgb or cmy == rgb:
        raise RuntimeError('Changing channel colours did not visibly change the real crops')
    capture('12_cmy_view')
    settings(primaries='rgb', frame='13_restore_rgb_settings')
    restored = pixels()
    write_json(captures / 'view_rgb_restored.json', {'pixels': restored, 'settings': vars(screen._settings)})
    capture('14_restored_rgb')
    if restored != rgb:
        raise RuntimeError('Restoring RGB did not restore the same displayed crops')
    folds = [b for b in screen.findChildren(FoldButton) if b.isVisible()]
    if [b.app_key for b in folds] != ['agreement']:
        raise RuntimeError('The current Annotate fold inventory changed')
    QTest.mouseMove(folds[0])
    settle(1)
    capture('15_agreement_fold')
    menu = screen._btn_train.menu()
    actions = [a.text() for a in menu.actions() if not a.isSeparator()]
    menu_seen = []

    def inspect_training_menu():
        if menu.isVisible():
            capture('16_training_routes')
            menu_seen.append(True)
        menu.close()

    QTimer.singleShot(600, inspect_training_menu)
    QTest.mouseClick(screen._btn_train, Qt.LeftButton)
    settle(0.9)
    if not menu_seen:
        raise RuntimeError('The actual Train menu did not open')
    QTest.mouseMove(screen._btn_generate)
    settle(1)
    capture('17_generate_database_entry')
    if not screen._console_switch.isChecked():
        QTest.mouseClick(screen._console_switch, Qt.LeftButton)
    settle()
    if not screen._console.isVisible():
        raise RuntimeError('The Console switch did not reveal the actual console')
    write_json(captures / 'annotation_console.json', {'text': screen._console.as_text()})
    capture('18_annotation_console')
    require_preserved(before, snapshot())
    if query(f'SELECT count(*) FROM png_list WHERE "{annotation}" IS NOT NULL')[0][0] != 0:
        raise RuntimeError('The demonstration left a label in the new column')
    write_json(captures / 'annotation_tour.json', {
        'accepted': True, 'source': str(source), 'database': str(database),
        'console_reset_before_annotation': True,
        'source_png_count': len(source_images), 'rows': before['row_count'],
        'visible_crops': len(screen._page_paths), 'annotation_column': annotation,
        'demonstration_label': {'png_path': first_path, 'persisted_transitions': [None, 1, None]},
        'original_columns_and_pixels_preserved': True,
        'view_changed_rgb_cmy_rgb': True, 'synthetic_images': False,
        'label_write_boundary': 'Actual Next and Back buttons; no private flush call',
        'folds': [b.app_key for b in folds], 'training_menu_actions': actions,
        'training_started': False, 'annotation_database_generation_started': False,
        'biological_classification_claim': False,
    })
    import os
    if os.environ.get('SPACR_TUTORIAL_ANNOTATE_SUGGEST') == '1':
        record_suggest_judgement(app, screen, captures, capture, settle, write_json,
                                 wait_for, page_ready, query, annotation)


def record_suggest_judgement(app, screen, captures, capture, settle, write_json,
                             wait_for, page_ready, query, annotation):
    """Label both classes in several wells, run a real Suggest round, judge two (512, 610).

    Runs after the preserving tour on the private copy only; the labels and
    suggestions stay in the tutorial column of that copy.
    """
    import re
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QMenu, QMessageBox

    def well(path):
        match = re.search(r'plate\d+_([A-Z]\d{2})_', Path(path).name)
        return match.group(1) if match else None

    labelled_wells = set()
    for _ in range(12):
        paths = [p for p, _ in screen._page_paths]
        wells = {}
        for index, path in enumerate(paths):
            wells.setdefault(well(path), []).append(index)
        for key, indices in wells.items():
            if key is None or key in labelled_wells or len(indices) < 6:
                continue
            for index in indices[:3]:
                QTest.mouseClick(screen._thumbs[index], Qt.LeftButton)
                settle(.05)
            for index in indices[3:6]:
                QTest.mouseClick(screen._thumbs[index], Qt.RightButton)
                settle(.05)
            labelled_wells.add(key)
        if len(labelled_wells) >= 3:
            break
        QTest.mouseClick(screen._btn_next, Qt.LeftButton)
        wait_for(page_ready, 'Next did not load while labelling')
    if len(labelled_wells) < 2:
        raise RuntimeError('Could not label crops from at least two wells')
    QTest.mouseClick(screen._btn_next, Qt.LeftButton)
    wait_for(page_ready, 'Next did not load after labelling')
    QTest.mouseClick(screen._btn_prev, Qt.LeftButton)
    wait_for(page_ready, 'Back did not reload the labelled page')
    wait_for(lambda: query(f'SELECT count(*) FROM png_list WHERE "{annotation}" = 1')[0][0] >= 6
             and query(f'SELECT count(*) FROM png_list WHERE "{annotation}" = 2')[0][0] >= 6,
             'The labels were not saved')
    capture('19_labels_for_suggest')
    seen = {}

    def pick():
        menus = [m for m in app.topLevelWidgets() if isinstance(m, QMenu) and m.isVisible()]
        if len(menus) != 1:
            seen['error'] = 'Suggest menu did not open'
            for m in menus:
                m.close()
            return
        menu = menus[0]
        capture('19_suggest_menu')
        action = next((a for a in menu.actions() if a.text() == 'Suggest for the images on this page'), None)
        if action is None:
            seen['error'] = 'No page suggestion entry'
            menu.close()
            return
        QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(action).center())
        seen['ok'] = True

    QTimer.singleShot(900, pick)
    QTest.mouseClick(screen._btn_suggest, Qt.LeftButton)
    settle(.5)
    if not seen.get('ok'):
        raise RuntimeError(seen.get('error', 'Suggest menu not handled'))
    settle(1.5)
    if screen._suggest_worker is not None:
        capture('19_suggest_running')
    boxes = []

    def no_hidden_box():
        for box in app.topLevelWidgets():
            if isinstance(box, QMessageBox) and box.isVisible():
                boxes.append(box.text())
                raise RuntimeError('Suggest raised a message box: ' + box.text())
        return screen._suggest_worker is None

    wait_for(no_hidden_box, 'The suggestion round did not finish')
    settle(2)
    wait_for(page_ready, 'The page did not reload with suggestions')
    suggested = [i for i in range(len(screen._page_paths))
                 if (screen._page_paths[i][1] or 0) >= 10]
    if len(suggested) < 2:
        raise RuntimeError(f'Too few suggestions on the page: {screen._status_label.text()}')
    capture('20_suggestions')
    QTest.mouseClick(screen._thumbs[suggested[0]], Qt.LeftButton)
    settle(.3)
    QTest.mouseClick(screen._thumbs[suggested[1]], Qt.RightButton)
    settle(.6)
    capture('21_judged')
    write_json(captures / 'suggest_tour.json', {
        'accepted': True, 'labelled_wells': sorted(labelled_wells),
        'suggested_on_page': len(suggested),
        'confirmed_index': suggested[0], 'rejected_index': suggested[1],
        'status': screen._status_label.text(), 'message_boxes': boxes})
