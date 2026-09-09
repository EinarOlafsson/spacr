"""Record actual named gates on a private copy of downloaded measurements."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time


def record_gates(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QDialogButtonBox, QInputDialog, QLineEdit, QPushButton

    write_json(captures / 'scientific_acceptance.json', {
        'accepted': False, 'reason': 'Gate tutorial recording is an unfinished chapter checkpoint',
        'published': False,
    })

    original = stage / 'annotate_fresh/example_data/plate1/measurements/measurements.db'
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    root = stage / 'gate_runs'
    root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='example-', dir=root))
    database = work / 'measurements.db'
    with sqlite3.connect(original.as_uri() + '?mode=ro', uri=True) as source:
        with sqlite3.connect(database) as destination:
            source.backup(destination)
    write_json(captures / 'input_manifest.json', {
        'original': str(original), 'original_sha256': before,
        'private_database': str(database), 'copy_method': 'SQLite read-only source backup',
        'invented_measurements': False,
    })

    def fill(edit, value):
        QTest.mouseClick(edit, Qt.LeftButton)
        QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(edit, str(value))
        QTest.keyClick(edit, Qt.Key_Tab)
        settle(0.15)

    def choose(box, value, *, data=False):
        index = box.findData(value) if data else box.findText(value)
        if index < 0:
            raise RuntimeError(f'Actual selector lacks {value!r}')
        box.setFocus()
        QTest.keyClick(box, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(box, Qt.Key_Down)
        QTest.keyClick(box, Qt.Key_Tab)
        settle(0.8)
        if (box.currentData() if data else box.currentText()) != value:
            raise RuntimeError('Actual axis/tool selector did not reach the requested entry')

    def wait():
        deadline = time.monotonic() + timeout
        while screen.is_busy() or screen.active_jobs():
            if time.monotonic() >= deadline:
                raise TimeoutError('Gate Editor did not finish its real job')
            settle(0.1)
        settle(0.8)

    def drag_handle(splitter, index, target):
        handle = splitter.handle(index)
        start = handle.rect().center()
        end = start + QPoint(target - handle.mapTo(window, start).x(), 0)
        QTest.mousePress(handle, Qt.LeftButton, pos=start)
        QTest.mouseMove(handle, end, delay=150)
        QTest.mouseRelease(handle, Qt.LeftButton, pos=end)
        settle(0.5)

    # Keep every real panel. Resize with the same handles a user can drag.
    drag_handle(screen.gates.parentWidget(), 1, 3220)
    drag_handle(screen.gates.body, 1, 2440)
    capture('01b_readable_panels')

    def file_dialog(button, path, frame, *, save=False):
        errors, accepted = [], []
        def pick():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise RuntimeError('The actual file picker did not open')
                watchdog = QTimer(dialog)
                watchdog.setSingleShot(True)
                watchdog.timeout.connect(dialog.reject)
                watchdog.start(12000)
                dialog.accepted.connect(lambda: accepted.append(True))
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), path)
                capture(frame)
                key = QDialogButtonBox.Save if save else QDialogButtonBox.Open
                QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(key), Qt.LeftButton)
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:
                    dialog.reject()
        QTimer.singleShot(400, pick)
        QTest.mouseClick(button, Qt.LeftButton)
        if errors or not accepted:
            raise RuntimeError('; '.join(errors) or 'File selection was cancelled')
        wait()

    buttons = [b for b in screen.findChildren(QPushButton) if b.isVisible() and b.text() == 'Load table…']
    if len(buttons) != 1:
        raise RuntimeError('Expected one real Load table button')
    file_dialog(buttons[0], database, '02_real_database_picker')
    if screen._frame is None or len(screen._frame) != 2341 or screen._table != 'cell':
        raise RuntimeError('Expected all 2341 downloaded cell measurements')
    capture('03_downloaded_cells')
    choose(screen._x, 'cell_area')
    choose(screen._y, 'cell_channel_1_mean_intensity')
    choose(screen.gates._tool, 'rectangle', data=True)
    canvas = screen.gates.canvas
    if len(canvas.population()) != 2341:
        raise RuntimeError('The initial view does not contain the full loaded table')
    capture('04_real_scatter')
    ax = canvas.axes_at()
    frame = screen._frame
    x0, x1 = frame['cell_area'].quantile([.25, .75])
    y0, y1 = frame['cell_channel_1_mean_intensity'].quantile([.25, .75])
    surface = canvas._canvas
    def pixel(x, y):
        px, py = ax.transData.transform((x, y))
        return QPoint(round(px), surface.height() - round(py))

    named, errors = [], []
    pending_name = 'tutorial_population'
    def name_gate():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, QInputDialog):
                raise RuntimeError('Drawing did not open the actual gate-name dialog')
            dialog.accepted.connect(lambda: named.append(True))
            fill(dialog.findChild(QLineEdit), pending_name)
            capture({'tutorial_population': '05_name_drawn_gate',
                     'tutorial_second': '13_name_second_gate',
                     'tutorial_volume': '17b_name_volume_gate'}[pending_name])
            QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok), Qt.LeftButton)
        except Exception as exc:
            errors.append(str(exc))
            if dialog is not None:
                dialog.reject()
    start, end = pixel(x0, y0), pixel(x1, y1)
    QTest.mousePress(surface, Qt.LeftButton, pos=start)
    QTest.mouseMove(surface, end, delay=200)
    QTimer.singleShot(500, name_gate)
    QTest.mouseRelease(surface, Qt.LeftButton, pos=end)
    settle(1)
    if errors or not named or screen.gates.gates.names != ('tutorial_population',):
        raise RuntimeError('; '.join(errors) or 'No actual named gate was created')
    gate = screen.gates.gates.get('tutorial_population')
    expected = frame['cell_area'].between(gate.x_low, gate.x_high) & frame[
        'cell_channel_1_mean_intensity'].between(gate.y_low, gate.y_high)
    actual = screen.gates.gates.mask(frame, gate.name)
    if not (expected.to_numpy() == actual).all() or not 0 < int(actual.sum()) < len(frame):
        raise RuntimeError('Actual gate membership does not match its drawn data bounds')
    count = int(actual.sum())
    capture('06_real_gate_and_counts')
    tree = screen.gates.tree.tree
    QTest.keyClick(tree, Qt.Key_Space)
    settle(0.5)
    if canvas.is_gate_enabled(gate.name):
        raise RuntimeError('The real checkbox did not hide the gate')
    capture('07_gate_hidden_not_deleted')
    QTest.keyClick(tree, Qt.Key_Space)
    settle(0.5)
    if not canvas.is_gate_enabled(gate.name) or int(screen.gates.gates.mask(frame, gate.name).sum()) != count:
        raise RuntimeError('Showing the gate did not preserve its exact membership')
    capture('08_gate_restored')

    saved = work / 'gates.json'
    strategy = screen.gates.gates.to_dict()
    file_dialog(screen._save_gates, saved, '09_save_strategy', save=True)
    if json.loads(saved.read_text()) != strategy:
        raise RuntimeError('The saved strategy differs from the actual gate')
    QTest.mouseClick(screen.gates.tree._remove, Qt.LeftButton)
    settle(0.5)
    if not screen.gates.gates.is_empty:
        raise RuntimeError('The real Delete gate did not remove the current gate')
    capture('10_gate_deleted_before_reload')
    file_dialog(screen._load_gates, saved, '11_reload_strategy')
    if screen.gates.gates.to_dict() != strategy:
        raise RuntimeError('Load gates did not restore the saved strategy')
    if int(screen.gates.gates.mask(frame, gate.name).sum()) != count:
        raise RuntimeError('Reloading the strategy changed its population')
    capture('12_strategy_restored')
    # Save the ACTUAL drawn graph and export gate membership to this private
    # database, never to the downloaded source.
    file_dialog(screen._save_graph, work / 'gate_graph.png', '12b_save_graph', save=True)
    if not (work / 'gate_graph.png').is_file():
        raise RuntimeError('The actual Save graph action produced no PNG')
    QTest.mouseClick(screen._export, Qt.LeftButton)
    wait()
    if not screen._source.text().startswith('wrote ') or 'could not export' in screen._source.text():
        raise RuntimeError('The real gate export did not finish completely: ' + screen._source.text())
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        columns = [row[1] for row in connection.execute('PRAGMA table_info(filters)')]
        gate_columns = [column for column in columns if 'tutorial_population' in column]
        if len(gate_columns) != 1:
            raise RuntimeError('The private filters table lacks the actual gate column')
        marked = connection.execute('SELECT COUNT(*) FROM filters WHERE "' + gate_columns[0] + '" = 1').fetchone()[0]
        if marked != count:
            raise RuntimeError('The saved gate flags do not match the actual drawn population')
        keys = ['plateID', 'rowID', 'columnID', 'fieldID', 'object_label']
        exported = list(connection.execute('SELECT ' + ','.join(keys) + ',object_type FROM filters WHERE "' + gate_columns[0] + '" = 1'))
        expected_keys = {(*row, 'cell') for row in frame.loc[actual, keys].itertuples(index=False, name=None)}
        from gate_evidence import check_exported_rows
        check_exported_rows(expected_keys, exported)
        with sqlite3.connect(original.as_uri() + '?mode=ro', uri=True) as source:
            if connection.execute('SELECT * FROM cell ORDER BY rowid').fetchall() != source.execute('SELECT * FROM cell ORDER BY rowid').fetchall():
                raise RuntimeError('Gate export changed the underlying raw cell measurements')
    capture('12c_exported_gate_membership')

    items = tree.findItems('tutorial_population', Qt.MatchExactly | Qt.MatchRecursive, 0)
    if len(items) != 1:
        raise RuntimeError('The reloaded parent is not present in the actual hierarchy')
    QTest.mouseClick(tree.viewport(), Qt.LeftButton, pos=tree.visualItemRect(items[0]).center())
    settle(0.3)
    ax = canvas.axes_at()
    if screen.gates.tree.active_gate() != gate.name or canvas.active_gate != gate.name:
        raise RuntimeError('The intended first gate was not actually selected before drawing the second')
    pending_name = 'tutorial_second'
    named.clear()
    errors.clear()
    start = pixel(gate.x_low - 1800, gate.y_low - 800)
    end = pixel((gate.x_low + gate.x_high) / 2, (gate.y_low + gate.y_high) / 2)
    QTest.mousePress(surface, Qt.LeftButton, pos=start)
    QTest.mouseMove(surface, end, delay=200)
    QTimer.singleShot(500, name_gate)
    QTest.mouseRelease(surface, Qt.LeftButton, pos=end)
    settle(0.6)
    if errors or not named or 'tutorial_second' not in screen.gates.gates.names:
        raise RuntimeError('; '.join(errors) or 'Drawing did not create a real second gate')
    second = screen.gates.gates.get('tutorial_second')
    if second.parent is not None:
        raise RuntimeError('Current drawing must not silently nest inside the selected first gate')
    second_mask = screen.gates.gates.mask(frame, second.name)
    if not (second_mask == second.mask(frame)).all() or not (second_mask & ~actual).any():
        raise RuntimeError('The second shape must retain its real rows outside the first gate')
    capture('14_second_gate_not_automatically_nested')
    file_dialog(screen._save_gates, work / 'two_gates.json', '15_save_two_gate_strategy', save=True)
    two_gate_strategy = screen.gates.gates.to_dict()

    QTest.mouseClick(screen.gates._mode_buttons['3D'], Qt.LeftButton)
    choose(screen._z, 'cell_perimeter')
    if screen.settings().gate_mode != '3D' or not hasattr(canvas.axes_at(), 'get_zlim'):
        raise RuntimeError('The actual three-measurement volume did not render')
    capture('16_actual_three_dimensional_view')
    QTest.mouseClick(screen.gates._spin_buttons['z'], Qt.LeftButton)
    ax = canvas.axes_at()
    angle_before = (ax.elev, ax.azim)
    center = surface.rect().center()
    QTest.mousePress(surface, Qt.LeftButton, pos=center)
    QTest.mouseMove(surface, center + QPoint(100, 35), delay=180)
    QTest.mouseRelease(surface, Qt.LeftButton, pos=center + QPoint(100, 35))
    settle(0.6)
    angle_after = (canvas.axes_at().elev, canvas.axes_at().azim)
    write_json(captures / 'volume_angles.json', {'before': angle_before, 'after': angle_after})
    if angle_before == angle_after:
        raise RuntimeError('The genuine 3D drag did not rotate the displayed volume')
    capture('17_rotated_volume')
    pending_name = 'tutorial_volume'
    named.clear()
    errors.clear()
    QTimer.singleShot(400, name_gate)
    QTest.mouseClick(screen.gates._box_gate, Qt.LeftButton)
    settle(0.7)
    if errors or not named or 'tutorial_volume' not in screen.gates.gates.names:
        raise RuntimeError('; '.join(errors) or 'From view did not create an actual volume gate')
    volume = screen.gates.gates.get('tutorial_volume')
    if volume.kind != 'box' or volume.parent is not None or len(volume.columns) != 3:
        raise RuntimeError('From view must record real bounds on three measurements')
    volume_count = int(screen.gates.gates.mask(frame, volume.name).sum())
    if volume_count <= 0:
        raise RuntimeError('The actual volume contains no measured objects')
    capture('17c_actual_volume_gate')
    file_dialog(screen._save_gates, work / 'three_gates.json', '17d_save_volume_strategy', save=True)
    three_gate_strategy = screen.gates.gates.to_dict()
    QTest.mouseClick(screen.gates._mode_buttons['2D'], Qt.LeftButton)
    settle(0.6)
    if screen.gates.gates.to_dict() != three_gate_strategy:
        raise RuntimeError('Changing view dimensions must not mutate saved gate boundaries')
    QTest.mouseClick(screen.gates._xd_button, Qt.LeftButton)
    wait()
    if not all(name in screen._frame.columns for name in ('PC1', 'PC2', 'PC3')):
        raise RuntimeError('The real projection did not create three actual component columns')
    if screen._x.currentText() != 'PC1' or screen._y.currentText() != 'PC2' or screen.settings().gate_mode != '2D':
        raise RuntimeError('xD must change the measurements without forcing the separate 2D/3D choice')
    capture('18_actual_projection_2d')
    QTest.mouseClick(screen.gates._mode_buttons['3D'], Qt.LeftButton)
    settle(0.8)
    capture('19_actual_projection_3d')
    QTest.mouseClick(screen.gates._xd_button, Qt.LeftButton)
    settle(0.6)
    if not all(name in screen._frame.columns for name in ('PC1', 'PC2', 'PC3')):
        raise RuntimeError('Turning off xD must not delete columns existing gates could depend on')
    capture('20_projection_off_keeps_columns')
    projection_label = screen._source.text()
    from capture_gate_search import record_search
    searches = record_search(app, screen, captures, capture, settle, write_json,
                             fill, choose, file_dialog, work)
    if hashlib.sha256(original.read_bytes()).hexdigest() != before:
        raise RuntimeError('The original downloaded measurement database changed')
    write_json(captures / 'gate_checkpoint.json', {
        'actual_named_gate': strategy, 'gate_count': count, 'all_rows': len(frame),
        'actual_save_delete_load_restores_membership': True,
        'checkbox_hides_without_removing_membership': True,
        'database': str(database), 'original_database_sha256': before,
        'original_database_unchanged': True, 'saved_strategy': str(saved),
        'chapter_complete': 'Actual drawing, strategy reuse, exports, 3D, projection and search',
        'second_gate_count': int(second_mask.sum()), 'automatic_nesting': False,
        'exported_gate_objects': marked,
        'exported_gate_object_identities_and_types_match': True,
        'private_raw_cell_measurements_unchanged': True,
        'volume_gate_count': volume_count,
        'two_gate_strategy': two_gate_strategy,
        'three_gate_strategy': three_gate_strategy,
        'actual_3d_rotation': {'before': angle_before, 'after': angle_after},
        'projection_source_label': projection_label,
        'projection_independent_of_dimensions': True,
        'search_outcomes': searches,
        'whole_lesson_complete': False, 'published': False,
    })
    print(f'Actual 2D gate checkpoint: {count}/2341, saved, deleted and reloaded', flush=True)
