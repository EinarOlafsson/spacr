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
    def name_gate():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, QInputDialog):
                raise RuntimeError('Drawing did not open the actual gate-name dialog')
            dialog.accepted.connect(lambda: named.append(True))
            fill(dialog.findChild(QLineEdit), 'tutorial_population')
            capture('05_name_drawn_gate')
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
    if hashlib.sha256(original.read_bytes()).hexdigest() != before:
        raise RuntimeError('The original downloaded measurement database changed')
    write_json(captures / 'gate_checkpoint.json', {
        'actual_named_gate': strategy, 'gate_count': count, 'all_rows': len(frame),
        'actual_save_delete_load_restores_membership': True,
        'checkbox_hides_without_removing_membership': True,
        'database': str(database), 'original_database_sha256': before,
        'original_database_unchanged': True, 'saved_strategy': str(saved),
        'chapter_complete': '2D gate, counts and strategy reuse',
        'whole_lesson_complete': False, 'published': False,
    })
    print(f'Actual 2D gate checkpoint: {count}/2341, saved, deleted and reloaded', flush=True)
