"""Record real column drags, plots, filtering and brushing on downloaded data.

Native XTest input is allowed only on the isolated tutorial desktop. No frame,
chart specification, MIME payload, filter or selection is injected into the app.
"""
from __future__ import annotations

import ctypes
import hashlib
import time
from dataclasses import asdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_graph(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    import numpy as np
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QLineEdit, QPushButton, QDialogButtonBox
    from capture_diagnostics import PrivateDesktop
    from spacr.qt.screens.graph_builder import FOLDED_APPS
    from spacr.qt.widgets.fold_strip import FoldButton

    database = stage / 'annotate_fresh/example_data/plate1/measurements/measurements.db'
    if not database.is_file():
        raise RuntimeError('Download the real Annotate example before recording Graph Builder')
    original_hash = digest(database)
    desktop = PrivateDesktop(stage)
    xtest = ctypes.CDLL('libXtst.so.6')
    xtest.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
    xtest.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]

    def motion(point):
        xtest.XTestFakeMotionEvent(desktop.display, -1, point.x(), point.y(), 0)
        desktop.x.XFlush(desktop.display)

    def button(pressed):
        xtest.XTestFakeButtonEvent(desktop.display, 1, int(pressed), 0)
        desktop.x.XFlush(desktop.display)

    def fill(widget, value):
        widget.setFocus()
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(value))
        QTest.keyClick(widget, Qt.Key_Tab)
        settle(0.2)

    def wait_loaded():
        deadline = time.monotonic() + timeout
        while screen.is_busy() or screen.active_jobs():
            if time.monotonic() >= deadline:
                raise TimeoutError('The actual measurement table did not finish loading')
            settle(0.1)
        settle(0.8)
        if screen._frame is None or screen._frame.empty:
            raise RuntimeError('No real measurement rows loaded')

    def drag(column, channel):
        panel = screen.builder
        fill(panel.well._search, column)
        listing = panel.well._list
        items = [listing.item(i) for i in range(listing.count())
                 if listing.item(i).data(Qt.UserRole) == column]
        if len(items) != 1:
            raise RuntimeError(f'The real column list does not contain exactly one {column}')
        listing.scrollToItem(items[0])
        start = listing.viewport().mapToGlobal(listing.visualItemRect(items[0]).center())
        zone = panel.zone(channel)
        end = zone.mapToGlobal(zone.rect().center())
        # Qt's genuine QDrag enters a nested event loop. Timed native input
        # completes that drag rather than substituting a synthetic drop event.
        motion(start)
        settle(0.15)
        button(True)
        QTimer.singleShot(120, lambda: motion(start + QPoint(18, 0)))
        QTimer.singleShot(260, lambda: motion(QPoint((start.x()+end.x())//2, (start.y()+end.y())//2)))
        QTimer.singleShot(420, lambda: motion(end))
        QTimer.singleShot(650, lambda: button(False))
        settle(1.4)
        if zone.column != column or panel.spec.column_for(channel) != column:
            raise RuntimeError(f'The real drag did not assign {column} to {channel}')
        if panel.canvas.render_data is None:
            raise RuntimeError('The actual chart did not render after the drop')

    try:
        for key in FOLDED_APPS:
            buttons = [b for b in screen.findChildren(FoldButton) if b.isVisible() and b.app_key == key]
            if len(buttons) != 1:
                raise RuntimeError(f'Expected one visible Graph Builder fold: {key}')
            QTest.mouseMove(buttons[0])
            settle(0.8)
            capture('02_fold_' + key)

        errors, accepted = [], []
        def choose():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise RuntimeError('Load table did not open the actual file picker')
                dialog.accepted.connect(lambda: accepted.append(True))
                dialog.resize(1400, 950)
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), database)
                capture('03_choose_database')
                QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open), Qt.LeftButton)
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:
                    dialog.reject()
        QTimer.singleShot(500, choose)
        buttons = [b for b in screen.findChildren(QPushButton) if b.isVisible() and b.text().startswith('Load table')]
        if len(buttons) != 1:
            raise RuntimeError('Expected one visible Load table control')
        QTest.mouseClick(buttons[0], Qt.LeftButton)
        if errors or not accepted:
            raise RuntimeError('; '.join(errors) or 'Database selection was cancelled')
        wait_loaded()
        if screen._table_picker.currentText() != 'cell':
            raise RuntimeError('Expected the actual cell measurement table')
        frame = screen._frame
        canvas = screen.builder.canvas
        capture('04_real_cell_table')
        drag('cell_area', 'x')
        capture('05_area_histogram')
        drag('cell_channel_1_mean_intensity', 'y')
        capture('06_real_scatter')
        drag('columnID', 'colour')
        capture('07_well_column_colour')
        initial_count = canvas.render_data.n_total
        if initial_count != len(frame):
            raise RuntimeError('The initial chart unexpectedly omits measurement rows')
        drag('rowID', 'facet_row')
        capture('08_acquisition_row_facets')
        picker = screen.filters._picker
        picker.setFocus()
        QTest.keyClick(picker, Qt.Key_Home)
        target = picker.findText('cell_area')
        if target < 0:
            raise RuntimeError('The actual filter menu has no cell_area entry')
        for _ in range(target):
            QTest.keyClick(picker, Qt.Key_Down)
        QTest.keyClick(picker, Qt.Key_Tab)
        if picker.currentText() != 'cell_area':
            raise RuntimeError(f'The visible filter picker selected {picker.currentText()!r}, not cell_area')
        print('Actual filter picker selected cell_area', flush=True)
        QTest.mouseClick(screen.filters.findChild(QPushButton, 'FilterAddButton'), Qt.LeftButton)
        settle()
        row = screen.filters._rows['cell_area']
        cutoff = float(np.ceil(frame['cell_area'].median()))
        fill(row._low.lineEdit(), str(cutoff))
        settle(1)
        expected = int((frame['cell_area'] >= cutoff).sum())
        filtered_count = canvas.render_data.n_total
        print(f'Actual range {row._low.value()} .. {row._high.value()}: {filtered_count}, expected {expected}', flush=True)
        if filtered_count != expected or not 0 < filtered_count < initial_count:
            raise RuntimeError('The live area filter did not select the expected real rows')
        capture('09_live_area_filter')
        QTest.mouseClick(screen.filters._clear, Qt.LeftButton)
        settle(1)
        if canvas.render_data.n_total != initial_count:
            raise RuntimeError('Clearing the live filter did not restore the actual rows')
        capture('10_filter_cleared')
        QTest.mouseClick(screen.builder.zone('facet_row')._clear, Qt.LeftButton)
        settle(1)
        ax = canvas.axes_at()
        x0, x1 = frame['cell_area'].quantile([.25, .75])
        y0, y1 = frame['cell_channel_1_mean_intensity'].quantile([.25, .75])
        def point(x, y):
            px, py = ax.transData.transform((x, y))
            return QPoint(int(px), int(canvas._canvas.height() - py))
        start, end = point(x0, y0), point(x1, y1)
        events = []
        def observed(event):
            events.append({'kind': event.name, 'x': event.xdata, 'y': event.ydata,
                           'in_current_axes': event.inaxes is ax})
        observers = [canvas._canvas.mpl_connect(kind, observed) for kind in
                     ('button_press_event', 'motion_notify_event', 'button_release_event')]
        print(f'Brush canvas {canvas._canvas.size()}, figure {canvas._figure.bbox.bounds}, points {start} -> {end}', flush=True)
        motion(canvas._canvas.mapToGlobal(start))
        settle(0.2)
        button(True)
        settle(0.2)
        motion(canvas._canvas.mapToGlobal(end))
        settle(0.3)
        button(False)
        settle(1)
        selected = canvas.selected_count()
        print(f'Actual brush: {selected}; handoff enabled={screen._to_annotate.isEnabled()}; notice={canvas.notice()}', flush=True)
        print(f'Observed genuine mouse events {events}; linked keys={len(canvas.link.selection)}; source={canvas.link.selection.source}', flush=True)
        for observer in observers:
            canvas._canvas.mpl_disconnect(observer)
        brush_proof = {
            'accepted': bool(0 < selected < initial_count and screen._to_annotate.isEnabled()),
            'rows': initial_count, 'visible_selected_rows': selected,
            'published_object_keys': len(canvas.link.selection),
            'handoff_enabled': screen._to_annotate.isEnabled(),
            'native_events': events, 'database_sha256_before': original_hash,
            'database_sha256_after': digest(database),
            'reason': 'The actual brush must visibly select rows and enable its handoff',
        }
        capture('11_actual_brush_state')
        write_json(captures / 'brush_outcome.json', brush_proof)
        if not 0 < selected < initial_count or not screen._to_annotate.isEnabled():
            write_json(captures / 'scientific_acceptance.json', brush_proof)
            raise RuntimeError('The actual brush did not create a partial linked selection')
        capture('11_real_brush_selection')
        QTest.mouseClick(canvas._canvas, Qt.LeftButton, pos=start)
        settle(0.8)
        if canvas.selected_count() != 0 or screen._to_annotate.isEnabled():
            raise RuntimeError('The actual click did not clear the linked selection')
        capture('12_selection_cleared')
        if digest(database) != original_hash:
            raise RuntimeError('Exploring the chart modified the downloaded database')
        write_json(captures / 'scientific_acceptance.json', {
            'accepted': True, 'database': str(database), 'database_sha256': original_hash,
            'table': 'cell', 'rows': initial_count, 'area_filter_cutoff': cutoff,
            'filtered_rows': filtered_count, 'filter_cleared': True,
            'brushed_rows': selected, 'selection_cleared': True,
            'source_database_unchanged': True, 'actual_native_column_drags': True,
            'folds': list(FOLDED_APPS), 'final_spec': asdict(screen.builder.spec),
            'notice': canvas.notice(), 'chart_or_spec_export_button_used': False,
            'annotation_handoff_clicked': False, 'published': False})
        print(f'Accepted Graph Builder: {initial_count} -> {filtered_count} -> {initial_count} rows', flush=True)
    finally:
        button(False)
        desktop.close()
