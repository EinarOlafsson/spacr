"""Record real column drags, plots, filtering and brushing on downloaded data.

Native XTest input is allowed only on the isolated tutorial desktop. No frame,
chart specification, MIME payload, filter or selection is injected into the app.
"""
from __future__ import annotations

import ctypes
import hashlib
import time
from dataclasses import asdict

import capture_ui
import capture_ui_drv_d as drv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inspect_trellis_fold(capture, screen, button, settle, timeout):
    """Open the actual empty Small Multiples page, then close its native tab."""
    from PySide6.QtWidgets import QTabBar

    human = capture_ui._h()
    human.click(button)
    deadline = time.monotonic() + timeout
    while True:
        openers = [opener for opener in screen._fold_openers if opener.key == 'trellis']
        child = openers[0].window if len(openers) == 1 else None
        pages = getattr(screen, '_fold_pages', None)
        if (child is not None and pages is not None and child.isVisible()
                and pages.currentWidget() is child):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('The native Small Multiples fold did not open its page')
        settle(.05)
    panel = getattr(child, 'panel', None)
    if (child.objectName() != 'TrellisScreen' or not child.isEnabled()
            or panel is None or not panel.isVisible() or not panel.isEnabled()
            or panel.width() <= 0 or panel.height() <= 0):
        raise RuntimeError('The actual Small Multiples page has no visible usable panel')
    splitter = panel._splitter
    channels = splitter.pane('Channels')
    heading = getattr(channels.widget, 'heading', None) if channels is not None else None
    handle = splitter.handle(1)
    if (channels is None or channels.is_collapsed() or heading is None
            or not heading.isVisible() or not heading.isEnabled()
            or handle is None or not handle.isVisible() or not handle.isEnabled()):
        raise RuntimeError('The actual Channels divider is not visible and usable')
    start = handle.rect().center()
    shelf_width = max(300, min(600, round(splitter.width() * .30)))
    delta = shelf_width - handle.mapTo(splitter, start).x()
    if delta > 4:
        human.drag(capture, handle, start, (delta, 0), duration=3.0)
        settle(.4)
    from capture_trellis import _require_visible_trellis
    _require_visible_trellis(panel)
    if (not channels.widget.isVisible() or not panel.well.isVisible()
            or channels.widget.width() <= 0 or channels.widget.height() <= 0
            or panel.well.width() <= 0 or panel.well.height() <= 0):
        raise RuntimeError('The actual Channels controls are not readable after the native drag')
    capture_ui.point_at(capture, child._header, dwell=2.0)
    index = pages.indexOf(child)
    if index <= 0:
        raise RuntimeError('Small Multiples must be a closable child of Graph Builder')
    bar = pages.tabBar()
    close = [bar.tabButton(index, side) for side in (QTabBar.RightSide, QTabBar.LeftSide)]
    close = [mark for mark in close if mark is not None and mark.isVisible() and mark.isEnabled()]
    if len(close) != 1:
        raise RuntimeError('The Small Multiples tab has no unique visible close mark')
    human.click(close[0])
    deadline = time.monotonic() + timeout
    while pages.currentIndex() != 0 or not screen.builder.isVisible():
        if time.monotonic() >= deadline:
            raise RuntimeError('The native tab close did not restore Graph Builder')
        settle(.05)


def record_graph(app, window, screen, stage, captures, capture, settle, write_json, timeout, *, review_handoff=False):
    import sqlite3

    import numpy as np
    import pandas as pd
    from capture_diagnostics import PrivateDesktop
    from capture_graph_annotations import record_annotations
    from graph_evidence import check_brush, check_brush_cleared, check_histogram, check_points
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QDialogButtonBox, QFileDialog, QLineEdit, QPushButton

    from spacr.qt.screens.graph_builder import FOLDED_APPS
    from spacr.qt.widgets.fold_strip import FoldButton

    database = stage / 'annotate_fresh/example_data/plate1/measurements/measurements.db'
    if not database.is_file():
        raise RuntimeError('Download the real Annotate example before recording Graph Builder')
    original_hash = digest(database)
    with sqlite3.connect('file:' + str(database) + '?mode=ro', uri=True) as connection:
        expected_frame = pd.read_sql_query('SELECT * FROM cell', connection)
    chart_checks = []
    live = capture_ui.live(capture)
    desktop = xtest = None
    if not live:        # the Xvfb recorder drives drags with XTest; the host uses its real pointer
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
        if live:
            capture_ui.set_text(capture, widget, str(value))
            return
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
        if live:
            # A real drag of the list entry onto the channel zone (QDrag runs its own loop).
            capture_ui._wheel_item_into_view(capture, listing, items[0])
            zone = panel.zone(channel)
            start_local = listing.visualItemRect(items[0]).center()
            delta = zone.mapToGlobal(zone.rect().center()) - listing.viewport().mapToGlobal(start_local)
            capture_ui._h().drag_qdrag(capture, listing.viewport(), start_local, (delta.x(), delta.y()), duration=1.2)
            capture_ui._h().pause(capture, 1.0)
            if zone.column != column or panel.spec.column_for(channel) != column:
                raise RuntimeError(f'The real drag did not assign {column} to {channel}')
            if panel.canvas.render_data is None:
                raise RuntimeError('The actual chart did not render after the drop')
            return
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

    def check_scatter(name, expected, *, faceted=False):
        canvas = screen.builder.canvas
        data = canvas.render_data
        if data.strategy != 'full' or data.n_total != len(expected) or data.n_shown != len(expected):
            raise RuntimeError('Unexpected sampled, binned or incomplete chart')
        pd.testing.assert_frame_equal(data.frame, expected)
        groups = set(expected['rowID'].unique()) if faceted else {None}
        counts, seen = [], set()
        for index in range(len(groups)):
            axes = canvas.axes_at(index, 0)
            group = axes.get_title().split('·')[0].strip() if faceted else None
            if group not in groups or group in seen:
                raise RuntimeError('Actual facet labels omit or duplicate a source group')
            seen.add(group)
            subset = expected if group is None else expected[expected['rowID'] == group]
            if faceted and axes.get_title().split('n=')[-1] != f'{len(subset):,}':
                raise RuntimeError('Actual facet label count differs from its source group')
            actual = axes.collections[0].get_offsets()
            points = subset[['cell_area', 'cell_channel_1_mean_intensity']].to_numpy()
            counts.append(check_points(points, actual))
            if screen.builder.spec.colour == 'columnID':
                from matplotlib.colors import to_rgb
                legend = canvas._figure.legends[0]
                labels = [label.get_text() for label in legend.get_texts()]
                handles = legend.legend_handles
                colours = {label:to_rgb(handle.get_markerfacecolor())
                           for label,handle in zip(labels,handles)}
                if set(labels) != set(expected_frame.columnID.unique()):
                    raise RuntimeError('Colour legend differs from source column groups')
                if not np.array_equal(points,actual):
                    raise RuntimeError('Colour audit requires the recorded source row order')
                wanted = [colours[label] for label in subset.columnID]
                shown = axes.collections[0].get_facecolors()[:,:3]
                if not np.allclose(wanted,shown,atol=1e-12,rtol=0):
                    raise RuntimeError('Point colours disagree with their labelled source groups')
        chart_checks.append({'scene': name, 'panel_point_counts': counts,
                             'all_rendered_frame_values_match_source': True,
                             'colour_assignments_checked':screen.builder.spec.colour == 'columnID'})

    try:
        for key in FOLDED_APPS:
            buttons = [b for b in screen.findChildren(FoldButton) if b.isVisible() and b.app_key == key]
            if len(buttons) != 1:
                raise RuntimeError(f'Expected one visible Graph Builder fold: {key}')
            with capture_ui.clip(capture, '02_fold_' + key):
                if live and key == 'trellis':
                    _inspect_trellis_fold(capture, screen, buttons[0], settle, timeout)
                elif live:
                    capture_ui.point_at(capture, buttons[0], dwell=1.5)
                else:
                    QTest.mouseMove(buttons[0])
                    settle(0.8)

        errors, accepted = [], []
        def choose():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise RuntimeError('Load table did not open the actual file picker')
                dialog.accepted.connect(lambda: accepted.append(True))
                capture_ui.dialog_geometry(capture, dialog, (1400, 950))
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), database)
                drv.snap(capture, '03_choose_database')
                capture_ui.wait(capture, 1.5, settle)
                QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open), Qt.LeftButton)
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:
                    dialog.reject()
        buttons = [b for b in screen.findChildren(QPushButton) if b.isVisible() and b.text().startswith('Load table')]
        if len(buttons) != 1:
            raise RuntimeError('Expected one visible Load table control')
        scene3 = drv.begin(capture, settle, '03_choose_database')
        capture_ui.when_modal(capture, 500, choose)
        QTest.mouseClick(buttons[0], Qt.LeftButton)
        drv.finish(capture, scene3)
        if errors or not accepted:
            raise RuntimeError('; '.join(errors) or 'Database selection was cancelled')
        scene4 = drv.begin(capture, settle, '04_real_cell_table')
        wait_loaded()
        if screen._table_picker.currentText() != 'cell':
            raise RuntimeError('Expected the actual cell measurement table')
        frame = screen._frame
        pd.testing.assert_frame_equal(frame, expected_frame)
        canvas = screen.builder.canvas
        capture_ui.point_at(capture, screen._table_picker, screen.builder.well._list)
        drv.snap(capture, '04_real_cell_table')
        drv.finish(capture, scene4)
        scene5 = drv.begin(capture, settle, '05_area_histogram')
        drag('cell_area', 'x')
        axes = canvas.axes_at()
        edges = canvas.scales.x_edges
        bars = axes.patches
        if len(bars) != len(edges)-1:
            raise RuntimeError('Expected one actual bar for each histogram bin')
        for i, bar in enumerate(bars):
            if not np.isclose(bar.get_x()+bar.get_width()/2, (edges[i]+edges[i+1])/2):
                raise RuntimeError('Actual histogram bar locations differ')
        histogram = check_histogram(expected_frame.cell_area, edges, [bar.get_height() for bar in bars])
        chart_checks.append({'scene':'05_area_histogram', 'actual_bar_counts':histogram})
        capture_ui.point_at(capture, canvas._canvas)
        drv.snap(capture, '05_area_histogram')
        drv.finish(capture, scene5)
        scene6 = drv.begin(capture, settle, '06_real_scatter')
        drag('cell_channel_1_mean_intensity', 'y')
        check_scatter('06_real_scatter', expected_frame)
        capture_ui.point_at(capture, canvas._canvas)
        drv.snap(capture, '06_real_scatter')
        drv.finish(capture, scene6)
        scene7 = drv.begin(capture, settle, '07_well_column_colour')
        drag('columnID', 'colour')
        check_scatter('07_well_column_colour', expected_frame)
        capture_ui.point_at(capture, canvas._canvas)
        drv.snap(capture, '07_well_column_colour')
        drv.finish(capture, scene7)
        initial_count = canvas.render_data.n_total
        if initial_count != len(frame):
            raise RuntimeError('The initial chart unexpectedly omits measurement rows')
        scene8 = drv.begin(capture, settle, '08_acquisition_row_facets')
        drag('rowID', 'facet_row')
        check_scatter('08_acquisition_row_facets', expected_frame, faceted=True)
        capture_ui.point_at(capture, canvas._canvas)
        drv.snap(capture, '08_acquisition_row_facets')
        drv.finish(capture, scene8)
        scene9 = drv.begin(capture, settle, '09_live_area_filter')
        picker = screen.filters._picker
        target = picker.findText('cell_area')
        if target < 0:
            raise RuntimeError('The actual filter menu has no cell_area entry')
        if live:
            capture_ui.select_combo(capture, picker, target)
        else:
            picker.setFocus()
            QTest.keyClick(picker, Qt.Key_Home)
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
        check_scatter('09_live_area_filter', expected_frame[expected_frame.cell_area >= cutoff], faceted=True)
        capture_ui.point_at(capture, canvas._canvas)
        drv.snap(capture, '09_live_area_filter')
        drv.finish(capture, scene9)
        scene10 = drv.begin(capture, settle, '10_filter_cleared')
        QTest.mouseClick(screen.filters._clear, Qt.LeftButton)
        settle(1)
        if canvas.render_data.n_total != initial_count:
            raise RuntimeError('Clearing the live filter did not restore the actual rows')
        check_scatter('10_filter_cleared', expected_frame, faceted=True)
        capture_ui.point_at(capture, canvas._canvas)
        drv.snap(capture, '10_filter_cleared')
        drv.finish(capture, scene10)
        scene11 = drv.begin(capture, settle, '11_actual_brush_state')
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
        if live:
            capture_ui._h().drag(capture, canvas._canvas, start, (end.x() - start.x(), end.y() - start.y()),
                                 duration=1.2)
        else:
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
        press = [event for event in events if event['kind'] == 'button_press_event']
        release = [event for event in events if event['kind'] == 'button_release_event']
        if len(press) != 1 or len(release) != 1 or not all(e['in_current_axes'] for e in press + release):
            raise RuntimeError('Expected one genuine rectangle inside the plotted axes')
        lo_x, hi_x = sorted([press[0]['x'],release[0]['x']])
        lo_y, hi_y = sorted([press[0]['y'],release[0]['y']])
        rows = expected_frame[expected_frame.cell_area.between(lo_x,hi_x) &
                              expected_frame.cell_channel_1_mean_intensity.between(lo_y,hi_y)]
        if not 0 < len(rows) < initial_count:
            raise RuntimeError('Expected a genuine partial rectangle of the unchanged source table')
        # This source's literal untyped identifiers contain no escaped parts.
        columns = ['plateID','rowID','columnID','fieldID','object_label']
        parts = rows[columns].astype(str)
        if parts.apply(lambda col: col.str.contains('_|%')).any().any():
            raise RuntimeError('Escaped identities require a separate source audit')
        expected_keys = parts.agg('_'.join,axis=1).tolist()
        brush_review = check_brush(expected_keys,canvas.link.selection.keys,selected,screen._to_annotate.isEnabled())
        if canvas.link.selection.source != canvas.link_source:
            raise RuntimeError('The actual brush publication belongs to a different view')
        current_axes = canvas.axes_at()
        if len(current_axes.collections) != 2 or not current_axes.collections[-1].get_visible():
            raise RuntimeError('The actual full scatter highlight artist is missing')
        highlight_points = rows[['cell_area', 'cell_channel_1_mean_intensity']].to_numpy()
        check_points(highlight_points, current_axes.collections[-1].get_offsets())
        if f'{selected:,} highlighted' not in canvas.notice():
            raise RuntimeError('The actual visible brush count notice is stale')
        check_scatter('11_brush_keeps_all_points', expected_frame)
        brush_review['highlight_points_independently_checked'] = len(rows)
        brush_proof = {
            'accepted': True, 'brush_review': brush_review,
            'rows': initial_count, 'visible_selected_rows': selected,
            'published_object_keys': len(canvas.link.selection),
            'handoff_enabled': screen._to_annotate.isEnabled(),
            'native_events': events, 'database_sha256_before': original_hash,
            'database_sha256_after': digest(database),
            'reason': 'The actual brush must visibly select rows and enable its handoff',
        }
        capture_ui.point_at(capture, screen._to_annotate)
        drv.snap(capture, '11_actual_brush_state')
        drv.finish(capture, scene11)
        write_json(captures / 'brush_outcome.json', brush_proof)
        if review_handoff:
            # A plain native click clears publication too, not only its stale display.
            with capture_ui.clip(capture, '12_selection_cleared'):
                QTest.mouseClick(canvas._canvas, Qt.LeftButton, pos=start)
                settle(0.8)
                clear_review = check_brush_cleared(canvas.link.selection.keys, canvas.selected_count(),
                                                  screen._to_annotate.isEnabled(), canvas.link.selection.is_active)
                check_points([], canvas.axes_at().collections[-1].get_offsets())
            extras = {'schema': stage / 'graph_schema' / captures.name / 'conditions.schema.json'}
            extras['schema'].parent.mkdir(parents=True, exist_ok=True)
            annotation_proof = record_annotations(app, screen, capture, settle, timeout, extras=extras)
            if digest(database) != original_hash:
                raise RuntimeError('Chart exploration changed the source database')
            from capture_graph_annotations import record_graph_extras
            annotation_proof['writes'] = record_graph_extras(app, screen, capture, settle, timeout,
                                                             write_json, captures)
            with sqlite3.connect('file:' + str(database) + '?mode=ro', uri=True) as connection:
                after = pd.read_sql_query('SELECT * FROM cell', connection)
            pd.testing.assert_frame_equal(after, expected_frame)
            annotation_proof['cell_table_unchanged_after_writes'] = True
            write_json(captures/'scientific_acceptance.json', {
                'accepted':True, 'scope':'Actual native charts, reversible filters and coherent brush handoff availability',
                'condition_annotations': annotation_proof,
                'brush_review':brush_review, 'clear_review':clear_review,
                'annotation_handoff_clicked':False,
                'database':str(database),'database_sha256':original_hash,'rows':initial_count,
                'area_filter_cutoff':cutoff,'filtered_rows':filtered_count,'filter_cleared':True,
                'selection_publication_cleared':True,'chart_checks':chart_checks,
                'folds':list(FOLDED_APPS),'final_spec':asdict(screen.builder.spec),
                'source_database_unchanged':True,'app_source_modified':False,
                'annotation_handoff_available':True,'chart_export_button_used':False,'published':False})
            print('Verified native graph, exact brush highlights and reversible handoff availability',flush=True)
            return
        if not 0 < selected < initial_count or not screen._to_annotate.isEnabled():
            write_json(captures / 'scientific_acceptance.json', brush_proof)
            raise RuntimeError('The actual brush did not create a partial linked selection')
        with capture_ui.clip(capture, '11_real_brush_selection'):
            capture_ui.point_at(capture, screen._to_annotate)
        with capture_ui.clip(capture, '12_selection_cleared'):
            QTest.mouseClick(canvas._canvas, Qt.LeftButton, pos=start)
            settle(0.8)
            clear_review = check_brush_cleared(canvas.link.selection.keys, canvas.selected_count(),
                                              screen._to_annotate.isEnabled(), canvas.link.selection.is_active)
            check_points([], canvas.axes_at().collections[-1].get_offsets())
        annotation_proof = record_annotations(app, screen, capture, settle, timeout)
        if digest(database) != original_hash:
            raise RuntimeError('Exploring the chart modified the downloaded database')
        write_json(captures / 'scientific_acceptance.json', {
            'accepted': True, 'database': str(database), 'database_sha256': original_hash,
            'table': 'cell', 'rows': initial_count, 'area_filter_cutoff': cutoff,
            'filtered_rows': filtered_count, 'filter_cleared': True,
            'condition_annotations': annotation_proof,
            'brushed_rows': selected, 'selection_cleared': True,
            'brush_review':brush_review, 'clear_review':clear_review,
            'source_database_unchanged': True, 'actual_native_column_drags': True,
            'folds': list(FOLDED_APPS), 'final_spec': asdict(screen.builder.spec),
            'notice': canvas.notice(), 'chart_or_spec_export_button_used': False,
            'annotation_handoff_clicked': False, 'published': False})
        print(f'Accepted Graph Builder: {initial_count} -> {filtered_count} -> {initial_count} rows', flush=True)
    finally:
        if desktop is not None:
            button(False)
            desktop.close()
