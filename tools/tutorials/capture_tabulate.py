"""Record genuine Tabulate pivots, exports and plot hand-offs on real cells."""
import ctypes
from pathlib import Path
import tempfile
import time

from capture_database import prepare_database_copy, require_unchanged_source, _digest
from capture_diagnostics import PrivateDesktop
from feature_explorer_evidence import read_measurements
from tabulate_evidence import verify_pivot, verify_csv


def record_tabulate(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer, QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QDialogButtonBox, QLineEdit, QPushButton
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.screens.tabulate import TabulateScreen
    import pandas as pd

    parent = Path(stage)/'tabulate_runs'; parent.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='real-measurements-', dir=parent))
    source = Path(stage)/'annotate_fresh/example_data/plate1/measurements/measurements.db'
    original = prepare_database_copy(source, work/'measurements.db')
    records, _ = read_measurements(work/'measurements.db')
    if len(records) != 2341:
        raise ValueError('Expected full downloaded cell table')
    proof = dict(lesson='61_tabulate', accepted=False, source=original,
                 synthetic_measurements=False, app_source_modified=False, published=False)
    panel = None; events = []; deadline = time.monotonic()+timeout
    desktop = PrivateDesktop(stage)
    xtest = ctypes.CDLL('libXtst.so.6')
    xtest.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
    xtest.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]

    def tick():
        if time.monotonic() > deadline: raise TimeoutError('Bounded Tabulate capture timed out')

    def click(widget):
        tick()
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Actual Tabulate control is not available: '+widget.objectName())
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.2)

    def idle():
        settle(.35)
        while panel.is_busy() or panel.active_jobs() or panel._refilter.isActive() or panel.pivot._debounce.isActive():
            tick(); settle(.1)
        settle(.3)

    def fill(widget, value):
        click(widget); QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        if str(value): QTest.keyClicks(widget, str(value))
        else: QTest.keyClick(widget, Qt.Key_Backspace)
        QTest.keyClick(widget, Qt.Key_Tab); settle(.2)

    def button(pressed):
        xtest.XTestFakeButtonEvent(desktop.display, 1, int(pressed), 0)
        desktop.x.XFlush(desktop.display)

    def motion(point):
        xtest.XTestFakeMotionEvent(desktop.display, -1, point.x(), point.y(), 0)
        desktop.x.XFlush(desktop.display)

    def drag(column, axis, graph=False):
        owner = panel.graph if graph else panel.pivot
        fill(owner.well._search, column)
        listing = owner.well._list
        items = [listing.item(i) for i in range(listing.count()) if listing.item(i).data(Qt.UserRole) == column]
        if len(items) != 1: raise ValueError('No unique actual column: '+column)
        listing.scrollToItem(items[0]); settle(.15)
        start = listing.viewport().mapToGlobal(listing.visualItemRect(items[0]).center())
        zone = owner.zone(axis) if graph else owner.wells[axis]._list.viewport()
        end = zone.mapToGlobal(zone.visibleRegion().boundingRect().center())
        motion(start); settle(.15); button(True)
        QTimer.singleShot(120, lambda: motion(start+QPoint(18, 0)))
        QTimer.singleShot(260, lambda: motion(QPoint((start.x()+end.x())//2, (start.y()+end.y())//2)))
        QTimer.singleShot(420, lambda: motion(end))
        QTimer.singleShot(650, lambda: button(False))
        settle(1.4); idle()
        selected = owner.spec.column_for(axis) == column if graph else column in owner.wells[axis].columns()
        if not selected: raise ValueError('Genuine native drag did not assign '+column)
        fill(owner.well._search, '')

    def clear_axis(axis):
        click(panel.pivot.wells[axis].findChild(QPushButton, 'PivotWellClear')); idle()
        if panel.pivot.wells[axis].columns(): raise ValueError('Axis clear did not empty actual well')

    def resize_splitter(splitter, index, target, vertical=False):
        handle = splitter.handle(index); start = handle.rect().center()
        current = handle.mapTo(window, start)
        end = start + (QPoint(0, target-current.y()) if vertical else QPoint(target-current.x(), 0))
        QTest.mousePress(handle, Qt.LeftButton, pos=start)
        QTest.mouseMove(handle, end, delay=150)
        QTest.mouseRelease(handle, Qt.LeftButton, pos=end); settle(.4)

    def picker(button_widget, path, name):
        accepted, errors = [], []
        timer, watchdog = QTimer(window), QTimer(window)
        timer.setSingleShot(True); watchdog.setSingleShot(True)
        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog): raise ValueError('Expected actual file picker')
                dialog.accepted.connect(lambda: accepted.append(True))
                dialog.resize(1400, 950)
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), path); capture(name)
                box = dialog.findChild(QDialogButtonBox)
                accept = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
                if len(accept) != 1: raise ValueError('No unique picker accept button')
                click(accept[0])
            except Exception as error:
                errors.append(str(error))
                if dialog is not None: dialog.reject()
        def abort():
            errors.append('Actual file picker timed out')
            dialog = app.activeModalWidget()
            if dialog is not None: dialog.reject()
        timer.timeout.connect(handle); watchdog.timeout.connect(abort)
        timer.start(300); watchdog.start(15000)
        try: click(button_widget)
        finally:
            timer.stop(); watchdog.stop(); timer.deleteLater(); watchdog.deleteLater()
        if errors or not accepted: raise ValueError('; '.join(errors) or 'No accepted file')
        idle()

    def check(name, population=None, cols=(), values=(), aggs=('n', 'mean', 'sd'), quantile=.75,
              rows=('plateID', 'rowID', 'columnID')):
        idle()
        if panel.pivot.result is None: raise ValueError('Actual pivot produced no result')
        p = verify_pivot(panel.pivot.result, records if population is None else population,
                         rows=rows, cols=cols,
                         values=values, aggs=aggs, quantile=quantile)
        p.update(source_label=panel._source.text(), notice=panel.pivot.notice.text())
        proof.setdefault('pivots', {})[name] = p; capture(name)
        return p

    def hover_cell(name, row, column, expected_text):
        table = panel.pivot.table
        item = table.item(row, column)
        if item is None: raise ValueError('No actual pivot cell to inspect')
        table.scrollToItem(item); settle(.2)
        motion(panel._source.mapToGlobal(panel._source.rect().center())); settle(.2)
        motion(table.viewport().mapToGlobal(table.visualItemRect(item).center())); settle(1.5)
        tips = [w for w in app.topLevelWidgets() if w.isVisible() and w.windowType() == Qt.ToolTip]
        if len(tips) != 1 or expected_text not in tips[0].text():
            raise ValueError('Actual cell hover did not show the expected full statistics')
        proof.setdefault('cell_tooltips', {})[name] = tips[0].text()
        capture(name, desktop=True)
        motion(panel._source.mapToGlobal(panel._source.rect().center())); settle(.3)

    def export(name, checked):
        path = work/(name+'.csv')
        picker(panel.pivot._export, path, name+'_picker')
        p = verify_csv(path, checked)
        p.update(path=str(path), sha256=_digest(path))
        proof.setdefault('exports', {})[name] = p; capture(name+'_complete')

    def graph_means(checked):
        frame = panel.graph.canvas._frame
        keys = checked['row_keys']
        mean_at = checked['csv_headers'].index('mean(cell_area)')
        wanted = {tuple(line[:len(keys)]): line[mean_at]
                  for line in checked['csv_rows'] if line[mean_at] is not None}
        if frame is None or len(frame) != len(wanted): raise ValueError('Summary plot must carry actual groups, not individual objects')
        if list(frame['value_column'].unique()) != ['cell_area']: raise ValueError('Wrong summary value in graph')
        for row in frame.to_dict('records'):
            key = tuple(row[k] for k in keys)
            if abs(row['mean']-wanted[key]) > 1e-8: raise ValueError('Plot summary did not receive independently verified group means')
        return dict(rows=len(frame), means=[{**{k:r[k] for k in keys}, 'mean':r['mean']}
                                          for r in frame.to_dict('records')], notice=panel.graph.canvas.notice())

    try:
        actions = [a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help']
        if len(actions) != 1: raise ValueError('No unique actual Help menu')
        menu = actions[0].menu()
        choices = [a for a in menu.actions() if a.text().replace('&', '') == 'Database browser']
        if len(choices) != 1: raise ValueError('No actual Database browser route')
        QTest.mouseClick(window.menuBar(), Qt.LeftButton, pos=window.menuBar().actionGeometry(actions[0]).center())
        settle(.3); capture('01_help_database_route')
        QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(choices[0]).center()); settle(1)
        host = window._screens['db_browser']
        folds = [w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key == 'tabulate']
        if len(folds) != 1: raise ValueError('No unique Database Browser Tabulate fold')
        capture('02_database_host'); click(folds[0]); settle(.8)
        panels = [w for w in window.findChildren(TabulateScreen) if w.isVisible()]
        if len(panels) != 1: raise ValueError('Actual Tabulate fold did not open')
        panel = panels[0]
        panel.pivot.computed.connect(lambda r: events.append(dict(source_rows=r.n_source_rows, shape=list(r.shape))))
        resize_splitter(panel.pivot.parentWidget(), 1, 1300, vertical=True)
        resize_splitter(panel.pivot.well.parentWidget().parentWidget(), 1, 1000)
        loads = [b for b in panel.findChildren(QPushButton) if b.text() == 'Load table…']
        picker(loads[0], work/'measurements.db', '03_choose_real_measurements')
        if panel._table_picker.currentText() != 'cell' or len(panel._frame) != 2341:
            raise ValueError('Actual table picker did not load the full cell table')
        capture('04_loaded_needs_axes')
        preset = [b for b in panel.pivot.findChildren(QPushButton) if b.text() == 'Plate / row / column']
        click(preset[0]); baseline_count = check('05_well_count_preset')
        drag('fieldID', 'cols'); field = check('06_real_missing_well_field', cols=('fieldID',))
        if field['empty'] != [dict(row=['plate1', 'r5', 'c2'], col=['f20'])]:
            raise ValueError('Unexpected absent real well/field combination')
        hover_cell('06b_blank_not_zero_tooltip', 1, 14, 'No objects here at all')
        export('07_field_counts_with_blank', field)
        clear_axis('cols'); drag('cell_area', 'values')
        base = check('08_area_mean_sample_sd', values=('cell_area',))
        hover_cell('08b_read_full_cell_statistics', 0, 3, 'n(cell_area) = 615')
        for agg in ('median', 'sem', 'quantile'): click(panel.pivot._agg_boxes[agg])
        expanded = ('n', 'mean', 'median', 'sd', 'sem', 'quantile')
        check('09_median_sem_upper_quartile', values=('cell_area',), aggs=expanded)
        hover_cell('09b_full_median_sem_quantile', 0, 3, 'sem(cell_area)')
        fill(panel.pivot._quantile.lineEdit(), .5)
        check('10_quantile_half_is_median', values=('cell_area',), aggs=expanded, quantile=.5)
        hover_cell('10b_half_quantile_tooltip', 0, 3, 'quantile(cell_area) = 22,400')
        for agg in ('median', 'sem', 'quantile'): click(panel.pivot._agg_boxes[agg])
        fill(panel.pivot._quantile.lineEdit(), .75)
        clear_axis('values'); drag('cell_channel_0_homogeneity_distance_32', 'values')
        missing = check('11_finite_feature_count', values=('cell_channel_0_homogeneity_distance_32',))
        if sum(line[3] for line in missing['csv_rows']) != 2334: raise ValueError('Expected seven real missing feature values')
        hover_cell('11b_source_count_vs_feature_n', 3, 3, 'n(cell_channel_0_homogeneity_distance_32) = 498')
        clear_axis('values'); drag('cell_area', 'values')
        base = check('12_restored_area', values=('cell_area',)); export('13_all_well_statistics', base)
        click(panel.pivot._plot); idle(); proof['initial_graph'] = graph_means(base)
        drag('columnID', 'x', graph=True); drag('mean', 'y', graph=True); drag('rowID', 'colour', graph=True)
        capture('14_four_group_means_not_2341_objects')
        proof['automatic_four_well_plot'] = dict(kinds=dict(panel.graph.canvas._kinds),
            kind=panel.graph.spec.resolved_kind(panel.graph.canvas._kinds), rows=len(panel.graph.canvas._frame))
        before = panel.graph.canvas._frame.copy(deep=True)
        picker_widget = panel.filters._picker
        click(picker_widget); QTest.keyClick(picker_widget.view(), Qt.Key_Home)
        index = picker_widget.findText('cell_area')
        if index < 0: raise ValueError('No actual area filter')
        for _ in range(index): QTest.keyClick(picker_widget.view(), Qt.Key_Down)
        QTest.keyClick(picker_widget.view(), Qt.Key_Return)
        click(panel.filters.findChild(QPushButton, 'FilterAddButton')); idle()
        fill(panel.filters._rows['cell_area']._low.lineEdit(), 21925); idle()
        selected = [r for r in records if r['cell_area'] >= 21925]
        if len(selected) != 1171: raise ValueError('Expected real median-threshold population')
        changed = check('15_live_pivot_old_plot', population=selected, values=('cell_area',))
        pd.testing.assert_frame_equal(panel.graph.canvas._frame, before)
        if changed['csv_rows'] == base['csv_rows']: raise ValueError('Positive filter did not change actual pivot means')
        proof['filter_requires_explicit_plot_handoff'] = True
        click(panel.pivot._plot); idle(); proof['filtered_graph'] = graph_means(changed)
        capture('16_plot_this_table_refreshes_summary'); export('17_filtered_well_statistics', changed)
        click(panel.filters._clear); idle()
        restored = check('18_clear_restores_cells', values=('cell_area',))
        if restored['csv_rows'] != base['csv_rows']: raise ValueError('Clear did not restore all real means')
        click(panel.pivot._plot); idle(); proof['restored_graph'] = graph_means(restored)
        capture('19_replot_restores_original_summary')
        drag('fieldID', 'rows')
        field_summary = check('20_field_level_is_a_different_unit', values=('cell_area',),
                              rows=('plateID', 'rowID', 'columnID', 'fieldID'))
        click(panel.pivot._plot); idle(); proof['field_graph'] = graph_means(field_summary)
        drag('fieldID', 'x', graph=True)
        combo = panel.graph._kind
        index = combo.findData('scatter')
        if index < 0: raise ValueError('Actual Plot menu has no Scatter choice')
        click(combo); capture('21_actual_plot_choices', desktop=True)
        QTest.keyClick(combo.view(), Qt.Key_Home)
        for _ in range(index): QTest.keyClick(combo.view(), Qt.Key_Down)
        QTest.keyClick(combo.view(), Qt.Key_Return); idle()
        canvas = panel.graph.canvas
        if canvas._kinds.get('mean') != 'continuous' or panel.graph.spec.kind != 'scatter':
            raise ValueError('Final field plot does not have a genuine continuous mean axis')
        observed = sorted(float(y) for ax in canvas.figure().axes for artist in ax.collections
                          for x,y in artist.get_offsets())
        expected = sorted(row['mean'] for row in canvas._frame.to_dict('records'))
        if len(observed) != 51 or len(expected) != 51 or any(abs(a-b)>1e-8 for a,b in zip(observed,expected)):
            raise ValueError('The 51 actual rendered scatter ordinates differ from checked group means')
        proof['field_graph'].update(kind=panel.graph.spec.kind, kinds=dict(canvas._kinds),
            independently_checked_rendered_points=len(observed),
            max_rendered_y_error=max(abs(a-b) for a,b in zip(observed, expected)))
        capture('22_fifty_one_actual_field_means')
        export('23_field_level_statistics', field_summary)
        proof['accepted'] = True
    finally:
        button(False); desktop.close()
        require_unchanged_source(source, original['source_bundle'])
        proof['original_unchanged'] = True
        proof['private_database_unchanged'] = _digest(work/'measurements.db') == original['database_sha256']
        proof['computed_events'] = events
        if panel is not None:
            idle(); proof['remaining_workers'] = panel.active_jobs()
        write_json(Path(captures)/'tabulate_acceptance.json', proof)
    if not proof['private_database_unchanged']: raise ValueError('Private measurement database changed')
