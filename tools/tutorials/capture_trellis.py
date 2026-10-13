"""Record real Small Multiples controls and independently check painted points.

No result/spec injection, synthetic cells, application edits or source writes.
All data changes use real selectors, native column drags or formula controls.
"""
import ctypes
from pathlib import Path
import tempfile
import time

from capture_database import prepare_database_copy, require_unchanged_source, _digest
from capture_diagnostics import PrivateDesktop
from feature_explorer_evidence import read_measurements
from trellis_evidence import verify_trellis, verify_rendered_axes


def record_trellis(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer, QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QAbstractButton, QFileDialog, QDialogButtonBox,
                                  QLineEdit, QPushButton, QScrollArea, QTabWidget)
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.screens.trellis import TrellisScreen

    parent = Path(stage)/'trellis_runs'; parent.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='real-measurements-', dir=parent))
    source = Path(stage)/'annotate_fresh/example_data/plate1/measurements/measurements.db'
    original = prepare_database_copy(source, work/'measurements.db')
    rows, _ = read_measurements(work/'measurements.db')
    if len(rows) != 2341: raise ValueError('Expected full downloaded cell table')
    proof = dict(lesson='63_small_multiples', accepted=False, source=original,
                 synthetic_measurements=False, app_source_modified=False, published=False)
    screen = None; events = []; deadline = time.monotonic()+timeout
    desktop = PrivateDesktop(stage)
    xtest = ctypes.CDLL('libXtst.so.6')
    xtest.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
    xtest.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]

    def tick():
        if time.monotonic() > deadline: raise TimeoutError('Bounded Small Multiples capture timed out')

    def click(widget):
        tick()
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('Actual Small Multiples control unavailable: '+widget.objectName())
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.2)

    def idle():
        settle(.3)
        while screen.is_busy() or screen.active_jobs() or screen.filters._debounce.isActive() or screen.panel.canvas._debounce.isActive():
            tick(); settle(.1)
        settle(.3)

    def fill(widget, value):
        click(widget); QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        if str(value): QTest.keyClicks(widget, str(value))
        else: QTest.keyClick(widget, Qt.Key_Backspace)
        QTest.keyClick(widget, Qt.Key_Tab); settle(.2)

    def button(pressed):
        xtest.XTestFakeButtonEvent(desktop.display, 1, int(pressed), 0); desktop.x.XFlush(desktop.display)

    def motion(point):
        xtest.XTestFakeMotionEvent(desktop.display, -1, point.x(), point.y(), 0); desktop.x.XFlush(desktop.display)

    def drag(column, axis):
        owner = screen.panel
        fill(owner.well._search, column)
        listing = owner.well._list
        items = [listing.item(i) for i in range(listing.count()) if listing.item(i).data(Qt.UserRole) == column]
        if len(items) != 1: raise ValueError('No unique actual column: '+column)
        listing.scrollToItem(items[0]); settle(.15)
        start = listing.viewport().mapToGlobal(listing.visualItemRect(items[0]).center())
        zone = owner.zone(axis); end = zone.mapToGlobal(zone.visibleRegion().boundingRect().center())
        motion(start); settle(.15); button(True)
        QTimer.singleShot(120, lambda: motion(start+QPoint(18, 0)))
        QTimer.singleShot(260, lambda: motion(QPoint((start.x()+end.x())//2, (start.y()+end.y())//2)))
        QTimer.singleShot(420, lambda: motion(end)); QTimer.singleShot(650, lambda: button(False))
        settle(1.4); idle()
        if owner.spec.graph.column_for(axis) != column: raise ValueError('Native drag did not assign '+column)
        fill(owner.well._search, '')

    def combo(box, value, data=False, frame=None):
        index = box.findData(value) if data else box.findText(value)
        if index < 0: raise ValueError('Actual selector does not offer '+str(value))
        click(box)
        if frame: capture(frame, desktop=True)
        QTest.keyClick(box.view(), Qt.Key_Home)
        for _ in range(index): QTest.keyClick(box.view(), Qt.Key_Down)
        QTest.keyClick(box.view(), Qt.Key_Return); idle()
        if (box.currentData() if data else box.currentText()) != value:
            raise ValueError('Actual selector did not change as requested')

    def check(name, population=None, *, x='cell_area', row='rowID', col='columnID', mode='shared', wrap=0):
        idle(); records = rows if population is None else population
        result = screen.panel.canvas.trellis
        if result is None: raise ValueError('Actual grid produced no result')
        p = verify_trellis(result, records, x=x, y='cell_channel_1_mean_intensity',
                           facet_row=row, facet_col=col, scale_x=mode, scale_y=mode, wrap=wrap)
        p.update(verify_rendered_axes(screen.panel.canvas, result, records, p))
        p.update(notice=screen.panel.canvas.notice(), source_label=screen._source.text())
        proof.setdefault('checks', {})[name] = p; capture(name)
        return p

    def add_filter(column):
        combo(screen.filters._picker, column)
        click(screen.filters.findChild(QPushButton, 'FilterAddButton')); idle()

    def category(column, keep):
        add_filter(column); row = screen.filters._rows[column]
        for box in row._boxes:
            if box.text() in keep: continue
            if box.visibleRegion().boundingRect().height() < box.height():
                scrolls = row.findChildren(QScrollArea)
                if len(scrolls) != 1: raise ValueError('No unique actual category scrollbar')
                bar = scrolls[0].verticalScrollBar()
                bar.setFocus(); QTest.keyClick(bar, Qt.Key_Home); settle(.1)
                while box.visibleRegion().boundingRect().height() < box.height():
                    tick(); before = bar.value(); QTest.keyClick(bar, Qt.Key_Down); settle(.03)
                    if bar.value() == before: raise ValueError('Actual scrolling did not reveal category')
            click(box); idle()
            if box.isChecked(): raise ValueError('Actual category toggle did not clear')
        if {b.text() for b in row._boxes if b.isChecked()} != set(keep):
            raise ValueError('Actual category selection differs')

    def picker(pressed, path):
        accepted, errors = [], []
        timer, watchdog = QTimer(window), QTimer(window)
        timer.setSingleShot(True); watchdog.setSingleShot(True)
        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog): raise ValueError('Expected genuine file picker')
                dialog.accepted.connect(lambda: accepted.append(True)); dialog.resize(1400, 950)
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), path); capture('03_choose_real_measurements')
                box = dialog.findChild(QDialogButtonBox)
                choices = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
                if len(choices) != 1: raise ValueError('No unique picker accept action')
                click(choices[0])
            except Exception as error:
                errors.append(str(error))
                if dialog is not None: dialog.reject()
        def abort():
            errors.append('Actual file picker timed out'); dialog = app.activeModalWidget()
            if dialog is not None: dialog.reject()
        timer.timeout.connect(handle); watchdog.timeout.connect(abort)
        timer.start(300); watchdog.start(15000)
        try: click(pressed)
        finally:
            timer.stop(); watchdog.stop(); timer.deleteLater(); watchdog.deleteLater()
        if errors or not accepted: raise ValueError('; '.join(errors) or 'No accepted file')
        idle()

    try:
        tiles = [w for w in window.findChildren(QAbstractButton) if w.isVisible() and
                 (w.property('moduleAppKey') == 'graph_builder' or w.property('navKey') == 'graph_builder')]
        if not tiles: raise ValueError('No actual Home Graph Builder tile')
        click(max(tiles, key=lambda w: w.width()*w.height())); settle(.8)
        host = window._screens['graph_builder']
        folds = [w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key == 'trellis']
        if len(folds) != 1: raise ValueError('No unique Graph Builder Small Multiples fold')
        capture('01_graph_builder_host'); click(folds[0]); settle(.8)
        panels = [w for w in window.findChildren(TrellisScreen) if w.isVisible()]
        if len(panels) != 1: raise ValueError('Actual Small Multiples fold did not open')
        screen = panels[0]; panel = screen.panel
        panel.canvas.trellis_rendered.connect(lambda r: events.append(dict(rows=len(r.frame), shape=list(r.shape))))
        splitter = panel.well.parentWidget().parentWidget(); handle = splitter.handle(1)
        start = handle.rect().center(); end = start+QPoint(1000-handle.mapTo(window, start).x(), 0)
        QTest.mousePress(handle, Qt.LeftButton, pos=start); QTest.mouseMove(handle, end, delay=150)
        QTest.mouseRelease(handle, Qt.LeftButton, pos=end); settle(.4); capture('02_current_small_multiples')
        loads = [b for b in screen.findChildren(QPushButton) if b.text() == 'Load table…']
        picker(loads[0], work/'measurements.db')
        if screen._table_picker.currentText() != 'cell' or len(screen._frame) != 2341:
            raise ValueError('Actual picker did not load full cell table')
        drag('cell_area', 'x'); drag('cell_channel_1_mean_intensity', 'y')
        combo(panel._kind, 'scatter', data=True, frame='04_actual_plot_choices')
        drag('rowID', 'facet_row'); drag('columnID', 'facet_col')
        baseline = check('05_four_wells_shared_axes')
        for mode, number in [('free', '06'), ('row', '07'), ('col', '08')]:
            combo(panel._scale_x, mode, data=True, frame=number+'_scale_menu')
            combo(panel._scale_y, mode, data=True)
            check(number+'_both_axes_'+mode, mode=mode)
        combo(panel._scale_x, 'shared', data=True); combo(panel._scale_y, 'shared', data=True)
        check('09_shared_restored')
        add_filter('cell_area'); fill(screen.filters._rows['cell_area']._low.lineEdit(), 21925)
        selected = [r for r in rows if r['cell_area'] >= 21925]
        if len(selected) != 1171: raise ValueError('Unexpected real area-filter population')
        check('10_live_area_filter', selected)
        click(screen.filters._clear); restored = check('11_all_cells_restored')
        if restored['panels'] != baseline['panels']: raise ValueError('Clear did not restore all real points and scales')
        category('columnID', {'c2'}); category('fieldID', {'f20', 'f21'})
        drag('fieldID', 'facet_col')
        selected = [r for r in rows if r['columnID'] == 'c2' and r['fieldID'] in {'f20', 'f21'}]
        p = check('12_real_empty_group', selected, col='fieldID')
        if p['rows'] != 174 or p['empty_real_panels'] != 1: raise ValueError('Expected one genuine missing group')
        click(screen.filters._clear); drag('columnID', 'facet_col'); check('13_unfiltered_four_wells')
        fill(panel._wrap.lineEdit(), 5); p = check('14_two_way_wrap_ignored', wrap=5)
        if p['shape'] != [2, 2] or 'ignored' not in p['notice'].lower():
            raise ValueError('Expected explicit ignored two-way wrap notice')
        click(panel.zone('facet_row')._clear); drag('fieldID', 'facet_col')
        p = check('15_twelve_fields_and_three_unused_slots', row=None, col='fieldID', wrap=5)
        if (p['shape'], p['real_panels'], p['unused_wrap_slots'], p['outside_displayed_levels']) != ([3,5],12,3,264):
            raise ValueError('Expected actual twelve-level cap, three padding slots and 264 omitted points')
        fill(panel._wrap.lineEdit(), 0); drag('rowID', 'facet_row'); drag('columnID', 'facet_col')
        check('16_original_groups_restored')
        tabs = screen.formulas.parentWidget().parentWidget()
        if not isinstance(tabs, QTabWidget): raise ValueError('Expected actual Filter/Columns tabs')
        QTest.mouseClick(tabs.tabBar(), Qt.LeftButton, pos=tabs.tabBar().tabRect(tabs.indexOf(screen.formulas)).center())
        settle(.3); fill(screen.formulas._name, 'area_k'); fill(screen.formulas._expression, 'cell_area/1e3')
        while not screen.formulas._add.isEnabled(): tick(); settle(.1)
        capture('17_actual_area_formula'); click(screen.formulas._add); idle()
        computed = screen.formulas.computed_frame()
        if len(computed) != 2341 or any(a != b/1000 for a,b in zip(computed['area_k'], computed['cell_area'])):
            raise ValueError('Actual formula differs from independent scaling')
        drag('area_k', 'x'); scaled = [dict(r, area_k=r['cell_area']/1000) for r in rows]
        check('18_rescaled_area_not_new_measurement', scaled, x='area_k')
        drag('cell_area', 'x'); items = screen.formulas._list
        QTest.mouseClick(items.viewport(), Qt.LeftButton, pos=items.visualItemRect(items.item(0)).center())
        remove = next(b for b in screen.formulas.findChildren(QPushButton) if b.text() == 'Remove')
        click(remove); idle(); check('19_formula_removed_original_points')
        if 'area_k' in screen.formulas.computed_frame(): raise ValueError('Derived column was not removed')
        proof['accepted'] = True
    finally:
        button(False); desktop.close()
        require_unchanged_source(source, original['source_bundle']); proof['original_unchanged'] = True
        proof['private_database_unchanged'] = _digest(work/'measurements.db') == original['database_sha256']
        proof['render_events'] = events
        if screen is not None:
            idle(); proof['remaining_workers'] = screen.active_jobs()
        write_json(Path(captures)/'trellis_acceptance.json', proof)
    if not proof['private_database_unchanged']: raise ValueError('Private measurement database changed')
