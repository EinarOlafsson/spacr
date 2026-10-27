"""Record real Small Multiples controls and independently check painted points.

No result/spec injection, synthetic cells, application edits or source writes.
All data changes use real selectors, native column drags or formula controls.
"""
import ctypes
from pathlib import Path
import tempfile
import time

import capture_ui
import capture_ui_drv_d as drv
from capture_database import prepare_database_copy, require_unchanged_source, _digest
from capture_diagnostics import PrivateDesktop
from feature_explorer_evidence import read_measurements
from trellis_evidence import verify_trellis, verify_rendered_axes


def _splitter_of(widget):
    """The nearest QSplitter above ``widget`` (the page may sit in a wrapper QWidget since the Home layout change)."""
    from PySide6.QtWidgets import QSplitter
    w = widget
    while w is not None and not isinstance(w, QSplitter):
        w = w.parentWidget()
    if w is None:
        raise RuntimeError('No splitter above ' + type(widget).__name__)
    return w


def _require_visible_trellis(panel, panels=None):
    """Reject an offscreen/folded result even when Matplotlib's objects exist."""
    pane = panel._splitter.pane('Trellis')
    canvas = panel.canvas._canvas
    if pane is None or pane.is_collapsed():
        raise ValueError('Actual foreground Trellis section is collapsed or missing')
    region = canvas.visibleRegion()
    if (not canvas.isVisible() or not canvas.window().isActiveWindow()
            or canvas.width() < 480 or canvas.height() < 320
            or region.rectCount() != 1 or region.boundingRect() != canvas.rect()):
        raise ValueError('Actual foreground Trellis canvas is hidden, clipped or too small')
    if panels is not None:
        # Matplotlib bounds use physical pixels; Qt geometry uses logical pixels.
        scale = float(canvas.devicePixelRatioF())
        frame = panel.canvas._figure.bbox
        for p in panels:
            if not p['occupied']:
                continue
            ax = panel.canvas._axes[(p['row'], p['col'])]
            box = ax.get_window_extent()
            if (not ax.get_visible() or box.width < 64*scale or box.height < 48*scale
                    or box.x0 < frame.x0 or box.y0 < frame.y0
                    or box.x1 > frame.x1 or box.y1 > frame.y1):
                raise ValueError('An actual Trellis result panel is hidden, clipped or too small')
    return dict(foreground_canvas_visible=True,
                canvas_logical_size=[canvas.width(), canvas.height()],
                occupied_panels_visible=(sum(bool(p['occupied']) for p in panels)
                                         if panels is not None else None))


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
    live = capture_ui.live(capture)
    desktop = xtest = None
    if not live:        # the Xvfb recorder drives drags with XTest; the host uses its real pointer
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

    def ensure_plot():
        splitter = screen.panel._splitter
        pane = splitter.pane('Trellis')
        if pane is None:
            raise ValueError('Actual Small Multiples Trellis section is missing')
        if pane.is_collapsed():
            heading = getattr(pane.widget, 'heading', None)
            if heading is None:
                raise ValueError('Actual Trellis fold heading is missing')
            click(heading); idle()
        return _require_visible_trellis(screen.panel)

    def fill(widget, value):
        if live:
            if str(value):
                capture_ui.set_text(capture, widget, str(value))
            else:
                h = capture_ui._h(); h.click(widget); h.key(widget, Qt.Key_A, Qt.ControlModifier)
                h.key(widget, Qt.Key_Backspace); h.key(widget, Qt.Key_Tab)
            return
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
        zone = owner.zone(axis); end = zone.mapToGlobal(zone.visibleRegion().boundingRect().center())
        if live:
            # A real drag of the list entry onto the axis zone (QDrag runs its own loop).
            capture_ui._wheel_item_into_view(capture, listing, items[0])
            start_local = listing.visualItemRect(items[0]).center()
            delta = end - listing.viewport().mapToGlobal(start_local)
            capture_ui._h().drag_qdrag(capture, listing.viewport(), start_local, (delta.x(), delta.y()), duration=1.2)
            capture_ui._h().pause(capture, 1.0)
        else:
            listing.scrollToItem(items[0]); settle(.15)
            start = listing.viewport().mapToGlobal(listing.visualItemRect(items[0]).center())
            motion(start); settle(.15); button(True)
            QTimer.singleShot(120, lambda: motion(start+QPoint(18, 0)))
            QTimer.singleShot(260, lambda: motion(QPoint((start.x()+end.x())//2, (start.y()+end.y())//2)))
            QTimer.singleShot(420, lambda: motion(end)); QTimer.singleShot(650, lambda: button(False))
            settle(1.4)
        idle()
        if owner.spec.graph.column_for(axis) != column: raise ValueError('Native drag did not assign '+column)
        fill(owner.well._search, '')

    def open_popup(box, frame, act=None):
        """Clip ``frame``: optional action, then the selector's real popup opened and read."""
        ensure_plot()
        started = drv.begin(capture, settle, frame)
        if act is not None: act()
        click(box); capture_ui.wait(capture, 1.5, settle)
        if not live: capture(frame, desktop=True)
        drv.finish(capture, started)

    def combo(box, value, data=False, popup_open=False):
        index = box.findData(value) if data else box.findText(value)
        if index < 0: raise ValueError('Actual selector does not offer '+str(value))
        if not popup_open: click(box)
        QTest.keyClick(box.view(), Qt.Key_Home)
        for _ in range(index): QTest.keyClick(box.view(), Qt.Key_Down)
        QTest.keyClick(box.view(), Qt.Key_Return); idle()
        if (box.currentData() if data else box.currentText()) != value:
            raise ValueError('Actual selector did not change as requested')

    def check(name, population=None, *, x='cell_area', row='rowID', col='columnID', mode='shared', wrap=0, act=None):
        ensure_plot()
        started = drv.begin(capture, settle, name)
        if act is not None: act()
        idle(); records = rows if population is None else population
        result = screen.panel.canvas.trellis
        if result is None: raise ValueError('Actual grid produced no result')
        p = verify_trellis(result, records, x=x, y='cell_channel_1_mean_intensity',
                           facet_row=row, facet_col=col, scale_x=mode, scale_y=mode, wrap=wrap)
        p.update(verify_rendered_axes(screen.panel.canvas, result, records, p))
        p.update(_require_visible_trellis(screen.panel, p['panels']))
        p.update(notice=screen.panel.canvas.notice(), source_label=screen._source.text())
        proof.setdefault('checks', {})[name] = p
        capture_ui.point_at(capture, screen.panel.canvas); drv.snap(capture, name); drv.finish(capture, started)
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
                if live:
                    capture_ui.scroll_to(capture, scrolls[0], box)
                else:
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
                dialog.accepted.connect(lambda: accepted.append(True)); capture_ui.dialog_geometry(capture, dialog, (1400, 950))
                fill(dialog.findChild(QLineEdit, 'fileNameEdit'), path); drv.snap(capture, '03_choose_real_measurements'); capture_ui.wait(capture, 1.5, settle)
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
        watchdog.timeout.connect(abort)
        watchdog.start(120000 if getattr(capture, 'host', None) is not None else 15000)
        ensure_plot()
        started = drv.begin(capture, settle, '03_choose_real_measurements')
        capture_ui.when_modal(capture, 300, handle)
        try: click(pressed)
        finally:
            drv.finish(capture, started)
            timer.stop(); watchdog.stop(); timer.deleteLater(); watchdog.deleteLater()
        if errors or not accepted: raise ValueError('; '.join(errors) or 'No accepted file')
        idle()

    try:
        tiles = [w for w in window.findChildren(QAbstractButton) if w.isVisible() and
                 (w.property('moduleAppKey') == 'graph_builder' or w.property('navKey') == 'graph_builder')]
        if not tiles: raise ValueError('No actual Home Graph Builder tile')
        with capture_ui.clip(capture, '01_graph_builder_host'):
            click(max(tiles, key=lambda w: w.width()*w.height())); settle(.8)
            host = window._screens['graph_builder']
            folds = [w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key == 'trellis']
            if len(folds) != 1: raise ValueError('No unique Graph Builder Small Multiples fold')
            capture_ui.point_at(capture, folds[0])
        scene2 = drv.begin(capture, settle, '02_current_small_multiples')
        click(folds[0]); settle(.8)
        panels = [w for w in window.findChildren(TrellisScreen) if w.isVisible()]
        if len(panels) != 1: raise ValueError('Actual Small Multiples fold did not open')
        screen = panels[0]; panel = screen.panel
        panel.canvas.trellis_rendered.connect(lambda r: events.append(dict(rows=len(r.frame), shape=list(r.shape))))
        splitter = panel._splitter
        # Open the real canvas, then leave it usable after resizing Channels.
        # A global x=1000 drag could squeeze/re-collapse it in the floating page.
        pane = splitter.pane('Trellis')
        if pane is None: raise ValueError('Actual Small Multiples Trellis section is missing')
        if pane.is_collapsed():
            heading = getattr(pane.widget, 'heading', None)
            if heading is None: raise ValueError('Actual Trellis fold heading is missing')
            click(heading); settle(.6)
        handle = splitter.handle(1)
        start = handle.rect().center()
        shelf_width = max(300, min(600, round(splitter.width()*.30)))
        end = start+QPoint(shelf_width-handle.mapTo(splitter, start).x(), 0)
        QTest.mousePress(handle, Qt.LeftButton, pos=start); QTest.mouseMove(handle, end, delay=150)
        QTest.mouseRelease(handle, Qt.LeftButton, pos=end); settle(.4)
        ensure_plot()
        capture_ui.point_at(capture, panel.well); drv.snap(capture, '02_current_small_multiples'); drv.finish(capture, scene2)
        loads = [b for b in screen.findChildren(QPushButton) if b.text() == 'Load table…']
        picker(loads[0], work/'measurements.db')
        if screen._table_picker.currentText() != 'cell' or len(screen._frame) != 2341:
            raise ValueError('Actual picker did not load full cell table')
        ensure_plot()
        scene4 = drv.begin(capture, settle, '04_actual_plot_choices')
        drag('cell_area', 'x'); drag('cell_channel_1_mean_intensity', 'y')
        click(panel._kind); capture_ui.wait(capture, 1.5, settle)
        if not live: capture('04_actual_plot_choices', desktop=True)
        drv.finish(capture, scene4)
        def four_wells():
            combo(panel._kind, 'scatter', data=True, popup_open=True)
            drag('rowID', 'facet_row'); drag('columnID', 'facet_col')
        baseline = check('05_four_wells_shared_axes', act=four_wells)
        for mode, number in [('free', '06'), ('row', '07'), ('col', '08')]:
            open_popup(panel._scale_x, number+'_scale_menu')
            def both(mode=mode):
                combo(panel._scale_x, mode, data=True, popup_open=True)
                combo(panel._scale_y, mode, data=True)
            check(number+'_both_axes_'+mode, mode=mode, act=both)
        check('09_shared_restored', act=lambda: (combo(panel._scale_x, 'shared', data=True),
                                                 combo(panel._scale_y, 'shared', data=True)))
        def area_filter():
            add_filter('cell_area'); fill(screen.filters._rows['cell_area']._low.lineEdit(), 21925)
        selected = [r for r in rows if r['cell_area'] >= 21925]
        if len(selected) != 1171: raise ValueError('Unexpected real area-filter population')
        check('10_live_area_filter', selected, act=area_filter)
        restored = check('11_all_cells_restored', act=lambda: click(screen.filters._clear))
        if restored['panels'] != baseline['panels']: raise ValueError('Clear did not restore all real points and scales')
        def empty_group():
            category('columnID', {'c2'}); category('fieldID', {'f20', 'f21'})
            drag('fieldID', 'facet_col')
        selected = [r for r in rows if r['columnID'] == 'c2' and r['fieldID'] in {'f20', 'f21'}]
        p = check('12_real_empty_group', selected, col='fieldID', act=empty_group)
        if p['rows'] != 174 or p['empty_real_panels'] != 1: raise ValueError('Expected one genuine missing group')
        check('13_unfiltered_four_wells', act=lambda: (click(screen.filters._clear), drag('columnID', 'facet_col')))
        p = check('14_two_way_wrap_ignored', wrap=5, act=lambda: fill(panel._wrap.lineEdit(), 5))
        if 'ignored' not in p['notice'].lower():
            # The spin box takes the typed 5 and the grid redraws on a coalesced timer; give both a turn, then read again.
            for _ in range(40):
                if panel._wrap.value() == 5 and 'ignored' in screen.panel.canvas.notice().lower():
                    break
                settle(.25); idle()
            p['notice'] = screen.panel.canvas.notice()
        if p['shape'] != [2, 2] or 'ignored' not in p['notice'].lower():
            raise ValueError('Expected explicit ignored two-way wrap notice (shape %s, wrap box %s, notice %r)'
                             % (p['shape'], panel._wrap.value(), p['notice']))
        p = check('15_twelve_fields_and_three_unused_slots', row=None, col='fieldID', wrap=5,
                  act=lambda: (click(panel.zone('facet_row')._clear), drag('fieldID', 'facet_col')))
        if (p['shape'], p['real_panels'], p['unused_wrap_slots'], p['outside_displayed_levels']) != ([3,5],12,3,264):
            raise ValueError('Expected actual twelve-level cap, three padding slots and 264 omitted points')
        check('16_original_groups_restored', act=lambda: (fill(panel._wrap.lineEdit(), 0),
              drag('rowID', 'facet_row'), drag('columnID', 'facet_col')))
        tabs = screen.formulas.parentWidget().parentWidget()
        if not isinstance(tabs, QTabWidget): raise ValueError('Expected actual Filter/Columns tabs')
        ensure_plot()
        with capture_ui.clip(capture, '17_actual_area_formula'):
            if live:
                capture_ui.select_tab(capture, tabs, tabs.indexOf(screen.formulas))
            else:
                QTest.mouseClick(tabs.tabBar(), Qt.LeftButton, pos=tabs.tabBar().tabRect(tabs.indexOf(screen.formulas)).center())
            settle(.3); fill(screen.formulas._name, 'area_k'); fill(screen.formulas._expression, 'cell_area/1e3')
            while not screen.formulas._add.isEnabled(): tick(); settle(.1)
        def add_formula():
            click(screen.formulas._add); idle()
            computed = screen.formulas.computed_frame()
            if len(computed) != 2341 or any(a != b/1000 for a,b in zip(computed['area_k'], computed['cell_area'])):
                raise ValueError('Actual formula differs from independent scaling')
            drag('area_k', 'x')
        scaled = [dict(r, area_k=r['cell_area']/1000) for r in rows]
        check('18_rescaled_area_not_new_measurement', scaled, x='area_k', act=add_formula)
        def remove_formula():
            drag('cell_area', 'x'); items = screen.formulas._list
            QTest.mouseClick(items.viewport(), Qt.LeftButton, pos=items.visualItemRect(items.item(0)).center())
            remove = next(b for b in screen.formulas.findChildren(QPushButton) if b.text() == 'Remove')
            click(remove); idle()
        check('19_formula_removed_original_points', act=remove_formula)
        if 'area_k' in screen.formulas.computed_frame(): raise ValueError('Derived column was not removed')
        proof['accepted'] = True
    finally:
        if desktop is not None:
            button(False); desktop.close()
        require_unchanged_source(source, original['source_bundle']); proof['original_unchanged'] = True
        proof['private_database_unchanged'] = _digest(work/'measurements.db') == original['database_sha256']
        proof['render_events'] = events
        if screen is not None:
            idle(); proof['remaining_workers'] = screen.active_jobs()
        write_json(Path(captures)/'trellis_acceptance.json', proof)
    if not proof['private_database_unchanged']: raise ValueError('Private measurement database changed')
