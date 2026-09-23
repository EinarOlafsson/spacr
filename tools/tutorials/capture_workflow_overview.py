"""Record workflow-map lessons through actual Home and folded-module controls.

This is a navigation and data-handoff lesson. Worked experiments belong to
the linked module lessons; this recording starts no analysis or download.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def record_overview(app, window, captures, capture, settle, write_json,
                    lesson_id='78_spacr_screens'):
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QDialog, QTableView, QTextBrowser
    from spacr.qt.widgets.fold_strip import FoldButton

    root = Path(__file__).resolve().parents[2]
    map_path = root / 'spacr/resources/module_workflows.json'
    if lesson_id not in ('78_spacr_screens', '79_module_inputs_outputs', '80_image_analysis_pathways',
                         '81_sequencing_pathways'):
        raise ValueError('This recorder has not validated that workflow lesson')
    lesson_path = root / 'tools/tutorials/lessons' / (lesson_id + '.json')
    data = json.loads(map_path.read_text())
    lesson = json.loads(lesson_path.read_text())
    keys = data['tutorials'][lesson['id']]['modules']
    expected = ['home', *('module_' + key for key in keys), 'home_summary']
    if [scene['visual'] for scene in lesson['scenes']] != expected:
        raise ValueError('The workflow lesson and map disagree')

    def click(button):
        visible = button.visibleRegion()
        if not button.isVisible() or not button.isEnabled() or visible.isEmpty():
            raise ValueError('A required navigation control is not usable')
        QTest.mouseClick(button, Qt.LeftButton, pos=visible.boundingRect().center())
        settle(.6)

    def home():
        if window._startup.isVisible():
            return
        choices = [button for button in window.findChildren(QAbstractButton)
                   if button.property('navKey') == '__home__' and button.isVisible()]
        if not choices:
            raise ValueError('The visible Home navigation button is missing')
        click(max(choices, key=lambda button: button.width() * button.height()))
        if not window._startup.isVisible():
            raise ValueError('The Home button did not show Home')

    def search_module(key, visual):
        from spacr.qt.help_search import field_of

        field = field_of(window)
        if field is None or not field.isVisible():
            raise ValueError('The native Help search is unavailable')
        query = data['modules'][key]['name']
        field.setFocus()
        QTest.keyClick(field, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(field, query)
        deadline = time.monotonic() + 30
        while True:
            matches = [index for index, entry in enumerate(field.results())
                       if entry.kind == 'module' and entry.payload.get('app') == key]
            if (len(matches) == 1 and not field._debounce.isActive()
                    and field.popup().isVisible()):
                break
            if time.monotonic() >= deadline:
                raise ValueError('The actual Help index did not find module ' + key)
            settle(.1)
        for _ in range(matches[0]):
            QTest.keyClick(field, Qt.Key_Down)
        if field._list.currentRow() != matches[0]:
            raise ValueError('The intended native Help result is not selected')
        settle(.4)
        capture(visual, desktop=True)
        QTest.keyClick(field, Qt.Key_Return)
        settle(2)
        screen = window._screens.get(key)
        if screen is None or not screen.isVisible():
            raise ValueError('The actual Help result did not open ' + key)
        if getattr(screen, '_worker_thread_is_running', lambda: False)():
            raise ValueError(f'Opening {key} unexpectedly started an analysis')
        field.clear()
        return screen

    home()
    capture('home')
    routes = []
    for key in keys:
        home()
        module = data['modules'][key]
        scene_captured = False
        if module.get('api_entry'):
            from spacr.qt.help_search import field_of

            field = field_of(window)
            if field is None or not field.isVisible():
                raise ValueError('The native Help search is unavailable')
            symbol = module['api_entry']
            query = symbol.rsplit('.', 1)[-1]
            field.setFocus()
            QTest.keyClick(field, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(field, query)
            deadline = time.monotonic() + 30
            while not any(entry.kind == 'api' and entry.payload.get('symbol') == symbol
                          for entry in field.results()):
                if time.monotonic() >= deadline:
                    raise ValueError('The actual Help index did not find ' + symbol)
                settle(.1)
            if not field.popup().isVisible():
                raise ValueError('The API search result is not visible')
            settle(.5)
            capture('api_search_' + key, desktop=True)
            QTest.keyClick(field, Qt.Key_Return)
            deadline = time.monotonic() + 15
            dialog = None
            while dialog is None:
                dialogs = [widget for widget in app.topLevelWidgets()
                           if isinstance(widget, QDialog) and widget.isVisible()
                           and widget.objectName() == 'HelpSearchApiEntry'
                           and widget.windowTitle() == symbol]
                if len(dialogs) == 1:
                    dialog = dialogs[0]
                elif time.monotonic() >= deadline:
                    raise ValueError('The genuine API result did not open its local reference')
                else:
                    settle(.1)
            body = dialog.findChild(QTextBrowser)
            from spacr.qt.help_search import local_docstring

            if body is None or body.toPlainText() != local_docstring(symbol):
                raise ValueError('The displayed API reference differs from the current source')
            dialog.resize(2200, 1400)
            dialog.move(window.mapToGlobal(QPoint(800, 350)))
            settle(.8)
            capture('module_' + key)
            routes.append({'module': key, 'api_symbol': symbol, 'search_query': query,
                           'real_help_result_visible': True, 'api_executed': False,
                           'local_docstring_displayed': True,
                           'displayed_docstring_sha256': hashlib.sha256(body.toPlainText().encode()).hexdigest(),
                           'home_tile_clicked': False})
            QTest.keyClick(dialog, Qt.Key_Escape)
            settle(.3)
            QTest.keyClick(field, Qt.Key_Escape)
            field.clear()
            settle(.3)
            continue
        host_key = module['home']
        if not host_key:
            screen = search_module(key, 'help_search_' + key)
            capture('module_' + key)
            routes.append({'module': key, 'home_tile_clicked': False,
                           'search_query': module['name'], 'real_help_result_visible': True,
                           'module_result_selected': True, 'screen_visible': True,
                           'screen_class': type(screen).__name__})
            continue
        tiles = [button for button in window._startup.findChildren(QAbstractButton)
                 if button.property('moduleAppKey') == host_key]
        tile = None
        if not tiles and module['parent'] and data['modules'][host_key]['home'] is None:
            screen = search_module(host_key, 'help_host_' + key)
        else:
            tabs = window._startup._tabs
            for index in range(tabs.count()):
                candidates = [button for button in tiles if tabs.widget(index).isAncestorOf(button)]
                if candidates:
                    QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                                     pos=tabs.tabBar().tabRect(index).center())
                    settle(.4)
                    tile = max(candidates, key=lambda button: button.width() * button.height())
                    break
            if tile is None:
                raise ValueError(f'No category contains the native {key} tile')
            click(tile)
            settle(2)
            screen = window._screens.get(host_key)
        if screen is None or not screen.isVisible():
            raise ValueError(f'The native Home tile did not open {host_key}')
        pages = getattr(screen, '_fold_pages', None)
        if pages is not None and pages.currentIndex() != 0:
            QTest.mouseClick(pages.tabBar(), Qt.LeftButton,
                             pos=pages.tabBar().tabRect(0).center())
            settle(.4)
        if module['parent']:
            expected_screen_key = key
            if module['parent'] == 'toxoplasma':
                assay_tiles = [button for button in screen.findChildren(QAbstractButton)
                               if button.property('organismModuleKey') == key]
                if len(assay_tiles) != 1:
                    raise ValueError('No unique organism assay tile for ' + key)
                screen._module_scroll.ensureWidgetVisible(assay_tiles[0])
                settle(.4)
                capture('organism_route_' + key)
                click(assay_tiles[0])
                settle(1)
                screen = window._screens.get(key)
            elif key == 'ops':
                from spacr.qt.screens.mask import ops_page

                switch = screen._ops_switch
                if not switch.isChecked():
                    click(switch)
                settle(1)
                manager = ops_page(screen)
                screen = manager.page if manager is not None else None
            elif key in ('classify', 'ml_analyze'):
                from PySide6.QtWidgets import QComboBox

                selector = screen._settings_model._widgets.get('classifier_family')
                family = 'cv' if key == 'classify' else 'ml'
                if not isinstance(selector, QComboBox) or selector.findData(family) < 0:
                    raise ValueError('The native classifier family selector is missing')
                section = next(section for section in screen._settings_sections
                               if section.isAncestorOf(selector))
                if not section.is_expanded():
                    screen._settings_scroll.ensureWidgetVisible(section.header())
                    settle(.2)
                    click(section.header())
                screen._settings_scroll.ensureWidgetVisible(selector)
                click(selector)
                QTest.keyClick(selector.view(), Qt.Key_Home)
                for _ in range(selector.findData(family)):
                    QTest.keyClick(selector.view(), Qt.Key_Down)
                QTest.keyClick(selector.view(), Qt.Key_Return)
                settle(.7)
                if screen._settings_model.collect().get('classifier_family') != family:
                    raise ValueError('The actual family selector did not reach ' + family)
                expected_screen_key = 'classify_merged'
            elif key == 'parameter_sweep':
                folder = screen._actions_folder
                if folder.shut:
                    click(folder.heading)
                switch = getattr(screen, '_sweep_switch', None)
                if switch is None:
                    raise ValueError('The native Parameter sweep switch is missing')
                if not switch.isChecked():
                    click(switch)
                settle(.8)
                if not screen._sweep_card.isVisible():
                    raise ValueError('The Parameter sweep card did not open')
                expected_screen_key = 'regression'
            elif key == 'regression_diagnostics':
                from PySide6.QtWidgets import QMessageBox

                buttons = [button for button in screen.findChildren(FoldButton)
                           if button.isVisible() and button.app_key == key]
                if len(buttons) != 1:
                    raise ValueError('The native Diagnostics control is unavailable')
                observed = {}
                diagnostics_deadline = time.monotonic() + 30

                def record_empty_diagnostics():
                    dialog = app.activeModalWidget()
                    if dialog is None and time.monotonic() < diagnostics_deadline:
                        QTimer.singleShot(100, record_empty_diagnostics)
                        return
                    try:
                        if not isinstance(dialog, QMessageBox) or dialog.windowTitle() != 'Diagnostics':
                            raise ValueError('The native missing-diagnostics dialog did not appear')
                        if dialog.text() != ('This project has no regression diagnostics yet. They '
                                             'are written when a regression finishes.'):
                            raise ValueError('Unexpected regression-diagnostics state')
                        capture('module_regression_diagnostics', desktop=True)
                        observed['captured'] = True
                    except Exception as error:
                        observed['error'] = str(error)
                    finally:
                        if dialog is not None:
                            dialog.accept()

                QTimer.singleShot(500, record_empty_diagnostics)
                click(buttons[0])
                if not observed.get('captured'):
                    raise ValueError(observed.get('error', 'The diagnostics dialog was not captured'))
                scene_captured = True
                expected_screen_key = 'regression'
            else:
                buttons = [button for button in screen.findChildren(FoldButton)
                           if button.isVisible() and button.app_key == key]
                if len(buttons) != 1:
                    raise ValueError('No unique visible folded route for ' + key)
                if key == 'cellpose_all' and (screen._folder or screen._image_files):
                    raise ValueError('The navigation capture must never mask a loaded folder')
                click(buttons[0])
                settle(1)
                if key == 'timelapse':
                    if not screen._settings_model.collect().get('timelapse'):
                        raise ValueError('The native Timelapse switch did not enable tracking')
                    expected_screen_key = 'mask'
                elif key == 'cellpose_all':
                    if screen._status_label.text() != 'Open a folder of images before masking it.':
                        raise ValueError('The folder action did not retain its empty-input guard')
                    expected_screen_key = 'make_masks'
                elif module['parent'] == 'make_masks':
                    screen = screen._fold_dialogs.get(key)
                elif key == 'hit_list':
                    from spacr.qt.screens.regression import results_panel

                    panel = results_panel(screen)
                    if panel is None or panel.tabs.currentWidget() is not panel.hits:
                        raise ValueError('The native Hits control did not select its result tab')
                    screen = panel.hits
                else:
                    openers = [opener for opener in getattr(screen, '_fold_openers', ())
                               if opener.key == key]
                    visible = [opener.window for opener in openers
                               if opener.window is not None and opener.window.isVisible()]
                    if not openers:
                        visible = [widget for widget in app.allWidgets()
                                   if not isinstance(widget, QAbstractButton)
                                   and getattr(widget, 'app_key', None) == key and widget.isVisible()]
                    if len(visible) != 1:
                        raise ValueError('The folded route did not show exactly one ' + key)
                    screen = visible[0]
            if screen is None or not screen.isVisible() or getattr(screen, 'app_key', expected_screen_key) != expected_screen_key:
                raise ValueError('The native folded control did not open ' + key)
        if getattr(screen, '_worker_thread_is_running', lambda: False)():
            raise ValueError(f'Opening {key} unexpectedly started an analysis')
        if key == 'barcode_qc':
            section = next(section for section in screen._settings_sections
                           if 'count_data' in {name for name, widget in screen._settings_model._widgets.items()
                                               if section.isAncestorOf(widget)})
            if not section.is_expanded():
                screen._settings_scroll.ensureWidgetVisible(section.header())
                settle(.2)
                click(section.header())
            input_widget = screen._settings_model._widgets['count_data']
            clear = [button for button in input_widget.findChildren(QAbstractButton)
                     if button.text() == 'Clear' and button.isVisible()]
            if len(clear) != 1:
                raise ValueError('The native count-input Clear control is missing')
            click(clear[0])
            if screen._settings_model.collect()['count_data']:
                raise ValueError('The placeholder count path was not cleared')
        actions_folded = False
        if key == 'ops':
            folder = screen._actions_folder
            if not folder.shut:
                click(folder.heading)
            if not folder.shut:
                raise ValueError('The native Actions fold did not close')
            actions_folded = True
        splitter = getattr(screen, '_body_splitter', None)
        target = {'power': .4, 'dose_response': .5, 'parameter_sweep': .4}.get(key, .62)
        minimum = target - .07
        if (key in ('regression', 'mask', 'measure', 'classify_merged', 'map_barcodes', 'power', 'dose_response', 'parameter_sweep')
                and splitter is not None and splitter.count() == 2
                and (key in ('dose_response', 'parameter_sweep') or splitter.sizes()[0] < splitter.width() * minimum)):
            # The default narrow Settings column clips the Regression input
            # table. Drag its actual handle so the handoff controls can be read.
            handle = splitter.handle(1)
            start = handle.rect().center()
            delta = int(splitter.width() * target) - splitter.sizes()[0]
            QTest.mousePress(handle, Qt.LeftButton, pos=start)
            QTest.mouseMove(handle, start + QPoint(delta, 0), delay=100)
            QTest.mouseRelease(handle, Qt.LeftButton, pos=handle.rect().center())
            settle(.6)
            if splitter.sizes()[0] < splitter.width() * minimum:
                raise ValueError(f'The visible Settings splitter did not expand for {key}')
        if key == 'regression':
            for table in screen._settings_panel.findChildren(QTableView):
                if table.isVisible():
                    table.resizeColumnsToContents()
            settle(.4)
        if key == 'experiment_design':
            table = screen._table
            table.resizeColumnsToContents()
            table.resizeRowsToContents()
            for row in range(table.rowCount()):
                for column in range(table.columnCount()):
                    widget = table.cellWidget(row, column)
                    if widget is not None:
                        table.setColumnWidth(column, max(table.columnWidth(column), widget.sizeHint().width() + 32))
                        table.setRowHeight(row, max(table.rowHeight(row), widget.sizeHint().height() + 8))
            settle(.4)
        if key == 'dose_response':
            for table in screen.findChildren(QTableView):
                if table.isVisible():
                    table.resizeColumnsToContents()
            settle(.4)
        if key == 'gate_editor':
            splitter = screen._body
            handle = splitter.handle(1)
            start = handle.rect().center()
            delta = splitter.sizes()[1] - 850
            QTest.mousePress(handle, Qt.LeftButton, pos=start)
            QTest.mouseMove(handle, start + QPoint(delta, 0), delay=100)
            QTest.mouseRelease(handle, Qt.LeftButton, pos=handle.rect().center())
            settle(.6)
            if screen.side_tabs.width() < 800:
                raise ValueError('The Gate Editor filter pane did not widen')
        if not scene_captured:
            capture('module_' + key)
        routes.append({'module': key, 'home_host': host_key, 'parent': module['parent'],
                       'tile_text': tile.text() if tile is not None else None,
                       'screen_class': type(screen).__name__,
                       'home_tile_clicked': tile is not None, 'screen_visible': True,
                       'host_opened_through_help': tile is None,
                       'navigation_only_empty_folder_guard': key == 'cellpose_all',
                       'timelapse_switch_enabled_without_running': key == 'timelapse',
                       'parameter_sweep_opened_without_running': key == 'parameter_sweep',
                       'native_no_diagnostics_message': key == 'regression_diagnostics',
                       'default_count_path_cleared': key == 'barcode_qc',
                       'actions_folded_for_navigation_only': actions_folded})
    home()
    capture('home_summary')
    write_json(captures / 'workflow_acceptance.json', {
        'accepted': True, 'scope': 'Real Home/fold navigation and indexed API search; data handoffs explained by the shared map',
        'lesson': lesson['id'], 'workflow_map_sha256': hashlib.sha256(map_path.read_bytes()).hexdigest(),
        'lesson_sha256': hashlib.sha256(lesson_path.read_bytes()).hexdigest(),
        'visuals': expected, 'routes': routes, 'analysis_requested': False,
        'download_requested': False, 'worked_experiment_completed': False,
        'narration_or_media_accepted': False})
