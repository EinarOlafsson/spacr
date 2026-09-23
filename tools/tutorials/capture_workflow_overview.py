"""Record the pooled-screen map lesson through actual Home module controls.

This is a navigation and data-handoff lesson. Worked experiments belong to
the linked module lessons; this recording starts no analysis or download.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def record_overview(app, window, captures, capture, settle, write_json):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QTableView

    root = Path(__file__).resolve().parents[2]
    map_path = root / 'spacr/resources/module_workflows.json'
    lesson_path = root / 'tools/tutorials/lessons/78_spacr_screens.json'
    data = json.loads(map_path.read_text())
    lesson = json.loads(lesson_path.read_text())
    keys = data['tutorials'][lesson['id']]['modules']
    expected = ['home', *('module_' + key for key in keys), 'home_summary']
    if [scene['visual'] for scene in lesson['scenes']] != expected:
        raise ValueError('The pooled-screen lesson and workflow map disagree')
    if any(data['modules'][key]['home'] != key or data['modules'][key]['parent']
           for key in keys):
        raise ValueError('This recorder requires the current direct Home routes')

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

    home()
    capture('home')
    routes = []
    for key in keys:
        home()
        tiles = [button for button in window._startup.findChildren(QAbstractButton)
                 if button.property('moduleAppKey') == key]
        if not tiles:
            raise ValueError(f'No native Home tile for {key}')
        tabs = window._startup._tabs
        tile = None
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
        screen = window._screens.get(key)
        if screen is None or not screen.isVisible():
            raise ValueError(f'The native Home tile did not open {key}')
        if getattr(screen, '_worker_thread_is_running', lambda: False)():
            raise ValueError(f'Opening {key} unexpectedly started an analysis')
        splitter = getattr(screen, '_body_splitter', None)
        if (splitter is not None and splitter.count() == 2
                and splitter.sizes()[0] < splitter.width() * .55):
            # The default narrow Settings column clips the Regression input
            # table. Drag its actual handle so the handoff controls can be read.
            handle = splitter.handle(1)
            start = handle.rect().center()
            delta = int(splitter.width() * .62) - splitter.sizes()[0]
            QTest.mousePress(handle, Qt.LeftButton, pos=start)
            QTest.mouseMove(handle, start + QPoint(delta, 0), delay=100)
            QTest.mouseRelease(handle, Qt.LeftButton, pos=handle.rect().center())
            settle(.6)
            if splitter.sizes()[0] < splitter.width() * .55:
                raise ValueError(f'The visible Settings splitter did not expand for {key}')
        if key == 'regression':
            for table in screen._settings_panel.findChildren(QTableView):
                if table.isVisible():
                    table.resizeColumnsToContents()
            settle(.4)
        capture('module_' + key)
        routes.append({'module': key, 'tile_text': tile.text(),
                       'screen_class': type(screen).__name__,
                       'home_tile_clicked': True, 'screen_visible': True})
    home()
    capture('home_summary')
    write_json(captures / 'workflow_acceptance.json', {
        'accepted': True, 'scope': 'Real Home navigation; data handoffs explained by the shared map',
        'lesson': lesson['id'], 'workflow_map_sha256': hashlib.sha256(map_path.read_bytes()).hexdigest(),
        'lesson_sha256': hashlib.sha256(lesson_path.read_bytes()).hexdigest(),
        'visuals': expected, 'routes': routes, 'analysis_requested': False,
        'download_requested': False, 'worked_experiment_completed': False,
        'narration_or_media_accepted': False})
