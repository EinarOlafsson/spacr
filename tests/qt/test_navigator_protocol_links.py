"""Navigator exposes published lessons and a click-only protocol browser."""
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow

from spacr.qt import app
from spacr.qt.widgets import home, sample_project
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


class Links(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.urls = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.urls.append(dict(attrs).get('href'))


def test_guidance_links_use_all_seven_published_lessons():
    from spacr.qt.tutorials import tutorial_url
    keys = ('mask', 'measure', 'annotate', 'classify_merged', 'map_barcodes', 'regression', 'make_masks')
    urls = Links(home._navigator_learning_links()).urls
    assert urls == [*(tutorial_url(key) for key in keys), 'spacr://protocols']
    assert all(url and '#lesson=' in url for url in urls[:-1])
    assert Links(home._navigator_learning_links('mask')).urls == [tutorial_url('mask'), 'spacr://protocols']


@pytest.mark.parametrize('phrase', ['Protocols', 'PROTOCOL', 'walkthroughs', 'protokoll', 'Analyseanleitungen',
                                  'protocolos', 'protocoles', '操作流程', 'प्रोटोकॉल', '프로토콜', 'verkferlar'])
def test_protocol_search_uses_command_card_not_direct_module(phrase):
    assert app._easy_shortcut_match(phrase) == 'protocols'
    if phrase == 'Analyseanleitungen':
        assert app._easy_shortcut_match('Protokolle') == 'logs'
    host = SimpleNamespace()
    rows = app.MainWindow._easy_navigation_suggestions(host, phrase)
    assert len(rows) == 1 and rows[0][0] == '__easy_command__:Protocols'


@pytest.mark.parametrize('navigation', [False, True])
def test_protocol_popup_receives_only_real_inspected_input(monkeypatch, navigation):
    captured = []
    context = {'action_groups': {'mask': {'kind': 'navigation' if navigation else 'images'}}}
    host = SimpleNamespace(_startup=SimpleNamespace(_last_result={
        'path': 'Protocols' if navigation else '/data/experiment/plate1', 'context': context}))
    monkeypatch.setattr(sample_project, '_show_protocols',
                        lambda *args, **kwargs: captured.append((args, kwargs)), raising=False)
    app.MainWindow._open_navigator_protocols(host)
    assert captured == [((host,), {'source_path': '' if navigation else '/data/experiment/plate1',
                                  'context': context})]


def test_navigator_protocol_link_requires_explicit_valid_activation(qtbot, private_ai_profile, monkeypatch):
    window = QMainWindow()
    qtbot.addWidget(window)
    opened = []
    window._open_navigator_protocols = lambda: opened.append(True)
    details = home._NavigatorDetailsBrowser(window)
    window.setCentralWidget(details)
    details.setHtml(home._navigator_learning_links())
    assert opened == []
    details.anchorClicked.emit(QUrl('spacr://protocols'))
    assert opened == [True]
    rejected = []
    from spacr.qt.widgets import ai_chat_panel
    monkeypatch.setattr(ai_chat_panel, '_activate_ai_link', lambda owner, url: rejected.append(url.toString()))
    details.anchorClicked.emit(QUrl('spacr://protocols?run=true'))
    assert opened == [True] and rejected == ['spacr://protocols?run=true']


def test_actual_help_and_navigator_share_browser_and_native_measure_input(qtbot, private_ai_profile, tmp_path):
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    plate = tmp_path / 'plate1'
    (plate / 'merged').mkdir(parents=True)
    context = {'generated_outputs': {'artifacts': [{'kind': 'merged'}]}}
    window = app.MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window._startup._last_result = {'path': str(plate), 'context': context}
    try:
        action = window.findChild(QAction, 'ProtocolsAction')
        assert action is not None
        action.trigger()
        dialog = window._protocol_browser_dialog
        qtbot.addWidget(dialog)
        assert dialog._source_path == str(plate)
        root_children = dialog._trail[-1]['children']
        assert root_children[dialog._choice_list.currentRow()]['id'] == 'images_masks'
        assert dialog._selected_protocol is None
        def find_measure_path(node):
            if node.get('protocol') == 'simple_imaging' and node.get('start_module') == 'measure':
                return []
            for child in node.get('children', ()):
                path = find_measure_path(child)
                if path is not None:
                    return [child['id'], *path]
            return None
        path = find_measure_path(dialog._tree)
        assert path
        for node_id in path:
            children = dialog._trail[-1]['children']
            dialog._choice_list.setCurrentRow(next(i for i, child in enumerate(children) if child['id'] == node_id))
            dialog._next.click()
        assert dialog._active_route['start_module'] == 'measure'
        assert window._stack.currentWidget() is window._startup
        window._open_easy_command('Protocols')
        assert window._protocol_browser_dialog is dialog
        window._startup._last_result = {}
        assert dialog._activate_link(QUrl('spacr://module/measure'))
        qtbot.waitUntil(lambda: 'measure' in window._screens, timeout=15000)
        screen = window._screens['measure']
        assert screen._settings_model.collect()['src'] == str(plate)
        assert window._stack.currentWidget() is screen
    finally:
        window.close()
