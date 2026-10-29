"""Navigator and console answers navigate through explicit native link clicks."""
from types import SimpleNamespace

from PySide6.QtCore import QUrl, Qt
from PySide6.QtWidgets import QWidget

from spacr.qt import app
from spacr.qt.ai import prompts, settings
from spacr.qt.widgets.ai_chat_panel import _ai_reply_html
from tests.qt.test_733_navigator_ai import ai_page, _enable
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


def test_navigator_answer_click_opens_the_real_measure_module(qtbot, private_ai_profile):
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    window = app.MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.show()
    browser = window._startup.field.details
    browser.setHtml(_ai_reply_html('[Measure](spacr://module/measure)'))
    assert window._stack.currentWidget() is window._startup
    cursor = browser.document().find('Measure')
    assert not cursor.isNull()
    cursor.setPosition(cursor.selectionStart() + 2)
    position = browser.cursorRect(cursor).center()
    assert browser.anchorAt(position) == 'spacr://module/measure'
    qtbot.mouseClick(browser.viewport(), Qt.LeftButton, pos=position)
    qtbot.waitUntil(lambda: 'measure' in window._screens, timeout=15000)
    assert window._stack.currentWidget() is not window._startup
    window.close()


def test_navigator_dispatch_keeps_custom_prompt_and_navigation_guidance(ai_page, monkeypatch):
    from spacr.qt.ai import worker

    page, calls, _ = ai_page
    captured = []
    original = worker.make_stream_thread
    monkeypatch.setattr(settings, 'get_system_prompt', lambda: 'Use short answers.')
    monkeypatch.setattr(settings, 'is_system_prompt_overridden', lambda: True)
    def stream(*args, **kwargs):
        captured.append(kwargs['system'])
        return original(*args, **kwargs)
    monkeypatch.setattr(worker, 'make_stream_thread', stream)
    _enable(page)
    page._ask_navigator_ai('Explain an unfamiliar analysis workflow')
    assert len(calls) == 1
    assert captured[0].startswith('Use short answers.')
    assert 'spacr://module/' in captured[0] and 'Navigator' in captured[0]
    assert 'Do not invoke tools' in captured[0]


def test_console_explanation_links_use_native_click_dispatch(qtbot, monkeypatch):
    from spacr.qt.widgets.console_panel import _AiReplyBlock

    owner = QWidget()
    qtbot.addWidget(owner)
    opened = []
    owner._open_easy_named_module = opened.append
    block = _AiReplyBlock()
    block.setParent(owner)
    block.show_markdown('[Measure](spacr://module/measure)')
    assert not opened
    block._rich_view.anchorClicked.emit(QUrl('spacr://module/measure'))
    assert opened == ['measure']
    block._rich_view.anchorClicked.emit(QUrl('spacr://module/measure?run=1'))
    assert opened == ['measure']


def test_internal_link_catalog_is_read_once_per_render_and_rechecked_next_time(qapp, monkeypatch):
    calls, current = [], ['measure']
    def catalog():
        calls.append(True)
        return tuple((key, key, '', '', '') for key in current)
    monkeypatch.setattr(prompts, '_navigation_modules', catalog)
    text = ' '.join('[Measure](spacr://module/measure)' for _ in range(20))
    assert 'href="spacr://module/measure"' in _ai_reply_html(text)
    assert len(calls) == 1
    current.clear()
    assert 'href="spacr://module/measure"' not in _ai_reply_html(text)
    assert len(calls) == 2


def test_special_native_ai_routes_use_existing_organism_and_overview_openers(monkeypatch):
    from spacr.qt.widgets import qc_field_browser

    calls = []
    owner = SimpleNamespace(_settings_model=SimpleNamespace(collect=lambda: {'src': 'plate'}))
    class Host:
        _open_easy_named_module = app.MainWindow._open_easy_named_module
        def _open_easy_suggestion(self, key, path):
            calls.append(('organism', key, path))
    host = Host()
    host._open_easy_named_module('parascape')
    assert calls == [('organism', 'parascape', '')]
    original = app.MainWindow._open_easy_named_module
    def named(key):
        if key == 'mask':
            return owner
        return original(host, key)
    host._open_easy_named_module = named
    monkeypatch.setattr(qc_field_browser, '_open_overview_module',
                        lambda screen, values: calls.append(('overview', screen, values)))
    host._open_easy_named_module('overview')
    assert calls[-1] == ('overview', owner, {'src': 'plate'})
