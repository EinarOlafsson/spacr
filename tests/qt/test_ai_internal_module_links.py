"""Assistant links open only catalogued modules after an explicit user action."""
import pytest
from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices, QTextDocument
from PySide6.QtWidgets import QWidget

from spacr.qt.ai import prompts
from spacr.qt.widgets import ai_chat_panel as panel


@pytest.fixture
def navigation_catalog(monkeypatch):
    rows = (('measure', 'Measure', 'Measure images', '', ''),
            ('3d', '3D', 'Review volumes', '', ''))
    monkeypatch.setattr(prompts, '_navigation_modules', lambda: rows, raising=False)


@pytest.mark.parametrize('value', [
    'spacr://other/measure', 'spacr://module/unknown', 'spacr://module/measure/run',
    'spacr://module/measure?src=/tmp', 'spacr://module/measure#run',
    'spacr://module/measure?', 'spacr://module/measure#',
    'spacr://module/%6deasure', 'spacr://module/measure%2Frun',
    'spacr://module@other/measure', 'spacr://module:80/measure',
    'spacr://module/../measure', 'file:///measure',
])
def test_internal_module_parser_refuses_noncanonical_and_unknown_urls(navigation_catalog, value):
    assert panel._ai_module_key(value) is None


def test_internal_module_parser_uses_the_current_catalog(navigation_catalog, monkeypatch):
    assert panel._ai_module_key('spacr://module/measure') == 'measure'
    assert panel._ai_module_key(QUrl('spacr://module/3d')) == '3d'
    assert panel._ai_module_key(QUrl('spacr://module/measure%2Frun')) is None
    monkeypatch.setattr(prompts, '_navigation_modules', lambda: ())
    assert panel._ai_module_key('spacr://module/measure') is None


def test_bubble_links_are_passive_until_clicked_and_use_native_navigation(
        qtbot, monkeypatch, navigation_catalog):
    window = QWidget()
    qtbot.addWidget(window)
    opened, external = [], []
    window._open_easy_named_module = opened.append
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: external.append(url.toString()) or True)
    bubble = panel._MessageBubble('assistant', '[Measure](spacr://module/measure)', window)
    assert opened == external == []
    assert not bubble._text.openExternalLinks()
    assert 'spacr://module/measure' in bubble._text.text()
    bubble._text.linkActivated.emit('spacr://module/measure')
    assert opened == ['measure'] and not external
    for value in ('spacr://module/unknown', 'spacr://module/measure?run=1', 'javascript:alert(1)'):
        bubble._text.linkActivated.emit(value)
    assert opened == ['measure'] and not external
    bubble._text.linkActivated.emit('https://example.org/guide')
    bubble._text.linkActivated.emit('mailto:help@example.org')
    assert external == ['https://example.org/guide', 'mailto:help@example.org']


def test_renderer_keeps_only_approved_internal_and_external_anchors(qapp, navigation_catalog):
    html = panel._ai_reply_html('''[Measure](spacr://module/measure)
[3D](spacr://module/3d) [unknown](spacr://module/unknown)
[query](spacr://module/measure?run=1) [encoded](spacr://module/measure%2Frun)
[guide](https://example.org/guide) ![image](https://example.org/image.png)
<img src="https://example.org/raw.png">''')
    document = QTextDocument()
    document.setHtml(html)
    anchors = set()
    block = document.begin()
    while block.isValid():
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid():
                style = fragment.charFormat()
                assert not style.isImageFormat()
                if style.isAnchor() and style.anchorHref():
                    anchors.add(style.anchorHref())
            iterator += 1
        block = block.next()
    assert anchors == {'spacr://module/measure', 'spacr://module/3d', 'https://example.org/guide'}
    assert '<img' not in html.lower()


def test_floating_chat_dock_routes_only_to_its_owning_main_window(
        qtbot, monkeypatch, navigation_catalog):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDockWidget, QMainWindow

    owner = QMainWindow()
    other = QMainWindow()
    qtbot.addWidget(owner)
    qtbot.addWidget(other)
    opened, unrelated, external = [], [], []
    owner._open_easy_named_module = opened.append
    other._open_easy_named_module = unrelated.append
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: external.append(url))
    dock = QDockWidget(owner)
    owner.addDockWidget(Qt.RightDockWidgetArea, dock)
    bubble = panel._MessageBubble('assistant', '[Measure](spacr://module/measure)')
    dock.setWidget(bubble)
    owner.show()
    dock.setFloating(True)
    dock.show()
    assert bubble.window() is dock
    assert opened == unrelated == external == []
    bubble._text.linkActivated.emit('spacr://module/measure')
    assert opened == ['measure'] and not unrelated and not external


def test_answer_detached_from_its_window_cannot_navigate_until_it_has_a_native_owner(qtbot, navigation_catalog):
    orphan = QWidget()
    qtbot.addWidget(orphan)
    bubble = panel._MessageBubble('assistant', '[Measure](spacr://module/measure)', orphan)
    assert panel._activate_ai_link(bubble, 'spacr://module/measure') is False
    owner = QWidget()
    qtbot.addWidget(owner)
    opened = []
    owner._open_easy_named_module = opened.append
    bubble.setParent(owner)
    bubble._text.linkActivated.emit('spacr://module/measure')
    assert opened == ['measure']
    assert bubble._text.textInteractionFlags() & Qt.TextSelectableByMouse


def test_malformed_external_authority_remains_readable_without_an_active_link(qapp, navigation_catalog):
    html = panel._ai_reply_html('[Read the guide](https://[invalid])')
    document = QTextDocument()
    document.setHtml(html)
    assert document.toPlainText().strip() == 'Read the guide'
    assert 'href=' not in html
    assert '<img' not in html.lower()
