"""Console AI uses real safe rich formatting while pipeline output remains literal."""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices, QFont

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


@pytest.fixture
def panel(qtbot, private_ai_profile):
    from spacr.qt.widgets.console_panel import ConsolePanel
    value = ConsolePanel(follow_log=False)
    qtbot.addWidget(value)
    value.resize(900, 700)
    value.show()
    yield value
    value.close()


def test_saved_reply_has_native_bold_lists_code_and_safe_click_only_links(panel, qtbot, monkeypatch):
    from spacr.qt.widgets.console_panel import _AiReplyBlock
    opened = []
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda url: opened.append(url.toString()) or True)
    source = ('## Settings\n\n**Choose channels**\n\n1. Load images\n2. Review masks\n\n'
              '- Keep originals\n- Save outputs\n\n```python\nprint("safe <code>")\n```\n\n'
              '[Docs](https://example.org/spacr) [Unsafe](file:///tmp/private)\n\n'
              '![blocked](https://example.org/image.png) <img src="file:///tmp/private.png">')
    block = panel.fill_ai_entry(panel.begin_ai_entry(False), source)
    assert isinstance(block, _AiReplyBlock)
    qtbot.wait(20)
    document = block.document()
    assert document.find('Choose channels').charFormat().fontWeight() == QFont.Bold
    items = []
    current = document.begin()
    while current.isValid():
        if current.textList() is not None:
            items.append(current.text())
        current = current.next()
    assert items == ['Load images', 'Review masks', 'Keep originals', 'Save outputs']
    assert 'print("safe <code>")' in block.toPlainText()
    anchors, images = [], []
    current = document.begin()
    while current.isValid():
        iterator = current.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            fmt = fragment.charFormat()
            if fmt.isAnchor():
                anchors.append(fmt.anchorHref())
            if fmt.isImageFormat():
                images.append(fmt.toImageFormat().name())
            iterator += 1
        current = current.next()
    assert 'https://example.org/spacr' in anchors
    assert not any(url.startswith('file:') for url in anchors)
    assert not images and not opened
    assert '<img src="file:///tmp/private.png">' in block.toPlainText()
    cursor = document.find('Docs')
    cursor.setPosition(cursor.selectionStart() + 1)
    block._rich_view.ensureCursorVisible()
    point = block._rich_view.cursorRect(cursor).center()
    qtbot.mouseClick(block._rich_view.viewport(), Qt.LeftButton, pos=point)
    assert opened == ['https://example.org/spacr']


def test_stream_chunks_are_coalesced_and_section_font_height_export_are_preserved(panel, qtbot, monkeypatch):
    from spacr.qt.widgets import ai_chat_panel
    from spacr.qt.widgets.console_panel import _AiReplyBlock
    rendered = []
    real_render = ai_chat_panel._ai_reply_html
    monkeypatch.setattr(ai_chat_panel, '_ai_reply_html', lambda text: (
        rendered.append(text) or real_render(text)))
    block = _AiReplyBlock()
    panel._insert_entry(block)
    block.set_user_height(180)
    block.set_console_font_pt(18)
    for chunk in ('**Rich', ' answer**\n\n', '- first\n', '- second'):
        block.append(chunk)
    assert not rendered
    qtbot.waitUntil(lambda: bool(rendered))
    assert rendered == ['**Rich answer**\n\n- first\n- second']
    assert block._user_height == 180 and block.height() == 180
    assert block._font_pt == 18 and block._rich_view.font().pointSize() == 18
    assert 'Rich answer' in panel.as_text() and '**Rich' not in panel.as_text()
    panel.fill_ai_entry(block, '**Replacement**\n\n1. Next')
    assert block.height() == 180 and block._font_pt == 18
    assert block.document().find('Replacement').charFormat().fontWeight() == QFont.Bold


def test_pipeline_append_and_traceback_document_remain_plain(qtbot):
    from spacr.qt.widgets.console_panel import _StdoutBlock
    source = '**literal**\n- not a rich list\nTraceback (most recent call last):\n<img src="file:///tmp/x">'
    block = _StdoutBlock(error=True)
    qtbot.addWidget(block)
    for part in source.splitlines(keepends=True):
        block.append(part)
    assert block.toPlainText() == source
    assert block.document().find('literal').charFormat().fontWeight() != QFont.Bold
    current = block.document().begin()
    while current.isValid():
        assert current.textList() is None
        current = current.next()
