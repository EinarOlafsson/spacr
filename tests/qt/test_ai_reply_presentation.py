"""Assistant replies display Markdown while user text and resources stay passive."""
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QFont, QTextDocument, QTextListFormat

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile

_EXAMPLE = r'''1\. \*\*Channels\*\*: set \`cell_channel\` to the cell image.
2\. \*\*Masks\*\*: inspect \`mask_channel\` before measuring.'''


def _document(text):
    from spacr.qt.widgets.ai_chat_panel import _ai_reply_html

    doc = QTextDocument()
    doc.setHtml(_ai_reply_html(text))
    return doc


def _blocks(document):
    block = document.begin()
    while block.isValid():
        yield block
        block = block.next()


def _assert_example(document):
    blocks = [block for block in _blocks(document) if block.text().strip()]
    assert len(blocks) == 2
    assert all(block.textList() is not None for block in blocks)
    assert all(block.textList().format().style() == QTextListFormat.ListDecimal for block in blocks)
    assert all('\\' not in block.text() and '**' not in block.text() for block in blocks)
    for word in ('Channels', 'Masks'):
        cursor = document.find(word)
        assert not cursor.isNull() and cursor.charFormat().fontWeight() >= QFont.Bold
    for word in ('cell_channel', 'mask_channel'):
        cursor = document.find(word)
        assert not cursor.isNull()
        assert any('mono' in family.casefold() or 'courier' in family.casefold()
                   for family in cursor.charFormat().fontFamilies())


def test_user_escaped_markdown_is_real_numbered_lists_bold_and_inline_code(qapp):
    _assert_example(_document(_EXAMPLE))


def test_regular_markdown_keeps_lists_code_and_windows_backslashes(qapp):
    text = '''- **Review channels**
- Preserve `C:\\images\\cell_channel.npy`

```text
1\\. \\*\\*literal\\*\\* \\`cell_channel\\`
C:\\images\\raw
```
'''
    document = _document(text)
    blocks = list(_blocks(document))
    assert sum(block.textList() is not None for block in blocks) == 2
    assert document.find('Review channels').charFormat().fontWeight() >= QFont.Bold
    plain = document.toPlainText()
    assert r'C:\images\cell_channel.npy' in plain
    assert r'C:\images\raw' in plain
    assert r'1\. \*\*literal\*\* \`cell_channel\`' in plain


def test_html_images_and_dangerous_links_never_become_active_or_fetch_resources(qapp, monkeypatch):
    from PySide6 import QtGui

    from spacr.qt.widgets.ai_chat_panel import _ai_reply_html

    original = QtGui.QTextDocument.setMarkdown
    resources = []
    def markdown(document, text, *args):
        for url in ('https://example.invalid/tracking.png', 'file:///private/secret.png'):
            resources.append(document.loadResource(QTextDocument.ImageResource, QUrl(url)))
        return original(document, text, *args)
    monkeypatch.setattr(QtGui.QTextDocument, 'setMarkdown', markdown)
    text = '''<script>alert('bad')</script><img src="https://example.invalid/html.png">
![tracking](https://example.invalid/pixel.png)
[ok](https://example.org/docs) [mail](mailto:help@example.org)
[bad](javascript:alert) [file](file:///private/secret) [data](data:text/html,bad)
[relative](../private) [network](//example.invalid/path)
'''
    html = _ai_reply_html(text)
    assert resources == [None, None]
    assert '<img' not in html.casefold() and '<script' not in html.casefold()
    assert '&lt;img src=' in html.casefold()
    document = QTextDocument()
    document.setHtml(html)
    anchors = set()
    for block in _blocks(document):
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid():
                fmt = fragment.charFormat()
                assert not fmt.isImageFormat()
                if fmt.isAnchor() and fmt.anchorHref():
                    anchors.add(fmt.anchorHref())
            iterator += 1
    assert anchors == {'https://example.org/docs', 'mailto:help@example.org'}


def test_message_bubble_formats_assistant_but_keeps_raw_user_literal(qtbot):
    from spacr.qt.widgets.ai_chat_panel import _MessageBubble

    assistant = _MessageBubble('assistant', _EXAMPLE)
    user = _MessageBubble('user', _EXAMPLE + '<b>literal</b>')
    qtbot.addWidget(assistant)
    qtbot.addWidget(user)
    assistant.show()
    user.show()
    assert assistant._text.textFormat() == Qt.RichText
    document = QTextDocument()
    document.setHtml(assistant._text.text())
    _assert_example(document)
    assert user._text.textFormat() == Qt.PlainText
    assert user._text.text() == _EXAMPLE + '<b>literal</b>'
    assert not assistant.grab().isNull() and not user.grab().isNull()
    assistant.set_text('A plain answer.')
    assert assistant._text.textFormat() == Qt.PlainText
    assert assistant._text.text() == 'A plain answer.'


def test_navigator_lower_box_uses_actual_shared_reply_renderer(qtbot, private_ai_profile):
    from spacr.qt.widgets.home import _EasyHomePage

    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1000, 800)
    page.show()
    page._ai_text = _EXAMPLE
    page._render_ai_reply()
    _assert_example(page.field.details.document())
    assert page.field.details.isVisible()
    assert not page.field.details.grab().isNull()
    page.close()


def test_legacy_answer_layout_preserves_heading_body_lists_and_inline_emphasis(qtbot):
    from spacr.qt.widgets.console_panel import _StdoutBlock

    block = _StdoutBlock()
    qtbot.addWidget(block)
    block.show_markdown('**Summary**: Review\n\nUse **masks** safely.\n'
                        '- first\n* second\n\n# Details\n**Next steps**\nKeep originals.\n\n')
    block.resize(440, 320)
    block.show()
    assert block.toPlainText() == ('Summary\nReview\n\nUse masks safely.\n'
                                   '• first\n• second\n\nDetails\n\nNext steps\nKeep originals.')
    assert block.document().find('masks').charFormat().fontWeight() >= QFont.Bold
    assert block.document().find('Summary').charFormat().fontWeight() >= QFont.Bold
    assert not block.grab().isNull()
    block.set_user_height(180)
    assert block.height() == 180
    block.reset_user_height()
    assert block.maximumHeight() > 180
    block.show_markdown('**Use** caution. 1. Inspect masks. 2. Keep originals.')
    assert block.toPlainText() == 'Use caution.\n1. Inspect masks.\n2. Keep originals.'
    assert block.document().find('Use').charFormat().fontWeight() >= QFont.Bold


def test_streamed_reply_cap_copy_passive_resources_and_close_form_one_lifecycle(qtbot):
    from PySide6.QtCore import QPoint
    from spacr.qt.widgets.console_panel import _AiReplyBlock

    block = _AiReplyBlock()
    qtbot.addWidget(block)
    block.resize(440, 200)
    block.show()
    block.append('')
    assert not block._reply_flush.isActive()
    # Retain the real production cap: a long response must lose its oldest text.
    block.append('discarded prefix\n' + ('older words ' * (block.MAX_CHARS // 12 + 1)))
    block.append('\n**Final answer**\n- Keep `raw.npy`.\n- Inspect **masks**.')
    qtbot.waitUntil(lambda: 'Inspect masks.' in block.toPlainText())
    assert 'discarded prefix' not in block.toPlainText()
    assert block._reply_chars <= block.MAX_CHARS
    assert 'Keep raw.npy.' in block.toPlainText()
    assert block._rich_view.loadResource(QTextDocument.ImageResource,
                                         QUrl('file:///private/not-a-resource.png')) is None
    assert block.link_at(QPoint(0, 0)) == ''
    block.set_user_height(180)
    assert block.sizeHint().height() == 180
    block.reset_user_height()
    assert block.sizeHint().height() >= 32
    block.append('\nPending final chunk')
    assert block._reply_flush.isActive()
    block.close()
    assert not block._reply_flush.isActive()


def test_qt_child_observer_can_measure_and_show_a_reply_during_native_construction(qtbot, qapp):
    from PySide6.QtCore import QEvent, QObject
    from spacr.qt.widgets.console_panel import _AiReplyBlock

    observations = []

    class InitialLayout(QObject):
        def eventFilter(self, watched, event):
            if event.type() != QEvent.ChildAdded:
                return False
            parent = watched.parent()
            if (isinstance(parent, _AiReplyBlock) and watched is parent.viewport()
                    and hasattr(parent, '_reply_parts') and parent._rich_view is None):
                observations.append((parent.toPlainText(), parent.sizeHint()))
                parent.resize(480, 160)
                parent.show()
            return False

    observer = InitialLayout()
    qapp.installEventFilter(observer)
    try:
        block = _AiReplyBlock()
        qtbot.addWidget(block)
    finally:
        qapp.removeEventFilter(observer)
    assert len(observations) == 1
    text, initial_size = observations[0]
    assert text == '' and initial_size.width() > 0 and initial_size.height() >= 32
    block.resize(500, 220)
    block.show_markdown('**Completed answer**\n\n- Keep original images.\n- Inspect masks.')
    assert 'Completed answer' in block.toPlainText()
    assert 'Keep original images.' in block.toPlainText()
    assert block._rich_view.geometry() == block.viewport().rect()
    assert block.sizeHint().height() >= 32
    assert not block.grab().isNull()
    block.close()
