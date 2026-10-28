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
