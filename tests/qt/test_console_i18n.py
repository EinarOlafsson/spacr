"""Presentation-only localization for chat and console surfaces.

The central contract is negative as well as positive: spaCR-authored chrome and
notices follow the selected language, while user text, provider replies,
pipeline stdout, tracebacks, paths and identifiers remain byte-for-byte intact.
"""
from __future__ import annotations


def _console_text(panel) -> str:
    from spacr.qt.widgets.console_panel import _StdoutBlock

    return "".join(block.toPlainText()
                   for block in panel.findChildren(_StdoutBlock))


def test_ai_and_live_have_explicit_ten_language_catalog_rows():
    from spacr.qt.i18n import CATALOGS, VALID_LANGUAGE_CODES

    for code in VALID_LANGUAGE_CODES[1:]:
        assert CATALOGS[code]["AI"].strip()
        assert CATALOGS[code]["Live"].strip()
    assert CATALOGS["de"]["AI"] == "KI"
    assert CATALOGS["es"]["Live"] == "En vivo"
    assert CATALOGS["zh_CN"]["Live"] == "实时"


def test_ai_toggle_retranslates_without_changing_state_or_emitting(
    qtbot, qt_theme_applied,
):
    from spacr.qt.i18n import retranslate_widget_tree
    from spacr.qt.widgets import AiToggleLabel

    toggle = AiToggleLabel(text="AI")
    qtbot.addWidget(toggle)
    emissions = []
    toggle.toggled.connect(emissions.append)
    toggle.setChecked(True)
    emissions.clear()

    retranslate_widget_tree(toggle, "de")
    assert toggle.text() == "KI"
    assert toggle.isChecked() is True
    assert emissions == []

    retranslate_widget_tree(toggle, "zh_CN")
    assert toggle.text() == "人工智能"
    assert toggle.isChecked() is True
    assert emissions == []


def test_qtextedit_placeholder_translates_but_typed_text_does_not(
    qtbot, qt_theme_applied,
):
    from PySide6.QtWidgets import QTextEdit
    from spacr.qt.i18n import retranslate_widget_tree

    editor = QTextEdit()
    qtbot.addWidget(editor)
    editor.setPlaceholderText(
        "Ask a question (Enter to send · Shift+Enter for newline)")
    editor.setPlainText("Run /data/Cell output — 用户输入")

    retranslate_widget_tree(editor, "sv")
    assert editor.placeholderText().startswith("Ställ en fråga")
    assert editor.toPlainText() == "Run /data/Cell output — 用户输入"

    retranslate_widget_tree(editor, "ko")
    assert editor.placeholderText().startswith("질문하기")
    assert editor.toPlainText() == "Run /data/Cell output — 用户输入"


def test_console_chrome_localizes_while_raw_streams_stay_exact(
    qtbot, qt_theme_applied, monkeypatch,
):
    from spacr.qt.i18n import retranslate_widget_tree
    from spacr.qt.widgets.console_panel import ConsolePanel

    monkeypatch.setenv("SPACR_LANGUAGE", "sv")
    panel = ConsolePanel(active_app_label="Mask")
    qtbot.addWidget(panel)
    assert panel._input.placeholderText().startswith("Skriv här")

    stdout = "Run Cell output | /tmp/plate_A | function=measure\n"
    traceback = "Traceback: ValueError: Cell output /tmp/plate_A\n"
    panel.append_stdout(stdout)
    panel.append_error(traceback)
    before = _console_text(panel)

    retranslate_widget_tree(panel, "ko")
    assert panel._input.placeholderText().startswith("여기에 입력")
    assert _console_text(panel) == before
    assert stdout in before
    assert traceback in before


def test_legacy_chat_body_and_cancel_state_survive_runtime_retranslation(
    qtbot, qt_theme_applied,
):
    from spacr.qt.i18n import retranslate_widget_tree, set_translatable_text
    from spacr.qt.widgets.ai_chat_panel import AIChatPanel, _MessageBubble

    panel = AIChatPanel()
    qtbot.addWidget(panel)
    bubble = _MessageBubble("assistant", "Run")
    panel._chat_layout.insertWidget(0, bubble)
    panel._set_send_mode("cancel")
    set_translatable_text(
        panel._status, "Connecting to {provider}…", provider="Claude")

    retranslate_widget_tree(panel, "ko")
    assert panel._btn_send.text() == "취소"
    assert "Claude" in panel._status.text()
    assert panel._status.text() != "Connecting to Claude…"
    assert bubble._text.text() == "Run"

    retranslate_widget_tree(panel, "en")
    assert panel._btn_send.text() == "Cancel"
    assert panel._status.text() == "Connecting to Claude…"
    assert bubble._text.text() == "Run"


def test_append_notice_translates_only_the_template(
    qtbot, qt_theme_applied, monkeypatch,
):
    from spacr.qt.widgets.console_panel import ConsolePanel

    monkeypatch.setenv("SPACR_LANGUAGE", "de")
    panel = ConsolePanel()
    qtbot.addWidget(panel)
    path = "/data/Cell_Output/plate_A"
    panel.append_notice(
        "Loaded {count} settings from {path}\n", count=12, path=path)
    rendered = _console_text(panel)
    assert path in rendered
    assert "12" in rendered
    assert "Einstellungen" in rendered


def test_append_notice_from_worker_thread_is_queued_and_preserves_values(
    qtbot, qt_theme_applied, monkeypatch,
):
    import threading
    from spacr.qt.widgets.console_panel import ConsolePanel

    monkeypatch.setenv("SPACR_LANGUAGE", "fr")
    panel = ConsolePanel()
    qtbot.addWidget(panel)
    path = "/data/plate_A/measurements.db"
    worker = threading.Thread(
        target=lambda: panel.append_notice(
            "Loaded {count} settings from {path}\n", count=7, path=path),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    qtbot.waitUntil(lambda: path in _console_text(panel), timeout=2000)
    rendered = _console_text(panel)
    assert path in rendered
    assert "7 paramètres" in rendered
