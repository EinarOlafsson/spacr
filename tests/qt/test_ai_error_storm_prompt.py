"""Error-storm controls act only after an explicit GUI choice."""
import weakref
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtGui import QRegion, QTextDocument, QTextOption
from PySide6.QtWidgets import (QLabel, QMainWindow, QMessageBox, QWidget, QVBoxLayout,
                              QStyle, QStyleOptionButton)

from spacr import crashreport as cr
from spacr.qt.widgets import console_panel as console
from spacr.qt.ai import providers
from tests.test_ai_error_storm_control import batcher

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(cr, "_reports_dir", lambda: tmp_path)
    monkeypatch.setattr(cr, "_ERROR_POLICY_CACHE", {})
    widget = console.ConsolePanel(follow_log=False, chat=False)
    qtbot.addWidget(widget)
    return widget


@pytest.mark.parametrize("choice", ("Cancel", "Force quit", "Force restart"))
def test_prompt_offers_one_explicit_action_with_cancel_default(panel, qtbot, monkeypatch, choice):
    selected = []
    monkeypatch.setattr(panel, "_force_error_storm", selected.append)
    panel._show_error_report_result(dict(storm=True))
    box = panel._error_storm_box
    assert box.defaultButton().text() == "Cancel"
    assert box.escapeButton().text() == "Cancel"
    assert selected == []
    panel._show_error_report_result(dict(storm=True))
    assert panel._error_storm_box is box
    button = next(button for button in box.buttons() if button.text() == choice)
    qtbot.mouseClick(button, Qt.LeftButton)
    assert selected == ([] if choice == "Cancel" else ["quit" if choice == "Force quit" else "restart"])


@pytest.mark.parametrize("font_scale,expected_pixels", ((.9, 11), (2., 23)))
def test_storm_prompt_initial_frame_and_text_fit(qtbot, qapp, monkeypatch, tmp_path,
                                               record_property, font_scale, expected_pixels):
    from shiboken6 import isValid
    from spacr.qt import gui_scale, preferences

    monkeypatch.setattr(cr, "_reports_dir", lambda: tmp_path)
    monkeypatch.setattr(cr, "_ERROR_POLICY_CACHE", {})
    window = QMainWindow()
    qtbot.addWidget(window)
    old_live = gui_scale.current_scale()
    old_gui, old_font = preferences.get_gui_scale(), preferences.get_font_scale()
    old_sheet, old_app_font = qapp.styleSheet(), qapp.font()
    old_palette = qapp.palette()
    box = None
    try:
        gui_scale.install_scaling_layer()
        gui_scale.apply_saved_gui_scale()
        preferences.apply_preferences_to_app()
        gui_scale.change_scales(window, gui=.9, font=font_scale, ask=False)
        assert gui_scale.current_scale() == .9
        assert preferences.get_font_scale() == font_scale
        widget = console.ConsolePanel(parent=window, follow_log=False, chat=False)
        window.setCentralWidget(widget)
        monkeypatch.setattr(widget, "_force_error_storm",
                            lambda action: pytest.fail("Cancel invoked destructive action: " + action))
        window.show()
        widget._show_error_report_result(dict(storm=True))
        box = widget._error_storm_box
        qtbot.waitUntil(lambda: box.windowHandle() is not None
                        and box.windowHandle().isExposed(), timeout=10000)
        assert qapp.platformName() == os.environ.get("QT_QPA_PLATFORM", qapp.platformName())
        handle = box.windowHandle()
        frame = handle.frameGeometry()
        work = handle.screen().availableGeometry()
        record_property("native_prompt", dict(platform=qapp.platformName(),
            frame=frame.getRect(), work=work.getRect(), dpr=handle.devicePixelRatio(),
            font_scale=font_scale))
        assert work.contains(frame)
        message = box.findChild(QLabel, "qt_msgbox_label")
        assert message is not None and message.font().pixelSize() == expected_pixels
        for label in box.findChildren(QLabel):
            if not label.text() or not label.isVisibleTo(box):
                continue
            assert QRegion(label.rect()).subtracted(label.visibleRegion()).isEmpty()
            document = QTextDocument()
            document.setDefaultFont(label.font())
            document.setDocumentMargin(0)
            option = document.defaultTextOption()
            option.setWrapMode(QTextOption.WordWrap if label.wordWrap() else QTextOption.NoWrap)
            document.setDefaultTextOption(option)
            if label.textFormat() == Qt.RichText:
                document.setHtml(label.text())
            else:
                document.setPlainText(label.text())
            area = label.contentsRect().adjusted(label.margin(), label.margin(),
                                                 -label.margin(), -label.margin())
            document.setTextWidth(area.width() if label.wordWrap() else -1)
            tolerance = 1 / handle.devicePixelRatio()
            assert document.idealWidth() <= area.width() + tolerance
            assert document.size().height() <= area.height() + tolerance
        assert {button.text() for button in box.buttons()} == {"Cancel", "Force quit", "Force restart"}
        for button in box.buttons():
            assert QRegion(button.rect()).subtracted(button.visibleRegion()).isEmpty()
            option = QStyleOptionButton()
            option.initFrom(button)
            option.rect = button.rect()
            option.text = button.text()
            option.icon = button.icon()
            option.iconSize = button.iconSize()
            if button.isDefault():
                option.features |= QStyleOptionButton.DefaultButton
            if button.autoDefault():
                option.features |= QStyleOptionButton.AutoDefaultButton
            if button.isFlat():
                option.features |= QStyleOptionButton.Flat
            contents = button.style().subElementRect(QStyle.SE_PushButtonContents, option, button)
            text = button.style().itemTextRect(button.fontMetrics(), contents,
                int(Qt.AlignCenter | Qt.TextSingleLine), button.isEnabled(), button.text())
            assert contents.contains(text), (button.text(), contents, text)
        assert box.defaultButton().text() == box.escapeButton().text() == "Cancel"
        widget._show_error_report_result(dict(storm=True))
        assert widget._error_storm_box is box
        qtbot.mouseClick(box.defaultButton(), Qt.LeftButton)
        qtbot.waitUntil(lambda: not isValid(box) or not box.isVisible(), timeout=10000)
    finally:
        try:
            if box is not None and isValid(box) and box.isVisible():
                box.reject()
            gui_scale.change_scales(window, gui=old_gui, font=old_font, ask=False)
        finally:
            preferences.set_gui_scale(old_gui)
            preferences.set_font_scale(old_font)
            gui_scale.set_gui_scale_live(old_live)
            qapp.setStyleSheet(old_sheet)
            qapp.setFont(old_app_font)
            qapp.setPalette(old_palette)
            window.close()


def test_hidden_console_receives_storm_prompt_on_visible_window(qtbot, monkeypatch):
    window = QWidget()
    qtbot.addWidget(window)
    layout = QVBoxLayout(window)
    widget = console.ConsolePanel(parent=window, follow_log=False, chat=False)
    layout.addWidget(widget)
    widget.hide()
    window.show()
    shown = []
    monkeypatch.setattr(widget, "_show_error_storm_prompt", lambda: shown.append(True))
    relay = console._ErrorReportRelay(window)
    relay.consoles = [weakref.ref(widget)]
    relay._receive(dict(storm=True))
    assert shown == [True]


def test_force_restart_uses_verified_launcher_and_does_not_quit_after_failure(panel, monkeypatch):
    from spacr.qt import shutdown
    calls, notices = [], []
    monkeypatch.setattr(shutdown, "restart_spacr", lambda *args: calls.append(args) or False)
    monkeypatch.setattr(shutdown, "force_quit_now", lambda: pytest.fail("unconfirmed force quit"))
    monkeypatch.setattr(panel, "append_notice", notices.append)
    panel._force_error_storm("restart")
    assert calls == [("", {})]
    assert "remains open" in notices[0]


def test_manual_error_batching_works_with_automatic_off_and_rechecks_provider(panel, monkeypatch):
    control, clock, _, _, _, _ = batcher()
    monkeypatch.setattr(cr, "_ERROR_BATCHER", control)
    monkeypatch.setattr(cr, "network_allowed", lambda: True)
    cr._update_state(mode="off", affirmed_policy={"provider": "hosted"})
    current = [SimpleNamespace(name="hosted")]
    monkeypatch.setattr(panel, "_current_provider", lambda: current[0])
    sent = []
    monkeypatch.setattr(panel, "_open_error_flow_now", lambda *args: sent.append(args))
    panel.open_error_flow("ValueError: first", "mask")
    assert len(sent) == 1
    clock.now = 1
    panel.open_error_flow("ValueError: pending", "mask")
    assert len(sent) == 1
    current[0] = SimpleNamespace(name="different")
    clock.now = 61
    control._tick()
    assert len(sent) == 1


def test_manual_duplicate_retains_owner_after_automatic_permission_revoked(panel, monkeypatch):
    control, clock, _, _, _, _ = batcher()
    monkeypatch.setattr(cr, "_ERROR_BATCHER", control)
    monkeypatch.setattr(cr, "network_allowed", lambda: True)
    monkeypatch.setattr(panel, "_current_provider", lambda: SimpleNamespace(name="hosted"))
    control.submit(dict(text="first", provider="hosted"), lambda *args: None)
    clock.now = 1
    consent = [True]
    control.submit(dict(text="ValueError: pending", module="mask", provider="hosted",
                        eligible=lambda: consent[0]), lambda *args: pytest.fail("revoked automatic request"))
    consent[0] = False
    sent = []
    monkeypatch.setattr(panel, "_open_error_flow_now", lambda *args: sent.append(args))
    panel.open_error_flow("ValueError: pending", "mask")
    clock.now = 61
    control._tick()
    assert len(sent) == 1 and sent[0][0] == "ValueError: pending"


def test_saved_explanation_never_reserves_actual_request(monkeypatch):
    monkeypatch.setattr(cr, "saved_explanation", lambda signature: dict(text="saved", tier="best"))
    monkeypatch.setattr(cr, "_reserve_hosted_request", lambda **kwargs: pytest.fail("cached result charged"))
    assert providers.explain_failure({}, providers.HostedProvider())["saved"] == "local"


def test_manual_eight_error_details_survive_hosted_context_filtering(panel, monkeypatch):
    control, clock, _, _, _, _ = batcher()
    monkeypatch.setattr(cr, "_ERROR_BATCHER", control)
    monkeypatch.setattr(cr, "network_allowed", lambda: True)
    provider = providers.HostedProvider()
    monkeypatch.setattr(panel, "_current_provider", lambda: provider)
    monkeypatch.setattr(panel, "_start_stream", lambda **kwargs: None)
    monkeypatch.setattr(cr, "saved_explanation", lambda signature: None)
    calls = []
    def diagnose(summary, question):
        calls.append((summary, question))
        return dict(advice="answer", model="test", tier="best")
    monkeypatch.setattr(cr, "diagnose_remotely", diagnose)
    panel.open_error_flow("ValueError: initial", "mask")
    clock.now = 1
    texts = [f"KeyError: missing column condition_{index}" for index in range(8)]
    for text in texts:
        panel.open_error_flow(text, "mask")
    clock.now = 61
    control._tick()
    with cr._hosted_error_context():
        list(provider.stream_chat(panel._ai_messages))
    assert len(calls) == 1
    assert [entry["error"] for entry in calls[0][0]["_batch_errors"]] == texts
    assert all(text in panel._ai_messages[-1]["content"] for text in texts)


def test_global_ai_disable_revokes_delayed_manual_request_even_with_stale_widget_provider(panel, monkeypatch):
    control, clock, _, _, _, _ = batcher()
    monkeypatch.setattr(cr, "_ERROR_BATCHER", control)
    monkeypatch.setattr(cr, "network_allowed", lambda: True)
    monkeypatch.setattr(panel, "_current_provider", lambda: SimpleNamespace(name="hosted"))
    sent = []
    monkeypatch.setattr(panel, "_open_error_flow_now", lambda *args: sent.append(args))
    panel.open_error_flow("ValueError: first", "mask")
    clock.now = 1
    panel.open_error_flow("ValueError: delayed", "mask")
    cr._update_state(affirmed_policy=dict(provider="none"))
    clock.now = 61
    control._tick()
    assert len(sent) == 1
