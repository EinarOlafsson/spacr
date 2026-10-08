"""Action help stays below the controls without moving the page."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QPushButton

from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.widgets.hint_bar import HintBar


def _screen(qtbot, key):
    screen = AppScreen(app_key=key)
    qtbot.addWidget(screen)
    screen.resize(1200, 850)
    screen.show()
    qtbot.wait(10)
    return screen


def test_each_primary_action_explains_itself_in_the_fixed_footer(
        qtbot, immediate_hover_help):
    screen = _screen(qtbot, "measure")
    controls = (
        screen._btn_run, screen._btn_stop, screen._btn_import,
        screen._btn_analysis_lock, screen._btn_remote, screen._btn_clear,
        screen._btn_copy_console, screen._btn_preferences,
    )
    footer = screen._hint_strip
    height = footer.height()
    positions = (screen._btn_run.pos(), screen._btn_stop.pos())
    immediate_hover_help(
        screen._btn_run, lambda: footer.text() != screen._default_hint())
    messages = []
    for control in controls:
        assert control in screen._action_hints
        assert control.accessibleDescription()
        assert screen.eventFilter(control, QEvent(QEvent.ToolTip)) is True
        screen._show_hover_hint(control)
        messages.append(footer.text())
        assert footer.text() != screen._default_hint()
        assert footer.height() == height
        assert (screen._btn_run.pos(), screen._btn_stop.pos()) == positions
    assert len(set(messages)) == len(controls)
    assert screen._progress.parentWidget() is screen._action_status


def test_progress_and_status_text_do_not_move_the_action_help(qtbot):
    screen = _screen(qtbot, "measure")
    before = (screen._actions_row.height(), screen._hint_strip.y(),
              screen._btn_run.pos(), screen._action_status.height())
    screen._progress.setVisible(True)
    screen._gpu_progress.setText("GPU progress 25%")
    screen._gpu_progress.setVisible(True)
    qtbot.wait(1)
    assert (screen._actions_row.height(), screen._hint_strip.y(),
            screen._btn_run.pos(), screen._action_status.height()) == before
    assert screen._progress.isVisibleTo(screen)
    assert screen._gpu_progress.isVisibleTo(screen)


@pytest.mark.parametrize("key", ("mask", "measure", "umap"))
def test_mode_switches_use_the_same_footer_without_native_popups(qtbot, key):
    screen = _screen(qtbot, key)
    controls = [screen._ai_switch]
    controls.extend(
        control for control in (
            getattr(screen, "_ops_switch", None),
            getattr(screen, "_gpu_switch", None),
            getattr(screen, "_interactive_switch", None),
        ) if control is not None)
    controls.extend(screen._dimension_switches.values())
    if getattr(screen, "_preview_switch", None) is not None:
        controls.append(screen._preview_switch)
    if key in ("mask", "measure"):
        assert set(screen._dimension_switches) == {"z", "t"}
    if key == "mask":
        assert screen._ops_switch in controls
    if key == "umap":
        assert screen._interactive_switch in controls
        assert screen._gpu_switch in controls
    for control in controls:
        assert control is not None
        assert control in screen._action_hints
        assert screen.eventFilter(control, QEvent(QEvent.ToolTip)) is True
        screen._show_hover_hint(control)
        assert screen._hint_strip.text() != screen._default_hint()


def test_preferences_hint_height_survives_long_text_and_font_change(qtbot):
    bar = HintBar()
    qtbot.addWidget(bar)
    button = QPushButton("Apply")
    qtbot.addWidget(button)
    bar.explain(button, "Apply saved settings without moving the dialog.")
    bar.show()
    initial = bar.height()
    bar.eventFilter(button, QEvent(QEvent.Enter))
    assert bar.minimumHeight() == bar.maximumHeight() == initial
    bar.explain(button, "A much longer action explanation. " * 30)
    bar.eventFilter(button, QEvent(QEvent.Enter))
    assert bar.height() == initial
    font = bar.font()
    font.setPointSize(max(1, font.pointSize()) + 4)
    bar.setFont(font)
    margins = bar.contentsMargins()
    lines = bar.fontMetrics().lineSpacing() * 4
    expected = max(28, lines + 12,
                   lines + margins.top() + margins.bottom()
                   + bar._resize_handle.height())
    assert bar.minimumHeight() == bar.maximumHeight() == expected


@pytest.mark.parametrize('key', ['mask', 'measure', 'analyze_plaques'])
def test_real_action_hover_stays_inline_after_translation(qtbot, monkeypatch, immediate_hover_help, key):
    from PySide6.QtGui import QHelpEvent
    from PySide6.QtWidgets import QApplication, QToolTip

    from spacr.qt import tooltip_policy
    from spacr.qt.i18n import retranslate_widget_tree

    screen = _screen(qtbot, key)
    retranslate_widget_tree(screen, language='en')
    tooltip_policy.install_tooltip_policy()
    shown = []
    monkeypatch.setattr(QToolTip, 'showText', lambda *args: shown.append(args))
    controls = [screen._btn_run, screen._btn_stop, screen._btn_import, screen._ai_switch]
    controls.extend(screen._dimension_switches.values())
    controls.extend(control for control in (getattr(screen, '_ops_switch', None),
                                           getattr(screen, '_preview_switch', None)) if control)
    for control in controls:
        if not control.isVisible():
            control.show()
    for _ in range(3):
        QApplication.processEvents()
    footer = screen._hint_strip
    height = footer.height()
    positions = screen._actions_row.geometry(), screen._action_status.geometry()
    for control in controls:
        assert control.toolTip() == ''
        assert control.accessibleDescription()
        screen._write_hint(screen._default_hint())
        immediate_hover_help(control, lambda: footer.text() != screen._default_hint())
        local = control.rect().center()
        QApplication.sendEvent(control, QHelpEvent(QEvent.ToolTip, local, control.mapToGlobal(local)))
        qtbot.wait(1)
        assert shown == []
        assert footer.height() == height
        assert (screen._actions_row.geometry(), screen._action_status.geometry()) == positions
        QApplication.sendEvent(control, QEvent(QEvent.Leave))


def test_preview_refresh_action_uses_footer_through_real_hover(qtbot, immediate_hover_help):
    screen = _screen(qtbot, 'mask')
    screen._live_preview_card.show()
    qtbot.wait(1)
    button = screen._live_preview_card._refresh_button
    assert button in screen._action_hints
    assert button.toolTip() == ''
    assert button.accessibleDescription()
    immediate_hover_help(button, lambda: screen._hint_strip.text() != screen._default_hint())


def test_action_help_retranslates_from_english_through_language_changes(
        qtbot, monkeypatch, immediate_hover_help):
    from PySide6.QtGui import QHelpEvent
    from PySide6.QtWidgets import QApplication, QToolTip

    from spacr.qt import i18n, tooltip_policy

    screen = _screen(qtbot, "measure")
    button = screen._ai_switch
    source = screen._action_hints[button]
    assert source.startswith("Click to toggle AI.")
    shown = []
    monkeypatch.setattr(QToolTip, "showText", lambda *args: shown.append(args))
    tooltip_policy.install_tooltip_policy()
    messages = []
    for language in ("en", "fr", "ko"):
        monkeypatch.setattr(i18n, "current_language", lambda: language)
        i18n.retranslate_widget_tree(screen, language=language)
        screen._install_action_hints()
        assert screen._action_hints[button] == source
        assert button.toolTip() == ""
        assert button.accessibleDescription()
        screen._write_hint(screen._default_hint())
        immediate_hover_help(button, lambda: screen._hint_strip.text() != screen._default_hint())
        assert screen._hint_strip.toolTip() == i18n.tr(source, language)
        messages.append(screen._hint_strip.toolTip())
        local = button.rect().center()
        QApplication.sendEvent(button, QHelpEvent(QEvent.ToolTip, local, button.mapToGlobal(local)))
        QApplication.sendEvent(button, QEvent(QEvent.Leave))
    assert len(set(messages)) == 3
    assert shown == []


def test_inline_registration_preserves_ordinary_setting_checkbox_help(qtbot):
    from PySide6.QtWidgets import QCheckBox

    from spacr.qt.tooltip_policy import OPT_OUT_PROPERTY

    screen = _screen(qtbot, "measure")
    checkbox = QCheckBox("setting", screen)
    checkbox.setProperty("settingKey", "scientific_setting")
    checkbox.setProperty("_spacr_i18n_tooltip", "Setting help")
    checkbox.setToolTip("Setting help")
    screen._install_action_hints()
    assert checkbox not in screen._action_hints
    assert checkbox.toolTip() == "Setting help"
    assert checkbox.property("_spacr_i18n_tooltip") == "Setting help"
    assert not checkbox.property(OPT_OUT_PROPERTY)


def test_first_use_preview_commands_register_inline_help(qtbot, immediate_hover_help):
    from PySide6.QtWidgets import QPushButton, QToolButton

    from spacr.qt.screens.app_screen import _LIVE_PREVIEW

    screen = _screen(qtbot, "mask")
    assert screen._part_is_owed(_LIVE_PREVIEW)
    screen._build_owed_part(_LIVE_PREVIEW)
    screen._live_preview_card.show()
    screen._live_preview.show()
    qtbot.wait(1)
    commands = screen._live_preview.findChildren(QPushButton)
    commands += screen._live_preview.findChildren(QToolButton)
    inline = [control for control in commands if control.accessibleDescription()]
    assert screen._live_preview._cancel_btn in inline
    for control in inline:
        assert control in screen._action_hints
        assert control.toolTip() == ""
    button = screen._live_preview._cancel_btn
    button.setEnabled(True)
    immediate_hover_help(button, lambda: screen._hint_strip.text() != screen._default_hint())
