"""Hover tooltips and the Navigator placeholder are drawn whole at every Zoom.

The maintainer (2026-10-31): tooltips were wide boxes with the text cut in
half at their font scale, and every piece of user-facing text should be
shown whole. Qt sized a tooltip in the unstyled application font before the
per-window sheet gave it the zoomed ``QToolTip`` font and padding, so the
wrapped text needed more lines than the box had. The Navigator composer kept
a 720 px field while its placeholder grew with Zoom, and elided it.

Tooltips can also be turned off: the Show tooltips switch hides every
tooltip and the hover help surfaces, and keeps accessible text.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile  # noqa: E402,F401

from PySide6.QtCore import QPoint, QRect  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QToolTip, QWidget  # noqa: E402

LONG_TIP = (
    "spaCR keeps logs and a folder per run in its home folder. Pruning "
    "deletes the oldest of them once a folder is over its size cap, and "
    "never anything newer than the age you keep, the run in progress or a "
    "log that is open. You see the list before anything is deleted.")

LANGUAGES = ("en", "de", "es", "fr", "hi", "is", "ko", "pt", "sv", "zh_CN")


@pytest.fixture
def window_sheets(qapp):
    """Production styling: the sheet on each window, none on the app."""
    from spacr.qt import theme

    before = qapp.styleSheet()
    applied = []

    def apply(scale, name="dark"):
        qapp.setStyleSheet("")
        theme.set_widget_qss_context(qapp, name, scale, None)
        theme.apply_stylesheet_per_window(
            qapp, theme.stylesheet(theme=name, font_scale=scale,
                                   load_widget_registrars=False))
        applied.append(scale)

    yield apply
    QToolTip.hideText()
    QApplication.processEvents()
    theme._forget_window_stylesheets(qapp)
    qapp.setStyleSheet(before)


def _shown_tip_label():
    for widget in QApplication.topLevelWidgets():
        if widget.objectName() == "qtooltip_label" and widget.isVisible():
            return widget
    return None


def _text(language):
    if language == "en":
        return LONG_TIP
    from spacr.qt.i18n_catalogs import ui_text

    translated = ui_text(LONG_TIP, language)
    assert translated and translated != LONG_TIP, language
    return translated


def _assert_tip_holds(label, text):
    assert label is not None, "no tooltip was shown"
    contents = label.contentsRect()
    needed = label.heightForWidth(label.width())
    assert needed <= label.height(), (
        f"the wrapped text needs {needed} px and the box is "
        f"{label.height()} px: the bottom is cut off")
    document_width = label.fontMetrics().horizontalAdvance(text.split()[0])
    assert document_width <= contents.width()


@pytest.mark.parametrize("scale", [0.5, 0.9, 1.0, 1.5, 2.0])
def test_a_long_tooltip_is_whole_at_every_zoom(qapp, window_sheets, scale):
    from spacr.qt.tooltip_policy import _wrapped_tooltip

    window_sheets(scale)
    QToolTip.showText(QPoint(40, 40), _wrapped_tooltip(LONG_TIP), None,
                      QRect(), 60000)
    for _ in range(3):
        QApplication.processEvents()
    _assert_tip_holds(_shown_tip_label(), LONG_TIP)


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_a_long_translated_tooltip_is_whole(qapp, window_sheets, scale,
                                            language):
    from spacr.qt.tooltip_policy import _wrapped_tooltip

    window_sheets(scale)
    text = _text(language)
    QToolTip.showText(QPoint(40, 40), _wrapped_tooltip(text), None, QRect(),
                      60000)
    for _ in range(3):
        QApplication.processEvents()
    _assert_tip_holds(_shown_tip_label(), text)


def test_a_reused_tooltip_is_measured_in_its_styled_font(qapp, window_sheets):
    from spacr.qt.tooltip_policy import _wrapped_tooltip

    window_sheets(2.0)
    QToolTip.showText(QPoint(40, 40), "short", None, QRect(), 60000)
    QApplication.processEvents()
    QToolTip.showText(QPoint(40, 40), _wrapped_tooltip(LONG_TIP), None,
                      QRect(), 60000)
    for _ in range(3):
        QApplication.processEvents()
    _assert_tip_holds(_shown_tip_label(), LONG_TIP)


def test_a_tooltip_on_screen_follows_a_zoom_change(qapp, window_sheets):
    from spacr.qt.tooltip_policy import _wrapped_tooltip

    window_sheets(1.0)
    QToolTip.showText(QPoint(40, 40), _wrapped_tooltip(LONG_TIP), None,
                      QRect(), 60000)
    for _ in range(3):
        QApplication.processEvents()
    window_sheets(2.0)
    for _ in range(3):
        QApplication.processEvents()
    _assert_tip_holds(_shown_tip_label(), LONG_TIP)


@pytest.mark.parametrize("language", ("en", "de", "hi", "ko"))
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_the_settings_box_holds_its_whole_help(qapp, qtbot, window_sheets,
                                               private_preferences, scale,
                                               language):
    from spacr.qt.widgets.hover_tooltip import HoverTooltip

    private_preferences.set_font_scale(scale)
    window_sheets(scale)
    anchor = QLabel("Anchor")
    qtbot.addWidget(anchor)
    anchor.show()
    text = " ".join([_text(language)] * 3)
    box = HoverTooltip.instance()
    try:
        box.show_for(anchor, "<p>" + text + "</p>", animation=None,
                     immediate=True)
        for _ in range(3):
            QApplication.processEvents()
        label = box._label
        assert label.heightForWidth(label.width()) <= label.height(), (
            f"{language} at {scale:.0%}: the help needs "
            f"{label.heightForWidth(label.width())} px and has {label.height()}")
        area = box.screen().availableGeometry()
        assert box.height() <= area.height() and box.width() <= area.width()
    finally:
        box.hide()


def test_the_fit_leaves_every_other_window_alone(qapp):
    from spacr.qt.theme import _fit_a_tooltip_to_its_styled_text

    plain = QLabel("not a tooltip")
    plain.resize(10, 10)
    try:
        assert _fit_a_tooltip_to_its_styled_text(plain) is False
        assert plain.size().toTuple() == (10, 10)
    finally:
        plain.deleteLater()


@pytest.fixture
def private_preferences(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from spacr.qt import preferences, tooltip_policy

    path = str(tmp_path / "prefs.ini")
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(path, QSettings.IniFormat))
    tooltip_policy.invalidate_tooltip_policy()
    yield preferences
    tooltip_policy.invalidate_tooltip_policy()


def test_off_shows_no_tooltip_and_no_hover_help(qtbot, private_preferences):
    from spacr.qt import tooltip_policy
    from spacr.qt.widgets.hover_tooltip import HoverTooltip

    private_preferences.set_tooltips_enabled(False)
    assert tooltip_policy.tooltips_enabled() is False
    anchor = QLabel("Anchor")
    anchor.setToolTip("Explains the anchor")
    anchor.setAccessibleDescription("Explains the anchor")
    qtbot.addWidget(anchor)
    anchor.show()
    called = []
    delay = tooltip_policy.HoverDelay(anchor)
    delay.schedule(anchor, lambda: called.append(True))
    box = HoverTooltip.instance()
    box.show_for(anchor, "<p>Help</p>")
    qtbot.wait(max(50, tooltip_policy._preferred_delay_ms() + 50))
    assert called == []
    assert not box.isVisible()
    assert not QToolTip.isVisible()
    assert anchor.accessibleDescription() == "Explains the anchor"


def test_off_greys_the_two_surface_switches(qtbot, private_preferences):
    from PySide6.QtWidgets import QWidget as _QWidget

    dialog = private_preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    master = dialog.findChild(_QWidget, "TooltipsEnabled")
    box = dialog.findChild(_QWidget, "TooltipsBox")
    bottom = dialog.findChild(_QWidget, "TooltipsBottom")
    if master is None:
        pytest.skip("the Appearance page is built on demand")
    master.setChecked(True)
    assert box.isEnabled() and bottom.isEnabled()
    master.setChecked(False)
    assert not box.isEnabled() and not bottom.isEnabled()
    master.setChecked(True)
    assert box.isEnabled() and bottom.isEnabled()


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_the_navigator_placeholder_is_whole(qapp, qtbot, private_ai_profile,
                                            monkeypatch, scale, language):
    from spacr.qt import theme
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_font_scale(scale)
    monkeypatch.setenv("SPACR_LANGUAGE", language)
    before = qapp.styleSheet()
    qapp.setStyleSheet(theme.stylesheet("dark", scale))
    try:
        page = _EasyHomePage()
        qtbot.addWidget(page)
        page.resize(1600, 900)
        page.show()
        for _ in range(5):
            QApplication.processEvents()
        field = page.field.panel
        expected = "Drop a file or folder | Type a Path | Ask a Question \u25c9"
        if language != "en":
            from spacr.qt.i18n_catalogs import ui_text
            expected = ui_text(expected, language)
        assert field.placeholderText() == expected
        margins = field.textMargins()
        room = field.width() - margins.left() - margins.right() - 4
        needed = field.fontMetrics().horizontalAdvance(field.placeholderText())
        assert needed <= room, (
            f"{language} at {scale:.0%}: the placeholder needs {needed} px "
            f"and has {room}")
    finally:
        qapp.setStyleSheet(before)
