"""Accessibility pass: accessible names, Tab order and the high-contrast theme.

The a11y lint sweeps every registered module screen offscreen, the way
``MainWindow`` opens one (build, language pass, show), and fails when a
visible control would be announced by a screen reader with no name. The name
is read through ``QAccessible`` -- what assistive technology is actually
given -- so a button named by its caption passes without an explicit
``setAccessibleName`` and an icon-only one does not.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAccessible
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QLineEdit,
    QPlainTextEdit,
    QScrollBar,
    QTableWidget,
    QToolButton,
    QWidget,
)

from spacr.qt import i18n
from spacr.qt import preferences as prefs
from spacr.qt import theme as theme_mod
from spacr.qt.app import APPS, MainWindow

from .test_all_module_smoke import _FactoryHost, _retire_graphics_menus

#: The screens a keyboard user lands on first: the Core section of the dock.
MAIN_SCREENS = tuple(key for key, _n, _d, section in APPS
                     if section == "Core")


def _accessible_name(widget) -> str:
    interface = QAccessible.queryAccessibleInterface(widget)
    return str(interface.text(QAccessible.Name) or "") if interface else ""


def _open(qtbot, monkeypatch, app_key):
    """Build, translate and show ``app_key`` as ``MainWindow`` does."""
    if app_key == "run_history":
        monkeypatch.setattr(
            "spacr.qt.screens.run_history.search_runs", lambda: [])
    screen = MainWindow._build_screen(_FactoryHost(), app_key)
    qtbot.addWidget(screen, before_close_func=_retire_graphics_menus)
    i18n.retranslate_widget_tree(screen)
    screen.resize(1200, 720)
    screen.show()
    QApplication.processEvents()
    return screen


@pytest.mark.parametrize("app_key", [key for key, *_ in APPS])
def test_every_visible_control_has_an_accessible_name(
        qtbot, qt_theme_applied, monkeypatch, app_key):
    """The a11y lint: no visible control on any module screen is nameless."""
    screen = _open(qtbot, monkeypatch, app_key)
    controls = [
        widget for widget in screen.findChildren(QWidget)
        if isinstance(widget, i18n._a11y_interactive_types())
        and not isinstance(widget, QScrollBar)
        and widget.isVisibleTo(screen)
    ]
    assert controls, f"{app_key}: no visible controls to check"
    nameless = [
        f"{type(widget).__name__}#{widget.objectName() or '?'} in "
        f"{type(widget.parentWidget()).__name__}"
        for widget in controls if not _accessible_name(widget).strip()
    ]
    assert not nameless, (
        f"{app_key}: {len(nameless)} visible control(s) without an "
        f"accessible name: {nameless[:10]}")


@pytest.mark.parametrize("app_key", MAIN_SCREENS)
def test_tab_reaches_every_control_on_a_main_screen_and_never_sticks(
        qtbot, qt_theme_applied, monkeypatch, app_key):
    """Tab visits every focusable control once round, and none traps it."""
    screen = _open(qtbot, monkeypatch, app_key)
    screen.activateWindow()
    focusable = [
        widget for widget in screen.findChildren(QWidget)
        if widget.isVisibleTo(screen) and widget.isEnabled()
        and widget.focusProxy() is None
        and (widget.focusPolicy() & Qt.TabFocus) == Qt.TabFocus
    ]
    assert focusable, f"{app_key}: nothing on the screen takes focus"
    focusable[0].setFocus(Qt.TabFocusReason)
    QApplication.processEvents()
    seen, stuck = set(), []
    for _press in range(2 * len(focusable) + 5):
        before = QApplication.focusWidget()
        seen.add(id(before))
        QTest.keyClick(before or screen, Qt.Key_Tab)
        QApplication.processEvents()
        if QApplication.focusWidget() is before:
            stuck.append(f"{type(before).__name__}#{before.objectName()}")
    missed = [f"{type(w).__name__}#{w.objectName()}"
              for w in focusable if id(w) not in seen]
    assert not stuck, f"{app_key}: Tab is trapped in {sorted(set(stuck))}"
    assert not missed, f"{app_key}: Tab never reaches {missed[:10]}"


def test_an_icon_only_button_is_named_after_its_tooltip(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    button = QToolButton(host)
    button.setToolTip("Browse for a folder. Opens the file picker.")
    i18n.retranslate_widget_tree(host)
    assert button.accessibleName() == "Browse for a folder"
    button.setToolTip("Clear the list")
    i18n.retranslate_widget_tree(host)
    assert button.accessibleName() == "Clear the list"


def test_a_field_is_named_after_its_label_and_described_by_its_help(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    form = QFormLayout(host)
    field = QLineEdit()
    form.addRow("Cell diameter", field)
    form.labelForField(field).setToolTip("Expected diameter in pixels.")
    i18n.retranslate_widget_tree(host)
    assert _accessible_name(field) == "Cell diameter"
    assert field.accessibleDescription() == "Expected diameter in pixels."


def test_a_field_beside_a_plain_label_is_named_after_it(qtbot):
    from PySide6.QtWidgets import QHBoxLayout, QLabel
    host = QWidget()
    qtbot.addWidget(host)
    row = QHBoxLayout(host)
    row.addWidget(QLabel("Threshold"))
    field = QLineEdit()
    row.addWidget(field)
    i18n.retranslate_widget_tree(host)
    assert field.accessibleName() == "Threshold"


def test_a_name_set_by_the_code_is_never_replaced(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    button = QToolButton(host)
    button.setToolTip("Zoom in")
    button.setAccessibleName("Magnify")
    i18n.retranslate_widget_tree(host)
    assert button.accessibleName() == "Magnify"


def test_tab_leaves_text_boxes_and_tables(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    text = QPlainTextEdit(host)
    table = QTableWidget(2, 2, host)
    keeps = QPlainTextEdit(host)
    keeps.setProperty("a11yKeepsTab", True)
    i18n.retranslate_widget_tree(host)
    assert text.tabChangesFocus()
    assert not table.tabKeyNavigation()
    assert not keeps.tabChangesFocus()


@pytest.fixture
def store(qapp, tmp_path, monkeypatch):
    """A throwaway INI store, so no test touches the real preferences."""
    settings = QSettings(str(tmp_path / "spacr-qt.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    monkeypatch.setattr(prefs, "_SAFE_MODE", False)
    return settings


def test_high_contrast_is_offered_in_appearance_and_persists(store):
    assert "high_contrast" in theme_mod.THEMES
    assert "high_contrast" in prefs.PALETTE_THEMES
    tokens = [token for _label, token in prefs.theme_choices()]
    assert "high_contrast" in tokens
    assert prefs.theme_description("high_contrast")
    prefs.set_theme_choice("high_contrast")
    assert prefs.get_theme_choice() == "high_contrast"
    assert prefs.resolve_effective_theme() == "high_contrast"


def test_high_contrast_text_clears_the_enhanced_level():
    """Every text role reaches 7:1 on every surface it is painted on."""
    assert theme_mod.contrast_failures("high_contrast") == []
    assert theme_mod.page_separation_failures("high_contrast") == []
    palette = theme_mod.palette_for("high_contrast")
    for ink in ("fg", "fg_muted", "fg_dim", "accent", "error", "warning",
                "success", "info"):
        for ground in ("bg", "surface", "surface_alt", "surface_hi"):
            ratio = theme_mod.contrast_ratio(palette[ink], palette[ground])
            assert ratio >= 7.0, f"{ink} on {ground} is {ratio:.2f}:1"
    assert theme_mod.stylesheet("high_contrast")
