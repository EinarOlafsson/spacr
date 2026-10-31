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
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QLabel,
    QMenu,
    QTabWidget,
    QFormLayout,
    QLineEdit,
    QPlainTextEdit,
    QScrollBar,
    QTableWidget,
    QTextBrowser,
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


def _focusable_unnamed(host):
    """Visible, enabled, focusable controls under ``host`` with no name."""
    out = []
    for widget in host.findChildren(QWidget):
        if isinstance(widget, (QScrollBar, QHeaderView)):
            continue
        if not widget.isVisibleTo(host) or widget.focusProxy() is not None:
            continue
        interactive = isinstance(widget, i18n._a11y_interactive_types()
                                 + (QAbstractItemView,))
        if not interactive and widget.focusPolicy() == Qt.NoFocus:
            continue
        if widget.objectName().startswith("qt_") and not interactive:
            continue
        if not _accessible_name(widget).strip():
            out.append(f"{type(widget).__name__}#{widget.objectName() or '?'}"
                       f" in {type(widget.parentWidget()).__name__}")
    return out


def _tab_walk(host):
    """Tab round ``host``; return the widgets that held focus against Tab.

    A rich text widget keeps focus while Tab steps through its links, and
    Qt spends one press wrapping from the last control round to the first, so
    a widget counts as holding focus only past one press per link plus two.
    """
    host.activateWindow()
    focusable = [
        widget for widget in host.findChildren(QWidget)
        if widget.isVisibleTo(host) and widget.isEnabled()
        and widget.focusProxy() is None
        and (widget.focusPolicy() & Qt.TabFocus) == Qt.TabFocus
        and not widget.objectName().startswith("qt_")
    ]
    assert focusable, "nothing in the host takes focus"
    focusable[0].setFocus(Qt.TabFocusReason)
    QApplication.processEvents()
    stuck, run = set(), 0
    for _press in range(3 * len(focusable) + 5):
        before = QApplication.focusWidget()
        QTest.keyClick(before or host, Qt.Key_Tab)
        QApplication.processEvents()
        if QApplication.focusWidget() is before:
            run += 1
            links = (before.text().count("<a ") if isinstance(before, QLabel) else
                     before.toHtml().count("<a ") if isinstance(before, QTextBrowser) else 0)
            if run > links + 2:
                stuck.add(f"{type(before).__name__}#{before.objectName()}")
        else:
            run = 0
    return sorted(stuck)


@pytest.fixture
def main_window(qtbot, qt_theme_applied, monkeypatch, tmp_path):
    monkeypatch.setenv(i18n.ENV_LANGUAGE, "en")
    monkeypatch.setattr(prefs, "_SAFE_MODE", False)
    settings = QSettings(str(tmp_path / "spacr-qt.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    prefs.set_ambient_animation("none")
    prefs.set_preload_policy("on_demand")
    prefs._set_restore_session(False)
    settings.setValue("onboarding/first_run_tour_seen", True)
    window = MainWindow()
    qtbot.addWidget(window, before_close_func=_retire_graphics_menus)
    window.resize(1400, 900)
    window.show()
    i18n.retranslate_widget_tree(window)
    QApplication.processEvents()
    return window


def test_main_window_chrome_controls_are_named(main_window, qtbot):
    """Toolbars, dock and sidebar controls and the status bar are named."""
    from spacr.qt.widgets.loading_screen import LoadingScreen

    for cover in main_window.findChildren(LoadingScreen):
        assert cover.accessibleName() == i18n.tr("spaCR")
    qtbot.waitUntil(lambda: not main_window.findChildren(LoadingScreen),
                    timeout=10000)
    assert not _focusable_unnamed(main_window)
    assert main_window.statusBar() is not None


def test_main_window_menus_name_every_action(main_window):
    """Every reachable menu and every visible action in it has a name."""
    nameless = []
    pending = [action.menu() for action in main_window.menuBar().actions()]
    seen = 0
    while pending:
        menu = pending.pop()
        if menu is None:
            continue
        seen += 1
        if not _accessible_name(menu).strip():
            nameless.append(f"menu {menu.title()!r}")
        for action in menu.actions():
            if action.isSeparator() or not action.isVisible():
                continue
            if not action.text().strip():
                nameless.append(f"action {action.objectName() or '?'}")
            pending.append(action.menu())
    assert seen > 3
    assert not nameless, nameless[:10]


def test_main_window_tab_is_never_trapped(main_window):
    assert _tab_walk(main_window) == []


def test_main_window_item_views_are_named(main_window):
    views = [view for view in main_window.findChildren(QAbstractItemView)
             if view.isVisibleTo(main_window)
             and not isinstance(view, QHeaderView)]
    unnamed = [f"{type(v).__name__}#{v.objectName()}" for v in views
               if not _accessible_name(v).strip()]
    assert not unnamed, unnamed


def test_every_preferences_tab_names_each_focusable_control_and_frees_tab(
        main_window):
    dialog = prefs.PreferencesDialog(main_window)
    try:
        dialog.resize(1100, 1000)
        dialog.show()
        from spacr.qt.widgets.section import Section
        for section in dialog.findChildren(Section):
            section.set_expanded(True)
        tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
        assert tabs is not None and tabs.count() >= 8
        for index in range(tabs.count()):
            tabs.setCurrentIndex(index)
            i18n.retranslate_widget_tree(dialog)
            QApplication.processEvents()
            title = tabs.tabText(index)
            assert not _focusable_unnamed(dialog), title
            assert _tab_walk(dialog) == [], title
    finally:
        dialog.reject()
