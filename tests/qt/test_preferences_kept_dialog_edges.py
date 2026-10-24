"""The edges of keeping Preferences warm and of applying it in pieces.

The kept dialog (6be4c1d70) and the sliced Apply decide, at every step,
whether what they hold is still good: a dialog may be dead, out of date,
waiting for a sheet, or the store and the world may fail to answer. Each of
those answers is exercised here through the function that makes it, with
the expensive parts (building the real dialog) replaced where the decision
does not depend on them.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QDialog, QMenu, QScrollArea, QTabWidget, QWidget,
)

from spacr.qt import preferences, theme  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private preference store, and warming allowed."""
    path = tmp_path / "preferences.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    monkeypatch.setenv("SPACR_NETWORK_CONFIG", str(tmp_path / "network.json"))
    monkeypatch.delenv("SPACR_BENCHMARK_JSON", raising=False)
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    yield path
    theme.finish_the_sliced_sheeting()


def _destroyed(widget):
    """``widget`` with its C++ half gone and the Python wrapper kept."""
    from shiboken6 import delete

    delete(widget)
    return widget


def _dead_widget(qtbot):
    return _destroyed(QWidget())


@pytest.fixture
def owner(qtbot, store):
    """A shown window with warming state and nothing built."""
    window = QWidget()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    window._spacr_preferences_warmth = {
        "dialog": None, "returned": None, "retired": [], "timer": None}
    yield window
    state = preferences._preferences_warmth(window)
    timer = state.get("timer")
    try:
        if timer is not None:
            timer.stop()
    except RuntimeError:
        pass


def _bare_window():
    """The Preferences window class, empty: no build behind it."""
    return preferences._preferences_window_class()(None)


def test_a_stored_value_reads_the_same_however_it_was_stored():
    assert preferences._plain([1, "a"]) == "[1.0,a]"
    assert preferences._plain(("TRUE",)) == "[true]"
    assert preferences._plain(True) == "true"
    assert preferences._plain(None) == "None"


def test_a_dialog_that_cannot_be_shown_again_says_so(store):
    world = preferences._preferences_world()
    assert not preferences._still_shows_the_store(SimpleNamespace())
    assert not preferences._still_shows_the_store(
        SimpleNamespace(_spacr_shows=({}, world, ()), _reusable=False))
    asking = SimpleNamespace(_spacr_shows=({}, world, ()), _reusable=True,
                             _apply_confirmation=object())
    assert not preferences._still_shows_the_store(asking)
    preferences._settings().setValue("prefs/pane_opacity", 0.5)
    changed = SimpleNamespace(
        _spacr_shows=({"prefs/pane_opacity": "0.9"}, world, ()),
        _reusable=True)
    assert not preferences._still_shows_the_store(changed, saved_by_it=True)


def test_the_world_and_the_draft_are_read_even_when_parts_will_not_answer(
        qtbot, store, monkeypatch):
    from spacr import updater
    from spacr.qt import ai

    def _refuse(*_a, **_k):
        raise OSError("no")

    monkeypatch.setattr(updater, "_read_network_config", _refuse)
    monkeypatch.setattr(ai, "configured_providers", _refuse)
    world = preferences._preferences_world()
    assert world[-2] is None and world[-1] is None

    dialog = _bare_window()
    qtbot.addWidget(dialog)
    dialog._draft_extras = lambda: 1 / 0
    assert preferences._preferences_draft(dialog)[1] is None


@pytest.mark.parametrize("busy", ["mouse", "popup", "modal"])
def test_warming_waits_while_the_user_is_doing_something(
        owner, qtbot, monkeypatch, busy):
    assert not preferences._the_user_is_busy(owner)
    if busy == "mouse":
        monkeypatch.setattr(QApplication, "mouseButtons",
                            staticmethod(lambda: Qt.MouseButton.LeftButton))
    elif busy == "popup":
        monkeypatch.setattr(QApplication, "activePopupWidget",
                            staticmethod(lambda: QMenu()))
    else:
        monkeypatch.setattr(QApplication, "activeModalWidget",
                            staticmethod(lambda: QDialog()))
    assert preferences._the_user_is_busy(owner)


def test_a_slice_without_warming_state_does_nothing(qtbot, store):
    window = QWidget()
    qtbot.addWidget(window)
    preferences._warm_one_slice(window)
    assert preferences._preferences_warmth(window) is None


def test_dead_dialogs_are_dropped_and_a_failed_build_waits(
        owner, qtbot, monkeypatch):
    monkeypatch.setattr(preferences, "_the_user_is_busy", lambda _w: False)
    state = preferences._preferences_warmth(owner)
    state["returned"] = _dead_widget(qtbot)
    preferences._warm_one_slice(owner)
    assert state["returned"] is None and state["retired"] == []

    state["retired"] = [_dead_widget(qtbot)]
    preferences._warm_one_slice(owner)
    assert state["retired"] == []

    def _fails(_parent):
        raise RuntimeError("cannot build")

    state["dialog"] = _dead_widget(qtbot)
    monkeypatch.setattr(preferences, "PreferencesDialog", _fails)
    preferences._warm_one_slice(owner)
    assert state["dialog"] is None


def test_a_kept_dialog_is_retired_restyled_or_rechecked(
        owner, qtbot, monkeypatch):
    monkeypatch.setattr(preferences, "_the_user_is_busy", lambda _w: False)
    state = preferences._preferences_warmth(owner)
    kept = QWidget()
    qtbot.addWidget(kept)
    kept._spacr_shown_off_screen = True

    state["dialog"] = kept
    monkeypatch.setattr(preferences, "_still_shows_the_store",
                        lambda _d, **_k: False)
    preferences._warm_one_slice(owner)
    assert state["dialog"] is None and state["retired"] == [kept]
    state["retired"] = []

    applied = []
    monkeypatch.setattr(preferences, "_still_shows_the_store",
                        lambda _d, **_k: True)
    monkeypatch.setattr(preferences, "_the_application_sheet_is_behind",
                        lambda: True)
    monkeypatch.setattr(preferences, "apply_preferences_to_app",
                        lambda *a, **k: applied.append(1))
    state["dialog"] = kept
    preferences._warm_one_slice(owner)
    assert applied == [1] and state["dialog"] is kept

    sheeted = []
    monkeypatch.setattr(preferences, "_the_application_sheet_is_behind",
                        lambda: False)
    monkeypatch.setattr(preferences, "_sheet_the_kept_dialog",
                        lambda d: sheeted.append(d) or True)
    preferences._warm_one_slice(owner)
    assert sheeted == [kept]


def test_a_kept_dialog_that_will_not_take_the_sheet_is_left_alone(
        monkeypatch):
    def _refuse(_root):
        raise ValueError("no sheet")

    monkeypatch.setattr(theme, "_sheet_one_window", _refuse)
    dialog = SimpleNamespace(_window_only=0)
    assert preferences._sheet_the_kept_dialog(dialog) is False
    assert dialog._window_only == 0


def test_a_second_close_retires_the_first_and_a_lone_dead_one_is_left(
        owner, qtbot, store):
    first, second = _bare_window(), _bare_window()
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    preferences._put_the_preferences_away(owner, first)
    preferences._put_the_preferences_away(owner, second)
    state = preferences._preferences_warmth(owner)
    assert state["retired"] == [first] and state["returned"] is second

    loose = QWidget()
    qtbot.addWidget(loose)
    gone = _destroyed(_bare_window())
    preferences._put_the_preferences_away(loose, gone)


def test_the_first_tab_is_chosen_only_where_there_are_tabs(qtbot):
    dialog = _bare_window()
    qtbot.addWidget(dialog)
    preferences._back_to_the_first_tab(dialog)
    assert getattr(dialog, "_page_tabs", None) is None

    tabs = QTabWidget(dialog)
    tabs.addTab(QWidget(), "one")
    tabs.addTab(QWidget(), "two")
    tabs.setCurrentIndex(1)
    dialog._page_tabs = tabs
    preferences._back_to_the_first_tab(dialog)
    assert tabs.currentIndex() == 0


def test_a_window_whose_look_cannot_be_read_is_redrawn(qtbot, monkeypatch):
    def _refuse():
        raise RuntimeError("no signature")

    monkeypatch.setattr(preferences, "_window_signature", _refuse)
    assert preferences._the_window_needs_refreshing(
        SimpleNamespace(_window_signature_seen=(1,)))

    dialog = _bare_window()
    qtbot.addWidget(dialog)
    dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
    dialog.show()
    assert dialog._window_signature_seen is None
    dialog.hide()

    class _Owner(QWidget):
        refreshes = 0

        def refresh_theme(self):
            _Owner.refreshes += 1

    window = _Owner()
    qtbot.addWidget(window)

    class _Stubborn:
        _window_signature_seen = (1,)

        def __setattr__(self, name, value):
            raise AttributeError(name)

    preferences._refresh_owner_window(window, _Stubborn())
    assert _Owner.refreshes == 1


def test_a_page_coming_back_survives_its_translation_and_glass_failing(
        qtbot, monkeypatch):
    from spacr.qt import i18n
    from spacr.qt.widgets import glass

    def _refuse(*_a, **_k):
        raise RuntimeError("no")

    monkeypatch.setattr(i18n, "retranslate_widget_tree", _refuse)
    monkeypatch.setattr(glass, "_glass_a_part_that_came_later", _refuse)
    dialog = _bare_window()
    qtbot.addWidget(dialog)
    tabs = QTabWidget(dialog)
    scroll = QScrollArea()
    tabs.addTab(scroll, "one")
    page = QWidget()
    dialog._page_tabs = tabs
    dialog._pages_away = {0: page}
    assert dialog._bring_the_page_back(0)
    assert scroll.widget() is page
    assert not dialog._bring_the_page_back(0)


def test_a_dialog_kept_per_language_is_translated_once(qtbot, monkeypatch):
    from spacr.qt import i18n

    app = QApplication.instance()
    i18n.install_dialog_translation(app)
    walked = []
    monkeypatch.setattr(i18n, "retranslate_widget_tree",
                        lambda widget: walked.append(widget))
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.setProperty(i18n.TRANSLATE_ONCE_PER_LANGUAGE, True)
    dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
    for _ in range(2):
        dialog.show()
        QApplication.processEvents()
        dialog.hide()
    assert walked.count(dialog) == 1


def test_the_window_is_redrawn_in_place_or_rebuilt_by_what_changed(
        qtbot, qt_theme_applied, store, monkeypatch):
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    rebuilt, docked, surfaces = [], [], []
    monkeypatch.setattr(window, "refresh_theme", lambda: rebuilt.append(1))
    monkeypatch.setattr(window, "apply_dock_mode",
                        lambda *a: docked.append(1))
    monkeypatch.setattr(window._startup, "refresh_pane_surfaces",
                        lambda: surfaces.append(1))
    assert window.refresh_after_preferences(()) == "nothing"
    assert window.refresh_after_preferences({"theme_choice"}) == "rebuilt"
    assert window.refresh_after_preferences(
        {"dock_mode", "pane_opacity", "tooltips_enabled"}) == "in place"
    assert (rebuilt, docked, surfaces) == ([1], [1], [1])

    def _refuse(*_a):
        raise RuntimeError("no")

    monkeypatch.setattr(window, "apply_dock_mode", _refuse)
    monkeypatch.setattr(window._startup, "refresh_pane_surfaces", _refuse)
    assert window.refresh_after_preferences(
        {"dock_width", "pane_opacity"}) == "in place"


def test_a_sliced_redraw_skips_a_window_that_went_away(qtbot, store,
                                                        monkeypatch):
    calls = []

    class _Owner(QWidget):
        def refresh_theme(self):
            calls.append("theme")

        def refresh_after_preferences(self, changes):
            calls.append(changes)
            raise RuntimeError("half drawn")

    monkeypatch.setattr(preferences, "_PREFERENCES_REDRAW_AFTER_MS", 0)
    kept = _Owner()
    qtbot.addWidget(kept)
    preferences._refresh_owner_window(
        kept, SimpleNamespace(_window_signature_seen=None), in_slices=True)
    qtbot.waitUntil(lambda: len(calls) == 1)
    assert calls[0] == frozenset(preferences._WINDOW_SIGNATURE_NAMES)

    gone = _Owner()
    preferences._refresh_owner_window(
        gone, SimpleNamespace(_window_signature_seen=None), in_slices=True)
    _destroyed(gone)
    qtbot.wait(30)
    assert len(calls) == 1


def test_a_sliced_sheet_passes_over_roots_that_are_gone(qtbot,
                                                        qt_theme_applied):
    app = QApplication.instance()
    preferences.apply_preferences_to_app()
    gone = _destroyed(QWidget())
    shown = QWidget()
    qtbot.addWidget(shown)
    shown.show()
    hidden = QWidget()
    qtbot.addWidget(hidden)
    setattr(app, theme._WINDOW_SHEET_SERIAL,
            getattr(app, theme._WINDOW_SHEET_SERIAL, 0) + 1)
    assert theme._sheet_the_windows_in_slices([gone, shown, hidden]) == 2
    theme._SLICED_SHEETING["owed"].insert(0, gone)
    theme._sheet_the_next_window()
    assert theme.sliced_sheeting_owed() == 1
    theme._SLICED_SHEETING["owed"].insert(0, gone)
    assert theme.finish_the_sliced_sheeting() == 1
    assert theme.sliced_sheeting_owed() == 0
    theme._sheet_the_next_window()


def test_home_sets_again_only_the_rules_that_changed(
        qtbot, qt_theme_applied, store, monkeypatch):
    from spacr.qt.app import MainWindow
    from spacr.qt.widgets import home as home_module

    preferences.set_pane_opacity(0.8)
    window = MainWindow()
    qtbot.addWidget(window)
    page = window._startup
    preferences.set_pane_opacity(0.4)
    owed = page.refresh_pane_surfaces()
    assert owed >= 2
    bar = page._tabs.tabBar()
    qtbot.waitUntil(lambda: bar.styleSheet() == home_module._selected_tab_qss(
        page._P, page._pane_alpha()))
    assert (page.findChild(QWidget, "Hero").styleSheet()
            == home_module.hero_qss())
    assert page.refresh_pane_surfaces() == 0

    def _refuse():
        raise RuntimeError("no")

    monkeypatch.setattr(home_module, "hero_qss", _refuse)
    preferences.set_pane_opacity(0.6)
    assert page.refresh_pane_surfaces() >= 1


def test_the_same_application_font_is_not_set_again(qapp, monkeypatch):
    from spacr.qt.app import _use_open_sans

    if not _use_open_sans(qapp, "regular"):
        pytest.skip("Open Sans is not registered here")
    sets = []
    real = type(qapp).setFont
    monkeypatch.setattr(type(qapp), "setFont",
                        staticmethod(lambda *a: sets.append(a) or real(*a)))
    _use_open_sans(qapp, "regular")
    assert sets == []


def test_an_in_place_redraw_with_no_home_or_no_opacity_change(
        qtbot, qt_theme_applied, store, monkeypatch):
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    surfaces = []
    monkeypatch.setattr(window._startup, "refresh_pane_surfaces",
                        lambda: surfaces.append(1))
    assert window.refresh_after_preferences({"tooltips_enabled"}) == "in place"
    assert surfaces == []
    monkeypatch.setattr(window, "_startup", None)
    assert window.refresh_after_preferences({"pane_opacity"}) == "in place"


def test_home_surfaces_tolerate_a_missing_masthead_tabs_or_a_dying_box(
        qtbot, qt_theme_applied, store):
    from PySide6.QtWidgets import QFrame

    from spacr.qt.theme import active_palette
    from spacr.qt.widgets import home as home_module

    class _Dying(QFrame):
        def styleSheet(self):                          # noqa: N802
            raise RuntimeError("gone")

    class _Bare(QWidget):
        _P = active_palette()
        _pane_alpha = staticmethod(lambda: 0.5)

    bare = _Bare()
    qtbot.addWidget(bare)
    _Dying(bare).setObjectName("HomePanelBox")
    plain = QFrame(bare)
    plain.setObjectName("HomePanelBox")
    owed = home_module.HomePage.refresh_pane_surfaces(bare)
    assert owed == 1
    qtbot.waitUntil(
        lambda: plain.styleSheet() == home_module.panel_box_qss(bare._P))


def test_an_icon_rescale_or_a_redraw_that_fails_does_not_stop_the_rest(
        qtbot, qt_theme_applied, store, monkeypatch):
    def _refuse(*_a, **_k):
        raise RuntimeError("no")

    from spacr.qt import verbose_logger

    levels = []
    monkeypatch.setattr(preferences, "_rescale_icon_sizes", _refuse)
    monkeypatch.setattr(verbose_logger, "apply_verbose_logging", levels.append)
    preferences.apply_preferences_to_app()
    assert len(levels) == 1, "the steps after the icon rescale did not run"

    asked = []

    class _Broken(QWidget):
        def refresh_theme(self):
            asked.append(1)
            raise RuntimeError("cannot redraw")

    window = _Broken()
    qtbot.addWidget(window)
    preferences._refresh_owner_window(window)
    assert asked == [1]


def test_a_whole_sheet_counts_only_the_roots_that_took_it(
        qtbot, qt_theme_applied, monkeypatch):
    app = QApplication.instance()
    shown = QWidget()
    qtbot.addWidget(shown)
    shown.show()
    monkeypatch.setattr(theme, "_sheet_one_window", lambda _root: False)
    assert theme.apply_stylesheet_per_window(
        app, theme.window_stylesheet(app) or "") == 0


def test_the_shown_translation_pass_leaves_waiting_pages_away(qtbot, monkeypatch):
    """Translating a shown Preferences walks the pages it shows, not all nine.

    The after-show pass walked the dialog with ``findChildren``, which brings
    every waiting page back: about 0.5 s of every open on an Intel Mac. A
    page still waiting is translated when it comes back instead.
    """
    from spacr.qt import i18n

    translated = []
    monkeypatch.setattr(i18n, "retranslate_widget_tree",
                        lambda widget: (translated.append(widget),
                                        widget.findChildren(QWidget)))
    dialog = _bare_window()
    qtbot.addWidget(dialog)
    tabs = QTabWidget(dialog)
    scroll = QScrollArea()
    tabs.addTab(scroll, "one")
    page = QWidget()
    dialog._page_tabs = tabs
    dialog._pages_away = {0: page}
    dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
    dialog.show()

    i18n._translate_shown_dialog(dialog)
    assert translated == [dialog]
    assert dialog._pages_away == {0: page}
    assert not dialog.property(i18n.WALKING_SHOWN_TREE)

    dialog.findChildren(QWidget)
    assert not dialog._pages_away and scroll.widget() is page
    assert page in translated
