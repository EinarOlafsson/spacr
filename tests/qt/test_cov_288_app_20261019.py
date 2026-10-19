"""The window's recovery arms, measured against what each step should leave.

Item 288 holds ``spacr.qt.app`` to its coverage baseline. The statements
that were still dark are almost all one shape: a step that must not take
the window down when a collaborator fails, or an early return when there
is nothing to do. Each test here makes that collaborator fail (or be
absent) and checks what the window did instead: the value returned, the
state it reset, the message it showed, or the fallback it took.

Most tests call a ``MainWindow`` method on a small stand-in ``self`` so
that only the behaviour under test runs; two build a real window because
the step under test runs in its constructor or through Qt's ``super()``.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QRect, Qt
from PySide6.QtGui import QWindowStateChangeEvent
from PySide6.QtWidgets import QMessageBox, QStackedWidget, QVBoxLayout, QWidget

import spacr.restart_state as restart_state
import spacr.settings as settings_mod
import spacr.updater as updater
from spacr.qt import app as app_mod
from spacr.qt import iconset
from spacr.qt import preferences as preferences_mod
from spacr.qt import settings_search as settings_search_mod
from spacr.qt import theme as theme_mod
from spacr.qt.app import MainWindow, _CrampedScroll, _ScreenPrewarm
from spacr.qt.widgets import ambient as ambient_mod


def _boom(*_args, **_kwargs):
    raise RuntimeError("collaborator failed")


class _Status:
    """A status bar that remembers what it was asked to say."""

    def __init__(self):
        self.messages = []
        self.removed = []

    def showMessage(self, text, *_args):
        self.messages.append(text)

    def clearMessage(self):
        self.messages.append("")

    def removeWidget(self, widget):
        self.removed.append(widget)


def _fake(**attrs):
    return types.SimpleNamespace(**attrs)


class _Window:
    """Enough of a top-level window for ``_fit_to_the_work_area``."""

    def __init__(self, maximised=False, screen=None):
        self._maximised = maximised
        self._screen = screen
        self.minimum = None

    def screen(self):
        return self._screen

    def frameGeometry(self):
        return QRect(0, 0, 900, 700)

    def geometry(self):
        return QRect(0, 0, 900, 700)

    def minimumSize(self):
        from PySide6.QtCore import QSize
        return QSize(2000, 2000)

    def setMinimumSize(self, size):
        self.minimum = size

    def isMaximized(self):
        return self._maximised

    def isFullScreen(self):
        return False


class _Screen:
    def __init__(self, rect):
        self._rect = rect

    def availableGeometry(self):
        return self._rect


def test_an_empty_work_area_or_a_maximised_window_is_left_where_it_is():
    empty = _Window(screen=_Screen(QRect()))
    assert app_mod._fit_to_the_work_area(empty) is False
    assert empty.minimum is None

    maximised = _Window(maximised=True)
    assert app_mod._fit_to_the_work_area(
        maximised, screen=_Screen(QRect(0, 0, 1280, 680))) is False
    assert maximised.minimum.width() <= 1280
    assert maximised.minimum.height() <= 680


def test_a_window_whose_screen_cannot_be_asked_is_not_moved():
    window = types.SimpleNamespace(screen=_boom)
    assert app_mod._fit_to_the_work_area(window) is False


def test_a_cramped_scroll_without_content_only_clears_its_pending_fit(qapp):
    scroll = _fake(_fit_pending=True, widget=lambda: None)
    _CrampedScroll._fit_content(scroll)
    assert scroll._fit_pending is False


def test_a_screen_without_a_layout_or_already_wrapped_is_not_wrapped_again(qapp):
    bare = QWidget()
    assert app_mod._scroll_when_cramped(bare) is False
    assert bare.findChild(_CrampedScroll) is None

    screen = QWidget()
    QVBoxLayout(screen)
    assert app_mod._scroll_when_cramped(screen) is True
    assert app_mod._scroll_when_cramped(screen) is False
    assert len(screen.findChildren(_CrampedScroll)) == 1


def test_unreadable_alpha_preferences_hide_no_apps(monkeypatch):
    monkeypatch.setattr(settings_mod, "_alpha_names", _boom)
    assert app_mod._alpha_hidden_apps() == frozenset()


class _App:
    """A QApplication stand-in for the prewarm's input checks."""

    def __init__(self, buttons=Qt.MouseButton.NoButton, popup=None,
                 modal=None, remove_raises=False):
        self.buttons = buttons
        self.popup = popup
        self.modal = modal
        self.remove_raises = remove_raises
        self.removed = 0

    def mouseButtons(self):
        return self.buttons

    def activePopupWidget(self):
        return self.popup

    def activeModalWidget(self):
        return self.modal

    def removeEventFilter(self, _obj):
        self.removed += 1
        if self.remove_raises:
            raise RuntimeError("already deleted")


def _with_app(monkeypatch, app):
    monkeypatch.setattr(app_mod, "QApplication",
                        types.SimpleNamespace(instance=lambda: app))


def test_stopping_the_prewarm_survives_an_application_already_torn_down(monkeypatch):
    stopped = []
    app = _App(remove_raises=True)
    _with_app(monkeypatch, app)
    prewarm = _fake(_timer=_fake(stop=lambda: stopped.append(True)),
                    finished=False, _steps=None)
    _ScreenPrewarm.stop(prewarm)
    assert stopped == [True]
    assert prewarm.finished is True
    assert app.removed == 1
    assert prewarm._steps is None


@pytest.mark.parametrize("app, reason", [
    (_App(buttons=Qt.MouseButton.LeftButton), "input"),
    (_App(popup=object()), "popup"),
    (_App(modal=object()), "modal"),
])
def test_a_held_button_an_open_popup_or_a_modal_dialog_holds_the_prewarm(
        monkeypatch, app, reason):
    _with_app(monkeypatch, app)
    monkeypatch.setattr(app_mod, "_time_monotonic", lambda: 1e9)
    prewarm = _fake(_last_input=0.0, _window=object())
    assert _ScreenPrewarm.busy_reason(prewarm) == reason


def test_a_window_opens_without_its_module_hints(monkeypatch, qtbot,
                                                  qt_theme_applied):
    import spacr.qt.module_hints as module_hints

    monkeypatch.setattr(module_hints, "install_module_hints", _boom)
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._module_hints is None
    assert window.statusBar() is not None


def test_a_state_change_survives_a_broken_maximise_mark(qtbot, qt_theme_applied):
    window = MainWindow()
    qtbot.addWidget(window)
    window._maximise_button = object()
    window._drag_from = (1, 2)
    window.changeEvent(QWindowStateChangeEvent(Qt.WindowState.WindowNoState))
    assert window._drag_from is None


def test_following_the_work_area_needs_a_native_window_and_survives_one_that_fails():
    calls = []
    without_handle = _fake(windowHandle=lambda: None,
                           _watch_the_screen=calls.append)
    MainWindow._follow_the_work_area(without_handle)
    assert calls == []
    assert not hasattr(without_handle, "_follows_screen_changes")

    failing = _fake(windowHandle=_boom, _watch_the_screen=calls.append)
    MainWindow._follow_the_work_area(failing)
    assert calls == []


class _Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


def test_each_screen_is_watched_once_and_a_missing_screen_not_at_all():
    window = _fake(_on_work_area_changed=lambda *a: None)
    MainWindow._watch_the_screen(window, None)
    assert not hasattr(window, "_watched_screens")

    screen = _fake(availableGeometryChanged=_Signal())
    MainWindow._watch_the_screen(window, screen)
    MainWindow._watch_the_screen(window, screen)
    assert len(screen.availableGeometryChanged.slots) == 1
    assert window._watched_screens == {id(screen)}


def test_moving_to_another_screen_watches_it_and_refits(monkeypatch):
    events = []
    window = _fake(_watch_the_screen=lambda s: events.append(("watch", s)),
                   _on_work_area_changed=lambda: events.append("refit"))
    MainWindow._on_screen_changed(window, "screen-2")
    assert events == [("watch", "screen-2"), "refit"]

    scheduled = []
    monkeypatch.setattr(app_mod, "_after",
                        lambda ms, owner, fn: scheduled.append((ms, owner, fn)))
    MainWindow._on_work_area_changed(window)
    assert scheduled and scheduled[0][0] == 0 and scheduled[0][1] is window
    assert callable(scheduled[0][2])


def test_the_menu_bar_relay_tolerates_a_missing_or_broken_bar():
    updates = []
    no_bar = _fake(menuBar=lambda: None,
                   update=lambda *a: updates.append(a))
    MainWindow._relay_the_menu_bar(no_bar)

    broken = _fake(menuBar=lambda: _fake(cornerWidget=_boom),
                   update=lambda *a: updates.append(a))
    MainWindow._relay_the_menu_bar(broken)
    assert updates == []


def test_window_actions_are_not_lifted_before_they_exist():
    inserted = []
    menu = _fake(insertAction=lambda *a: inserted.append(a),
                 insertSeparator=lambda *a: inserted.append(a))
    MainWindow._lift_the_window_actions_into_the_spacr_menu(
        _fake(_act_minimise=object(), _act_maximise=None, _act_quit=object()),
        menu)
    assert inserted == []


def test_a_theme_refresh_carries_on_past_every_failing_step(monkeypatch):
    monkeypatch.setattr(preferences_mod, "_apply_alpha_widgets", _boom)
    screen = _fake(refresh_maturity_visibility=_boom,
                   _refresh_alpha_visibility=_boom)
    languages = []
    window = _fake(
        _sidebar=_fake(refresh_icons=_boom, refresh_visibility=_boom),
        _refresh_app_action_visibility=_boom,
        _backdrop_the_dock_column=_boom,
        _screens={"probe": screen},
        _drop_a_redundant_screen_backdrop=_boom,
        apply_dock_mode=_boom,
        _rebuild_startup_page=_boom,
        refresh_language=lambda: languages.append(True))
    MainWindow.refresh_theme(window)
    assert languages == [True]


def test_an_update_check_waits_for_a_frozen_update_already_running():
    status = _Status()
    window = _fake(_uninstall_coordinator=None,
                   _frozen_update_handshake=object(),
                   statusBar=lambda: status)
    MainWindow._check_for_updates(window)
    assert status.messages == ["An update operation is already running."]


class _Host(QWidget):
    """A real parent widget carrying the attributes a method reads."""


def test_whats_new_is_not_shown_while_closing_and_names_the_running_version(
        monkeypatch, qapp):
    closing = _Host()
    closing._closing = True
    assert MainWindow._show_whats_new(closing) is None

    asked = []

    def notes(old, new, releases=()):
        asked.append(old)
        return [{"name": "v9.9", "published": "2026-10-19", "body": "Faster."}]

    monkeypatch.setattr(updater, "_installed_version", lambda: "9.9")
    monkeypatch.setattr(updater, "_release_notes_between", notes)
    monkeypatch.setattr(preferences_mod, "_previous_version", lambda: "")
    host = _Host()
    host._closing = False
    dialog = MainWindow._show_whats_new(host)
    try:
        assert asked == ["0"]
        from PySide6.QtWidgets import QLabel, QTextBrowser
        headings = [label.text() for label in dialog.findChildren(QLabel)]
        assert "You are running spaCR 9.9." in headings
        assert "Faster." in dialog.findChild(QTextBrowser, "WhatsNewNotes").toPlainText()
    finally:
        dialog.close()
        host.deleteLater()
        closing.deleteLater()


def test_an_upgrade_stops_with_a_warning_when_its_helper_cannot_be_imported(
        monkeypatch):
    import spacr.qt.bridge as bridge

    monkeypatch.setattr(updater, "editable_install_location", lambda: None)
    monkeypatch.setattr(bridge, "registry",
                        lambda: _fake(_of_kind=lambda kind: []))
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warnings.append(a[2])))
    monkeypatch.setitem(sys.modules, "spacr.install_cleanup",
                        types.ModuleType("spacr.install_cleanup"))
    started = []
    window = _fake(_closing=False,
                   _start_update_worker=lambda *a: started.append(a))
    info = _fake(error="", latest_release="9.9", upgrade_available=True,
                 installed_version="9.8")
    MainWindow._on_update_check_done(window, info)
    assert started == []
    assert len(warnings) == 1 and warnings[0].startswith("Upgrade unavailable")
    assert not hasattr(window, "_update_version")


def test_a_clean_upgrade_restarts_and_a_finished_sequence_reports_its_install():
    restarts = []
    window = _fake(_closing=False,
                   _restart_after_package_upgrade=lambda: restarts.append(True))
    MainWindow._on_upgrade_done(window, (0, "ok"))
    assert restarts == [True]

    reported = []
    sequence = _fake(_closing=False, _on_upgrade_done=reported.append)
    MainWindow._on_update_sequence_done(sequence, ([], (0, "done")))
    assert reported == [(0, "done")]


def test_cancelling_a_frozen_update_clears_it_even_when_cancel_cannot_publish():
    hidden = []
    window = _fake(_hide_update_progress=lambda: hidden.append(True),
                   _frozen_update_handshake=_fake(cancel=_boom),
                   _frozen_update_timer=None,
                   _frozen_update_plan={"version": "9.9"},
                   _frozen_update_restart_saved=False)
    MainWindow._cancel_frozen_update(window)
    assert hidden == [True]
    assert window._frozen_update_handshake is None
    assert window._frozen_update_plan is None


def test_a_frozen_update_poll_during_close_cancels_instead_of_asking():
    cancelled = []
    asked = []
    window = _fake(_frozen_update_handshake=_fake(status=lambda: asked.append(1)),
                   _closing=True,
                   _cancel_frozen_update=lambda: cancelled.append(True))
    MainWindow._poll_frozen_update(window)
    assert cancelled == [True] and asked == []

    closing_now = _fake(_frozen_update_handshake=_fake(status=_boom),
                        _closing=False)

    def cancel():
        cancelled.append("error")
        closing_now._closing = True

    closing_now._cancel_frozen_update = cancel
    MainWindow._poll_frozen_update(closing_now)
    assert cancelled[-1] == "error"


def test_autosave_writes_nothing_while_the_window_closes():
    collected = []
    window = _fake(_closing=True,
                   _open_app_screens=lambda: collected.append(1) or {})
    assert MainWindow._autosave_settings(window) is False
    assert collected == []


def test_a_last_session_that_fails_to_reopen_is_not_offered(monkeypatch):
    monkeypatch.setattr(app_mod, "_OPEN_FRESH", [False])
    monkeypatch.setattr(preferences_mod, "_get_restore_session", lambda: True)
    monkeypatch.setattr(restart_state, "_last_session",
                        lambda: {"module": "probe", "settings": {"a": 1}})
    offered = []
    window = _fake(open_module=_boom,
                   _offer_open_fresh=lambda *a: offered.append(a))
    assert MainWindow._resume_last_session(window) == ""
    assert offered == []


def test_dropping_open_fresh_with_no_button_or_a_dead_one():
    status = _Status()
    window = _fake(_open_fresh_button=None, statusBar=lambda: status)
    MainWindow._drop_open_fresh(window)
    assert status.removed == []

    def dead_status():
        raise RuntimeError("deleted")

    window = _fake(_open_fresh_button=object(), statusBar=dead_status)
    MainWindow._drop_open_fresh(window)
    assert window._open_fresh_button is None


def test_open_fresh_still_goes_home_when_the_defaults_will_not_apply():
    status = _Status()
    went = []
    screen = _fake(apply_settings_dict=_boom)
    window = _fake(_fresh_settings=("probe", {"a": 1}),
                   _open_app_screens=lambda: {"probe": screen},
                   _drop_open_fresh=lambda: None,
                   statusBar=lambda: status,
                   _on_nav_selected=went.append)
    MainWindow._open_fresh(window)
    assert went == ["__home__"]
    assert window._restored_session_settings == ("", {})
    assert status.messages == [""]


def test_declining_the_drafts_forgets_them():
    restored = []
    window = _fake(_drafts_restore_button=object(),
                   _restore_settings_drafts=lambda: restored.append(True),
                   _pending_drafts={"probe": {"a": 1}})
    MainWindow._on_drafts_answered(window, object())
    assert restored == []
    assert window._pending_drafts == {}


def test_drafts_that_will_not_open_are_counted_out():
    window = _fake(_pending_drafts={"probe": {"a": 1}},
                   _stack=_fake(currentWidget=lambda: _fake(app_key="probe")),
                   open_module=_boom)
    assert MainWindow._restore_settings_drafts(window) == 0
    assert window._pending_drafts == {}


def test_a_module_that_will_not_open_is_named_by_its_key_without_a_title_table(
        monkeypatch, qapp):
    from spacr.qt.screens import app_screen

    monkeypatch.delattr(app_screen, "APP_TITLES", raising=False)
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(self.text()) or 0)

    class _NoStatus(QWidget):
        def statusBar(self):
            raise RuntimeError("no status bar")

    host = _NoStatus()
    try:
        said = MainWindow._say_a_module_would_not_open(
            host, "probe_key", ValueError("bad value"))
    finally:
        host.deleteLater()
    assert said == "probe_key could not open: ValueError: bad value"
    assert shown == [said]


def test_a_prewarm_that_fails_part_way_leaves_no_half_built_screen(qapp):
    stack = QStackedWidget()
    screen = QWidget()
    window = _fake(_build_screen=lambda key: screen,
                   _screen_scales={},
                   _screens={},
                   _theme_screen=_boom,
                   _stack=stack)
    steps = MainWindow._prewarm_steps(window, "probe")
    next(steps)
    assert "probe" in window._screen_scales
    with pytest.raises(StopIteration):
        next(steps)
    assert window._screen_scales == {}
    assert window._screens == {}
    assert stack.count() == 0
    stack.deleteLater()


def _failing(calls, name):
    def fail(*_args, **_kwargs):
        calls.append(name)
        raise RuntimeError(name)
    return fail


def test_handing_the_backdrop_back_survives_a_screen_that_cannot_repaint():
    calls = []
    screen = _fake(_clear_page_surfaces=_failing(calls, "clear"),
                   update=_failing(calls, "update"))
    MainWindow._stop_painting_over_the_window_backdrop(None, screen)
    assert calls == ["clear", "update"]


def test_a_page_joins_the_stack_even_when_marking_and_gating_fail(monkeypatch):
    calls = []
    monkeypatch.setattr(theme_mod, "mark_as_a_sheet_target",
                        _failing(calls, "mark"))
    monkeypatch.setattr(settings_search_mod, "install",
                        _failing(calls, "search"))
    monkeypatch.setattr(preferences_mod, "_apply_alpha_widgets",
                        _failing(calls, "alpha"))
    MainWindow._a_page_joined_the_stack(None, object())
    assert calls == ["mark", "search", "alpha"]


def test_the_dock_backdrop_waits_for_a_central_widget_and_the_import_lock(
        monkeypatch):
    retired = []
    no_host = _fake(centralWidget=lambda: None,
                    _retire_the_dock_backdrop=lambda: retired.append(True))
    MainWindow._backdrop_the_dock_column(no_host)
    assert retired == []

    monkeypatch.setattr(preferences_mod, "get_ambient_enabled", lambda: True)
    monkeypatch.setattr(ambient_mod, "_the_heavy_import_lock_is_free",
                        lambda: False)
    later = []
    from PySide6.QtCore import QTimer
    monkeypatch.setattr(QTimer, "singleShot",
                        staticmethod(lambda ms, fn: later.append((ms, fn))))
    installs = []
    monkeypatch.setattr(ambient_mod, "install_ambient",
                        lambda *a, **k: installs.append(a))
    busy = _fake(centralWidget=lambda: object(), _dock_backdrop=None)
    busy._backdrop_the_dock_column = lambda: None
    MainWindow._backdrop_the_dock_column(busy)
    assert installs == []
    assert later == [(400, busy._backdrop_the_dock_column)]
    assert busy._dock_backdrop is None

    monkeypatch.setattr(preferences_mod, "get_ambient_enabled", _boom)
    failing = _fake(centralWidget=lambda: object(), _dock_backdrop=None)
    MainWindow._backdrop_the_dock_column(failing)
    assert failing._dock_backdrop is None


class _Stubborn:
    """A screen whose attributes cannot be set."""

    __slots__ = ()


def test_retiring_the_dock_backdrop_survives_a_dead_backdrop_and_screen():
    doomed = _fake(setProperty=_boom, set_animating=_boom, setParent=_boom,
                   deleteLater=_boom)
    window = _fake(_dock_backdrop=doomed, _screens={"probe": _Stubborn()},
                   _startup=None)
    MainWindow._retire_the_dock_backdrop(window)
    assert window._dock_backdrop is None


class _Raising:
    def __getattr__(self, name):
        raise RuntimeError("half imported")


def test_the_pyside_probe_is_left_alone_when_shiboken_cannot_be_asked(monkeypatch):
    monkeypatch.setitem(sys.modules, "shibokensupport.feature", _Raising())
    assert app_mod._quicken_the_pyside_import_probe() is False


def test_a_module_whose_source_cannot_be_read_goes_to_the_original_probe(tmp_path):
    asked = []
    module = types.SimpleNamespace(__file__=str(tmp_path / "gone.py"))
    assert app_mod._module_source_mentions_pyside(
        lambda m: asked.append(m) or True, module) is True
    assert asked == [module]


def test_matplotlib_keeps_its_cache_in_the_home_folder_off_linux(monkeypatch):
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setattr(app_mod.sys, "platform", "darwin")
    assert app_mod._matplotlib_cache_dir() == Path.home() / ".matplotlib"


def test_no_font_cache_child_starts_when_the_cache_folder_cannot_be_found(
        monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(app_mod, "_matplotlib_cache_dir", _boom)
    assert app_mod._start_the_font_cache_in_a_child() is None


def test_the_glyph_font_and_the_icon_prewarm_survive_a_failing_iconset(monkeypatch):
    monkeypatch.setattr(preferences_mod, "in_safe_mode", lambda: False)
    monkeypatch.setattr(iconset, "icon", _boom)
    assert app_mod._register_the_glyph_font() is None

    monkeypatch.setattr(iconset, "active_theme", _boom)
    assert app_mod._start_icon_prewarm() is None

    monkeypatch.setattr(iconset, "active_theme", lambda: "dark")
    warmed = []

    def warm(theme):
        warmed.append(theme)
        raise OSError("bad icon file")

    monkeypatch.setattr(iconset, "_warm_the_bundled_icons", warm)
    thread = app_mod._start_icon_prewarm()
    thread.join(5)
    assert not thread.is_alive()
    assert warmed == ["dark"]
