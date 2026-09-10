"""The main window's refreshes, updater replies and fold hand-offs.

Each of these runs over parts that may not be there: a sidebar mid-rebuild,
a screen that does not do maturity, an updater helper that is not installed,
a window already closing. None of them is allowed to take the window with
it, and each has a different right answer -- which is why they are guarded
one at a time rather than in one blanket try.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt import app as qt_app  # noqa: E402

pytestmark = pytest.mark.qt


def _explode(*_args, **_kwargs):
    raise RuntimeError("this part has gone")


@pytest.fixture
def win(qtbot, qt_theme_applied):
    window = qt_app.MainWindow()
    qtbot.addWidget(window)
    return window


class _Warnings:
    """Every QMessageBox entry point, recorded rather than shown."""

    def __init__(self, monkeypatch):
        self.warnings = []
        self.informations = []
        monkeypatch.setattr(
            qt_app.QMessageBox, "warning",
            staticmethod(lambda _p, title, text: self.warnings.append(
                (title, text))))
        monkeypatch.setattr(
            qt_app.QMessageBox, "information",
            staticmethod(lambda _p, title, text: self.informations.append(
                (title, text))))


# --------------------------------------------------------------------------
# the refresh cascade
# --------------------------------------------------------------------------

def test_a_theme_refresh_finishes_even_when_every_step_of_it_fails(
        win, monkeypatch):
    """The language pass at the end is what proves it got there."""
    monkeypatch.setattr(win, "_refresh_app_action_visibility", _explode)
    monkeypatch.setattr(win, "apply_dock_mode", _explode)
    monkeypatch.setattr(win, "_rebuild_startup_page", _explode)
    stubborn = QWidget()
    stubborn.refresh_maturity_visibility = _explode
    win._screens["a stubborn screen"] = stubborn
    told = []
    monkeypatch.setattr(win, "refresh_language", lambda: told.append(1))

    win.refresh_theme()

    assert told == [1], "the language pass runs after everything that failed"


def test_a_screen_that_does_maturity_is_asked_to_do_it(win, monkeypatch):
    asked = []
    screen = QWidget()
    screen.refresh_maturity_visibility = lambda: asked.append(1)
    win._screens["a maturity-aware screen"] = screen
    monkeypatch.setattr(win, "refresh_language", lambda: None)

    win.refresh_theme()

    assert asked == [1]


def test_a_language_pass_that_cannot_run_still_rebuilds_the_demo_tips(
        win, monkeypatch, caplog):
    from spacr.qt import i18n

    monkeypatch.setattr(i18n, "retranslate_widget_tree", _explode)
    rebuilt = []
    monkeypatch.setattr(win, "_refresh_demo_status_tips",
                        lambda: rebuilt.append(1))

    with caplog.at_level(logging.ERROR, logger=qt_app.LOG.name):
        win.refresh_language()

    assert rebuilt == [1]
    assert any("UI language" in record.getMessage()
               for record in caplog.records)


def test_a_demo_action_deleted_during_shutdown_does_not_stop_the_rest(win):
    from shiboken6 import delete as _delete_cpp_side
    from PySide6.QtGui import QAction

    live = QAction("live", win)
    dead = QAction("dead", win)
    _delete_cpp_side(dead)
    win._demo_actions = {"gone": dead, "here": live}

    win._refresh_demo_status_tips()

    assert live.statusTip(), "the action after the deleted one was reached"


# --------------------------------------------------------------------------
# the updater
# --------------------------------------------------------------------------

def test_an_upgrade_result_arriving_during_shutdown_is_discarded(
        win, monkeypatch, caplog):
    recorder = _Warnings(monkeypatch)
    win._closing = True

    with caplog.at_level(logging.DEBUG, logger=qt_app.LOG.name):
        win._on_upgrade_done((1, "pip said no"))

    assert recorder.warnings == [], (
        "a dialog parented to a window that is going is a crash")


def test_a_failed_upgrade_shows_the_last_lines_of_what_pip_said(win,
                                                                monkeypatch):
    recorder = _Warnings(monkeypatch)
    win._closing = False

    win._on_upgrade_done((1, "line one\nline two\nERROR: no matching dist"))

    assert recorder.warnings, "a nonzero exit code is reported"
    assert "exit code 1" in recorder.warnings[0][1]
    assert "ERROR: no matching dist" in recorder.warnings[0][1]


def test_an_older_helper_returning_a_bare_code_is_still_understood(
        win, monkeypatch):
    recorder = _Warnings(monkeypatch)
    win._closing = False

    win._on_upgrade_done(1)

    assert recorder.warnings
    assert "No output was captured." in recorder.warnings[0][1]


def test_an_updater_exception_during_shutdown_is_logged_not_shown(
        win, monkeypatch, caplog):
    recorder = _Warnings(monkeypatch)
    win._closing = True

    with caplog.at_level(logging.DEBUG, logger=qt_app.LOG.name):
        win._on_update_worker_failed("check", "Traceback\nValueError: bad")

    assert recorder.warnings == []
    assert any("failed during shutdown" in record.getMessage()
               for record in caplog.records)


@pytest.mark.parametrize("operation,label", [("check", "Update check"),
                                             ("upgrade", "Upgrade")])
def test_an_updater_exception_is_named_by_the_operation_that_raised(
        win, monkeypatch, operation, label):
    recorder = _Warnings(monkeypatch)
    win._closing = False

    win._on_update_worker_failed(
        operation, "Traceback (most recent call last):\nValueError: bad\n\n")

    assert recorder.warnings
    assert recorder.warnings[0][1].startswith(f"{label} failed:")
    assert "ValueError: bad" in recorder.warnings[0][1], (
        "the last real line is the one worth showing")


def test_an_upgrade_with_no_helper_installed_says_so_and_starts_nothing(
        win, monkeypatch):
    recorder = _Warnings(monkeypatch)
    win._closing = False
    started = []
    monkeypatch.setattr(win, "_start_update_worker",
                        lambda *args: started.append(args))
    monkeypatch.setattr(
        qt_app.QMessageBox, "question",
        staticmethod(lambda *args, **kwargs: qt_app.QMessageBox.Yes))

    import sys
    monkeypatch.setitem(sys.modules, "spacr.updater", None)

    info = type("_Info", (), {
        "error": "", "latest_release": "9.9.9", "upgrade_available": True,
        "installed_version": "1.0.0"})()
    win._on_update_check_done(info)

    assert started == [], "nothing to run means nothing is started"
    assert recorder.warnings
    assert "Upgrade unavailable" in recorder.warnings[0][1]


# --------------------------------------------------------------------------
# opening a module that has been folded into another
# --------------------------------------------------------------------------

def test_a_host_that_was_never_built_has_no_switch_to_press(win):
    before = dict(win._screens)

    result = win._switch_a_fold_on("a host nobody opened", "timelapse")

    assert result is None
    assert win._screens == before, "a lookup must not manufacture the host"


def test_a_host_whose_fold_set_cannot_be_read_is_left_alone(win, monkeypatch,
                                                            caplog):
    from spacr.qt.screens import mask

    monkeypatch.setattr(mask, "fold_set", _explode)
    win._screens["mask"] = QWidget()

    with caplog.at_level(logging.DEBUG, logger=qt_app.LOG.name):
        win._switch_a_fold_on("mask", "timelapse")

    assert any("could not switch timelapse on in mask" in record.getMessage()
               for record in caplog.records)


# --------------------------------------------------------------------------
# coming back after a restart
# --------------------------------------------------------------------------

def test_a_restart_record_with_no_module_reopens_nothing(win, monkeypatch):
    from spacr import restart_state

    monkeypatch.setattr(restart_state, "take", lambda: {"settings": {}})
    opened = []
    monkeypatch.setattr(win, "open_module", lambda key: opened.append(key))

    assert win.resume_after_restart() == ""
    assert opened == []


def test_a_module_that_will_not_reopen_leaves_the_window_on_home(
        win, monkeypatch, caplog):
    from spacr import restart_state

    monkeypatch.setattr(restart_state, "take",
                        lambda: {"module": "mask", "settings": {"src": "/x"}})
    monkeypatch.setattr(win, "open_module", _explode)

    with caplog.at_level(logging.ERROR, logger=qt_app.LOG.name):
        assert win.resume_after_restart() == ""

    assert any("could not reopen mask" in record.getMessage()
               for record in caplog.records)


# --------------------------------------------------------------------------
# handing a hit to Investigate Hit
# --------------------------------------------------------------------------

def test_a_selected_hit_carries_its_direction_and_its_guides(win,
                                                              monkeypatch):
    handed = []
    monkeypatch.setattr(win, "_on_train_requested",
                        lambda key, settings: handed.append((key, settings)))

    win._on_investigate_hit_requested({
        "folder": "/runs/plate1", "gene": "TSG101",
        "guides": ["g1", "g2"], "effect": -1.5, "fdr": 0.01,
        "n_guides": 2, "well_support": 7, "phenotype": "infection",
    })

    key, settings = handed[0]
    assert key == "investigate_hit"
    assert settings["target_gene"] == "TSG101"
    assert settings["target_guides"] == ["g1", "g2"]
    assert settings["hit_direction"] == "negative", (
        "the sign of the effect is the direction, not a separate field")


def test_a_hit_with_a_positive_effect_is_called_positive(win, monkeypatch):
    handed = []
    monkeypatch.setattr(win, "_on_train_requested",
                        lambda key, settings: handed.append(settings))

    win._on_investigate_hit_requested({"effect": 0.0})

    assert handed[0]["hit_direction"] == "positive"


# --------------------------------------------------------------------------
# the setup slides
# --------------------------------------------------------------------------

def test_a_setup_screen_that_cannot_be_built_is_not_fatal(win, monkeypatch,
                                                           caplog):
    from spacr.qt.widgets import setup_slides

    monkeypatch.setattr(setup_slides, "SetupSlides", _explode)

    with caplog.at_level(logging.DEBUG, logger=qt_app.LOG.name):
        win._show_setup()

    assert any("setup screen" in record.getMessage()
               for record in caplog.records)


def test_the_hotkey_map_opens_over_the_window_it_was_asked_from(win):
    win.resize(900, 600)

    win._show_shortcuts()

    assert getattr(win, "_spacr_shortcut_overlay", None) is not None, (
        "one map, three doors")


# --------------------------------------------------------------------------
# the parts that are simply not there
# --------------------------------------------------------------------------

class _SidebarMidRebuild:
    """A sidebar whose refreshes all fail, as one being rebuilt does."""

    refresh_icons = staticmethod(_explode)
    refresh_visibility = staticmethod(_explode)


def test_a_sidebar_that_refuses_every_refresh_still_lets_the_theme_land(
        win, monkeypatch):
    monkeypatch.setattr(win, "_sidebar", _SidebarMidRebuild())
    told = []
    monkeypatch.setattr(win, "refresh_language", lambda: told.append(1))

    win.refresh_theme()

    assert told == [1], (
        "the sidebar is asked twice and neither refusal may end the pass")


def test_a_window_with_no_drawer_yet_has_no_dock_to_apply(win, monkeypatch):
    monkeypatch.setattr(win, "_app_drawer", None)
    before = win.dock_mode()

    win.apply_dock_mode("locked")

    assert win.dock_mode() == before, (
        "a mode written with nowhere to put the sidebar is a mode lost")


def test_a_home_page_that_cannot_be_refreshed_is_still_shown(win,
                                                             monkeypatch):
    monkeypatch.setattr(win._startup, "refresh", _explode, raising=False)

    win._on_nav_selected("__home__")

    assert win._stack.currentWidget() is win._startup
    assert win._status_app_label.text()


def test_a_screen_that_cannot_be_themed_or_translated_still_opens(
        win, monkeypatch, caplog):
    from spacr.qt import i18n

    monkeypatch.setattr(win, "_theme_screen", _explode)
    monkeypatch.setattr(i18n, "retranslate_widget_tree", _explode)

    with caplog.at_level(logging.ERROR, logger=qt_app.LOG.name):
        win._on_nav_selected("mask")

    assert win._stack.currentWidget() is win._screens["mask"], (
        "decoration must never stop a screen from opening")
    messages = [record.getMessage() for record in caplog.records]
    assert any("Could not theme the mask screen" in m for m in messages)
    assert any("Could not translate the mask screen" in m for m in messages)


def test_a_screen_that_animates_its_own_background_is_left_alone(win):
    from spacr.qt.screens.app_screen import uses_ambient_background

    key = next((k for k, *_rest in qt_app.APPS
                if not uses_ambient_background(k)), "")
    assert key, "some module draws its own background"
    screen = QWidget()

    win._theme_screen(screen, key)

    assert screen.findChildren(QWidget) == [], (
        "two independent animations would fight for the same pixels")


def test_the_backdrop_is_not_installed_when_it_is_switched_off(win,
                                                               monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_ambient_enabled", lambda: False)
    screen = QWidget()

    win._theme_screen(screen, "mask")

    assert screen.findChildren(QWidget) == []
