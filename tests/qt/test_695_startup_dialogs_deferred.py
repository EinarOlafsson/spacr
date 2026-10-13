"""Item 695 follow-up: no startup dialog opens over the logo sequence.

What's new, the installer consent, the tour and the restore-drafts question
are held until the logo has blown away (or a click or key skipped it) and then
open in the order they were asked for. With the sequence off they open at the
time they always did. Time is a fake clock throughout.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from spacr.qt.widgets import loading_screen as ls  # noqa: E402

pytestmark = pytest.mark.qt


class _Clock:
    def __init__(self, now: float = 500.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def on_a_display(monkeypatch):
    from PySide6.QtGui import QGuiApplication

    monkeypatch.delenv(ls.NO_SPLASH_ENV, raising=False)
    monkeypatch.setattr(QGuiApplication, "platformName",
                        staticmethod(lambda: "xcb"))


@pytest.fixture()
def seen(monkeypatch):
    from spacr.qt import first_run, install_consent

    calls = []
    monkeypatch.setattr(install_consent, "maybe_show_installer_consent",
                        lambda parent: calls.append("consent"))
    monkeypatch.setattr(first_run, "maybe_show_tour",
                        lambda window, force=False: calls.append("tour"))
    return calls


def _build(qtbot, monkeypatch, with_cover=True):
    from spacr.qt import app as app_mod
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_preload_policy", lambda: "lazy")
    clock = _Clock()
    made = []

    def install(self):
        if not with_cover:
            return None
        cover = ls.LoadingScreen(parent=self)
        cover._clock = clock
        cover.setGeometry(self.rect())
        cover.show_mosaic()
        cover.show()
        made.append(cover)
        return cover

    monkeypatch.setattr(app_mod.MainWindow, "_install_loading_screen", install)
    window = app_mod.MainWindow()
    qtbot.addWidget(window)
    for timer in (window._consent_timer, window._tour_timer):
        timer.stop()
    window._startup_deferred.clear()
    return window, made, clock


def _ask_for_all(window, seen, monkeypatch):
    monkeypatch.setattr(window, "_show_whats_new",
                        lambda previous=None, releases=(): seen.append(
                            "whats-new"))
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "_note_running_version", lambda v: "1.0")
    monkeypatch.setattr(preferences, "get_refresh_news", lambda: False)
    window._consent_timer.timeout.emit()
    window._tour_timer.timeout.emit()
    window._maybe_show_whats_new()


def test_nothing_opens_over_the_logo_and_all_open_in_order_after(
        qtbot, qt_theme_applied, monkeypatch, on_a_display, seen):
    window, made, clock = _build(qtbot, monkeypatch)
    cover = made[0]
    _ask_for_all(window, seen, monkeypatch)
    assert seen == [] and len(window._startup_deferred) == 3

    window._on_preload_done()
    cover.grab()
    assert cover._state in ("waiting", "holding") and seen == []
    clock.now = cover._first_paint_at + 2.0
    cover._begin()
    assert cover._state == "playing" and seen == []
    clock.now = cover._first_paint_at + 5.99
    cover._tick()
    assert seen == [] and window._startup_sequence_running()
    clock.now = cover._first_paint_at + 6.0
    cover._tick()
    assert cover._state == "done"
    assert seen == ["consent", "tour", "whats-new"]
    assert not window._startup_sequence_running()
    assert window._startup_deferred == []


def test_a_click_while_it_plays_shows_them_at_once(
        qtbot, qt_theme_applied, monkeypatch, on_a_display, seen):
    window, made, clock = _build(qtbot, monkeypatch)
    cover = made[0]
    _ask_for_all(window, seen, monkeypatch)
    window._on_preload_done()
    cover.grab()
    clock.now = cover._first_paint_at + 2.5
    cover._begin()
    assert cover._state == "playing" and seen == []
    cover.skip()
    assert seen == ["consent", "tour", "whats-new"]


def test_a_skip_asked_while_loading_releases_when_loading_ends(
        qtbot, qt_theme_applied, monkeypatch, on_a_display, seen):
    window, made, clock = _build(qtbot, monkeypatch)
    cover = made[0]
    _ask_for_all(window, seen, monkeypatch)
    assert seen == []
    cover.skip()
    assert not window._startup_sequence_running()
    window._on_preload_done()
    assert seen == ["consent", "tour", "whats-new"]


def test_the_drafts_question_waits_too(
        qtbot, qt_theme_applied, monkeypatch, on_a_display):
    from spacr import restart_state

    window, made, clock = _build(qtbot, monkeypatch)
    cover = made[0]
    monkeypatch.setattr(restart_state, "_take_drafts",
                        lambda: {"mask": {"a": 1}})
    opened = []
    monkeypatch.setattr(QMessageBox, "open",
                        lambda self, *a: opened.append(self.objectName()))
    window._offer_settings_drafts()
    assert opened == [] and window._drafts_box is not None
    window._on_preload_done()
    cover.grab()
    clock.now = cover._first_paint_at + 2.5
    cover._begin()
    cover.skip()
    assert opened == ["RestoreDraftsDialog"]


def test_the_whats_new_news_refresh_waits_for_the_sequence_too(
        qtbot, qt_theme_applied, monkeypatch, on_a_display, seen):
    from spacr import updater
    from spacr.qt import preferences

    window, made, clock = _build(qtbot, monkeypatch)
    cover = made[0]
    shown = []
    monkeypatch.setattr(window, "_show_whats_new",
                        lambda previous=None, releases=(): shown.append(
                            (previous, tuple(releases))))
    monkeypatch.setattr(preferences, "_note_running_version", lambda v: "1.0")
    monkeypatch.setattr(preferences, "get_refresh_news", lambda: True)
    monkeypatch.setattr(updater, "fetch_release_notes", lambda: ["r"])
    window._maybe_show_whats_new()
    qtbot.waitUntil(lambda: len(window._startup_deferred) == 1, timeout=5000)
    assert shown == []

    def broken():
        raise RuntimeError("offline")

    monkeypatch.setattr(updater, "fetch_release_notes", broken)
    window._maybe_show_whats_new()
    qtbot.waitUntil(lambda: len(window._startup_deferred) == 2, timeout=5000)
    assert shown == []
    cover.skip()
    window._on_preload_done()
    assert shown == [("1.0", ("r",)), ("1.0", ())]


def test_with_the_sequence_off_everything_opens_at_once(
        qtbot, qt_theme_applied, monkeypatch, on_a_display, seen):
    window, made, clock = _build(qtbot, monkeypatch)
    monkeypatch.setenv(ls.NO_SPLASH_ENV, "1")
    assert not ls.startup_animation_allowed()
    _ask_for_all(window, seen, monkeypatch)
    assert seen == ["consent", "tour", "whats-new"]
    assert window._startup_deferred == []


def test_with_the_animation_preference_off_timing_is_unchanged(
        qtbot, qt_theme_applied, monkeypatch, on_a_display, seen):
    from spacr.qt import preferences

    window, made, clock = _build(qtbot, monkeypatch)
    monkeypatch.setattr(preferences, "get_startup_animation", lambda: False)
    _ask_for_all(window, seen, monkeypatch)
    assert seen == ["consent", "tour", "whats-new"]


def test_offscreen_has_no_cover_and_nothing_is_held(
        qtbot, qt_theme_applied, monkeypatch, seen):
    window, made, clock = _build(qtbot, monkeypatch, with_cover=False)
    assert not window._startup_sequence_running()
    _ask_for_all(window, seen, monkeypatch)
    assert seen == ["consent", "tour", "whats-new"]


def test_a_failing_held_dialog_does_not_stop_the_rest(
        qtbot, qt_theme_applied, monkeypatch, on_a_display):
    window, made, clock = _build(qtbot, monkeypatch)
    ran = []

    def bad():
        raise RuntimeError("boom")

    window._after_the_startup_sequence(bad)
    window._after_the_startup_sequence(lambda: ran.append("ok"))
    made[0].skip()
    window._on_preload_done()
    assert ran == ["ok"]


def test_a_closing_window_opens_nothing(
        qtbot, qt_theme_applied, monkeypatch, on_a_display):
    window, made, clock = _build(qtbot, monkeypatch)
    ran = []
    window._after_the_startup_sequence(lambda: ran.append("x"))
    window._closing = True
    window._release_the_startup_dialogs()
    assert ran == []
    window._closing = False


def test_a_cover_that_is_gone_ends_the_hold(
        qtbot, qt_theme_applied, monkeypatch, on_a_display):
    import shiboken6

    window, made, clock = _build(qtbot, monkeypatch)
    window._loading_screen = None
    window._blowing_cover = made[0]
    assert window._startup_sequence_running()
    made[0]._state = "done"
    assert not window._startup_sequence_running()
    assert window._blowing_cover is None
    window._loading_screen = made[0]
    shiboken6.delete(made[0])
    assert not window._startup_sequence_running()
