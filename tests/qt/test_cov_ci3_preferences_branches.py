"""Preference edges: the Spaceout effect set, prewarm order and permission,
the run-finished tray relay, waiting Preferences pages and a twice-finished
Apply question."""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

import PySide6.QtCore as QtCore
import PySide6.QtWidgets as QtWidgets
from PySide6.QtCore import QSettings
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QDialogButtonBox,
                               QLabel, QMessageBox, QScrollArea, QSlider,
                               QTabWidget, QWidget)

from spacr import run_journal
from spacr.qt import preferences as prefs

pytestmark = pytest.mark.qt


@pytest.fixture
def store(tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "ci3_branches.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    return settings


def test_an_incomplete_spaceout_effect_set_is_refused_and_writes_nothing(store):
    keys = list(prefs._SPACEOUT_FIELD_EFFECT_KEYS)
    partial = {key: False for key in keys[:-1]}
    with pytest.raises(ValueError, match="every known key"):
        prefs._set_spaceout_field_effects(partial)
    with pytest.raises(ValueError):
        prefs._set_spaceout_field_effects({**{k: True for k in keys}, "bogus": True})
    assert not any(key.startswith("spaceout/field/") for key in store.allKeys())
    assert all(prefs._spaceout_field_effects().values())


def test_a_stored_prewarm_order_list_is_joined_and_deduplicated(store, monkeypatch):
    monkeypatch.delenv("SPACR_PREWARM_ORDER", raising=False)
    store.setValue(prefs._KEY_SCREEN_PREWARM_ORDER, ["mask", " measure ", "mask", ""])
    assert prefs._screen_prewarm_order() == ("mask", "measure")
    store.setValue(prefs._KEY_SCREEN_PREWARM_ORDER, [])
    assert prefs._screen_prewarm_order() == prefs._SCREEN_PREWARM_ORDER


@pytest.mark.parametrize("env,allowed,reason", [
    ("1", True, ""), ("on", True, ""),
    ("", False, "laptop or low-memory machine"),
])
def test_forcing_prewarm_on_skips_only_the_laptop_check(
        store, monkeypatch, env, allowed, reason):
    from spacr.qt import laptop_mode

    monkeypatch.setattr(prefs, "_SAFE_MODE", False)
    monkeypatch.setenv("SPACR_PREWARM", env)
    asked = []

    def wanted():
        asked.append(True)
        return True, "two cores"

    monkeypatch.setattr(laptop_mode, "wanted", wanted)
    store.setValue(prefs._KEY_PERFORMANCE_LEVEL, "balanced")
    assert prefs._screen_prewarm_allowed() == (allowed, reason)
    assert bool(asked) is (env == "")
    store.setValue(prefs._KEY_PERFORMANCE_LEVEL, "laptop")
    assert prefs._screen_prewarm_allowed() == (False, "performance level laptop")


def test_an_unreadable_laptop_check_does_not_block_prewarm(store, monkeypatch, caplog):
    from spacr.qt import laptop_mode

    monkeypatch.setattr(prefs, "_SAFE_MODE", False)
    monkeypatch.delenv("SPACR_PREWARM", raising=False)

    def broken():
        raise RuntimeError("no /proc")

    monkeypatch.setattr(laptop_mode, "wanted", broken)
    store.setValue(prefs._KEY_PERFORMANCE_LEVEL, "workstation")
    with caplog.at_level(logging.DEBUG, logger=prefs.LOG.name):
        assert prefs._screen_prewarm_allowed() == (True, "")
    assert any("could not read laptop mode" in r.getMessage() for r in caplog.records)


class _FakeTray:
    available = True
    made = []

    class MessageIcon:
        Critical = "critical"
        Information = "information"

    @staticmethod
    def isSystemTrayAvailable():
        return _FakeTray.available

    def __init__(self, icon, parent):
        self.icon = icon
        self.parent = parent
        self.tooltip = None
        self.shown = 0
        self.messages = []
        self.hidden = 0
        _FakeTray.made.append(self)

    def setToolTip(self, text):
        self.tooltip = text

    def show(self):
        self.shown += 1

    def showMessage(self, title, body, icon, msecs):
        self.messages.append((title, body, icon, msecs))

    def hide(self):
        self.hidden += 1


@pytest.fixture
def relay_env(qapp, monkeypatch):
    monkeypatch.setattr(run_journal, "_DESKTOP_NOTIFIER", [None])
    monkeypatch.setattr(QtWidgets, "QSystemTrayIcon", _FakeTray)
    _FakeTray.available = True
    _FakeTray.made = []
    timers = []
    monkeypatch.setattr(QtCore.QTimer, "singleShot",
                        staticmethod(lambda msecs, func: timers.append((msecs, func))))
    made = []
    yield made, timers
    for relay in made:
        relay.setParent(None)
        relay.deleteLater()


def test_the_relay_builds_one_tray_and_marks_failures_critical(relay_env, qapp):
    made, timers = relay_env
    previous = qapp.windowIcon()
    qapp.setWindowIcon(QIcon())
    try:
        relay = prefs._install_run_notifier(qapp)
        made.append(relay)
        relay.arrived.emit("Run failed", "mask stopped", True)
        relay.arrived.emit("Run done", "measure finished", False)
    finally:
        qapp.setWindowIcon(previous)
    assert len(_FakeTray.made) == 1
    tray = _FakeTray.made[0]
    assert tray.parent is relay and tray.tooltip == "spaCR"
    assert not tray.icon.isNull()
    assert tray.shown == 2
    assert tray.messages == [
        ("Run failed", "mask stopped", "critical", 15000),
        ("Run done", "measure finished", "information", 15000)]
    assert [msecs for msecs, _func in timers] == [20000, 20000]
    timers[0][1]()
    assert tray.hidden == 1


def test_the_tray_uses_the_application_icon_when_there_is_one(relay_env, qapp):
    made, _timers = relay_env
    pixmap = QPixmap(8, 8)
    pixmap.fill(QtCore.Qt.red)
    icon = QIcon(pixmap)
    previous = qapp.windowIcon()
    qapp.setWindowIcon(icon)
    try:
        expected = qapp.windowIcon().cacheKey()
        relay = prefs._install_run_notifier(qapp)
        made.append(relay)
        relay.arrived.emit("t", "b", False)
    finally:
        qapp.setWindowIcon(previous)
    tray = _FakeTray.made[0]
    assert tray.icon.cacheKey() == expected
    assert tray.messages == [("t", "b", "information", 15000)]


def test_a_second_install_reuses_the_relay_and_rewires_the_journal(relay_env, qapp):
    made, _timers = relay_env
    first = prefs._install_run_notifier(qapp)
    made.append(first)
    run_journal._DESKTOP_NOTIFIER[0] = None
    second = prefs._install_run_notifier()
    assert second is first
    assert run_journal._DESKTOP_NOTIFIER[0] is not None
    run_journal._DESKTOP_NOTIFIER[0]("again", "body", False)
    assert _FakeTray.made[0].messages[-1][:2] == ("again", "body")
    assert len(qapp.findChildren(QtCore.QObject, "RunFinishedNotifier")) == 1


def test_without_an_application_nothing_is_installed(relay_env, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(QApplication, "instance", staticmethod(lambda: None))
        assert prefs._install_run_notifier() is None
    assert run_journal._DESKTOP_NOTIFIER == [None]


def test_a_failing_os_notification_is_logged_not_raised(relay_env, qapp, qtbot,
                                                        monkeypatch, caplog):
    made, _timers = relay_env
    _FakeTray.available = False

    def refuse(title, body):
        raise OSError("no notify-send")

    monkeypatch.setattr(run_journal, "_desktop_os_notify", refuse)
    relay = prefs._install_run_notifier(qapp)
    made.append(relay)
    with caplog.at_level(logging.INFO, logger=prefs.LOG.name):
        relay.arrived.emit("t", "b", True)
        qtbot.waitUntil(lambda: any("no desktop notification shown" in r.getMessage()
                                    for r in caplog.records), timeout=5000)
    assert _FakeTray.made == []


def _scroll(page=None):
    scroll = QScrollArea()
    if page is not None:
        scroll.setWidget(page)
    return scroll


def test_a_window_without_tabs_sends_no_page_away(qapp, qtbot):
    window = prefs._preferences_window_class()()
    qtbot.addWidget(window)
    assert window._send_the_unseen_pages_away() == 0
    assert window._pages_away == {}
    assert window._bring_the_page_back(3) is False


def test_empty_tabs_are_skipped_and_a_page_returns_without_a_stand_in(qapp, qtbot):
    window = prefs._preferences_window_class()()
    qtbot.addWidget(window)
    tabs = QTabWidget(window)
    current = QWidget()
    away = QWidget()
    QLabel("one", away)
    QLabel("two", away)
    tabs.addTab(_scroll(current), "current")
    tabs.addTab(_scroll(), "empty")
    tabs.addTab(_scroll(away), "away")
    tabs.setCurrentIndex(0)
    window._show_only_the_open_page_at_first(tabs)

    assert window._send_the_unseen_pages_away() == 3
    assert list(window._pages_away) == [2]
    assert tabs.widget(1).widget() is None
    stand_in = tabs.widget(2).widget()
    assert isinstance(stand_in, prefs._page_stand_in_class())
    assert window._send_the_unseen_pages_away() == 0

    tabs.widget(2).takeWidget().deleteLater()
    assert tabs.widget(2).widget() is None
    tabs.setCurrentIndex(2)
    assert tabs.widget(2).widget() is away
    assert window._pages_away == {}
    assert away._spacr_detached_from is None
    tabs.setCurrentIndex(1)
    assert tabs.widget(1).widget() is None


@pytest.fixture
def private_preferences(tmp_path, monkeypatch):
    from spacr import updater

    path = tmp_path / "preferences.ini"
    monkeypatch.setattr(prefs, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    monkeypatch.setenv("SPACR_NETWORK_CONFIG", str(tmp_path / "network.json"))
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(prefs, "_SAFE_MODE", False)
    prefs.set_language("en")
    prefs.set_theme_choice("dark")
    prefs.set_ambient_animation("none")
    yield
    updater._undo_network_exports()


def test_a_second_finish_of_the_apply_question_reverts_nothing(
        private_preferences, qtbot):
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    theme = next(combo for combo in dialog.findChildren(QComboBox)
                 if combo.findData("glass") >= 0)
    theme.setCurrentIndex(theme.findData("light"))
    dialog.findChild(QSlider, "PaneOpacity").setValue(63)
    box = dialog.findChild(QDialogButtonBox)
    apply_button = box.button(QDialogButtonBox.Apply)
    apply_button.click()
    qtbot.waitUntil(lambda: dialog._apply_confirmation is not None)
    question = dialog._apply_confirmation
    assert isinstance(question, QMessageBox)
    assert prefs.get_theme_choice() == "light"
    assert not apply_button.isEnabled()

    question.finished.emit(int(QMessageBox.Rejected))
    assert prefs.get_theme_choice() == "dark"
    assert dialog._apply_confirmation is None
    assert apply_button.isEnabled()

    prefs.set_theme_choice("light")
    question.finished.emit(int(QMessageBox.Rejected))
    assert prefs.get_theme_choice() == "light"
    assert dialog._apply_confirmation is None


def test_a_tray_that_refuses_a_message_is_logged_and_the_relay_survives(
        relay_env, qapp, monkeypatch, caplog):
    made, timers = relay_env
    calls = []

    def refuse(self, title, body, icon, msecs):
        calls.append(title)
        if title == "first":
            raise RuntimeError("tray gone")
        self.messages.append((title, body, icon, msecs))

    monkeypatch.setattr(_FakeTray, "showMessage", refuse)
    relay = prefs._install_run_notifier(qapp)
    made.append(relay)
    with caplog.at_level(logging.DEBUG, logger=prefs.LOG.name):
        relay.arrived.emit("first", "b", True)
    assert any("could not show the run notification" in r.getMessage()
               for r in caplog.records)
    assert timers == []
    relay.arrived.emit("second", "b", False)
    assert calls == ["first", "second"]
    assert _FakeTray.made[0].messages == [("second", "b", "information", 15000)]
