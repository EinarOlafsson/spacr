"""Apply, Keep and Revert in Preferences do not stall the GUI thread.

The tutorial capture gate (30 fps, real app) found app-freeze seconds where
Apply was pressed and again where the preview was reverted and applied
again. Offscreen with the real MainWindow on a loaded machine (load 70-78
on 32 cores), Apply of a page-opacity change held the GUI thread 6.8-8.9 s
(5.4-6.9 s of it inside the click): the new sheet put on every window at
once (2.2-4.7 s, some fifty hidden combo-box popups and parked windows
among them), then ``refresh_theme`` -- Home rebuilt, sheeted again, the
window retranslated -- because the opacity is in the window's signature
(2.6-3.3 s); and every Apply set the application font again, which sends a
font change to every widget. Revert held it 5.9-7.2 s.

What replaced it, and what is tested here:

* the sheet goes on one root per pass of the event loop, after the next
  paint, the windows on show first and the hidden ones last;
* the window redraws only what changed: a new opacity is three rules on
  Home set again in place; a new theme, scale or language still rebuilds;
* Apply, Keep and Revert mean what they meant.

Timing is measured against the same steps run the old way in the same test,
on the same dialog and window, so a loaded runner slows both sides alike.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QComboBox, QDialogButtonBox, QSlider, QWidget,
)

from spacr.qt import preferences, theme  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private preference store."""
    path = tmp_path / "preferences.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    monkeypatch.setenv("SPACR_NETWORK_CONFIG", str(tmp_path / "network.json"))
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("SPACR_BENCHMARK_JSON", raising=False)
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    preferences.set_language("en")
    preferences.set_theme_choice("dark")
    preferences.set_ambient_animation("none")
    preferences.set_pane_opacity(0.8)
    preferences._settings().sync()
    yield path
    theme.finish_the_sliced_sheeting()
    from spacr import updater

    updater._undo_network_exports()


class _Heartbeat:
    """The longest the event loop went without running a 1 ms timer."""

    def __init__(self):
        self.beats = []
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(
            lambda: self.beats.append(time.perf_counter()))

    def __enter__(self):
        self.beats = [time.perf_counter()]
        self.timer.start(1)
        return self

    def __exit__(self, *_exc):
        self.timer.stop()
        self.beats.append(time.perf_counter())

    def longest(self) -> float:
        return max((b - a for a, b in zip(self.beats, self.beats[1:])),
                   default=0.0)


def _settled(qtbot):
    """Wait, beating, until no slice and no redraw is still owed."""
    deadline = time.perf_counter() + 60
    while time.perf_counter() < deadline:
        qtbot.wait(60)
        if theme.sliced_sheeting_owed() == 0:
            qtbot.wait(120)
            if theme.sliced_sheeting_owed() == 0:
                return
    raise AssertionError("the sliced sheet never finished")


def _button(dialog, which):
    return dialog.findChild(QDialogButtonBox).button(which)


def _answer(dialog, qtbot, text):
    qtbot.waitUntil(lambda: dialog._apply_confirmation is not None)
    question = dialog._apply_confirmation
    next(b for b in question.buttons() if b.text() == text).click()
    qtbot.waitUntil(lambda: dialog._apply_confirmation is None)


@pytest.fixture
def window(qtbot, qt_theme_applied, store):
    """A shown MainWindow, sheeted per window the way launch sheets it."""
    from spacr.qt.app import MainWindow

    preferences.apply_preferences_to_app()
    from spacr.qt.app import _use_open_sans

    _use_open_sans(QApplication.instance())
    main = MainWindow()
    qtbot.addWidget(main)
    main.resize(1200, 800)
    main.show()
    qtbot.waitExposed(main)
    qtbot.wait(50)
    yield main
    theme.finish_the_sliced_sheeting()


@pytest.fixture
def classic_window(store, request):
    """Select ordinary Home before its own Apply/Revert surface is built."""
    preferences.set_easy_mode(False)
    return request.getfixturevalue("window")


def _open_on_appearance(window, qtbot):
    dialog = preferences.PreferencesDialog(window)
    dialog.show()
    qtbot.waitExposed(dialog)
    slider = dialog.findChild(QSlider, "PaneOpacity")
    assert slider is not None
    return dialog, slider


def _stall(qtbot, act):
    """The longest the event loop stalled from ``act`` until all was drawn."""
    with _Heartbeat() as beat:
        act()
        _settled(qtbot)
    return beat.longest()


def test_apply_keep_and_revert_hold_the_loop_a_fraction_of_the_old_apply(
        window, qtbot, monkeypatch):
    """Each step's longest stall is well under what the same step used to cost.

    The reference is the same Apply and Revert on the same dialog and
    window, run the old way -- the whole sheet on every window at once, then
    ``refresh_theme`` -- by taking ``in_slices`` away from both.
    """
    real_apply = preferences.apply_preferences_to_app
    real_redraw = preferences._refresh_owner_window
    rebuilt = []
    real_refresh = window.refresh_theme
    monkeypatch.setattr(window, "refresh_theme",
                        lambda: rebuilt.append(1) or real_refresh())
    home = window._startup
    dialog, slider = _open_on_appearance(window, qtbot)
    apply = _button(dialog, QDialogButtonBox.Apply)
    old, new = {}, {}
    try:
        preferences.apply_preferences_to_app = (
            lambda app=None, **_kw: real_apply(app))
        preferences._refresh_owner_window = (
            lambda parent, dialog=None, **_kw: real_redraw(parent, dialog))
        try:
            slider.setValue(55)
            old["apply"] = _stall(qtbot, apply.click)
            old["revert"] = _stall(
                qtbot, lambda: _answer(dialog, qtbot, "Revert"))
        finally:
            preferences.apply_preferences_to_app = real_apply
            preferences._refresh_owner_window = real_redraw
        assert rebuilt, "the reference did not take the old path"
        rebuilt.clear()
        home = window._startup

        new["apply"] = _stall(qtbot, apply.click)
        assert abs(preferences.get_pane_opacity() - 0.55) < 1e-9
        new["revert"] = _stall(qtbot, lambda: _answer(dialog, qtbot, "Revert"))
        assert abs(preferences.get_pane_opacity() - 0.8) < 1e-9
        assert dialog.isVisible(), "Revert closed Preferences"
        new["apply again"] = _stall(qtbot, apply.click)
        new["keep"] = _stall(qtbot, lambda: _answer(dialog, qtbot, "Keep"))
        assert abs(preferences.get_pane_opacity() - 0.55) < 1e-9
        assert dialog.isVisible(), "Keep closed Preferences"
    finally:
        dialog.reject()
        dialog.deleteLater()

    assert rebuilt == [], "an opacity Apply rebuilt the window"
    assert window._startup is home, "an opacity Apply rebuilt Home"
    shown = {k: round(v * 1000) for k, v in new.items()}
    was = {k: round(v * 1000) for k, v in old.items()}
    bounds = {"apply": (0.5, "apply"), "apply again": (0.5, "apply"),
              "revert": (0.5, "revert"), "keep": (0.15, "apply")}
    for step, (fraction, against) in bounds.items():
        assert new[step] < fraction * old[against], (
            f"{step} held the event loop {shown[step]} ms against "
            f"{was[against]} ms for the old {against}; now {shown}, "
            f"before {was}")


def test_an_opacity_apply_redraws_home_in_place_and_revert_puts_it_back(
        classic_window, qtbot):
    """Home's own surfaces follow Apply and Revert without a rebuild."""
    window = classic_window
    from spacr.qt.widgets.home import _selected_tab_qss, hero_qss

    home = window._startup
    hero = home.findChild(QWidget, "Hero")
    before = hero.styleSheet()
    dialog, slider = _open_on_appearance(window, qtbot)
    try:
        slider.setValue(40)
        _button(dialog, QDialogButtonBox.Apply).click()
        qtbot.waitUntil(lambda: hero.styleSheet() == hero_qss())
        assert hero.styleSheet() != before
        bar = home._tabs.tabBar()
        qtbot.waitUntil(lambda: bar.styleSheet() == _selected_tab_qss(
            home._P, home._pane_alpha()))
        _answer(dialog, qtbot, "Revert")
        qtbot.waitUntil(lambda: hero.styleSheet() == before)
        assert window._startup is home
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_a_theme_apply_still_rebuilds_the_window(window, qtbot, monkeypatch):
    """Only what changed is redrawn; a theme is drawn into Home's build."""
    rebuilt = []
    monkeypatch.setattr(window, "refresh_theme", lambda: rebuilt.append(1))
    dialog, _slider = _open_on_appearance(window, qtbot)
    try:
        combo = next(c for c in dialog.findChildren(QComboBox)
                     if c.findData("glass") >= 0)
        combo.setCurrentIndex(combo.findData("light"))
        _button(dialog, QDialogButtonBox.Apply).click()
        qtbot.waitUntil(lambda: rebuilt == [1])
        _answer(dialog, qtbot, "Keep")
        assert preferences.get_theme_choice() == "light"
    finally:
        dialog.reject()
        dialog.deleteLater()


def test_the_window_is_told_which_places_changed():
    names = preferences._WINDOW_SIGNATURE_NAMES
    now = tuple(range(len(names)))
    moved = list(now)
    moved[names.index("pane_opacity")] = -1
    assert preferences._window_changes(now, tuple(moved)) == {"pane_opacity"}
    assert preferences._window_changes(now, now) == frozenset()
    assert preferences._window_changes(None, now) == frozenset(names)


def test_the_window_signature_and_its_names_agree(qapp, store):
    assert len(preferences._window_signature()) == len(
        preferences._WINDOW_SIGNATURE_NAMES)


def test_a_sliced_sheet_goes_on_shown_windows_first_and_reaches_all(
        qtbot, qt_theme_applied, store):
    """Nothing is sheeted in the call; every root is sheeted in the end."""
    app = QApplication.instance()
    preferences.apply_preferences_to_app()
    shown = QWidget()
    hidden = QWidget()
    qtbot.addWidget(shown)
    qtbot.addWidget(hidden)
    shown.show()
    hidden.ensurePolished()
    qtbot.waitExposed(shown)
    serial = getattr(app, theme._WINDOW_SHEET_SERIAL, 0)
    order = []
    real = theme._sheet_one_window

    def _counting(root):
        done = real(root)
        if done and root in (shown, hidden):
            order.append(root)
        return done

    theme._sheet_one_window = _counting
    try:
        owed = theme.apply_stylesheet_per_window(
            app, theme.window_stylesheet(app) + "\n", in_slices=True)
        assert owed >= 1
        assert shown.property(theme._WINDOW_SHEET_SERIAL) != serial + 1
        qtbot.waitUntil(lambda: theme.sliced_sheeting_owed() == 0,
                        timeout=30000)
    finally:
        theme._sheet_one_window = real
    assert shown.property(theme._WINDOW_SHEET_SERIAL) == serial + 1
    assert hidden.property(theme._WINDOW_SHEET_SERIAL) == serial + 1
    assert order and order[0] is shown, "a hidden window went first"
