"""The Sound tab: last in Preferences, off, and driven the way a user drives it.

Filed 2026-09-18: "A Sound tab in Preferences, LAST tab, everything OFF BY
DEFAULT", "the tab is last because it is the least important thing in
Preferences". Decided 2026-09-19: master volume, and per-event switches for
click, hover (debounced), run finished, run failed and an ambient music bed.

HANDOFF 0b: a model that works is not a feature that works until a test
presses the control. So the last test here opens Preferences, switches
sound on with the mouse, presses Save, and then clicks a button in another
window and hears -- through a stand-in sink -- the click.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QPushButton, QSlider, QTabWidget, QWidget

from spacr.qt import sound as snd
from spacr.qt.sound_synth import CACHE_ENV


@pytest.fixture(autouse=True)
def _in_spaceout_mode(monkeypatch):
    """Sound exists only in spaceout mode since 2026-09-21 ("in normal
    spacr sound should be off by default and there should be no sound tab
    in preferences"), so everything here runs as the ``spaceout`` launcher
    would. Ordinary spaCR is covered by
    ``test_sound_lives_only_in_spaceout.py``."""
    from spacr.qt import theme

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)

EVENT_SWITCHES = ("SoundClick", "SoundHover", "SoundRunFinished",
                  "SoundRunFailed", "SoundMusicBed")


class _Effect:
    """A stand-in sound effect that records what it is asked to play."""

    def __init__(self, log):
        self.log = log
        self.name = ""
        self.volume = None
        self.playing = False

    def setSource(self, url):                  # noqa: N802 - Qt naming
        from pathlib import Path
        self.name = Path(url.toLocalFile()).stem

    def setVolume(self, value):                # noqa: N802 - Qt naming
        self.volume = value

    def setLoopCount(self, _count):            # noqa: N802 - Qt naming
        pass

    def play(self):
        self.playing = True
        self.log.append(self.name)

    def stop(self):
        self.playing = False


class _Heard(list):
    """The sounds played, in order, and every effect that played one.

    A list, so a test still reads `heard == ["click-0"]`; `effects` is for
    the two tests that ask whether something is STILL playing, which a log
    of what started cannot answer.
    """

    effects: list = []


@pytest.fixture
def heard(monkeypatch, tmp_path, qapp):
    """Every sound played, by name. The engine the app would build is
    replaced by one that runs inline and plays into this list."""
    from spacr.qt import preferences as prefs

    path = str(tmp_path / "prefs.ini")
    monkeypatch.setattr(prefs, "_settings",
                        lambda: QSettings(path, QSettings.IniFormat))
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "sounds"))
    log = _Heard()
    log.effects = []

    def make(_parent):
        effect = _Effect(log)
        log.effects.append(effect)
        return effect

    def create(app):
        engine = snd.SoundEngine(None, threaded=False,
                                 effect_factory=make,
                                 cache_root=tmp_path / "sounds")
        snd._ENGINE = engine
        return engine

    monkeypatch.setattr(snd, "_create_engine", create)
    yield log
    snd.shutdown_sound()


def _open_preferences(qtbot):
    """A second Preferences dialog, opened after sound is already on."""
    from spacr.qt.preferences import PreferencesDialog

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    dlg.resize(900, 700)
    dlg.show()
    qtbot.waitExposed(dlg)
    _show_sound_tab(dlg)
    return dlg


@pytest.fixture
def dialog(heard, qtbot, qt_theme_applied):
    from spacr.qt.preferences import PreferencesDialog

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    dlg.resize(900, 700)
    dlg.show()
    qtbot.waitExposed(dlg)
    return dlg


def _show_sound_tab(dialog):
    tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
    tabs.setCurrentIndex(tabs.count() - 1)
    return tabs


def _press(qtbot, widget):
    qtbot.mouseClick(widget, Qt.MouseButton.LeftButton,
                     pos=QPoint(widget.width() // 4, widget.height() // 2))


def test_sound_is_the_last_tab(dialog):
    tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
    assert tabs.tabText(tabs.count() - 1) == "Sound"
    page = tabs.widget(tabs.count() - 1)
    assert page.findChild(QWidget, "PreferencesTabSound") is not None


def test_sound_stays_last_when_the_fractal_tab_is_offered(heard, qtbot,
                                                          monkeypatch):
    """The launcher-only Fractal tab is built after AI; Sound after that."""
    from spacr.qt import theme
    from spacr.qt.preferences import PreferencesDialog

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    tabs = dlg.findChild(QTabWidget, "PreferencesTabs")
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert "Fractal" in titles and titles[-1] == "Sound", titles


def test_everything_is_off_and_greyed_until_sound_is_switched_on(dialog):
    master = dialog.findChild(QWidget, "SoundEnabled")
    assert master.isChecked() is False
    for name in EVENT_SWITCHES:
        assert not dialog.findChild(QWidget, name).isEnabled(), name
        assert not dialog.findChild(QPushButton, f"{name}Preview").isEnabled()
    assert not dialog.findChild(QSlider, "SoundVolume").isEnabled()
    assert not dialog.findChild(QComboBox, "SoundTheme").isEnabled()


def test_switching_sound_on_offers_the_rest(dialog, qtbot):
    _show_sound_tab(dialog)
    master = dialog.findChild(QWidget, "SoundEnabled")
    _press(qtbot, master)
    assert master.isChecked()
    for name in EVENT_SWITCHES:
        assert dialog.findChild(QWidget, name).isEnabled(), name
    hover = dialog.findChild(QWidget, "SoundHover")
    bed = dialog.findChild(QWidget, "SoundMusicBed")
    assert not hover.isChecked() and not bed.isChecked(), (
        "hover and music must each be asked for by name")
    assert dialog.findChild(QWidget, "SoundClick").isChecked()


def test_the_sound_set_offers_the_reference_set(dialog):
    combo = dialog.findChild(QComboBox, "SoundTheme")
    keys = [combo.itemData(i) for i in range(combo.count())]
    assert "orbit" in keys
    assert combo.currentData() == "orbit"


def test_save_writes_every_sound_setting(dialog, qtbot):
    from spacr.qt import preferences as prefs

    _show_sound_tab(dialog)
    _press(qtbot, dialog.findChild(QWidget, "SoundEnabled"))
    _press(qtbot, dialog.findChild(QWidget, "SoundHover"))
    dialog.findChild(QSlider, "SoundVolume").setValue(30)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()

    assert prefs.get_sound_enabled() is True
    assert prefs.get_sound_volume() == pytest.approx(0.30)
    assert prefs.get_sound_event_enabled("hover") is True
    assert prefs.get_sound_event_enabled("bed") is False


def test_cancel_writes_nothing(dialog, qtbot):
    from spacr.qt import preferences as prefs

    _show_sound_tab(dialog)
    _press(qtbot, dialog.findChild(QWidget, "SoundEnabled"))
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Cancel).click()
    assert prefs.get_sound_enabled() is False
    assert snd.sound_engine() is None


def test_reset_puts_sound_back_off(dialog, qtbot):
    from spacr.qt import preferences as prefs

    prefs.set_sound_enabled(True)
    prefs.set_sound_event_enabled("hover", True)
    prefs.set_sound_volume(0.9)
    from spacr.qt.preferences import PreferencesDialog

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    assert dlg.findChild(QWidget, "SoundEnabled").isChecked()
    dlg.findChild(QPushButton, "PreferencesReset").click()
    assert not dlg.findChild(QWidget, "SoundEnabled").isChecked()
    assert not dlg.findChild(QWidget, "SoundHover").isChecked()
    assert dlg.findChild(QSlider, "SoundVolume").value() == 50
    assert not dlg.findChild(QWidget, "SoundClick").isEnabled()
    assert prefs.get_sound_enabled() is True, "Reset wrote before Save"


def test_a_preview_press_plays_that_sound(dialog, heard, qtbot):
    _show_sound_tab(dialog)
    _press(qtbot, dialog.findChild(QWidget, "SoundEnabled"))
    dialog.findChild(QSlider, "SoundVolume").setValue(100)
    button = dialog.findChild(QPushButton, "SoundRunFinishedPreview")
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert heard == ["run_finished"]


def test_preview_is_dead_while_sound_is_off(dialog, heard, qtbot):
    button = dialog.findChild(QPushButton, "SoundClickPreview")
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert heard == []
    assert snd.sound_engine() is None


def test_a_preview_press_is_not_also_a_click(heard, qtbot, qt_theme_applied):
    """A user who already has sound on opens Preferences to try a Preview.

    The app-wide filter hears the press, so without the Preview button's
    `spacrSilentPress` the Click row answers a press with its own pluck AND
    the previewed one: the same sound twice. The switch beside it still
    clicks, which is what says the filter is alive and the property is
    doing the work.
    """
    from spacr.qt import preferences as prefs
    from spacr.qt.preferences import apply_preferences_to_app

    prefs.set_sound_enabled(True)
    apply_preferences_to_app()
    engine = snd.sound_engine()
    assert engine is not None and engine.filter_installed

    dlg = _open_preferences(qtbot)
    heard.clear()
    qtbot.mouseClick(dlg.findChild(QPushButton, "SoundClickPreview"),
                     Qt.MouseButton.LeftButton)
    assert heard == ["click-0"]
    heard.clear()
    _press(qtbot, dlg.findChild(QWidget, "SoundHover"))
    assert heard == ["click-1"], "an ordinary control still clicks"


def test_switching_the_master_off_ends_a_preview(heard, qtbot,
                                                 qt_theme_applied):
    """Greying the Preview buttons does nothing about the nine seconds of
    music bed already playing, which is the one sound on this page long
    enough to outlive the switch that started it."""
    from spacr.qt import preferences as prefs
    from spacr.qt.preferences import apply_preferences_to_app

    prefs.set_sound_enabled(True)
    apply_preferences_to_app()

    dlg = _open_preferences(qtbot)
    master = dlg.findChild(QWidget, "SoundEnabled")
    assert master.isChecked()
    qtbot.mouseClick(dlg.findChild(QPushButton, "SoundMusicBedPreview"),
                     Qt.MouseButton.LeftButton)
    bed = [e for e in heard.effects if e.name == "bed"][0]
    assert bed.playing

    _press(qtbot, master)
    assert not master.isChecked()
    assert not bed.playing


def test_a_saved_switch_is_heard_in_another_window(dialog, heard, qtbot):
    """The whole path, the way a user takes it."""
    _show_sound_tab(dialog)
    _press(qtbot, dialog.findChild(QWidget, "SoundEnabled"))
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    engine = snd.sound_engine()
    assert engine is not None and engine.filter_installed

    other = QWidget()
    button = QPushButton("Run", other)
    qtbot.addWidget(other)
    other.show()
    qtbot.waitExposed(other)
    heard.clear()
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert heard == ["click-0"]


def test_switching_sound_off_again_silences_and_unhooks(dialog, heard, qtbot):
    from spacr.qt import preferences as prefs
    from spacr.qt.preferences import PreferencesDialog, apply_preferences_to_app

    prefs.set_sound_enabled(True)
    apply_preferences_to_app()
    engine = snd.sound_engine()
    assert engine is not None and engine.filter_installed

    dlg = PreferencesDialog()
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    _show_sound_tab(dlg)
    _press(qtbot, dlg.findChild(QWidget, "SoundEnabled"))
    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert not engine.filter_installed
    assert engine.play("click") is False


def test_the_music_file_row_is_empty_and_greyed_until_sound_is_on(dialog):
    """Part B's one new row, and it follows the master like the rest."""
    from PySide6.QtWidgets import QLineEdit

    _show_sound_tab(dialog)
    field = dialog.findChild(QLineEdit, "SoundMusicFile")
    browse = dialog.findChild(QPushButton, "SoundMusicFileBrowse")
    assert field is not None and browse is not None
    assert field.text() == "", "a fresh install plays spaCR's own music"
    assert field.placeholderText()
    assert not field.isEnabled() and not browse.isEnabled()


def test_a_music_file_typed_into_the_row_is_what_the_bed_plays(dialog, heard,
                                                               qtbot,
                                                               tmp_path):
    """HANDOFF 0b: driven through the dialog, not through the setter.

    A path typed in, Save pressed with the mouse, and then the question
    that matters -- is that file what the effect was pointed at, and is it
    what the Resonance backdrop is being driven by.
    """
    import math
    from pathlib import Path

    import numpy as np
    from PySide6.QtWidgets import QLineEdit

    from spacr.qt import resonance as rs
    from spacr.qt import sound_synth as ss

    chosen = tmp_path / "mine.wav"
    samples = np.sin(2.0 * math.pi * 330.0
                     * np.arange(int(1.2 * ss.SAMPLE_RATE)) / ss.SAMPLE_RATE)
    ss.write_wav(chosen, np.vstack([samples, samples]) * 0.5)

    _show_sound_tab(dialog)
    _press(qtbot, dialog.findChild(QWidget, "SoundEnabled"))
    _press(qtbot, dialog.findChild(QWidget, "SoundMusicBed"))
    field = dialog.findChild(QLineEdit, "SoundMusicFile")
    field.setFocus()
    qtbot.keyClicks(field, str(chosen))
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()

    from spacr.qt import preferences as prefs
    assert prefs.get_sound_music_file() == str(chosen)
    bed = [e for e in heard.effects if e.name == "mine"]
    assert bed and bed[-1].playing, "the chosen file is not what plays"

    record = rs.now_playing()
    assert record is not None
    assert record.duration == pytest.approx(1.2, abs=0.01)
    assert Path(record.analysis).exists()
    rs.clear_now_playing()
