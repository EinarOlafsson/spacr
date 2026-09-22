"""spaCR's sounds: silent until asked, never on the GUI thread, never a crash.

Decided 2026-09-19: "Ok build as recomended" -- off by default; click, hover
and run-finished sounds synthesized in code and played with Qt's
QSoundEffect, falling back to silence. Filed with the rules this file holds
the engine to: EVERYTHING OFF BY DEFAULT; hover debounced hard, only on
interactive controls, never on a disabled one, with its own switch; audio
must not touch the run, and a missing audio device must degrade to silence
without a warning dialog and without an exception.

The audio sink is faked throughout: a stand-in effect records what it was
asked to do, and on which thread. Nothing here can make a sound, and every
rendered file goes under ``tmp_path``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, QSettings, Qt, QThread
from PySide6.QtGui import QEnterEvent
from PySide6.QtWidgets import QApplication, QPushButton, QSlider, QVBoxLayout, QWidget

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

ROOT = Path(__file__).resolve().parents[2]


class FakeEffect:
    """Stands in for ``QSoundEffect``; records, never plays."""

    def __init__(self, log):
        self.log = log
        self.source = ""
        self.volume = None
        self.loops = 1
        self.playing = False
        self.deleted = False
        self.made_on = QThread.currentThread()
        self.played_on = None

    @property
    def name(self) -> str:
        return Path(self.source).stem

    def setSource(self, url):                  # noqa: N802 - Qt naming
        self.source = url.toLocalFile()

    def setVolume(self, value):                # noqa: N802 - Qt naming
        self.volume = float(value)

    def setLoopCount(self, count):             # noqa: N802 - Qt naming
        self.loops = int(count)

    def play(self):
        self.playing = True
        self.played_on = QThread.currentThread()
        self.log.append(("play", self.name, self.volume, self.loops))

    def stop(self):
        if self.playing:
            self.log.append(("stop", self.name))
        self.playing = False

    def deleteLater(self):                     # noqa: N802 - Qt naming
        self.deleted = True


@pytest.fixture
def store(monkeypatch, tmp_path):
    """A private preference store, and a private sound cache."""
    from spacr.qt import preferences as prefs

    path = str(tmp_path / "prefs.ini")
    monkeypatch.setattr(prefs, "_settings",
                        lambda: QSettings(path, QSettings.IniFormat))
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "sounds"))
    return prefs


@pytest.fixture
def sink():
    """Every effect made, and a log of every play and stop, in order."""
    log = []
    effects = []

    def factory(_parent):
        effect = FakeEffect(log)
        effects.append(effect)
        return effect

    factory.log = log
    factory.effects = effects
    return factory


@pytest.fixture(autouse=True)
def _no_engine_outlives_the_test():
    yield
    snd.shutdown_sound()


@pytest.fixture
def make_engine(qapp, sink, tmp_path):
    made = []

    def build(threaded=False, factory=None, **fields):
        engine = snd.SoundEngine(None, threaded=threaded,
                                 effect_factory=factory or sink,
                                 cache_root=tmp_path / "sounds")
        made.append(engine)
        if fields:
            engine.apply(snd.SoundSettings(**fields))
        return engine

    yield build
    for engine in made:
        engine.shutdown()


def _same(a, b) -> bool:
    """Whether two wrappers are the same C++ object."""
    import shiboken6

    return shiboken6.getCppPointer(a)[0] == shiboken6.getCppPointer(b)[0]


def _plays(sink):
    return [row[1] for row in sink.log if row[0] == "play"]


@pytest.fixture
def window(qtbot):
    host = QWidget()
    layout = QVBoxLayout(host)
    buttons = [QPushButton(f"b{i}") for i in range(10)]
    for button in buttons:
        layout.addWidget(button)
    slider = QSlider(Qt.Orientation.Horizontal)
    layout.addWidget(slider)
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    host.buttons = buttons
    host.slider = slider
    return host


def _enter(widget):
    QApplication.sendEvent(widget, QEnterEvent(QPointF(3, 3), QPointF(3, 3),
                                               QPointF(3, 3)))


def _leave(widget):
    QApplication.sendEvent(widget, QEvent(QEvent.Type.Leave))


class TestNothingExistsWhileOff:

    def test_a_fresh_install_is_silent(self, store):
        assert store.get_sound_enabled() is False
        settings = snd.read_sound_settings()
        assert settings.enabled is False
        assert not any(settings.wants(e) for e in
                       ("click", "hover", "run_finished", "run_failed", "bed"))

    def test_hover_and_the_bed_stay_off_even_when_sound_is_switched_on(
            self, store):
        store.set_sound_enabled(True)
        settings = snd.read_sound_settings()
        assert settings.wants("click") and settings.wants("run_finished")
        assert not settings.wants("hover"), "hover must be asked for by name"
        assert not settings.wants("bed"), "music must be asked for by name"

    def test_applying_the_preferences_builds_nothing(self, store, qapp):
        from spacr.qt.preferences import apply_preferences_to_app

        apply_preferences_to_app(qapp)
        assert snd.apply_sound_preferences(qapp) is None
        assert snd.sound_engine() is None
        assert not [c for c in qapp.children()
                    if isinstance(c, snd.InputSoundFilter)]

    def test_off_does_not_even_import_qt_multimedia(self, tmp_path):
        """Measured in a fresh interpreter, because this one has usually
        imported it already for some other test. Launch applies the
        preferences before the window exists, and importing this module
        cold was measured at 26 ms, so with sound off it is not imported."""
        script = (
            "import sys, threading\n"
            "import spacr\n"
            f"assert spacr.__file__.startswith({str(ROOT)!r}), spacr.__file__\n"
            "from PySide6.QtWidgets import QApplication\n"
            "app = QApplication([])\n"
            "before = threading.active_count()\n"
            "from spacr.qt.preferences import apply_preferences_to_app\n"
            "apply_preferences_to_app(app)\n"
            "assert 'spacr.qt.sound' not in sys.modules, 'imported while off'\n"
            "from spacr.qt.sound import apply_sound_preferences, sound_engine\n"
            "assert apply_sound_preferences(app) is None\n"
            "assert sound_engine() is None\n"
            "assert 'PySide6.QtMultimedia' not in sys.modules\n"
            "assert threading.active_count() == before\n"
            "print('SILENT')\n")
        env = dict(os.environ, HOME=str(tmp_path), QT_QPA_PLATFORM="offscreen",
                   XDG_CONFIG_HOME=str(tmp_path / "config"),
                   XDG_DATA_HOME=str(tmp_path / "data"),
                   SPACR_SOUND_CACHE=str(tmp_path / "sounds"))
        done = subprocess.run([sys.executable, "-c", script], cwd=str(ROOT),
                              env=env, capture_output=True, text=True,
                              timeout=120)
        assert "SILENT" in done.stdout, done.stdout + done.stderr

    def test_off_reads_the_master_switch_and_nothing_else(self, store, qapp,
                                                          monkeypatch):
        """`apply_preferences_to_app` runs at launch, after every Save and
        after every Z-and-wheel zoom; for the user who never asked for
        sound it must cost one read."""
        def refuse():
            raise AssertionError("read every sound setting while off")

        monkeypatch.setattr(snd, "read_sound_settings", refuse)
        assert snd.apply_sound_preferences(qapp) is None

    def test_the_run_sound_hook_is_silent_without_an_engine(self):
        assert snd.announce_run_end("success") is False

    def test_a_finishing_run_does_not_import_the_engine_to_stay_silent(
            self, tmp_path):
        """`announce_pipeline_finished` runs at the end of every run, and
        the module it reaches for costs 20.4 ms to import on the GUI
        thread. For a user with sound off there is nothing at the other end
        of it: the engine is built by `apply_sound_preferences`, so an
        unimported module IS the answer. A fresh interpreter, because this
        one imported `spacr.qt.sound` at the top of this file."""
        script = (
            "import sys\n"
            "import spacr\n"
            f"assert spacr.__file__.startswith({str(ROOT)!r}), spacr.__file__\n"
            "from PySide6.QtWidgets import QApplication\n"
            "app = QApplication([])\n"
            "from spacr.qt import notify\n"
            "notify.notify = lambda *a, **k: True\n"
            "notify.notify_tray = lambda *a, **k: True\n"
            "notify.announce_pipeline_finished('mask', 'success', 3.0)\n"
            "notify.announce_pipeline_finished('mask', 'failed', 3.0)\n"
            "assert 'spacr.qt.sound' not in sys.modules, 'imported while off'\n"
            "assert 'PySide6.QtMultimedia' not in sys.modules\n"
            "print('SILENT')\n")
        env = dict(os.environ, HOME=str(tmp_path), QT_QPA_PLATFORM="offscreen",
                   XDG_CONFIG_HOME=str(tmp_path / "config"),
                   XDG_DATA_HOME=str(tmp_path / "data"),
                   SPACR_SOUND_CACHE=str(tmp_path / "sounds"))
        done = subprocess.run([sys.executable, "-c", script], cwd=str(ROOT),
                              env=env, capture_output=True, text=True,
                              timeout=120)
        assert "SILENT" in done.stdout, done.stdout + done.stderr


class TestClicks:

    def test_pressing_a_button_plays_an_in_key_click(self, make_engine,
                                                     sink, window, qtbot):
        engine = make_engine(enabled=True)
        assert engine.filter_installed
        qtbot.mouseClick(window.buttons[0], Qt.MouseButton.LeftButton)
        assert _plays(sink) == ["click-0"]
        time.sleep(snd.CLICK_REPEAT_S * 2)
        qtbot.mouseClick(window.buttons[1], Qt.MouseButton.LeftButton)
        assert _plays(sink) == ["click-0", "click-1"], "the clicks rotate"

    def test_the_click_plays_at_the_volume_asked_for(self, make_engine,
                                                     sink, window, qtbot):
        make_engine(enabled=True, volume=0.5)
        qtbot.mouseClick(window.buttons[0], Qt.MouseButton.LeftButton)
        assert sink.log[-1][2] == pytest.approx(0.25)

    def test_a_disabled_control_is_silent(self, make_engine, sink, window,
                                          qtbot):
        make_engine(enabled=True)
        window.buttons[0].setEnabled(False)
        qtbot.mouseClick(window.buttons[0], Qt.MouseButton.LeftButton)
        assert _plays(sink) == []

    def test_a_right_click_is_not_a_click(self, make_engine, sink, window,
                                          qtbot):
        make_engine(enabled=True)
        qtbot.mouseClick(window.buttons[0], Qt.MouseButton.RightButton)
        assert _plays(sink) == []

    def test_a_slider_press_clicks(self, make_engine, sink, window, qtbot):
        make_engine(enabled=True)
        qtbot.mouseClick(window.slider, Qt.MouseButton.LeftButton)
        assert _plays(sink) == ["click-0"]

    def test_a_control_marked_silent_is_pressed_without_a_click(
            self, make_engine, sink, window, qtbot):
        """`SILENT_PRESS_PROPERTY` is how the Sound tab's Preview buttons
        stay out of the way: pressing Preview asks for ONE sound, and the
        Click row's own Preview would otherwise play the same pluck twice.
        The button beside it still clicks, so this is the property working
        and not the filter falling over."""
        make_engine(enabled=True)
        window.buttons[0].setProperty(snd.SILENT_PRESS_PROPERTY, True)
        qtbot.mouseClick(window.buttons[0], Qt.MouseButton.LeftButton)
        assert _plays(sink) == []
        qtbot.mouseClick(window.buttons[1], Qt.MouseButton.LeftButton)
        assert _plays(sink) == ["click-0"]

    def test_click_switched_off_is_silent(self, make_engine, sink, window,
                                          qtbot):
        engine = make_engine(enabled=True, click=False)
        assert not engine.filter_installed, (
            "nothing needs the filter, so it must not be taxing every event")
        qtbot.mouseClick(window.buttons[0], Qt.MouseButton.LeftButton)
        assert _plays(sink) == []

    def test_master_off_is_silent_whatever_the_events_say(
            self, make_engine, sink, window, qtbot):
        engine = make_engine(enabled=True)
        engine.apply(snd.SoundSettings(enabled=False, click=True, hover=True,
                                       bed=True))
        assert not engine.filter_installed
        qtbot.mouseClick(window.buttons[0], Qt.MouseButton.LeftButton)
        assert engine.play("run_finished") is False
        assert _plays(sink) == []


class TestHoverIsDebouncedHard:

    def test_resting_on_a_control_plays_one_quiet_hover(
            self, make_engine, sink, window, qtbot):
        make_engine(enabled=True, hover=True)
        _enter(window.buttons[0])
        assert _plays(sink) == [], "a hover must wait for the pointer to rest"
        qtbot.wait(snd.HOVER_SETTLE_MS + 120)
        assert _plays(sink) == ["hover-0"]

    def test_sweeping_across_a_panel_is_not_a_stream(
            self, make_engine, sink, window, qtbot):
        make_engine(enabled=True, hover=True)
        for button in window.buttons:
            _enter(button)
            qtbot.wait(20)
            _leave(button)
        qtbot.wait(snd.HOVER_SETTLE_MS + 120)
        assert _plays(sink) == []

    def test_two_rests_in_quick_succession_sound_once(
            self, make_engine, sink, window, qtbot):
        make_engine(enabled=True, hover=True)
        for button in window.buttons[:2]:
            _enter(button)
            qtbot.wait(snd.HOVER_SETTLE_MS + 60)
            _leave(button)
        assert len(_plays(sink)) == 1, "the cooldown did not hold"
        qtbot.wait(int(snd.HOVER_COOLDOWN_S * 1000) + 50)
        _enter(window.buttons[3])
        qtbot.wait(snd.HOVER_SETTLE_MS + 60)
        assert len(_plays(sink)) == 2

    def test_no_hover_right_after_a_click(self, make_engine, sink, window,
                                          qtbot):
        make_engine(enabled=True, hover=True, click=False)
        qtbot.mouseClick(window.buttons[0], Qt.MouseButton.LeftButton)
        _enter(window.buttons[1])
        qtbot.wait(snd.HOVER_SETTLE_MS + 60)
        assert _plays(sink) == []

    def test_a_disabled_control_never_hovers(self, make_engine, sink, window,
                                             qtbot):
        make_engine(enabled=True, hover=True)
        window.buttons[0].setEnabled(False)
        _enter(window.buttons[0])
        qtbot.wait(snd.HOVER_SETTLE_MS + 60)
        assert _plays(sink) == []

    def test_a_control_disabled_while_the_pointer_settles_stays_silent(
            self, make_engine, sink, window, qtbot):
        make_engine(enabled=True, hover=True)
        _enter(window.buttons[0])
        window.buttons[0].setEnabled(False)
        qtbot.wait(snd.HOVER_SETTLE_MS + 60)
        assert _plays(sink) == []

    def test_hover_has_its_own_switch(self, make_engine, sink, window, qtbot):
        make_engine(enabled=True, hover=False)
        _enter(window.buttons[0])
        qtbot.wait(snd.HOVER_SETTLE_MS + 60)
        assert _plays(sink) == []

    def test_a_slider_does_not_hover(self, make_engine, sink, window, qtbot):
        make_engine(enabled=True, hover=True)
        _enter(window.slider)
        qtbot.wait(snd.HOVER_SETTLE_MS + 60)
        assert _plays(sink) == []


class TestRunSounds:

    @pytest.fixture(autouse=True)
    def _no_desktop_notification(self, monkeypatch):
        from spacr.qt import notify

        monkeypatch.setattr(notify, "notify", lambda *a, **k: True)
        monkeypatch.setattr(notify, "notify_tray", lambda *a, **k: True)

    @pytest.fixture
    def installed(self, make_engine, monkeypatch):
        engine = make_engine(enabled=True)
        monkeypatch.setattr(snd, "_ENGINE", engine)
        return engine

    def test_a_finished_run_plays_the_rising_figure(self, installed, sink):
        from spacr.qt.notify import announce_pipeline_finished

        announce_pipeline_finished("mask", "success", 12.0)
        assert _plays(sink) == ["run_finished"]

    def test_a_failed_run_plays_the_falling_figure(self, installed, sink):
        from spacr.qt.notify import announce_pipeline_finished

        announce_pipeline_finished("mask", "failed", 12.0)
        assert _plays(sink) == ["run_failed"]

    def test_a_run_the_user_stopped_is_silent(self, installed, sink):
        from spacr.qt.notify import announce_pipeline_finished

        announce_pipeline_finished("mask", "cancelled", 12.0)
        assert _plays(sink) == []

    def test_each_run_sound_has_its_own_switch(self, installed, sink):
        from spacr.qt.notify import announce_pipeline_finished

        installed.apply(snd.SoundSettings(enabled=True, run_finished=False))
        announce_pipeline_finished("mask", "success", 1.0)
        announce_pipeline_finished("mask", "failed", 1.0)
        assert _plays(sink) == ["run_failed"]

    def test_the_real_run_end_reaches_the_sound(self, installed, sink,
                                                qtbot, monkeypatch):
        """Through `AppScreen._on_finished`, the door every run leaves by."""
        from spacr.qt.screens.app_screen import AppScreen

        monkeypatch.setattr("spacr.qt.ai.settings.get_route_errors_through_ai",
                            lambda: False)
        monkeypatch.setattr("spacr.qt.ai.settings.get_auto_file_issues",
                            lambda: False)
        screen = AppScreen("mask")
        qtbot.addWidget(screen)
        screen._on_finished(True)
        screen._on_pipeline_error("Traceback\nValueError: boom")
        screen._on_finished(False)

        class _Stopped:
            was_cancelled = True

        screen._worker = _Stopped()
        screen._on_finished(False)
        screen._worker = None
        assert _plays(sink) == ["run_finished", "run_failed"]


class TestTheMusicBed:

    def test_it_loops_until_switched_off(self, make_engine, sink):
        engine = make_engine(enabled=True, bed=True, volume=1.0)
        starts = [row for row in sink.log if row[0] == "play"]
        assert starts == [("play", "bed", pytest.approx(snd.BED_LEVEL),
                           snd.LOOP_FOREVER)]
        engine.apply(snd.SoundSettings(enabled=True, bed=False))
        assert sink.log[-1] == ("stop", "bed")

    def test_a_volume_change_does_not_restart_it(self, make_engine, sink):
        engine = make_engine(enabled=True, bed=True, volume=1.0)
        engine.apply(snd.SoundSettings(enabled=True, bed=True, volume=0.5))
        assert [row[0] for row in sink.log].count("play") == 1
        bed = [e for e in sink.effects if e.name == "bed"][0]
        assert bed.volume == pytest.approx(0.25 * snd.BED_LEVEL)

    def test_it_rests_at_the_levels_that_switch_the_backdrop_off(self, store):
        store.set_sound_enabled(True)
        store.set_sound_event_enabled("bed", True)
        store.set_performance_level("balanced")
        assert snd.read_sound_settings().wants("bed")
        for level in ("laptop", "extra_performance"):
            store.set_performance_level(level)
            assert not snd.read_sound_settings().wants("bed"), level

    def test_a_preview_fades_out_by_itself(self, make_engine, sink, qtbot,
                                           monkeypatch):
        monkeypatch.setattr(snd, "PREVIEW_BED_MS", 20)
        monkeypatch.setattr(snd, "FADE_MS", 60)
        engine = make_engine(enabled=True)
        engine.preview("bed", "orbit", 1.0)
        bed = [e for e in sink.effects if e.name == "bed"][0]
        assert bed.playing and bed.loops == snd.LOOP_FOREVER
        start = bed.volume
        qtbot.waitUntil(lambda: not bed.playing, timeout=5000)
        assert sink.log[-1] == ("stop", "bed")
        assert bed.volume < start, "it stopped without fading"

    def test_a_preview_hands_the_stored_bed_back_when_it_ends(
            self, make_engine, sink, qtbot, monkeypatch):
        """A user who already has the bed on presses Preview. The preview
        fades out -- and the bed the settings ask for has to come back. It
        used to stay silent until Preferences was closed, because the fade
        that ends a preview and the stop that ends the bed were the same
        stop."""
        monkeypatch.setattr(snd, "PREVIEW_BED_MS", 20)
        monkeypatch.setattr(snd, "FADE_MS", 60)
        engine = make_engine(enabled=True, bed=True, volume=1.0)
        bed = [e for e in sink.effects if e.name == "bed"][0]
        stored = snd.perceived_gain(1.0) * snd.BED_LEVEL
        assert bed.playing and bed.volume == pytest.approx(stored)

        engine.preview("bed", "orbit", 0.2)
        assert bed.volume < stored, "the preview took the dialog's volume"
        qtbot.waitUntil(lambda: bed.volume == pytest.approx(stored),
                        timeout=5000)
        assert bed.playing and bed.loops == snd.LOOP_FOREVER
        assert sink.log[-2:] == [
            ("stop", "bed"),
            ("play", "bed", pytest.approx(stored), snd.LOOP_FOREVER)], (
            "the preview has to end before the stored bed starts again")

    def test_closing_preferences_ends_a_preview(self, make_engine, sink,
                                                monkeypatch):
        engine = make_engine()
        monkeypatch.setattr(snd, "_ENGINE", engine)
        assert snd.preview_sound("bed", "orbit", 1.0) is True
        bed = [e for e in sink.effects if e.name == "bed"][0]
        assert bed.playing
        snd.stop_sound_preview()
        assert not bed.playing and bed.deleted, (
            "sound is off, so the preview's device must be let go")


class TestSilenceIsTheFallback:

    def test_no_audio_stack_is_silence_not_an_exception(
            self, make_engine, window, qtbot):
        def broken(_parent):
            raise ImportError("No module named 'PySide6.QtMultimedia'")

        engine = make_engine(enabled=True, factory=broken)
        assert engine.available is False
        qtbot.mouseClick(window.buttons[0], Qt.MouseButton.LeftButton)
        assert engine.play("run_finished") is True

    def test_a_sound_that_cannot_be_rendered_is_silence(
            self, make_engine, sink, monkeypatch):
        from spacr.qt import sound_synth

        def fails(*_a, **_k):
            raise OSError("disk full")

        monkeypatch.setattr(sound_synth, "write_wav", fails)
        engine = make_engine(enabled=True)
        assert engine.play("run_failed") is True
        assert _plays(sink) == []

    def test_the_filter_drops_a_freed_event(self, make_engine):
        import shiboken6

        engine = make_engine(enabled=True, hover=True)
        event = QEvent(QEvent.Type.Enter)
        shiboken6.delete(event)
        assert engine._filter.eventFilter(QPushButton(), event) is False

    def test_the_filter_drops_a_freed_receiver(self, make_engine):
        import shiboken6

        engine = make_engine(enabled=True, hover=True)
        button = QPushButton()
        shiboken6.delete(button)
        assert engine._filter.eventFilter(
            button, QEvent(QEvent.Type.Enter)) is False

    def test_the_filter_never_consumes_an_event(self, make_engine, window):
        engine = make_engine(enabled=True, hover=True)
        for kind in (QEvent.Type.MouseMove, QEvent.Type.Enter,
                     QEvent.Type.Leave, QEvent.Type.Paint):
            assert engine._filter.eventFilter(window.buttons[0],
                                              QEvent(kind)) is False

    def test_an_uninteresting_event_costs_microseconds(self, make_engine,
                                                       window):
        """Item 380 prices a do-nothing application filter at 0.93 us per
        event. This one is only installed while wanted; when it is, the path
        for an event it ignores is two liveness checks and four compares."""
        engine = make_engine(enabled=True, hover=True)
        event = QEvent(QEvent.Type.MouseMove)
        target = window.buttons[0]
        f = engine._filter.eventFilter
        n = 20000
        start = time.perf_counter()
        for _ in range(n):
            f(target, event)
        per_event_us = (time.perf_counter() - start) / n * 1e6
        assert per_event_us < 15.0, f"{per_event_us:.2f} us per event"


class TestTheAudioThread:

    def test_every_effect_is_made_and_played_on_the_gui_thread(
            self, make_engine, sink, qapp, qtbot):
        """Item 444, and the reverse of what this test asserted before it.

        Effects used to be built on the audio thread, to keep the sound
        server connection off the GUI thread. Qt Multimedia's device
        handling belongs to the thread that owns the event loop, so that
        enabled socket notifiers from the wrong one: reopening Preferences
        with sound on segfaulted three runs out of three, and wedged the
        maintainer's workstation until he force quit it. The thread the
        effects are made on is therefore pinned to the GUI thread, and the
        audio thread keeps only the rendering.
        """
        engine = make_engine(threaded=True)
        thread = engine.audio_thread()
        assert thread is not None and thread.objectName() == "spacr-sound"
        engine.apply(snd.SoundSettings(enabled=True))
        qtbot.waitUntil(lambda: engine.available is not None, timeout=20000)
        assert sink.effects, "nothing was prepared"
        gui = qapp.thread()
        assert all(_same(e.made_on, gui) and not _same(e.made_on, thread)
                   for e in sink.effects)
        engine.play("click")
        qtbot.waitUntil(lambda: bool(_plays(sink)), timeout=5000)
        played = [e for e in sink.effects if e.played_on is not None]
        assert played and all(_same(e.played_on, gui) for e in played)

    def test_no_qt_multimedia_symbol_is_reached_off_the_gui_thread(
            self, qapp, tmp_path, monkeypatch, qtbot):
        """The pin for item 444: the thread of every Qt Multimedia call.

        Stand-ins for ``QMediaDevices`` and ``QSoundEffect`` record the
        thread they are reached on, so this fails the moment the device
        listing or an effect moves back to the audio thread -- which is
        what wedged the application. Nothing here touches a real device.
        """
        import PySide6.QtMultimedia as mm

        threads = []

        class Devices:
            @staticmethod
            def defaultAudioOutput():          # noqa: N802 - Qt naming
                threads.append(QThread.currentThread())

        class Effect(FakeEffect):
            def __init__(self, _parent=None):
                threads.append(QThread.currentThread())
                super().__init__([])

            def setSource(self, url):          # noqa: N802 - Qt naming
                threads.append(QThread.currentThread())
                super().setSource(url)

            def play(self):
                threads.append(QThread.currentThread())
                super().play()

        monkeypatch.setattr(mm, "QMediaDevices", Devices)
        monkeypatch.setattr(mm, "QSoundEffect", Effect)
        engine = snd.SoundEngine(None, threaded=True,
                                 cache_root=tmp_path / "sounds")
        try:
            engine.apply(snd.SoundSettings(enabled=True, bed=True))
            qtbot.waitUntil(lambda: engine.available is not None,
                            timeout=60000)
            engine.play("click")
            engine.play("run_finished")
            qtbot.waitUntil(lambda: "bed" in engine.player()._effects,
                            timeout=60000)
        finally:
            audio = engine.audio_thread()
            assert engine.shutdown(20000) is True, (
                "the audio thread had to be parked")
        gui = qapp.thread()
        assert threads, "nothing reached Qt Multimedia at all"
        assert all(_same(t, gui) for t in threads), (
            "Qt Multimedia was reached off the GUI thread")
        assert not any(_same(t, audio) for t in threads)

    def test_quitting_ends_the_thread_and_lets_go_of_every_effect(
            self, make_engine, sink, qtbot):
        engine = make_engine(threaded=True)
        engine.apply(snd.SoundSettings(enabled=True, bed=True))
        qtbot.waitUntil(lambda: engine.available is not None, timeout=20000)
        qtbot.waitUntil(lambda: any(e.name == "bed" for e in sink.effects),
                        timeout=20000)
        assert engine.shutdown() is True
        assert not engine.audio_thread().isRunning()
        assert all(e.deleted and not e.playing for e in sink.effects)
        assert engine.shutdown() is True, "a second quit is harmless"
        assert not engine.filter_installed

    def test_the_device_is_connected_once_and_before_the_first_effect(
            self, qapp, tmp_path, monkeypatch):
        """The sound server is connected to ONCE, and never by surprise.

        Connecting cost 354 ms on the maintainer's workstation, and it is
        the GUI thread that has to pay it (item 444). So it is paid at one
        named place -- ``_SoundPlayer.warm`` -- ahead of the first effect,
        and not again. Stand-ins record the order; nothing here reaches a
        real device.
        """
        import PySide6.QtMultimedia as mm

        calls = []

        class Devices:
            @staticmethod
            def defaultAudioOutput():          # noqa: N802 - Qt naming
                calls.append("devices")

        class Effect(FakeEffect):
            def __init__(self, _parent=None):
                calls.append("effect")
                super().__init__([])

        monkeypatch.setattr(mm, "QMediaDevices", Devices)
        monkeypatch.setattr(mm, "QSoundEffect", Effect)
        engine = snd.SoundEngine(None, threaded=False,
                                 cache_root=tmp_path / "sounds")
        try:
            engine.apply(snd.SoundSettings(enabled=True, run_failed=False,
                                           run_finished=False))
            engine.play("click")
            engine._warm()
        finally:
            engine.shutdown()
        assert calls[0] == "devices" and calls.count("devices") == 1
        assert calls.count("effect") == 4, calls

    def test_switching_sound_on_warms_the_device_on_an_idle_timer(
            self, qapp, tmp_path, monkeypatch, qtbot):
        """Nobody waits for the sound server, because nobody is waiting.

        The connection is scheduled rather than made: `apply` returns
        without it, and it happens once the application has run out of
        other work. That is the whole of the item's third point -- the
        cost is real, so it is paid where it is not felt, and logged.
        """
        import PySide6.QtMultimedia as mm

        calls = []

        class Devices:
            @staticmethod
            def defaultAudioOutput():          # noqa: N802 - Qt naming
                calls.append("devices")

        monkeypatch.setattr(mm, "QMediaDevices", Devices)
        monkeypatch.setattr(mm, "QSoundEffect", FakeEffect)
        monkeypatch.setattr(snd, "DEVICE_WARM_DELAY_MS", 5)
        engine = snd.SoundEngine(None, threaded=True,
                                 cache_root=tmp_path / "sounds")
        try:
            engine.apply(snd.SoundSettings(enabled=True, click=False,
                                           run_finished=False,
                                           run_failed=False))
            assert calls == [], "the device was connected inside apply()"
            qtbot.waitUntil(lambda: calls == ["devices"], timeout=5000)
            engine.apply(snd.SoundSettings(enabled=True))
            qtbot.wait(60)
            assert calls == ["devices"], "connected more than once"
        finally:
            engine.shutdown()

    def test_the_gui_thread_never_waits_for_a_render(self, make_engine,
                                                     sink, qtbot, qapp):
        """Rendering the whole set on the audio thread while the GUI thread
        keeps turning: the longest gap between two GUI ticks stays short."""
        from PySide6.QtCore import QTimer

        ticks = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
        timer.start(5)
        engine = make_engine(threaded=True)
        engine.apply(snd.SoundSettings(enabled=True, hover=True, bed=True))
        qtbot.waitUntil(lambda: any(e.name == "bed" for e in sink.effects),
                        timeout=30000)
        timer.stop()
        gaps = [b - a for a, b in zip(ticks, ticks[1:])]
        assert len(ticks) > 20
        assert max(gaps) < 0.25, f"the GUI thread stalled {max(gaps):.3f} s"
        assert threading.current_thread() is threading.main_thread()
