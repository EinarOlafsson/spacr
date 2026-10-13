"""Unavailable files, analysis and audio sinks never turn sound into a crash."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QObject, Signal

from spacr.qt import sound as snd
from spacr.qt import resonance


class Effect(QObject):
    loadedChanged = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.loaded = False
        self.plays = 0
        self.failure = None

    def setSource(self, source):
        self.source = source.toLocalFile()

    def setVolume(self, gain):
        if self.failure == 'volume':
            raise RuntimeError('audio sink disconnected')
        self.gain = gain

    def setLoopCount(self, count):
        self.loops = count

    def play(self):
        if self.failure == 'play':
            raise RuntimeError('audio sink disconnected')
        self.plays += 1

    def stop(self):
        pass

    def isLoaded(self):
        if self.failure == 'loaded':
            raise RuntimeError('audio sink disconnected')
        return self.loaded


@pytest.fixture
def player(qapp, tmp_path):
    made = []
    def factory(parent):
        effect = Effect(parent)
        made.append(effect)
        return effect
    value = snd._SoundPlayer(effect_factory=factory)
    value._switch_theme(snd.DEFAULT_THEME)
    value._paths[snd.BED] = str(tmp_path / 'bed.wav')
    value._paths['click'] = str(tmp_path / 'click.wav')
    yield value, made
    value._drop_effects()


def test_bed_start_is_announced_only_when_the_file_is_loaded(player):
    value, effects = player
    started = []
    value.bed_started.connect(started.append)
    value.bed(snd.DEFAULT_THEME, True, 0.2)
    effect = effects[0]
    assert effect.plays == 1 and started == []
    effect.loadedChanged.emit()
    assert started == []
    effect.loaded = True
    effect.loadedChanged.emit()
    assert len(started) == 1 and started[0] > 0
    value.bed(snd.DEFAULT_THEME, False, 0.2)
    effect.loadedChanged.emit()
    assert len(started) == 1, 'a late loading signal must not revive a stopped bed'


def test_bed_readiness_failure_does_not_announce_a_false_start(player):
    value, effects = player
    started = []
    value.bed_started.connect(started.append)
    value.bed(snd.DEFAULT_THEME, True, 0.2)
    effects[0].failure = 'loaded'
    effects[0].loadedChanged.emit()
    assert started == []


@pytest.mark.parametrize('operation', ['play', 'bed', 'preview'])
@pytest.mark.parametrize('failure', ['volume', 'play'])
def test_disconnected_sink_does_not_escape_to_the_caller(player, operation, failure):
    value, effects = player
    name = 'click' if operation == 'play' else snd.BED
    effect = value._effect(name)
    effect.failure = failure
    started = []
    value.bed_started.connect(started.append)
    if operation == 'play':
        value.play(snd.DEFAULT_THEME, name, 0.2)
    elif operation == 'bed':
        value.bed(snd.DEFAULT_THEME, True, 0.2)
    else:
        value.preview_bed(snd.DEFAULT_THEME, 0.2, 100)
    assert effect.plays == 0 and started == []
    assert not value._bed_playing
    assert value._fade_timer is None


def test_failed_analysis_is_forgotten_and_visualiser_is_cleared(qapp, tmp_path, monkeypatch):
    renderer = snd._SoundRenderer(tmp_path)
    calls = []
    def fail(*args, **kwargs):
        raise OSError('analysis cache is not writable')
    monkeypatch.setattr(resonance, 'ensure_analysis', fail)
    monkeypatch.setattr(resonance, 'set_now_playing', calls.append)
    monkeypatch.setattr(snd, 'wav_seconds', lambda path: 3.0)
    renderer._measure_bed(tmp_path / 'bed.wav')
    assert renderer._bed_analysis is None and renderer._bed_seconds == 0
    renderer.announce(10.0)
    assert calls == [None]


def test_unreadable_user_music_falls_back_and_one_failed_sound_does_not_cost_the_rest(
    qapp, tmp_path, monkeypatch,
):
    renderer = snd._SoundRenderer(tmp_path)
    attempts, results = [], []
    renderer.rendered.connect(lambda theme, tried, ready: results.append((tried, ready)))
    def render(theme, names, **kwargs):
        name = names[0]
        attempts.append(name)
        if name == 'broken':
            raise OSError('disk full')
        return {name: tmp_path / (name + '.wav')}
    monkeypatch.setattr(snd, 'ensure_rendered', render)
    monkeypatch.setattr(renderer, '_measure_bed', lambda path: None)
    def unreadable(path):
        raise OSError('unreadable')
    with monkeypatch.context() as local:
        local.setattr(Path, 'is_file', unreadable)
        renderer.render(snd.DEFAULT_THEME, ['broken', snd.BED, 'click'], str(tmp_path / 'own.wav'))
    assert attempts == ['broken', snd.BED, 'click']
    assert results == [(['broken', snd.BED, 'click'], {
        snd.BED: str(tmp_path / (snd.BED + '.wav')), 'click': str(tmp_path / 'click.wav'),
    })]


def test_stopped_renderer_reports_an_empty_attempt_without_rendering(qapp, tmp_path, monkeypatch):
    renderer = snd._SoundRenderer(tmp_path)
    results = []
    renderer.rendered.connect(lambda theme, tried, ready: results.append((tried, ready)))
    monkeypatch.setattr(snd, 'ensure_rendered', lambda *args, **kwargs: pytest.fail('rendered after stop'))
    renderer.request_stop()
    renderer.render(snd.DEFAULT_THEME, ['click'])
    assert results == [([], {})]


def test_visualiser_announcement_failure_does_not_interrupt_audio(qapp, tmp_path, monkeypatch):
    renderer = snd._SoundRenderer(tmp_path)
    announcements = []
    def unavailable(value):
        announcements.append(value)
        raise RuntimeError('visualiser unavailable')
    monkeypatch.setattr(resonance, 'set_now_playing', unavailable)
    renderer.announce(10.0)
    assert announcements == [None]


def test_application_absence_never_creates_an_engine(monkeypatch):
    monkeypatch.setattr(snd, '_ENGINE', None)
    monkeypatch.setattr(snd, 'QCoreApplication', SimpleNamespace(instance=lambda: None))
    monkeypatch.setattr(snd, '_create_engine', lambda app: pytest.fail('created without an app'))
    assert snd.apply_sound_preferences(settings=snd.SoundSettings(enabled=True)) is None
    assert snd.preview_sound('click', snd.DEFAULT_THEME, 0.2) is False


def test_exit_hook_cannot_replace_an_application_exit_with_an_audio_error(monkeypatch):
    calls = []
    def fail():
        calls.append('shutdown')
        raise RuntimeError('audio already gone')
    monkeypatch.setattr(snd, 'shutdown_sound', fail)
    snd._stop_at_exit()
    assert calls == ['shutdown']


def test_exit_cleanup_is_registered_once_even_if_the_quit_signal_is_unavailable(monkeypatch):
    registered, parents = [], []
    class Engine:
        def __init__(self, parent):
            parents.append(parent)
        def shutdown(self):
            pass
    class BrokenSignal:
        def connect(self, callback):
            raise RuntimeError('application is quitting')
    monkeypatch.setattr(snd, 'SoundEngine', Engine)
    monkeypatch.setattr(snd, '_ENGINE', None)
    monkeypatch.setattr(snd, '_EXIT_HOOK_INSTALLED', False)
    monkeypatch.setattr(snd.atexit, 'register', registered.append)
    first = snd._create_engine(SimpleNamespace(aboutToQuit=BrokenSignal()))
    assert snd.sound_engine() is first
    second = snd._create_engine(SimpleNamespace())
    assert snd.sound_engine() is second and first is not second
    assert registered == [snd._stop_at_exit]
    assert parents == [None, None]
