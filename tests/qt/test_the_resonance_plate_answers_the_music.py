"""The Resonance backdrop: a real Chladni plate, driven by what is playing.

Part B of instruction 427, asked for 2026-09-19: "an audio-reactive
visualiser (Chladni-like particle figures) driven by the bed's precomputed
envelope/spectrum in sync with playback position, or a user-chosen local
WAV; no system-audio capture. It idles beautifully in silence, is
frame-budgeted inside the existing ambient producer architecture, and
respects the Animation preferences."

Five things are held down here and each one is a way the feature could be
true in the ledger and false on the screen:

* THE PHYSICS. The particles sit on the nodal set of a standing wave --
  measured as ``|w|`` at the particles, before and after.
* THE SYNC. The moment the visualiser reads is the moment of the audio at
  the playback position, and it wraps with the loop.
* THE ARCHITECTURE. The real-time signal enters in ``advance`` on the GUI
  thread and nowhere else, so a frame is still a pure function of the
  clock while nothing plays and two threads still shade the same bytes.
* THE IDLE. With nothing playing there is still a picture, and it moves.
* NO CAPTURE. Nothing here opens an audio input, and ``playing_moment``
  does no file access at all once something is playing.

The audio sink is faked throughout and every file is written under
``tmp_path``.
"""
from __future__ import annotations

import dataclasses
import math
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, QThread
from PySide6.QtGui import QColor, QImage, QPainter

from spacr.qt import resonance as rs
from spacr.qt import sound as snd
from spacr.qt import sound_synth as ss
from spacr.qt.widgets import ambient as amb

DARK = "#101418"
SR = ss.SAMPLE_RATE


@pytest.fixture(autouse=True)
def _nothing_is_playing_afterwards():
    """No test leaves a record behind for the next one to be driven by."""
    yield
    rs.clear_now_playing()
    snd.shutdown_sound()


def _tone(path: Path, seconds: float, hz, amplitude=0.5, sr: int = SR) -> Path:
    """Write a WAV of one or more sine tones, for the analysis to measure."""
    t = np.arange(int(seconds * sr)) / sr
    mono = np.zeros_like(t)
    for one in (hz if isinstance(hz, (list, tuple)) else [hz]):
        mono += np.sin(2.0 * math.pi * one * t)
    mono *= amplitude / max(1e-9, float(np.abs(mono).max()))
    ss.write_wav(path, np.vstack([mono, mono]), sr)
    return path


def _render(engine, width=320, height=200, background=DARK) -> QImage:
    """One frame, painted exactly as the widget paints it."""
    image = QImage(width, height, QImage.Format_RGB32)
    painter = QPainter(image)
    painter.fillRect(image.rect(), QColor(background))
    engine.paint(painter, width, height)
    painter.end()
    return image


# ---------------------------------------------------------------------------
# The plate
# ---------------------------------------------------------------------------

def test_the_field_is_the_chladni_superposition_and_vanishes_on_its_nodes():
    field, dx, dy = rs.lattice((3, 1, -1), (3, 1, -1), 0.0, 129)
    axis = np.linspace(0.0, 1.0, 129)
    m, n = 3, 1
    expected = (np.outer(np.cos(m * math.pi * axis), np.cos(n * math.pi * axis))
                - np.outer(np.cos(n * math.pi * axis),
                           np.cos(m * math.pi * axis))) * 0.5
    assert np.allclose(field, expected, atol=1e-5)
    # The minus family's defining property: the leading diagonal is a node.
    assert np.abs(np.diagonal(field)).max() < 1e-6
    # And the gradient is the gradient, checked against a finite difference.
    step = axis[1] - axis[0]
    numeric = np.gradient(field, step, axis=1)
    assert np.abs(numeric[2:-2, 2:-2] - dx[2:-2, 2:-2]).max() < 0.1
    numeric = np.gradient(field, step, axis=0)
    assert np.abs(numeric[2:-2, 2:-2] - dy[2:-2, 2:-2]).max() < 0.1


def test_the_plus_family_has_no_diagonal_and_both_families_are_offered():
    field, _dx, _dy = rs.lattice((3, 1, 1), (3, 1, 1), 0.0, 129)
    assert np.abs(np.diagonal(field)).max() > 0.2, \
        "the plus sign is not reaching the field"
    signs = {sign for _m, _n, sign in rs.MODES}
    assert signs == {1, -1}, "one whole family of figures is missing"
    assert all(m > n >= 1 for m, n, _s in rs.MODES)


def test_the_sand_settles_onto_the_nodal_lines():
    field, dx, dy = rs.lattice((5, 2, -1), (5, 2, -1), 0.0, 128)
    gen = np.random.default_rng(3)
    x = gen.random(900).astype(np.float32)
    y = gen.random(900).astype(np.float32)
    before = float(np.abs(rs.sample(field, x, y)).mean())
    px, py = rs.settle(x, y, field, dx, dy, 8, 0.9)
    after = float(np.abs(rs.sample(field, px, py)).mean())
    assert after < before / 5.0, f"{before:.3f} -> {after:.3f}"
    assert 0.0 <= px.min() and px.max() <= 1.0
    assert 0.0 <= py.min() and py.max() <= 1.0


def test_a_gentler_drive_leaves_the_sand_looser():
    """Tightness is the control that makes silence a cloud and loud a figure."""
    field, dx, dy = rs.lattice((5, 2, -1), (5, 2, -1), 0.0, 128)
    gen = np.random.default_rng(3)
    x, y = gen.random(600).astype(np.float32), gen.random(600).astype(np.float32)
    loose = rs.settle(x, y, field, dx, dy, 8, 0.15)
    tight = rs.settle(x, y, field, dx, dy, 8, 0.95)
    assert float(np.abs(rs.sample(field, *loose)).mean()) > \
        3.0 * float(np.abs(rs.sample(field, *tight)).mean())
    assert rs.settle(x, y, field, dx, dy, 0, 0.9)[0].tolist() == x.tolist()


def test_nothing_piles_up_on_the_rim_or_in_the_corners_of_the_plate():
    """The bug the step clamp and the reflection were written for.

    A Newton step is ``w * grad(w) / |grad(w)|^2`` and the denominator
    vanishes at an antinode, so an unclamped step flings those particles
    off the plate; CLIPPING the overshoot then parks them on the boundary
    for good, because the gradient there is near zero and nothing moves
    them off again. The first rendering drew a faint rectangle round every
    figure and four bright dots in its corners that no mode explained.

    Swept over every entry of :data:`spacr.qt.resonance.MODES` rather than
    one, because how bad it gets depends on the mode: one pair showed five
    corner grains where the sweep shows 162. Measured with each guard
    removed on its own, 1 200 grains per pair, 16 800 in all:

        shipped             3 corners, 0.85 % on the rim
        without the clamp  17 corners, 1.05 %
        clipping, not reflecting
                            7 corners, 1.97 %
        neither            162 corners, 4.15 %
    """
    corners = rim = total = 0
    for index in range(len(rs.MODES)):
        first, second, blend = rs.mode_blend(index + 0.35)
        field, dx, dy = rs.lattice(first, second, blend, 128)
        gen = np.random.default_rng(11)
        x = gen.random(1200).astype(np.float32)
        y = gen.random(1200).astype(np.float32)
        px, py = rs.settle(x, y, field, dx, dy, 8, 0.9)
        edge_x, edge_y = np.minimum(px, 1.0 - px), np.minimum(py, 1.0 - py)
        corners += int(((edge_x < 0.01) & (edge_y < 0.01)).sum())
        rim += int(((edge_x < 0.002) | (edge_y < 0.002)).sum())
        total += px.size
    assert corners <= 10, f"{corners} grains parked in the corners"
    assert rim <= 0.014 * total, f"{100.0 * rim / total:.2f} % on the rim"


def test_the_figure_morphs_rather_than_cutting_between_modes():
    """A blend a hair apart is a picture a hair apart."""
    here = rs.lattice(*rs.mode_blend(2.50)[:2], 0.50, 96)[0]
    near = rs.lattice(*rs.mode_blend(2.51)[:2], 0.51, 96)[0]
    far = rs.lattice(*rs.mode_blend(3.60)[:2], 0.60, 96)[0]
    assert np.abs(near - here).max() < 0.1
    assert np.abs(far - here).max() > 0.3
    first, second, blend = rs.mode_blend(len(rs.MODES) + 0.25)
    assert (first, second) == (rs.MODES[0], rs.MODES[1])
    assert blend == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def test_each_band_answers_its_own_part_of_the_spectrum(tmp_path):
    """Four tones, one per band, one after the other.

    NOT four files of one tone each, which is the test this started as and
    which said nothing: every row is scaled against ITS OWN loudest frame,
    so a band that is silent throughout and a band that is loud throughout
    both come back as a flat row and the largest of them is noise. A tone
    that comes and goes is the only thing the normalisation lets a band
    say -- and saying it is the point of normalising per band, because the
    shaker never rises within forty decibels of the kick.
    """
    span = 0.75
    parts = []
    for low, high in rs.BANDS:
        t = np.arange(int(span * SR)) / SR
        parts.append(0.6 * np.sin(2.0 * math.pi * math.sqrt(low * high) * t))
    mono = np.concatenate(parts)
    ss.write_wav(tmp_path / "ladder.wav", np.vstack([mono, mono]), SR)
    read, rate = rs.read_wav_mono(tmp_path / "ladder.wav")
    measured = rs.analyse(read, rate, 60.0, loop=False)
    for index in range(len(rs.BANDS)):
        middle = int((index + 0.5) * span * 60.0)
        strongest = int(np.argmax(measured.bands[:, middle]))
        assert strongest == index, (
            f"band {index}'s own tone lit band {strongest}")
        assert measured.bands[index, middle] > 0.8


def test_the_envelope_follows_the_loudness(tmp_path):
    t = np.arange(int(2.0 * SR)) / SR
    ramp = np.linspace(0.02, 0.9, t.size)
    mono = ramp * np.sin(2.0 * math.pi * 300.0 * t)
    ss.write_wav(tmp_path / "ramp.wav", np.vstack([mono, mono]), SR)
    read, rate = rs.read_wav_mono(tmp_path / "ramp.wav")
    measured = rs.analyse(read, rate, 60.0, loop=False)
    assert measured.level[10] < 0.3 < measured.level[-10]
    assert np.all(np.diff(measured.level[5:-5]) > -0.05), "not monotonic"


def test_an_onset_is_visible_for_longer_than_one_analysis_frame(tmp_path):
    """A 24 fps painter must not be able to miss a beat in a 60 fps row."""
    mono = np.zeros(int(2.0 * SR))
    hit = np.exp(-np.arange(int(0.05 * SR)) / (0.01 * SR))
    for at in (int(0.5 * SR), int(1.2 * SR)):
        mono[at:at + hit.size] += hit * np.sin(
            2.0 * math.pi * 900.0 * np.arange(hit.size) / SR)
    ss.write_wav(tmp_path / "hits.wav", np.vstack([mono, mono]), SR)
    read, rate = rs.read_wav_mono(tmp_path / "hits.wav")
    measured = rs.analyse(read, rate, 60.0, loop=False)
    peak = int(np.argmax(measured.onset[20:60])) + 20
    assert measured.onset[peak] > 0.5
    later = measured.onset[peak + 1:peak + int(0.2 * 60)]
    assert float(later.min()) > 0.15, "the onset vanished in one frame"
    assert rs.ONSET_RELEASE_S > 1.0 / 24.0


def test_a_looping_piece_is_analysed_as_a_circle(tmp_path):
    """The frames that straddle the join see the start, not silence."""
    mono, rate = rs.read_wav_mono(_tone(tmp_path / "steady.wav", 2.0, 440.0))
    measured = rs.analyse(mono, rate, 60.0, loop=True)
    assert measured.level[-1] == pytest.approx(measured.level[len(mono)
                                                              // 1600], 0.2)
    # `at` wraps BETWEEN the last frame and the first, not to the first.
    end = measured.at(measured.duration - 1e-4)
    start = measured.at(0.0)
    assert end.level == pytest.approx(start.level, abs=0.25)
    assert measured.at(measured.duration * 3.0 + 0.5).level == \
        pytest.approx(measured.at(0.5).level, abs=1e-6)


def test_the_analysis_is_written_once_and_re_read(tmp_path):
    source = _tone(tmp_path / "bed.wav", 1.0, 440.0)
    first = rs.ensure_analysis(source)
    assert first is not None and first.exists()
    assert first.name.endswith(rs.ANALYSIS_SUFFIX)
    stamp = first.stat().st_mtime_ns
    assert rs.ensure_analysis(source) == first
    assert first.stat().st_mtime_ns == stamp, "it analysed the file twice"

    _tone(source, 1.0, 880.0)
    assert rs.ensure_analysis(source) == first
    assert first.stat().st_mtime_ns != stamp, \
        "a replaced source was answered from the old analysis"


def test_a_file_that_cannot_be_analysed_is_refused_quietly(tmp_path):
    missing = tmp_path / "gone.wav"
    assert rs.ensure_analysis(missing) is None
    assert rs.load_analysis(missing) is None
    rubbish = tmp_path / "rubbish.wav"
    rubbish.write_bytes(b"not a wav at all")
    assert rs.ensure_analysis(rubbish) is None
    eight_bit = tmp_path / "eight.wav"
    with wave.open(str(eight_bit), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(SR)
        handle.writeframes(b"\x80" * 1000)
    assert rs.ensure_analysis(eight_bit) is None


# ---------------------------------------------------------------------------
# What is playing, and no capture anywhere
# ---------------------------------------------------------------------------

def test_nothing_playing_is_silence_and_not_an_error():
    assert rs.now_playing() is None
    assert rs.playing_moment() == rs.silence()
    assert rs.silence().level == 0.0


def test_the_moment_follows_the_playback_position(tmp_path):
    source = _tone(tmp_path / "two.wav", 2.0, [80.0, 6000.0])
    analysis = rs.ensure_analysis(source)
    rs.set_now_playing(rs.NowPlaying(str(analysis), 1000.0, 2.0, True))
    early = rs.playing_moment(1000.5)
    later = rs.playing_moment(1000.5 + 2.0)
    assert early.level > 0.0
    assert later == early, "the position did not wrap with the loop"
    assert rs.playing_moment(999.0) == rs.silence(), "before it started"
    rs.clear_now_playing()
    assert rs.playing_moment(1000.5) == rs.silence()


def test_reading_the_moment_never_touches_the_filesystem(tmp_path,
                                                         monkeypatch):
    """It is read once a frame from the GUI thread's own timer.

    A ``stat`` there is a filesystem call on the GUI thread, and on a
    network home directory that is the stall ``spacr.qt.path_probe``
    exists for. The guard is that the analysis is resolved in
    ``set_now_playing``, on the audio thread; here every filesystem entry
    point is made to raise, and the moment still has to come back.
    """
    source = _tone(tmp_path / "one.wav", 1.0, 400.0)
    analysis = rs.ensure_analysis(source)
    rs.set_now_playing(rs.NowPlaying(str(analysis), 0.0, 1.0, True))

    # Counted rather than refused: `rs.os` IS the os module, so making it
    # raise would fail whatever ran next in the process rather than this
    # call -- which it did, in the settings store's teardown.
    seen = []
    real_stat, real_load = rs.os.stat, rs.np.load
    rs.os.stat = lambda *a, **k: (seen.append("stat"), real_stat(*a, **k))[1]
    rs.np.load = lambda *a, **k: (seen.append("load"), real_load(*a, **k))[1]
    try:
        moment = rs.playing_moment(0.5)
    finally:
        rs.os.stat, rs.np.load = real_stat, real_load
    assert moment.level > 0.0
    assert seen == [], f"the GUI thread touched the filesystem: {seen}"


def test_no_audio_input_is_ever_opened():
    """"no system-audio capture" is a claim about what is imported."""
    import inspect

    for module in (rs, snd, amb):
        source = inspect.getsource(module)
        for banned in ("QAudioInput", "QAudioSource", "QMediaCaptureSession",
                       "audioInputs", "sounddevice", "pyaudio"):
            assert banned not in source, f"{module.__name__} uses {banned}"


# ---------------------------------------------------------------------------
# The engine: idle, driven, and still a pure function of the clock
# ---------------------------------------------------------------------------

def test_resonance_is_an_animation_the_preferences_offer():
    assert "resonance" in amb.AMBIENT_THEMES
    assert "resonance" in amb.ANIMATION_CHOICES
    assert amb.theme_label("resonance") == "Resonance"
    assert amb.theme_note("resonance").endswith(".")
    assert "okabe" in amb.palettes_for("resonance")
    assert "pastel" not in amb.palettes_for("resonance"), \
        "a pale low-contrast hue at one pixel is indistinguishable from the page"


def test_it_idles_beautifully_in_silence(qapp):
    """Nothing playing still has to be worth looking at, and to move."""
    engine = amb.make_engine("resonance", "spacr", DARK, seed=7)
    engine.set_time(4.0)
    assert engine.drive == rs.silence()
    assert engine.energy() > 0.2, "the plate went out"
    early = _render(engine)
    engine.set_time(30.0)
    assert bytes(_render(engine).constBits()) != bytes(early.constBits())
    grains = engine.geometry(320, 200)
    assert len(grains) == amb.RESONANCE_PARTICLES
    assert all(b > 0.0 for _x, _y, b in grains)


def test_the_music_reaches_the_picture(qapp, tmp_path):
    """The whole point, and the thing a mocked test would never catch."""
    source = _tone(tmp_path / "loud.wav", 2.0, [60.0, 500.0, 5000.0], 0.9)
    analysis = rs.ensure_analysis(source)
    engine = amb.make_engine("resonance", "spacr", DARK, seed=7)
    engine.set_time(6.0)
    engine.advance(0.0)
    silent = bytes(_render(engine).constBits())
    quiet_energy = engine.energy()

    rs.set_now_playing(rs.NowPlaying(str(analysis), 0.0, 2.0, True))
    engine.advance(0.0)
    assert engine.drive.level > 0.5
    assert engine.energy() > quiet_energy
    assert bytes(_render(engine).constBits()) != silent


def test_the_beat_throws_the_sand_off_the_lines(qapp):
    """The bounce the request asks for, measured on the positions."""
    engine = amb.make_engine("resonance", "spacr", DARK, seed=7)
    engine.set_time(5.0)
    still = engine.drive._replace(level=0.8, bands=(0.8, 0.8, 0.8, 0.8))
    engine.drive = still
    settled = engine.sand()
    engine.drive = still._replace(onset=1.0)
    thrown = engine.sand()
    moved = np.hypot(thrown[0] - settled[0], thrown[1] - settled[1])
    assert float(moved.mean()) == pytest.approx(amb.RESONANCE_THROW, rel=0.35)


def test_the_real_time_signal_enters_in_advance_and_nowhere_else(qapp,
                                                                 tmp_path):
    """The architecture rule, asserted rather than trusted.

    Shading runs on the shading thread and must not read the clock of the
    world: the music can start underneath it and the frame must not
    change until the GUI thread has stepped the engine. Otherwise the
    picture the shading thread produces is not the picture the GUI thread
    would have produced at that clock, which is the property
    ``test_the_backdrop_survives_a_run`` compares byte for byte.
    """
    source = _tone(tmp_path / "loud.wav", 1.5, [70.0, 700.0, 6000.0], 0.9)
    analysis = rs.ensure_analysis(source)
    engine = amb.make_engine("resonance", "spacr", DARK, seed=7)
    engine.set_time(8.0)
    first = engine.shade(320, 200).copy()

    rs.set_now_playing(rs.NowPlaying(str(analysis), 0.0, 1.5, True))
    assert rs.playing_moment(0.6).level > 0.5, "the test is not driving it"
    assert bytes(engine.shade(320, 200).constBits()) == \
        bytes(first.constBits()), "shade() read something outside the clock"

    engine.advance(0.0)
    assert engine.drive.level > 0.0
    assert bytes(engine.shade(320, 200).constBits()) != bytes(first.constBits())


def test_the_same_clock_shades_the_same_bytes_on_another_thread(qapp):
    engine = amb.make_engine("resonance", "spacr", DARK, seed=99)
    engine.set_time(12.5)
    here = engine.shade(480, 300)
    box = {}
    thread = threading.Thread(
        target=lambda: box.update(image=engine.shade(480, 300)))
    thread.start()
    thread.join(timeout=30.0)
    assert not thread.is_alive()
    assert bytes(box["image"].constBits()) == bytes(here.constBits())


def test_the_density_and_size_controls_reach_the_plate(qapp):
    engine = amb.make_engine("resonance", "spacr", DARK, seed=7, density=0.25)
    assert len(engine.geometry(320, 200)) < amb.RESONANCE_PARTICLES
    wide = amb.make_engine("resonance", "spacr", DARK, seed=7, size=2.5)
    narrow = amb.make_engine("resonance", "spacr", DARK, seed=7, size=0.25)
    assert wide.plate(320, 200)[2] > narrow.plate(320, 200)[2]
    assert narrow.plate(320, 200)[2] > 0


def test_a_hidden_backdrop_costs_nothing_and_this_one_is_no_different(qtbot):
    """The whole performance story, for the new theme as for the six."""
    widget = amb.AmbientWidget(theme="resonance", palette="spacr",
                               background=DARK, seed=7)
    qtbot.addWidget(widget)
    widget.resize(480, 320)
    widget.show()
    qtbot.waitUntil(lambda: widget.frames_shaded() > 0, timeout=10000)
    assert widget.shading_thread_alive()
    widget.hide()
    qtbot.waitUntil(lambda: not widget.shading_thread_alive(), timeout=10000)
    assert not widget.is_running()


# ---------------------------------------------------------------------------
# The sound engine tells the backdrop what is playing
# ---------------------------------------------------------------------------

class _Fake:
    """Stands in for ``QSoundEffect``; records, never plays."""

    def __init__(self, _parent=None):
        self.source = ""
        self.playing = False
        self.made_on = QThread.currentThread()

    def setSource(self, url):                  # noqa: N802 - Qt naming
        self.source = url.toLocalFile()

    def setVolume(self, value):                # noqa: N802 - Qt naming
        self.volume = float(value)

    def setLoopCount(self, count):             # noqa: N802 - Qt naming
        self.loops = int(count)

    def play(self):
        self.playing = True

    def stop(self):
        self.playing = False

    def deleteLater(self):                     # noqa: N802 - Qt naming
        self.playing = False


@pytest.fixture
def quick_bed(monkeypatch, tmp_path):
    """A four-bar reference set, so a test that plays the bed is not slow."""
    fast = dataclasses.replace(ss.ORBIT, bed_bars=4, reverb_seconds=0.6)
    monkeypatch.setattr(ss, "SOUND_THEMES", {fast.key: fast})
    monkeypatch.setattr(snd, "SOUND_THEMES", {fast.key: fast})
    monkeypatch.setenv(ss.CACHE_ENV, str(tmp_path / "sounds"))
    from spacr.qt import preferences as prefs
    monkeypatch.setattr(
        prefs, "_settings",
        lambda: QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat))
    return fast


def _engine(qapp, tmp_path):
    return snd.SoundEngine(None, threaded=False, effect_factory=_Fake,
                           cache_root=tmp_path / "sounds")


def test_starting_the_bed_tells_the_backdrop_what_is_playing(qapp, tmp_path,
                                                             quick_bed):
    engine = _engine(qapp, tmp_path)
    engine.apply(snd.SoundSettings(enabled=True, bed=True, click=False,
                                   run_finished=False, run_failed=False))
    record = rs.now_playing()
    assert record is not None
    assert Path(record.analysis).exists()
    assert record.loop is True
    assert record.duration == pytest.approx(
        quick_bed.bed_bars * 4 * quick_bed.beat, abs=0.01)
    assert rs.playing_moment(record.started + 0.5).level > 0.0

    engine.apply(snd.SoundSettings(enabled=True, bed=False))
    assert rs.now_playing() is None


def test_a_music_file_of_the_users_own_is_what_plays_and_what_is_seen(
        qapp, tmp_path, quick_bed):
    chosen = _tone(tmp_path / "mine.wav", 1.5, 300.0)
    engine = _engine(qapp, tmp_path)
    engine.apply(snd.SoundSettings(enabled=True, bed=True, click=False,
                                   run_finished=False, run_failed=False,
                                   music=str(chosen)))
    record = rs.now_playing()
    assert record is not None
    assert record.duration == pytest.approx(1.5, abs=0.01)
    assert engine._worker._effects[ss.BED].source == str(chosen)
    # The sidecar goes in spaCR's cache, NOT into the folder the user's
    # music is in: a tool asked to read a file must not leave litter
    # beside it, and that folder may not even be writable.
    analysis = Path(record.analysis)
    assert analysis.parent == tmp_path / "sounds"
    assert not rs.analysis_path_for(chosen).exists()
    assert analysis == rs.analysis_path_for(chosen, tmp_path / "sounds")


def test_a_chosen_file_that_has_gone_falls_back_to_spacrs_own(qapp, tmp_path,
                                                              quick_bed):
    """Silence because somebody moved a WAV is worse than spaCR's music."""
    engine = _engine(qapp, tmp_path)
    engine.apply(snd.SoundSettings(enabled=True, bed=True, click=False,
                                   run_finished=False, run_failed=False,
                                   music=str(tmp_path / "never.wav")))
    played = engine._worker._effects[ss.BED].source
    assert played.endswith("bed.wav")
    assert "sounds" in played
    assert rs.now_playing() is not None


def test_switching_music_file_builds_a_new_effect(qapp, tmp_path, quick_bed):
    """An effect holds its source for life, so a new file is a new effect."""
    first = _tone(tmp_path / "a.wav", 1.0, 300.0)
    second = _tone(tmp_path / "b.wav", 1.0, 700.0)
    engine = _engine(qapp, tmp_path)
    base = dict(enabled=True, bed=True, click=False, run_finished=False,
                run_failed=False)
    engine.apply(snd.SoundSettings(music=str(first), **base))
    was = engine._worker._effects[ss.BED]
    engine.apply(snd.SoundSettings(music=str(second), **base))
    now = engine._worker._effects[ss.BED]
    assert now is not was
    assert now.source == str(second)
    assert rs.now_playing().duration == pytest.approx(1.0, abs=0.01)


def test_the_music_setting_is_off_by_default_and_round_trips(tmp_path,
                                                             monkeypatch):
    from spacr.qt import preferences as prefs

    monkeypatch.setattr(
        prefs, "_settings",
        lambda: QSettings(str(tmp_path / "p.ini"), QSettings.IniFormat))
    assert prefs.get_sound_music_file() == ""
    assert snd.read_sound_settings().music == ""
    prefs.set_sound_music_file("  /tmp/a.wav  ")
    assert prefs.get_sound_music_file() == "/tmp/a.wav"
    prefs.set_sound_music_file(None)
    assert prefs.get_sound_music_file() == ""


def test_the_length_of_a_wav_comes_from_its_header(tmp_path):
    assert snd.wav_seconds(_tone(tmp_path / "len.wav", 1.25, 440.0)) == \
        pytest.approx(1.25, abs=0.01)
    assert snd.wav_seconds(tmp_path / "missing.wav") == 0.0
