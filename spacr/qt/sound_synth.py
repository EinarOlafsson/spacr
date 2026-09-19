"""Synthesize spaCR's interface sounds and music bed with numpy.

Every sound spaCR can play is made here, in code, from a
:class:`SoundTheme`: nothing is recorded, sampled or downloaded, so no
sound carries a licence question. A theme is a key, a scale, a tempo, a
chord progression and a handful of timbre numbers; the same theme always
renders the same samples, because every random choice is drawn from a
generator seeded by the theme itself.

The reference theme, :data:`ORBIT`, is melodic space house in A minor:
warm detuned-saw pads, plucked arpeggio notes with a dotted-eighth
ping-pong delay, a soft sub, and a long dark reverb. Clicks and hovers
are short plucks on notes of the scale, a finished run is an arpeggio
rising through the seventh chord into the tonic under a pad swell, and a
failed run is a slower falling figure over the minor fourth.

This module is Qt-free on purpose. It writes 16-bit PCM WAV files, which
is all :class:`PySide6.QtMultimedia.QSoundEffect` needs, into a cache
under the spaCR data directory (:func:`sound_cache_root`), one folder per
theme and fingerprint so a changed theme or a new synthesiser version can
never be answered from a stale file. :mod:`spacr.qt.sound` renders
through :func:`ensure_rendered` on its own audio thread and never on the
GUI thread.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BED",
    "CACHE_ENV",
    "CLICK_VARIANTS",
    "DEFAULT_THEME",
    "EVENTS",
    "FEEDBACK_EVENTS",
    "HOVER_VARIANTS",
    "NATURAL_MINOR",
    "ORBIT",
    "Rendered",
    "SAMPLE_RATE",
    "SOUND_THEMES",
    "SYNTH_VERSION",
    "SoundTheme",
    "chord_tones",
    "ensure_rendered",
    "midi_to_hz",
    "read_wav",
    "render",
    "scale_note",
    "sound_cache_root",
    "sound_names",
    "theme_cache_dir",
    "theme_fingerprint",
    "write_wav",
]

#: Samples per second of every file written. 48 kHz is what PipeWire,
#: PulseAudio, CoreAudio and WASAPI run at by default, so the audio device
#: plays the file without resampling it.
SAMPLE_RATE = 48000

#: Raised whenever a change here alters what a theme sounds like. It is part
#: of every cache fingerprint, so raising it retires every cached file.
SYNTH_VERSION = 1

#: Environment variable that moves the cache. Tests and probes point it at
#: a scratch folder so they never write into a real ``~/.spacr``.
CACHE_ENV = "SPACR_SOUND_CACHE"

#: The natural minor (Aeolian) scale, in semitones above the tonic.
NATURAL_MINOR: Tuple[int, ...] = (0, 2, 3, 5, 7, 8, 10)

#: How many different in-key plucks a click, and a hover, rotates through.
#: Successive clicks walk a short phrase rather than repeating one note.
CLICK_VARIANTS = 4
HOVER_VARIANTS = 4

#: The events a sound set answers, in the order Preferences lists them.
FEEDBACK_EVENTS: Tuple[str, ...] = ("click", "hover", "run_finished",
                                    "run_failed")
#: The looping music bed. Rendered only when somebody asks for it.
BED = "bed"
EVENTS: Tuple[str, ...] = FEEDBACK_EVENTS + (BED,)


@dataclass(frozen=True)
class SoundTheme:
    """Everything that decides how one sound set sounds.

    :param key: stable identifier, used in the cache path and the stored
        preference.
    :param label: name shown in Preferences.
    :param description: one sentence shown as the choice's explanation.
    :param tonic: MIDI note of the key's tonic, in the octave the pads sit
        in (57 is A3).
    :param scale: semitones above the tonic of each scale degree.
    :param tempo: beats per minute; sets the arpeggio rate and the delay.
    :param progression: one scale degree per bar of the music bed, each
        the root of a diatonic seventh chord (0-based: 0 is the tonic).
    :param pad_voices: detuned sawtooth oscillators per pad note.
    :param pad_detune_cents: spread of those oscillators, either side.
    :param pad_cutoff_hz: low-pass cutoff of the pad at rest; it breathes
        up to roughly twice this.
    :param pad_level: pad level in the mix.
    :param pad_pump: depth of the dip on every beat, the side-chained pad
        of house music without a kick drum to cause it (0 to 1).
    :param pluck_brightness: how slowly the pluck's harmonics fall away
        (0 to 1); higher is brighter.
    :param pluck_decay: time constant of the pluck's fundamental, seconds.
    :param pluck_level: arpeggio level in the mix.
    :param arp_pattern: indexes into the five arpeggio tones of a chord
        (four chord tones and the root an octave up), one per step.
    :param arp_division: arpeggio steps per beat (2 is eighth notes).
    :param delay_beats: echo spacing in beats; 0.75 is the dotted eighth.
    :param delay_feedback: level of each echo relative to the one before.
    :param delay_mix: how much of the echo reaches the output.
    :param sub_level: level of the sine sub under each chord.
    :param space: reverb send (0 to 1).
    :param reverb_seconds: time for the reverb tail to fall 60 dB.
    :param width: stereo spread of the pad voices (0 is mono).
    :param bed_bars: bars in one loop of the music bed.
    :param seed: seeds every random choice, so a theme always renders the
        same samples.
    """

    key: str
    label: str
    description: str
    tonic: int = 57
    scale: Tuple[int, ...] = NATURAL_MINOR
    tempo: float = 122.0
    progression: Tuple[int, ...] = (0, 5, 2, 6)
    pad_voices: int = 5
    pad_detune_cents: float = 16.0
    pad_cutoff_hz: float = 1100.0
    pad_level: float = 0.5
    pad_pump: float = 0.3
    pluck_brightness: float = 0.62
    pluck_decay: float = 0.32
    pluck_level: float = 0.5
    arp_pattern: Tuple[int, ...] = (0, 2, 1, 3, 2, 4, 3, 1)
    arp_division: int = 2
    delay_beats: float = 0.75
    delay_feedback: float = 0.42
    delay_mix: float = 0.38
    sub_level: float = 0.45
    space: float = 0.32
    reverb_seconds: float = 3.2
    width: float = 0.7
    bed_bars: int = 8
    seed: int = 427

    @property
    def beat(self) -> float:
        """Seconds per beat at this theme's tempo."""
        return 60.0 / float(self.tempo)


#: The reference sound set: melodic space house in A minor at 122 BPM.
#: The bed walks i - VI - III - VII (Am7, Fmaj7, Cmaj7, G7), the
#: progression most of the genre is built on.
ORBIT = SoundTheme(
    key="orbit",
    label="Orbit",
    description=("Melodic space house in A minor: warm detuned pads, "
                 "plucked arpeggios with a dotted-eighth echo, a soft sub."),
)

#: Every sound set spaCR can play, by key.
SOUND_THEMES: Dict[str, SoundTheme] = {ORBIT.key: ORBIT}

#: The sound set a fresh install uses.
DEFAULT_THEME = ORBIT.key


@dataclass
class Rendered:
    """One rendered sound and the notes it was made from.

    :param name: the sound's file stem, for example ``"click-2"``.
    :param audio: float samples, shape ``(2, n)``, peak at most 1.
    :param notes: ``(onset seconds, MIDI note, part)`` for every note
        played, in onset order. ``part`` is ``"pluck"``, ``"pad"`` or
        ``"sub"``. The schedule is what a test reads to check a figure
        rises or falls without having to transcribe the audio.
    :param loop: whether the sound is meant to repeat seamlessly.
    """

    name: str
    audio: np.ndarray
    notes: List[Tuple[float, int, str]]
    loop: bool = False


def midi_to_hz(note: float) -> float:
    """Frequency of a MIDI note number in equal temperament, A4 = 440 Hz.

    :param note: MIDI note number (69 is A4).
    :returns: frequency in hertz.
    """
    return 440.0 * 2.0 ** ((float(note) - 69.0) / 12.0)


def scale_note(theme: SoundTheme, degree: int, octave: int = 0) -> int:
    """The MIDI note of a scale degree, any number of octaves away.

    :param theme: supplies the tonic and the scale.
    :param degree: 0-based scale degree; values past the scale wrap into
        the next octave and negative values into the one below.
    :param octave: whole octaves added on top.
    :returns: a MIDI note number.
    """
    size = len(theme.scale)
    extra, step = divmod(int(degree), size)
    return int(theme.tonic + 12 * (extra + octave) + theme.scale[step])


def chord_tones(theme: SoundTheme, degree: int, count: int = 4) -> List[int]:
    """The diatonic chord built in thirds on a scale degree.

    :param theme: supplies the tonic and the scale.
    :param degree: 0-based scale degree of the chord's root.
    :param count: chord tones to stack; 4 is a seventh chord.
    :returns: MIDI notes, root first, ascending.
    """
    return [scale_note(theme, degree + 2 * k) for k in range(count)]


def _rooted(theme: SoundTheme, degree: int, count: int, centre: int) -> List[int]:
    """A chord in root position, its root in the octave nearest ``centre``.

    Stacked thirds keep every neighbouring pair of notes three or four
    semitones apart, so a detuned pad never smears two notes a semitone
    apart into a beat; and moving the whole block in parallel from chord to
    chord is how the pads of this genre move anyway.

    :param theme: supplies the tonic and the scale.
    :param degree: 0-based scale degree of the root.
    :param count: chord tones, 4 for a seventh chord.
    :param centre: MIDI note the root should sit closest to; a tie goes to
        the lower octave.
    :returns: MIDI notes, ascending.
    """
    tones = chord_tones(theme, degree, count)
    root = tones[0]
    shift = math.floor((centre - root) / 12.0 + 0.5)
    if abs(root + 12 * (shift - 1) - centre) <= abs(root + 12 * shift - centre):
        shift -= 1
    return [int(t + 12 * shift) for t in tones]


def _rng(theme: SoundTheme, salt: str) -> np.random.Generator:
    """A generator seeded by the theme and a label, stable across runs."""
    digest = hashlib.sha256(f"{theme.seed}:{theme.key}:{salt}".encode())
    return np.random.default_rng(int.from_bytes(digest.digest()[:8], "little"))


def _saw(freq: float, n: int, phase: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A band-limited sawtooth, corrected at each wrap by PolyBLEP.

    A naive sawtooth aliases audibly at pad pitches; the two-sample
    polynomial correction removes most of that for almost no cost.
    """
    dt = float(freq) / sr
    t = np.mod(phase + dt * np.arange(n, dtype=np.float64), 1.0)
    y = 2.0 * t - 1.0
    low = t < dt
    x = t[low] / dt
    y[low] -= x + x - x * x - 1.0
    high = t > 1.0 - dt
    x = (t[high] - 1.0) / dt
    y[high] -= x * x + x + x + 1.0
    return y


def _lowpass(signal: np.ndarray, cutoff: float, order: int = 2,
             sr: int = SAMPLE_RATE) -> np.ndarray:
    """Butterworth low-pass along the last axis."""
    from scipy.signal import butter, sosfilt

    cutoff = float(min(max(cutoff, 20.0), 0.45 * sr))
    sos = butter(order, cutoff, btype="low", fs=sr, output="sos")
    return sosfilt(sos, signal, axis=-1)


def _highpass(signal: np.ndarray, cutoff: float, order: int = 2,
              sr: int = SAMPLE_RATE) -> np.ndarray:
    """Butterworth high-pass along the last axis."""
    from scipy.signal import butter, sosfilt

    sos = butter(order, float(cutoff), btype="high", fs=sr, output="sos")
    return sosfilt(sos, signal, axis=-1)


def _envelope(n: int, attack: float, release: float, hold: Optional[float] = None,
              sr: int = SAMPLE_RATE) -> np.ndarray:
    """A raised-cosine attack, a flat hold, and a raised-cosine release.

    :param n: length in samples.
    :param attack: seconds to rise from silence.
    :param release: seconds to fall back to silence at the end.
    :param hold: seconds from the start at which the release begins;
        ``None`` releases at the very end.
    """
    env = np.ones(n, dtype=np.float64)
    a = max(1, min(n, int(attack * sr)))
    env[:a] = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, a))
    r = max(1, int(release * sr))
    start = n - r if hold is None else int(hold * sr)
    start = max(0, min(n, start))
    stop = min(n, start + r)
    if stop > start:
        env[start:stop] *= 0.5 + 0.5 * np.cos(
            np.linspace(0.0, math.pi, stop - start))
    env[stop:] = 0.0
    return env


def _pan(mono: np.ndarray, position: float) -> np.ndarray:
    """Equal-power pan of a mono signal; -1 is hard left, +1 hard right."""
    angle = (float(np.clip(position, -1.0, 1.0)) + 1.0) * math.pi / 4.0
    return np.vstack([mono * math.cos(angle), mono * math.sin(angle)])


def _pluck(freq: float, seconds: float, brightness: float, decay: float,
           sr: int = SAMPLE_RATE) -> np.ndarray:
    """A plucked note, built from harmonics that die faster the higher they are.

    That is what a sawtooth through a closing low-pass filter sounds like --
    the pluck of melodic house -- and it is exact in pitch, which a
    Karplus-Strong string at these frequencies is not. The two channels are
    detuned by a few cents so a single note already has width.

    :returns: stereo samples, shape ``(2, n)``.
    """
    n = max(1, int(seconds * sr))
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros((2, n))
    limit = 0.45 * sr
    tilt = float(np.clip(brightness, 0.05, 0.95))
    for channel, cents in ((0, -3.0), (1, 3.0)):
        f0 = freq * 2.0 ** (cents / 1200.0)
        harmonic = 1
        while harmonic * f0 < limit and harmonic <= 32:
            amp = tilt ** (harmonic - 1) / harmonic
            if amp < 2e-4:
                break
            tau = decay / (1.0 + 0.55 * (harmonic - 1))
            span = min(n, int(tau * 9.0 * sr) + 1)
            out[channel, :span] += (
                amp * np.exp(-t[:span] / tau)
                * np.sin(2.0 * math.pi * harmonic * f0 * t[:span]))
            harmonic += 1
    attack = max(1, int(0.0015 * sr))
    out[:, :attack] *= np.linspace(0.0, 1.0, attack)
    tail = max(1, int(0.01 * sr))
    out[:, -tail:] *= np.linspace(1.0, 0.0, tail)
    return out


def _supersaw(freq: float, n: int, voices: int, detune_cents: float,
              width: float, rng: np.random.Generator,
              sr: int = SAMPLE_RATE) -> np.ndarray:
    """Detuned sawtooth voices spread across the stereo field.

    :returns: stereo samples, shape ``(2, n)``, roughly unit RMS per voice.
    """
    voices = max(1, int(voices))
    out = np.zeros((2, n))
    for v in range(voices):
        spread = 0.0 if voices == 1 else 2.0 * v / (voices - 1) - 1.0
        detuned = freq * 2.0 ** (spread * detune_cents / 1200.0)
        wave_ = _saw(detuned, n, float(rng.random()), sr)
        out += _pan(wave_, spread * width)
    return out / math.sqrt(voices)


def _sub(freq: float, n: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A sine sub with a little second harmonic so small speakers hear it."""
    t = np.arange(n, dtype=np.float64) / sr
    tone = np.sin(2.0 * math.pi * freq * t) + 0.18 * np.sin(
        4.0 * math.pi * freq * t)
    return np.vstack([tone, tone])


def _ping_pong(signal: np.ndarray, delay_s: float, feedback: float,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    """Echoes of ``signal`` alternating right and left, each one darker.

    Computed tap by tap rather than as a recursive filter: the delay is
    thousands of samples long, so a handful of shifted, filtered copies is
    both exact and fast.

    :param signal: stereo input, shape ``(2, n)``; echoes are taken from
        its mono sum.
    :returns: the echoes alone, shape ``(2, n + taps * delay)``.
    """
    feedback = float(np.clip(feedback, 0.0, 0.9))
    d = max(1, int(round(delay_s * sr)))
    if feedback <= 0.0:
        return np.zeros((2, signal.shape[1]))
    taps = int(min(10, max(1, math.ceil(math.log(0.02) / math.log(feedback)))))
    n = signal.shape[1]
    out = np.zeros((2, n + taps * d))
    echo = signal.mean(axis=0)
    for k in range(1, taps + 1):
        echo = _lowpass(echo, 4200.0 / (1.0 + 0.25 * k), order=1, sr=sr) * feedback
        channel = 1 if k % 2 else 0
        out[channel, k * d:k * d + n] += echo
    return out


def _reverb_ir(seconds: float, rng: np.random.Generator,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    """A synthetic stereo impulse response: decaying, darkening noise.

    The two channels are independent noise, which is what makes the tail
    wide; the tail darkens as it decays, as a real room's does, by fading
    from a bright to a dark filtered copy.
    """
    n = max(1, int(seconds * sr))
    t = np.arange(n, dtype=np.float64) / sr
    decay = np.exp(-6.9 * t / max(seconds, 0.05))
    noise = rng.standard_normal((2, n))
    bright = _lowpass(noise, 7000.0, sr=sr)
    dark = _lowpass(noise, 1800.0, sr=sr)
    blend = np.clip(t / max(seconds * 0.5, 1e-3), 0.0, 1.0)
    ir = (bright * (1.0 - blend) + dark * blend) * decay
    onset = max(1, int(0.03 * sr))
    ir[:, :onset] *= np.linspace(0.0, 1.0, onset)
    predelay = np.zeros((2, int(0.018 * sr)))
    ir = np.concatenate([predelay, ir], axis=1)
    energy = np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    return ir / np.maximum(energy, 1e-9)


def _convolve(signal: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Convolve each channel of ``signal`` with the same channel of ``ir``."""
    from scipy.signal import fftconvolve

    return np.vstack([fftconvolve(signal[c], ir[c]) for c in range(2)])


def _add(bus: np.ndarray, part: np.ndarray, at: int) -> None:
    """Mix ``part`` into ``bus`` starting at sample ``at``, clipped to fit."""
    if at >= bus.shape[1]:
        return
    stop = min(bus.shape[1], at + part.shape[1])
    bus[:, at:stop] += part[:, :stop - at]


def _pad(length: int, extra: int = 0) -> np.ndarray:
    """An empty stereo bus."""
    return np.zeros((2, int(length) + int(extra)))


def _trim(audio: np.ndarray, floor_db: float = -66.0,
          sr: int = SAMPLE_RATE) -> np.ndarray:
    """Cut the silent end of a one-shot sound and fade its last 20 ms.

    A tail below ``floor_db`` of the peak cannot be heard at any volume the
    slider allows, and every second of it is a second the file takes to
    load and hold in memory.
    """
    peak = float(np.max(np.abs(audio))) or 1.0
    floor = peak * 10.0 ** (floor_db / 20.0)
    loud = np.nonzero(np.max(np.abs(audio), axis=0) > floor)[0]
    end = int(loud[-1]) + 1 if loud.size else audio.shape[1]
    end = min(audio.shape[1], end + int(0.02 * sr))
    audio = audio[:, :end].copy()
    fade = max(1, min(end, int(0.02 * sr)))
    audio[:, -fade:] *= np.linspace(1.0, 0.0, fade)
    return audio


def _level(audio: np.ndarray, peak_db: float, fade_in: bool = True,
           sr: int = SAMPLE_RATE) -> np.ndarray:
    """Round the peaks and set the level, sample by sample.

    ``tanh`` rounds only what is already loud, so the level can be set by
    the peak without one transient deciding it. Everything here is
    memoryless, which is what lets the music bed be levelled AFTER it is
    folded into a loop without putting a seam back in.

    :param fade_in: ramp the first 2 ms up from silence. A one-shot wants
        it; a loop must not have it, or every repeat would dip.
    """
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = np.tanh(1.2 * audio / peak) / math.tanh(1.2)
    target = 10.0 ** (peak_db / 20.0)
    audio = audio * (target / (float(np.max(np.abs(audio))) or 1.0))
    if fade_in:
        fade = max(1, int(0.002 * sr))
        audio[:, :fade] *= np.linspace(0.0, 1.0, fade)
    return audio


def _master(audio: np.ndarray, peak_db: float,
            sr: int = SAMPLE_RATE) -> np.ndarray:
    """High-pass the rumble out of a one-shot sound, then level it."""
    return _level(_highpass(audio, 28.0, sr=sr), peak_db, True, sr)


def _with_space(dry: np.ndarray, theme: SoundTheme, send: float,
                seconds: float, salt: str,
                sr: int = SAMPLE_RATE) -> np.ndarray:
    """``dry`` plus its reverb, the reverb scaled by ``send``."""
    if send <= 0.0:
        return dry
    wet = _convolve(dry, _reverb_ir(seconds, _rng(theme, salt), sr))
    out = _pad(wet.shape[1])
    out[:, :dry.shape[1]] += dry
    return out + send * wet


def _click(theme: SoundTheme, index: int, sr: int = SAMPLE_RATE) -> Rendered:
    """A short in-key pluck with one quiet echo, for a pressed control.

    The four variants are the tonic, third, fifth and second of the key in
    the octave above the pads, so a run of clicks spells a small phrase.
    """
    degree = (0, 2, 4, 1)[index % 4]
    note = scale_note(theme, degree, octave=1)
    dry = _pluck(midi_to_hz(note), 0.30, theme.pluck_brightness * 0.8, 0.06, sr)
    echo = _ping_pong(dry, theme.delay_beats * theme.beat, 0.18, sr)
    bus = _pad(echo.shape[1])
    _add(bus, dry, 0)
    bus += 0.6 * echo
    audio = _with_space(bus, theme, 0.10, 0.9, f"click-{index}", sr)
    audio = _trim(_master(audio, -12.0, sr), sr=sr)
    return Rendered(f"click-{index}", audio, [(0.0, note, "pluck")])


def _hover(theme: SoundTheme, index: int, sr: int = SAMPLE_RATE) -> Rendered:
    """A muted, very short pluck an octave above the click, and quieter.

    No echo: a hover can follow another within a fraction of a second, and
    echoes of echoes are what turn hover sounds into a stream.
    """
    degree = (4, 6, 7, 9)[index % 4]
    note = scale_note(theme, degree, octave=1)
    dry = _pluck(midi_to_hz(note), 0.16, theme.pluck_brightness * 0.5, 0.035, sr)
    audio = _with_space(dry, theme, 0.06, 0.6, f"hover-{index}", sr)
    audio = _trim(_master(audio, -20.0, sr), sr=sr)
    return Rendered(f"hover-{index}", audio, [(0.0, note, "pluck")])


def _pad_chord(theme: SoundTheme, notes: Sequence[int], n: int,
               rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Every note of a chord as a detuned-saw pad, summed, unfiltered."""
    bus = _pad(n)
    for note in notes:
        bus += _supersaw(midi_to_hz(note), n, theme.pad_voices,
                         theme.pad_detune_cents, theme.width, rng, sr)
    return bus / math.sqrt(max(1, len(notes)))


def _swelling_filter(pad: np.ndarray, cutoff: float, opening: np.ndarray,
                     sr: int = SAMPLE_RATE) -> np.ndarray:
    """Cross-fade a dark and an open copy of ``pad`` by ``opening`` (0 to 1).

    A time-varying low-pass without a sample-by-sample loop: two static
    filters, blended, sound like one filter sweeping between them.
    """
    closed = _lowpass(pad, cutoff, order=4, sr=sr)
    opened = _lowpass(pad, cutoff * 2.0, order=4, sr=sr)
    return closed * (1.0 - opening) + opened * opening


def _run_finished(theme: SoundTheme, sr: int = SAMPLE_RATE) -> Rendered:
    """A rising arpeggio through the seventh-degree chord into the tonic.

    Sixteenth notes climb the VII triad (G B D in A minor) for two octaves
    and step up a whole tone onto the tonic, where they stop; under them a
    pad swells from the VII chord into the tonic chord with its ninth, and
    a sub enters on the resolution. VII to i is the cadence this genre
    resolves with.
    """
    rng = _rng(theme, "run_finished")
    step = theme.beat / 4.0
    climb = [scale_note(theme, 6 + 2 * k, octave=-1) for k in range(3)]
    climb = climb + [n + 12 for n in climb] + [climb[0] + 24]
    tonic = int(theme.tonic)
    while tonic <= climb[-1]:
        tonic += 12
    arrival = step * len(climb)
    length = arrival + 2.4
    n = int(length * sr)

    plucks = _pad(n)
    notes: List[Tuple[float, int, str]] = []
    for i, note in enumerate(climb):
        at = i * step
        velocity = 0.55 + 0.45 * i / max(1, len(climb) - 1)
        plucks_note = _pluck(midi_to_hz(note), 0.9, theme.pluck_brightness,
                             theme.pluck_decay * 0.6, sr) * velocity
        _add(plucks, plucks_note, int(at * sr))
        notes.append((at, note, "pluck"))
    final = _pluck(midi_to_hz(tonic), 2.2, theme.pluck_brightness,
                   theme.pluck_decay * 1.8, sr)
    _add(plucks, final, int(arrival * sr))
    notes.append((arrival, tonic, "pluck"))
    body = _pluck(midi_to_hz(tonic - 12), 1.6, theme.pluck_brightness * 0.7,
                  theme.pluck_decay * 1.4, sr) * 0.45
    _add(plucks, body, int(arrival * sr))
    notes.append((arrival, tonic - 12, "pluck"))
    echoes = _ping_pong(plucks, theme.delay_beats * theme.beat,
                        theme.delay_feedback, sr)

    before = [scale_note(theme, d, octave=-1) for d in (6, 8, 10)]
    after = [scale_note(theme, d) for d in (0, 2, 4, 8)]
    a_n = int(arrival * sr)
    pad = _pad(n)
    first = _pad_chord(theme, before, a_n + int(0.25 * sr), rng, sr)
    first *= _envelope(first.shape[1], arrival * 0.9, 0.3, sr=sr) * 0.55
    _add(pad, first, 0)
    second = _pad_chord(theme, after, n - a_n, rng, sr)
    second *= _envelope(second.shape[1], 0.35, 1.5, sr=sr)
    _add(pad, second, a_n)
    opening = np.clip(np.arange(n) / max(1, a_n + int(0.4 * sr)), 0.0, 1.0)
    pad = _swelling_filter(pad, theme.pad_cutoff_hz, opening, sr)
    for note in before:
        notes.append((0.0, note, "pad"))
    for note in after:
        notes.append((arrival, note, "pad"))

    sub_note = theme.tonic - 24
    sub = _sub(midi_to_hz(sub_note), n - a_n, sr)
    sub *= _envelope(sub.shape[1], 0.08, 1.4, sr=sr)
    notes.append((arrival, sub_note, "sub"))

    bus = _pad(echoes.shape[1])
    _add(bus, plucks * theme.pluck_level, 0)
    bus += echoes * theme.pluck_level * theme.delay_mix * 1.6
    _add(bus, pad * theme.pad_level * 0.8, 0)
    wet_send = _with_space(bus, theme, theme.space, theme.reverb_seconds * 0.8,
                           "run_finished-space", sr)
    _add(wet_send, sub * theme.sub_level * 0.7, a_n)
    audio = _trim(_master(wet_send, -6.0, sr), sr=sr)
    notes.sort(key=lambda row: (row[0], row[2], row[1]))
    return Rendered("run_finished", audio, notes)


def _run_failed(theme: SoundTheme, sr: int = SAMPLE_RATE) -> Rendered:
    """A slower figure falling from the fifth to the tonic over the iv chord.

    Eighth notes step down five, four, three, one of the key (E D C A in
    A minor), darker and softer than the finished sound, over a closed
    minor-fourth pad (Dm7) that never resolves. It says "stopped" without
    an alarm in it.
    """
    rng = _rng(theme, "run_failed")
    step = theme.beat / 2.0
    fall = [scale_note(theme, d, octave=1) for d in (4, 3, 2, 0)]
    length = step * len(fall) + 2.2
    n = int(length * sr)
    plucks = _pad(n)
    notes: List[Tuple[float, int, str]] = []
    for i, note in enumerate(fall):
        at = i * step
        last = i == len(fall) - 1
        velocity = 0.9 - 0.12 * i
        note_audio = _pluck(midi_to_hz(note), 1.8 if last else 0.8,
                            theme.pluck_brightness * 0.6,
                            theme.pluck_decay * (1.4 if last else 0.8), sr)
        _add(plucks, note_audio * velocity, int(at * sr))
        notes.append((at, note, "pluck"))
    echoes = _ping_pong(plucks, theme.delay_beats * theme.beat,
                        theme.delay_feedback * 0.7, sr)

    chord = _rooted(theme, 3, 4, theme.tonic)
    pad = _pad_chord(theme, chord, n, rng, sr)
    pad *= _envelope(n, 0.5, 1.6, sr=sr)
    pad = _lowpass(pad, theme.pad_cutoff_hz * 0.7, order=4, sr=sr)
    for note in chord:
        notes.append((0.0, note, "pad"))
    sub_note = scale_note(theme, 3, octave=-2)
    sub = _sub(midi_to_hz(sub_note), n, sr) * _envelope(n, 0.3, 1.2, sr=sr)
    notes.append((0.0, sub_note, "sub"))

    bus = _pad(echoes.shape[1])
    _add(bus, plucks * theme.pluck_level, 0)
    bus += echoes * theme.pluck_level * theme.delay_mix * 1.2
    _add(bus, pad * theme.pad_level * 0.7, 0)
    audio = _with_space(bus, theme, theme.space, theme.reverb_seconds * 0.8,
                        "run_failed-space", sr)
    _add(audio, sub * theme.sub_level * 0.6, 0)
    audio = _trim(_master(audio, -7.0, sr), sr=sr)
    notes.sort(key=lambda row: (row[0], row[2], row[1]))
    return Rendered("run_failed", audio, notes)


def _pump(n: int, beat: float, depth: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """The side-chain dip: down on every beat, back up over half a beat."""
    t = np.arange(n, dtype=np.float64) / sr
    phase = np.mod(t / beat, 1.0)
    recovery = np.clip(phase / 0.5, 0.0, 1.0)
    shape = 0.5 - 0.5 * np.cos(math.pi * recovery)
    return 1.0 - float(np.clip(depth, 0.0, 1.0)) * (1.0 - shape)


def _fold(audio: np.ndarray, length: int) -> np.ndarray:
    """Wrap everything past ``length`` back onto the start.

    That is what makes the bed loop without a seam: the reverb and the
    echoes still ringing at the end of bar eight are exactly what should be
    sounding under bar one the second time round.
    """
    out = audio[:, :length].copy()
    at = length
    while at < audio.shape[1]:
        chunk = audio[:, at:at + length]
        out[:, :chunk.shape[1]] += chunk
        at += length
    return out


def _bed(theme: SoundTheme, sr: int = SAMPLE_RATE) -> Rendered:
    """The looping music bed: pads, an arpeggio with delay, and a sub.

    One chord per bar from ``theme.progression``. The pad pumps on every
    beat and its filter breathes over four bars; the arpeggio plays
    ``theme.arp_pattern`` over the chord an octave above the pads, lifted a
    further octave on its last notes of each half so the loop has a shape;
    the sub holds each chord's root two octaves down. No drums: this plays
    under somebody's work, not over it.
    """
    rng = _rng(theme, "bed")
    beat = theme.beat
    bar = 4.0 * beat
    bars = max(1, int(theme.bed_bars))
    length = int(round(bars * bar * sr))
    tail = int((theme.reverb_seconds + 2.0) * sr)
    pads = _pad(length, tail)
    plucks = _pad(length, tail)
    subs = _pad(length, tail)
    notes: List[Tuple[float, int, str]] = []
    pluck_cache: Dict[int, np.ndarray] = {}
    step = beat / max(1, int(theme.arp_division))
    steps_per_bar = int(round(bar / step))
    pattern = theme.arp_pattern or (0,)
    for b in range(bars):
        degree = theme.progression[b % len(theme.progression)]
        start = b * bar
        chord = _rooted(theme, degree, 4, theme.tonic + 2)
        span = int((bar + 0.35) * sr)
        chord_audio = _pad_chord(theme, chord, span, rng, sr)
        chord_audio *= _envelope(span, 0.25, 0.45, sr=sr)
        _add(pads, chord_audio, int(start * sr))
        for note in chord:
            notes.append((start, note, "pad"))

        tones = [note + 12 for note in chord]
        tones = tones + [tones[0] + 12]
        lift = 12 if b % 4 == 3 else 0
        for s in range(steps_per_bar):
            at = start + s * step
            index = pattern[(b * steps_per_bar + s) % len(pattern)]
            note = tones[int(index) % len(tones)]
            if lift and s >= steps_per_bar // 2:
                note += lift
            accent = 1.0 if s % max(1, int(theme.arp_division)) == 0 else 0.72
            accent *= 1.0 + 0.08 * (float(rng.random()) - 0.5)
            if note not in pluck_cache:
                pluck_cache[note] = _pluck(
                    midi_to_hz(note), theme.pluck_decay * 6.0,
                    theme.pluck_brightness, theme.pluck_decay, sr)
            _add(plucks, pluck_cache[note] * accent, int(at * sr))
            notes.append((at, note, "pluck"))

        root = scale_note(theme, degree, octave=-2)
        while root > theme.tonic - 17:
            root -= 12
        sub_span = int((bar + 0.1) * sr)
        sub_audio = _sub(midi_to_hz(root), sub_span, sr)
        sub_audio *= _envelope(sub_span, 0.06, 0.12, sr=sr)
        _add(subs, sub_audio, int(start * sr))
        notes.append((start, root, "sub"))

    total = pads.shape[1]
    t = np.arange(total, dtype=np.float64) / sr
    breathing = 0.5 - 0.5 * np.cos(2.0 * math.pi * t / (4.0 * bar))
    pads = _swelling_filter(pads, theme.pad_cutoff_hz, breathing * 0.8, sr)
    pump = _pump(total, beat, theme.pad_pump, sr)
    pads *= pump
    subs *= 0.5 + 0.5 * pump

    echoes = _ping_pong(plucks, theme.delay_beats * beat, theme.delay_feedback, sr)
    bus = _pad(echoes.shape[1])
    _add(bus, pads * theme.pad_level, 0)
    _add(bus, plucks * theme.pluck_level * 0.55, 0)
    bus += echoes * theme.pluck_level * 0.55 * theme.delay_mix * 1.6
    wet = _with_space(bus, theme, theme.space, theme.reverb_seconds, "bed-space",
                      sr)
    _add(wet, subs * theme.sub_level * 0.6, 0)
    looped = _fold(_highpass(wet, 28.0, sr=sr), length)
    audio = _level(looped, -13.0, fade_in=False, sr=sr)
    notes.sort(key=lambda row: (row[0], row[2], row[1]))
    return Rendered(BED, audio, notes, loop=True)


def sound_names(event: str) -> List[str]:
    """The file stems an event plays, in the order they rotate.

    :param event: one of :data:`EVENTS`.
    :returns: for example ``["click-0", ..., "click-3"]`` for a click.
    :raises KeyError: for an event this module does not know.
    """
    if event == "click":
        return [f"click-{i}" for i in range(CLICK_VARIANTS)]
    if event == "hover":
        return [f"hover-{i}" for i in range(HOVER_VARIANTS)]
    if event in ("run_finished", "run_failed", BED):
        return [event]
    raise KeyError(event)


def render(theme: SoundTheme, name: str, sr: int = SAMPLE_RATE) -> Rendered:
    """Synthesize one named sound of a theme.

    :param theme: the sound set.
    :param name: a stem from :func:`sound_names`.
    :param sr: sample rate.
    :returns: the rendered audio and its note schedule.
    :raises KeyError: for a name no event plays.
    """
    if name.startswith("click-"):
        return _click(theme, int(name.split("-", 1)[1]), sr)
    if name.startswith("hover-"):
        return _hover(theme, int(name.split("-", 1)[1]), sr)
    if name == "run_finished":
        return _run_finished(theme, sr)
    if name == "run_failed":
        return _run_failed(theme, sr)
    if name == BED:
        return _bed(theme, sr)
    raise KeyError(name)


def write_wav(path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    """Write stereo float samples as 16-bit PCM, atomically.

    The file appears under its final name only once it is complete, so a
    player that looks while it is being written sees the old file or none,
    never half of one.

    :param path: destination.
    :param audio: shape ``(2, n)``, values in -1 to 1.
    :param sr: sample rate.
    :returns: the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(audio, dtype=np.float64), -1.0, 1.0)
    frames = np.round(pcm.T * 32767.0).astype("<i2").tobytes()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(int(pcm.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(int(sr))
        handle.writeframes(frames)
    os.replace(temporary, path)
    return path


def read_wav(path) -> Tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV written by :func:`write_wav`.

    :param path: the file.
    :returns: ``(audio, sample rate)``, audio shaped ``(channels, n)`` in
        -1 to 1.
    """
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32767.0
    return data.reshape(-1, channels).T, rate


def sound_cache_root() -> Path:
    """Where rendered sounds are kept: ``~/.spacr/sounds`` by default.

    :data:`CACHE_ENV` overrides it. Read on every call, so a test that sets
    the variable is honoured without any reload.
    """
    override = os.environ.get(CACHE_ENV, "").strip()
    if override:
        return Path(override)
    return Path.home() / ".spacr" / "sounds"


def theme_fingerprint(theme: SoundTheme, sr: int = SAMPLE_RATE) -> str:
    """A short hash of everything that decides what a theme's files hold.

    :param theme: the sound set.
    :param sr: sample rate.
    :returns: twelve hex digits.
    """
    payload = json.dumps({"synth": SYNTH_VERSION, "rate": int(sr),
                          "theme": dataclasses.asdict(theme)},
                         sort_keys=True, default=list)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def theme_cache_dir(theme: SoundTheme, root: Optional[Path] = None,
                    sr: int = SAMPLE_RATE) -> Path:
    """The folder a theme's files live in, named by key and fingerprint.

    :param theme: the sound set.
    :param root: cache root; :func:`sound_cache_root` when omitted.
    :param sr: sample rate.
    :returns: the folder (not created here).
    """
    base = Path(root) if root is not None else sound_cache_root()
    return base / f"{theme.key}-{theme_fingerprint(theme, sr)}"


def _forget_other_versions(theme: SoundTheme, keep: Path) -> int:
    """Delete this theme's folders from earlier fingerprints.

    Only folders named ``<key>-<twelve hex digits>`` beside ``keep`` are
    touched, so nothing that is not a sound cache can be removed.
    """
    removed = 0
    prefix = f"{theme.key}-"
    try:
        siblings = list(keep.parent.iterdir())
    except OSError:
        return 0
    for folder in siblings:
        name = folder.name
        tail = name[len(prefix):]
        if (folder == keep or not folder.is_dir()
                or not name.startswith(prefix) or len(tail) != 12
                or any(ch not in "0123456789abcdef" for ch in tail)):
            continue
        shutil.rmtree(folder, ignore_errors=True)
        removed += 1
    return removed


def ensure_rendered(theme: SoundTheme, names: Iterable[str],
                    root: Optional[Path] = None,
                    sr: int = SAMPLE_RATE,
                    should_stop: Optional[Callable[[], bool]] = None,
                    ) -> Dict[str, Path]:
    """Make sure each named sound exists on disk, rendering what is missing.

    A file already in the theme's folder is trusted: the folder name is the
    fingerprint of everything that decides its contents. Folders left by an
    earlier fingerprint of the same theme are removed once the current one
    is in use.

    :param theme: the sound set.
    :param names: stems from :func:`sound_names`.
    :param root: cache root; :func:`sound_cache_root` when omitted.
    :param sr: sample rate.
    :param should_stop: polled between sounds; returning True abandons the
        rest, so a render can be cancelled at the next file boundary.
    :returns: stem -> path, for every file that exists afterwards.
    """
    folder = theme_cache_dir(theme, root, sr)
    ready: Dict[str, Path] = {}
    wrote = False
    for name in names:
        path = folder / f"{name}.wav"
        if path.is_file():
            ready[name] = path
            continue
        if should_stop is not None and should_stop():
            break
        write_wav(path, render(theme, name, sr).audio, sr)
        ready[name] = path
        wrote = True
    if wrote:
        _forget_other_versions(theme, folder)
    return ready
