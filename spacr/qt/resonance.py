"""What the Resonance backdrop is driven by, and the plate it draws.

Two halves, both Qt-free and both numpy-only, kept together because they
are the two ends of one idea: a sound is measured here, and a vibrating
plate is solved here, and the backdrop in
:class:`spacr.qt.widgets.ambient.ResonanceEngine` is only the painting.

**The measurement.** :func:`analyse` turns a rendered WAV into a few
hundred kilobytes of envelope, four band energies, a spectral centroid
and an onset strength, sixty times a second, and :func:`ensure_analysis`
keeps that beside the WAV so it is computed once and read thereafter.
:mod:`spacr.qt.sound` does the computing on its own audio thread when it
loads the music bed, and records what is playing and when it started
(:func:`set_now_playing`); the backdrop reads the small arrays and the
clock. **Nothing here listens to the machine.** There is no system-audio
capture anywhere in spaCR and there is not going to be: the only sounds
this can see are the ones spaCR is playing and the file the user chose.

**The plate.** Chladni figures are what sand does on a plate driven at one
of its resonances: it is thrown off the parts that move and comes to rest
along the lines that do not, so the figure is the nodal set of a standing
wave made visible. For a square plate the classical superposition is

.. math::

    w(x, y) = \\cos(n \\pi x)\\cos(m \\pi y) \\pm \\cos(m \\pi x)\\cos(n \\pi y)

and the sand settles where :math:`w = 0`. Both signs are real families of
figures and :data:`MODES` alternates them; see there for what leaving the
plus sign out looked like. :func:`lattice` evaluates that
field and its gradient on a small grid, and :func:`settle` walks particles
down it. Both are separable in x and y, so a whole lattice costs four
outer products and a frame costs a handful of array lookups -- which is
the only reason a particle field can afford to be a backdrop at all.

WHY A LATTICE AND NOT AN EVALUATION PER PARTICLE. Eight relaxation rounds
over 1 300 particles is 10 400 evaluations of eight trigonometric
functions each if the field is evaluated where the particle is; on a
128x128 lattice it is four outer products, once, and then 10 400 integer
lookups. Measured on this machine, eight rounds over 1 300 particles:
0.61 ms with the lattice against 0.76 ms without it, which is a modest
1.26x -- the point is not the ratio today but that the FIELD's share stops
depending on the particle count at all, so density can be raised without
the trigonometry following it.

AND WHY THE FIGURE IS RECOMPUTED FROM A FIXED START EVERY FRAME RATHER
THAN CARRIED FORWARD. Every ambient engine promises that a frame is a pure
function of ``(seed, clock, size)`` -- ``test_the_clock_alone_decides_the
_frame`` steps one engine twelve times and jumps another straight to the
same clock and demands the same picture, and
``test_the_backdrop_survives_a_run`` shades the same clock on two threads
and compares the bytes. A particle simulation carried forward from frame
to frame satisfies neither. Relaxation from a seeded start does, costs
0.61 ms, and has the better behaviour besides: the figure always belongs
to the sound that is playing now rather than to the last few seconds of
it.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

__all__ = [
    "ANALYSIS_FPS",
    "ANALYSIS_SUFFIX",
    "ANALYSIS_VERSION",
    "BANDS",
    "MODES",
    "Moment",
    "NowPlaying",
    "Resonance",
    "analyse",
    "analysis_path_for",
    "clear_now_playing",
    "ensure_analysis",
    "lattice",
    "load_analysis",
    "mode_blend",
    "now_playing",
    "playing_moment",
    "read_wav_mono",
    "set_now_playing",
    "settle",
    "ONSET_RELEASE_S",
    "sample",
    "silence",
]

#: Frames of analysis per second of audio. Sixty is above every frame rate
#: a backdrop runs at (the cap is 24), so the visualiser interpolates
#: between neighbouring frames rather than holding one for several of its
#: own -- and the whole 32-bar bed is still only about 105 kB.
ANALYSIS_FPS = 60.0

#: Raised when a change here alters what :func:`analyse` produces. It is
#: stored in the file, and a file that does not carry the current version
#: is recomputed rather than read.
ANALYSIS_VERSION = 1

#: What :func:`analysis_path_for` appends to a WAV's name.
ANALYSIS_SUFFIX = ".resonance.npz"

#: The four bands the visualiser separates, in hertz: the sub, the body,
#: the melody and the air. Chosen so the four parts of the music bed land
#: in four different ones -- the sub and kick below 120, the pads and the
#: chords up to 500, the arpeggio and its echoes above that, the shaker on
#: its own at the top -- so a band is a PART and not merely a slice.
BANDS: Tuple[Tuple[float, float], ...] = ((20.0, 120.0), (120.0, 500.0),
                                          (500.0, 2000.0), (2000.0, 16000.0))

#: Longest a single relaxation step may be, as a fraction of the plate.
#:
#: A Newton step is ``w * grad(w) / |grad(w)|^2`` and the denominator goes
#: to zero at an antinode, where the field is flat and its gradient has
#: nothing to say about where the nodal line is. Unclamped, the particles
#: that start there are flung off the plate and pile up on the rim: the
#: first rendering of this theme drew a faint rectangle around the figure
#: that no mode explained. Clamping the step keeps them wandering in the
#: middle instead, which is where sand on a flat part of a plate is.
_MAX_STEP = 0.11

#: Window length of the analysis, in samples at 48 kHz: 43 ms, which
#: resolves a kick from the pad under it and is still short enough that a
#: sixteenth note at 122 BPM is its own frame.
_WINDOW = 2048

#: The drives the plate is put through, as ``(m, n, sign)``, in order of
#: how busy the figure they make is. A visualiser that jumped between them
#: would read as a slideshow, so :func:`mode_blend` crossfades along this
#: list and the figure MORPHS: the nodal set of a blend of two modes is a
#: continuous deformation of both, which is why a blend is the right thing
#: to interpolate rather than the particle positions.
#:
#: THE SIGN IS NOT DECORATION AND LEAVING IT OUT COST A WHOLE FAMILY OF
#: FIGURES. With the minus sign the field vanishes identically along
#: ``y = x`` -- swap x and y and the two terms exchange -- so every single
#: figure carries the same diagonal, which is exactly what the first
#: rendering of this theme looked like. The plus sign has no such line and
#: draws the closed lens-and-ring figures the plus family is known for.
#: Alternating them along the list doubles the variety for one
#: multiplication.
#:
#: The pairs are all ``m > n`` and share no factor, because ``(4, 2)`` and
#: ``(2, 1)`` differ by a scale factor and draw nearly the same figure at
#: different sizes; the list is the ones that look different.
MODES: Tuple[Tuple[int, int, int], ...] = (
    (2, 1, -1), (2, 1, 1), (3, 1, -1), (3, 2, 1), (4, 1, -1), (4, 3, 1),
    (5, 2, -1), (5, 3, 1), (6, 1, -1), (6, 5, 1), (7, 4, -1), (8, 3, 1),
    (9, 5, -1), (7, 2, 1))


class Moment(NamedTuple):
    """One instant of whatever is playing, all of it 0 to 1.

    :param level: overall loudness against the loop's own loudest.
    :param bands: energy in each of :data:`BANDS`, each against its own
        loudest, so a band that is always quiet still says something.
    :param centroid: where the spectrum's centre of mass is, on a log
        scale between 60 Hz and 8 kHz.
    :param onset: spectral flux -- how much of the sound is NEW. This is
        what a beat looks like when it is measured rather than guessed at
        from a tempo the visualiser was told.
    """

    level: float
    bands: Tuple[float, float, float, float]
    centroid: float
    onset: float


def silence() -> Moment:
    """The moment that stands for nothing playing.

    Not zeros: a plate at rest still has a plate, and the backdrop has to
    idle beautifully rather than go out. The engine reads this as "breathe
    slowly on your own clock", which is what
    :data:`spacr.qt.widgets.ambient.RESONANCE_IDLE` is for.

    :returns: an all-zero moment.
    """
    return Moment(0.0, (0.0, 0.0, 0.0, 0.0), 0.0, 0.0)


@dataclass(frozen=True)
class Resonance:
    """Everything the visualiser knows about one piece of audio.

    :param fps: analysis frames per second.
    :param duration: length of the audio in seconds.
    :param level: per-frame loudness, 0 to 1.
    :param bands: ``(4, frames)``, each row 0 to 1.
    :param centroid: per-frame spectral centroid, 0 to 1.
    :param onset: per-frame spectral flux, 0 to 1.
    :param loop: whether the audio repeats seamlessly, which decides
        whether :meth:`at` wraps or clamps.
    """

    fps: float
    duration: float
    level: np.ndarray
    bands: np.ndarray
    centroid: np.ndarray
    onset: np.ndarray
    loop: bool = True

    @property
    def frames(self) -> int:
        """How many analysis frames there are."""
        return int(self.level.size)

    def at(self, seconds: float) -> Moment:
        """The moment at ``seconds`` into the audio.

        Linear between neighbouring frames, so a 24 fps backdrop reading a
        60 fps analysis moves smoothly instead of stepping. A looping piece
        wraps -- including BETWEEN its last frame and its first, which is
        the one interpolation a naive clamp gets wrong and the one a
        listener would hear as a stall every time round.

        :param seconds: position in the audio.
        :returns: the interpolated moment.
        """
        count = self.frames
        if count == 0:
            return silence()
        position = float(seconds) * self.fps
        if self.loop:
            position = math.fmod(position, count)
            if position < 0.0:
                position += count
            low = int(position)
            high = (low + 1) % count
        else:
            position = min(max(position, 0.0), count - 1.0)
            low = int(position)
            high = min(low + 1, count - 1)
        blend = position - int(position)
        return self._mix(low, high, blend)

    def _mix(self, low: int, high: int, blend: float) -> Moment:
        """Interpolate between two analysis frames."""
        keep = 1.0 - blend
        bands = self.bands[:, low] * keep + self.bands[:, high] * blend
        return Moment(
            float(self.level[low] * keep + self.level[high] * blend),
            (float(bands[0]), float(bands[1]), float(bands[2]),
             float(bands[3])),
            float(self.centroid[low] * keep + self.centroid[high] * blend),
            float(self.onset[low] * keep + self.onset[high] * blend),
        )


def read_wav_mono(path) -> Tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV down to one channel.

    Deliberately NOT :func:`spacr.qt.sound_synth.read_wav`: that module
    imports scipy, and this one is reached from the backdrop's shading
    thread by somebody who may have sound switched off entirely. Reading a
    WAV is the standard library and numpy, and that is all this needs.

    :param path: the file.
    :returns: ``(mono samples in -1..1, sample rate)``.
    :raises ValueError: for anything that is not 16-bit PCM.
    """
    with wave.open(str(path), "rb") as handle:
        channels = max(1, handle.getnchannels())
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"{path} is {width * 8}-bit; only 16-bit PCM is read")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    usable = (data.size // channels) * channels
    return data[:usable].reshape(-1, channels).mean(axis=1), int(rate)


def _windows(mono: np.ndarray, hop: float, first: int, count: int,
             loop: bool) -> np.ndarray:
    """Analysis windows ``first`` to ``first + count``, as one array.

    A looping piece takes its windows MODULO the length, so the frames
    that straddle the loop point see the start of the piece rather than
    silence -- the analysis is then as circular as the audio, and the
    backdrop does not dip once a minute.
    """
    starts = ((np.arange(count, dtype=np.float64) + first)
              * hop).astype(np.int64)
    offsets = np.arange(_WINDOW, dtype=np.int64)
    index = starts[:, None] + offsets[None, :]
    if loop:
        index %= mono.size
    else:
        index = np.clip(index, 0, mono.size - 1)
    return mono[index]


def analyse(mono: np.ndarray, sr: int, fps: float = ANALYSIS_FPS,
            loop: bool = True) -> Resonance:
    """Measure what a visualiser needs, frame by frame.

    Every row is normalised against its OWN 98th percentile rather than
    against the loudest thing in the file. A shaker that never rises above
    -40 dB is still the loudest thing in its own band, and a band scaled by
    the kick's peak would be a row of zeros -- which is how an audio
    visualiser ends up with three bars that move and one that does not.

    :param mono: one channel of samples.
    :param sr: sample rate.
    :param fps: analysis frames per second.
    :param loop: whether the audio repeats seamlessly.
    :returns: the measurement.
    """
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    if mono.size < 2:
        empty = np.zeros(1, dtype=np.float32)
        return Resonance(float(fps), 0.0, empty, np.zeros((4, 1), np.float32),
                         empty.copy(), empty.copy(), bool(loop))
    duration = mono.size / float(sr)
    hop = float(sr) / float(fps)
    count = max(1, int(round(duration * fps)))
    window = np.hanning(_WINDOW).astype(np.float32)
    freqs = np.fft.rfftfreq(_WINDOW, 1.0 / sr)

    level = np.zeros(count, dtype=np.float32)
    bands = np.zeros((len(BANDS), count), dtype=np.float32)
    centroid = np.zeros(count, dtype=np.float32)
    flux = np.zeros(count, dtype=np.float32)
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in BANDS]
    tone = np.clip(freqs, 20.0, 20000.0)
    previous: Optional[np.ndarray] = None
    opening: Optional[np.ndarray] = None
    step = 256
    for start in range(0, count, step):
        stop = min(count, start + step)
        block = _windows(mono, hop, start, stop - start, loop)
        spectrum = np.abs(np.fft.rfft(block * window,
                                      axis=1)).astype(np.float32)
        level[start:stop] = np.sqrt((block ** 2).mean(axis=1))
        for index, mask in enumerate(masks):
            bands[index, start:stop] = spectrum[:, mask].sum(axis=1)
        weight = spectrum.sum(axis=1)
        centroid[start:stop] = (spectrum * tone).sum(axis=1) \
            / np.maximum(weight, 1e-9)
        before = spectrum[0] if previous is None else previous
        rising = np.diff(np.vstack([before[None, :], spectrum]), axis=0)
        flux[start:stop] = np.maximum(rising, 0.0).sum(axis=1)
        if opening is None:
            opening = spectrum[0]
        previous = spectrum[-1]
    if loop and count > 1 and opening is not None and previous is not None:
        flux[0] = float(np.maximum(opening - previous, 0.0).sum())

    low, high = 60.0, 8000.0
    centroid = np.clip(np.log2(np.maximum(centroid, low) / low)
                       / math.log2(high / low), 0.0, 1.0)
    return Resonance(float(fps), duration, _unit(level),
                     np.vstack([_unit(row) for row in bands]),
                     centroid.astype(np.float32),
                     _release(_unit(flux), fps, loop), bool(loop))


#: How long an onset goes on being visible after the transient that caused
#: it, in seconds. A backdrop paints at 24 frames a second at most, so a
#: flux spike one analysis frame wide is a spike the picture would show or
#: miss depending on where the tick fell. Giving it a release turns it into
#: something a slower painter can see EVERY time, which is what makes the
#: figure bounce on the beat rather than flicker near it.
ONSET_RELEASE_S = 0.28


def _release(flux: np.ndarray, fps: float, loop: bool) -> np.ndarray:
    """Let each onset fall away over :data:`ONSET_RELEASE_S` instead of
    vanishing on the next frame.

    Run twice round for a looping piece, so the release that starts on the
    last beat is still falling under the first one.
    """
    out = np.asarray(flux, dtype=np.float32).copy()
    if out.size < 2:
        return out
    keep = float(math.exp(-1.0 / max(1e-6, ONSET_RELEASE_S * fps)))
    for _ in range(2 if loop else 1):
        carry = out[-1] if loop else 0.0
        for index in range(out.size):
            carry = max(float(out[index]), carry * keep)
            out[index] = carry
    return out


def _unit(row: np.ndarray) -> np.ndarray:
    """Scale a row to 0..1 against its own 98th percentile.

    The percentile and not the maximum, so one transient cannot flatten
    the other nine hundred frames; anything above it clips, which is what
    a peak should look like.
    """
    ceiling = float(np.percentile(row, 98.0)) if row.size else 0.0
    if ceiling <= 1e-9:
        return np.zeros_like(row, dtype=np.float32)
    return np.clip(row / ceiling, 0.0, 1.0).astype(np.float32)


def analysis_path_for(wav_path, out_dir=None) -> Path:
    """Where the analysis of ``wav_path`` is kept.

    Beside the file by default, which is right for spaCR's OWN rendered
    bed: it lives in the sound cache already and is swept away with the
    theme folder when the fingerprint changes.

    IT IS WRONG FOR A FILE THE USER CHOSE, and that is what ``out_dir`` is
    for. Writing a sidecar into somebody's music folder is a scientific
    tool leaving litter in a directory it was only asked to read from, and
    the folder may not even be writable. :mod:`spacr.qt.sound` passes the
    sound cache for a chosen file, and the name carries a hash of the
    source's absolute path so two files called ``loop.wav`` in different
    folders are two analyses.

    :param wav_path: the audio file.
    :param out_dir: a folder to keep the analysis in instead of beside it.
    :returns: the ``.resonance.npz`` path.
    """
    path = Path(wav_path)
    if out_dir is None:
        return path.with_name(path.name + ANALYSIS_SUFFIX)
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()
    return Path(out_dir) / f"{path.stem}-{digest[:12]}{ANALYSIS_SUFFIX}"


def ensure_analysis(wav_path, loop: bool = True,
                    out_dir=None) -> Optional[Path]:
    """Analyse ``wav_path`` unless a current analysis is already beside it.

    **Never call this from the GUI thread.** It reads a whole file and runs
    a few thousand FFTs; :mod:`spacr.qt.sound` calls it on the audio thread
    right after it has rendered or loaded the bed, which is where the rest
    of that work already is.

    The stored analysis carries the source's size and modification time, so
    a user who replaces their chosen WAV with a different one of the same
    name is analysed again rather than visualised as the old one.

    :param wav_path: the audio file.
    :param loop: whether the audio repeats seamlessly.
    :param out_dir: where to keep the analysis; see
        :func:`analysis_path_for`.
    :returns: the analysis path, or ``None`` when the audio could not be
        read or the analysis could not be written.
    """
    source = Path(wav_path)
    target = analysis_path_for(source, out_dir)
    try:
        stat = source.stat()
    except OSError:
        return None
    if _is_current(target, stat):
        return target
    try:
        mono, rate = read_wav_mono(source)
        measured = analyse(mono, rate, ANALYSIS_FPS, loop)
    except (OSError, ValueError, wave.Error, EOFError):
        LOG.debug("could not analyse %s", source, exc_info=True)
        return None
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(temporary, "wb") as handle:
            np.savez_compressed(
                handle, version=np.int32(ANALYSIS_VERSION),
                fps=np.float32(measured.fps),
                duration=np.float32(measured.duration),
                loop=np.int32(1 if measured.loop else 0),
                size=np.int64(stat.st_size),
                mtime=np.float64(stat.st_mtime),
                level=measured.level, bands=measured.bands,
                centroid=measured.centroid, onset=measured.onset)
        os.replace(temporary, target)
    except OSError:
        LOG.debug("could not write the analysis of %s", source, exc_info=True)
        try:
            temporary.unlink()
        except OSError:
            pass
        return None
    return target


def _is_current(target: Path, stat) -> bool:
    """Whether ``target`` already analyses a source with exactly this stat."""
    try:
        with np.load(target) as held:
            return (int(held["version"]) == ANALYSIS_VERSION
                    and int(held["size"]) == stat.st_size
                    and abs(float(held["mtime"]) - stat.st_mtime) < 1e-6)
    except (OSError, ValueError, KeyError, EOFError):
        return False


#: Analyses already read, by path, with the modification time they were
#: read at. The backdrop's shading thread asks for one sixty times a
#: minute and must not open a file each time.
_LOADED: Dict[str, Tuple[float, Optional[Resonance]]] = {}
_LOCK = threading.RLock()


def load_analysis(path) -> Optional[Resonance]:
    """Read an analysis written by :func:`ensure_analysis`, once.

    Cached on the path and its modification time, and a file that cannot be
    read is remembered as unreadable so a missing analysis costs one failed
    ``stat`` and not one per frame.

    :param path: the ``.resonance.npz``.
    :returns: the analysis, or ``None``.
    """
    key = str(path)
    try:
        mtime = os.stat(key).st_mtime
    except OSError:
        with _LOCK:
            _LOADED[key] = (0.0, None)
        return None
    with _LOCK:
        held = _LOADED.get(key)
        if held is not None and held[0] == mtime:
            return held[1]
    measured: Optional[Resonance] = None
    try:
        with np.load(key) as raw:
            if int(raw["version"]) == ANALYSIS_VERSION:
                measured = Resonance(
                    float(raw["fps"]), float(raw["duration"]),
                    np.asarray(raw["level"], dtype=np.float32),
                    np.asarray(raw["bands"], dtype=np.float32),
                    np.asarray(raw["centroid"], dtype=np.float32),
                    np.asarray(raw["onset"], dtype=np.float32),
                    bool(int(raw["loop"])))
    except (OSError, ValueError, KeyError, EOFError):
        measured = None
    with _LOCK:
        _LOADED[key] = (mtime, measured)
    return measured


@dataclass(frozen=True)
class NowPlaying:
    """What is coming out of the speakers, and when it started.

    THE CLOCK IS THE WHOLE MECHANISM, and it is worth saying why there is
    no other. ``QSoundEffect`` has no playback position to ask for, and
    capturing the machine's audio output is not something a scientific tool
    should be doing to somebody's computer. So the audio thread records
    the instant it called ``play()`` and the length of the loop, and the
    position is arithmetic. Both clocks are real time, so the two drift
    only by the audio device's own rate error -- tens of parts per million,
    a millisecond an hour -- and the loop wraps it out anyway.

    :param analysis: path to the ``.resonance.npz`` for what is playing.
    :param started: :func:`time.monotonic` at the moment playback began.
    :param duration: length of the piece in seconds.
    :param loop: whether it repeats.
    """

    analysis: str
    started: float
    duration: float
    loop: bool = True


_PLAYING: Optional[NowPlaying] = None
_PLAYING_ANALYSIS: Optional[Resonance] = None


def set_now_playing(record: Optional[NowPlaying]) -> None:
    """Record what is playing, from whichever thread started it.

    THE ANALYSIS IS READ HERE AND NOWHERE ELSE, which is the whole reason
    this is a setter rather than a variable. :func:`playing_moment` is
    called once a frame from the backdrop's clock, which runs on the GUI
    thread; a ``stat`` there is a filesystem call on the GUI thread, and on
    a network home directory that is exactly the stall spaCR has
    :mod:`spacr.qt.path_probe` for. This runs on the audio thread, which
    has just rendered the file and is the right place to open it.

    :param record: the piece, or ``None`` for silence.
    """
    global _PLAYING, _PLAYING_ANALYSIS
    measured = None if record is None else load_analysis(record.analysis)
    with _LOCK:
        _PLAYING = record
        _PLAYING_ANALYSIS = measured


def clear_now_playing() -> None:
    """Nothing is playing any more."""
    set_now_playing(None)


def now_playing() -> Optional[NowPlaying]:
    """What is playing, as last recorded.

    :returns: the record, or ``None``.
    """
    with _LOCK:
        return _PLAYING


def playing_moment(at: Optional[float] = None) -> Moment:
    """The moment of whatever is playing, right now.

    Safe on any thread and cheap enough for a frame: a lock, an arithmetic
    position and two array reads, with NO file access at all -- see
    :func:`set_now_playing`. Returns :func:`silence` whenever nothing is
    playing, the analysis is missing, or the piece has ended, so the caller
    has one code path and the backdrop idles rather than failing.

    :param at: the monotonic clock to read it at; now, by default.
    :returns: the moment.
    """
    with _LOCK:
        record = _PLAYING
        measured = _PLAYING_ANALYSIS
    if record is None or record.duration <= 0.0 or measured is None:
        return silence()
    elapsed = (time.monotonic() if at is None else float(at)) - record.started
    if elapsed < 0.0:
        return silence()
    if not record.loop and elapsed > record.duration:
        return silence()
    return measured.at(elapsed)


def mode_blend(position: float) -> Tuple[Tuple[int, int, int],
                                         Tuple[int, int, int], float]:
    """Which two entries of :data:`MODES` a figure is between.

    :param position: where along the list the figure sits; wrapped, so a
        drive that walks off the end comes back to the simplest figure
        rather than sticking on the busiest.
    :returns: ``(first drive, second drive, 0..1 between them)``.
    """
    count = len(MODES)
    place = math.fmod(float(position), count)
    if place < 0.0:
        place += count
    low = int(place)
    return MODES[low], MODES[(low + 1) % count], place - low


def lattice(first: Tuple[int, int, int], second: Tuple[int, int, int],
            blend: float,
            grid: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The plate's displacement and its gradient, on a ``grid`` x ``grid``.

    Both modes are evaluated and mixed rather than one mode being drawn at
    a time, because the nodal set of a mixture is a continuous deformation
    of the nodal sets of its parts: that is what makes one figure MORPH
    into the next instead of being replaced by it.

    Separable throughout -- every term is a product of one function of x
    and one of y -- so the whole lattice is eight one-dimensional cosines
    and four outer products.

    :param first: ``(m, n, sign)`` of the drive being left.
    :param second: ``(m, n, sign)`` of the drive being arrived at.
    :param blend: 0 at ``first``, 1 at ``second``.
    :param grid: lattice edge.
    :returns: ``(w, dw/dx, dw/dy)``, each ``(grid, grid)`` and indexed
        ``[y, x]``, with ``w`` scaled to -1..1.
    """
    grid = max(8, int(grid))
    axis = np.linspace(0.0, 1.0, grid, dtype=np.float32)
    blend = float(min(max(blend, 0.0), 1.0))
    field = np.zeros((grid, grid), dtype=np.float32)
    dx = np.zeros((grid, grid), dtype=np.float32)
    dy = np.zeros((grid, grid), dtype=np.float32)
    for (m, n, sign), weight in ((first, 1.0 - blend), (second, blend)):
        if weight <= 0.0:
            continue
        cn, sn = np.cos(n * math.pi * axis), np.sin(n * math.pi * axis)
        cm, sm = np.cos(m * math.pi * axis), np.sin(m * math.pi * axis)
        field += weight * (np.outer(cm, cn) + sign * np.outer(cn, cm))
        dx += weight * (-n * math.pi * np.outer(cm, sn)
                        - sign * m * math.pi * np.outer(cn, sm))
        dy += weight * (-m * math.pi * np.outer(sm, cn)
                        - sign * n * math.pi * np.outer(sn, cm))
    return field * 0.5, dx * 0.5, dy * 0.5


def settle(x: np.ndarray, y: np.ndarray, field: np.ndarray,
           dx: np.ndarray, dy: np.ndarray, rounds: int,
           tightness) -> Tuple[np.ndarray, np.ndarray]:
    """Walk particles down ``|w|`` toward the plate's nodal lines.

    Each round is one Newton step onto the linearised zero set --
    ``p -= t * w * grad(w) / |grad(w)|^2`` -- which lands on the nodal line
    in one step where the field is locally straight and converges in a
    handful where it is not. That is what the sand does, and it is cheaper
    than the gradient descent it replaces, which needs a step size that
    depends on the mode numbers.

    ``tightness`` below 1 leaves the particles short of the lines, which is
    what a plate driven gently looks like: the sand gathers but does not
    resolve. It is the control that makes silence read as a loose cloud and
    a loud passage as a figure. It may be one number for every particle or
    ONE PER PARTICLE, and per particle is what the backdrop uses: a plate
    where every grain converges equally draws a wire diagram, and a real
    one has a haze of sand that never quite arrives. The haze is what makes
    the picture look like sand rather than like a plot of an equation.

    :param x: particle x in 0..1. Copied, never modified in place: the
        caller's array is the seeded start it needs again next frame.
    :param y: particle y in 0..1.
    :param field: ``w`` from :func:`lattice`.
    :param dx: ``dw/dx``.
    :param dy: ``dw/dy``.
    :param rounds: how many Newton steps to take.
    :param tightness: fraction of each step to take, 0 to 1; a number or
        one value per particle.
    :returns: ``(x, y)``, new arrays in 0..1.
    """
    grid = field.shape[0]
    px = np.array(x, dtype=np.float32, copy=True)
    py = np.array(y, dtype=np.float32, copy=True)
    step = np.clip(np.asarray(tightness, dtype=np.float32), 0.0, 1.0)
    if rounds <= 0 or not float(step.max() if step.size else 0.0) > 0.0:
        return px, py
    scale = grid - 1
    for _ in range(int(rounds)):
        ix = np.clip((px * scale + 0.5).astype(np.int32), 0, grid - 1)
        iy = np.clip((py * scale + 0.5).astype(np.int32), 0, grid - 1)
        w = field[iy, ix]
        gx = dx[iy, ix]
        gy = dy[iy, ix]
        norm = gx * gx + gy * gy + 1e-4
        px -= np.clip(step * w * gx / norm, -_MAX_STEP, _MAX_STEP)
        py -= np.clip(step * w * gy / norm, -_MAX_STEP, _MAX_STEP)
        _reflect(px)
        _reflect(py)
    return px, py


def _reflect(values: np.ndarray) -> None:
    """Fold values back inside 0..1 at the plate's edge, in place.

    REFLECTION AND NOT CLIPPING, and the difference is visible. Clipping
    parks every particle that overshoots on the boundary, and the gradient
    at a corner is near zero so nothing ever moves it off again: the first
    rendering of this theme had four bright dots pinned in the corners of
    the plate that no figure ever cleared. A reflected particle lands back
    on the plate where a grain bouncing off the rim would.
    """
    np.abs(values, out=values)
    np.subtract(1.0, values, out=values)
    np.abs(values, out=values)
    np.subtract(1.0, values, out=values)
    np.clip(values, 0.0, 1.0, out=values)


def sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``field`` read at the particles, nearest lattice point.

    :param field: any ``(grid, grid)`` from :func:`lattice`.
    :param x: particle x in 0..1.
    :param y: particle y in 0..1.
    :returns: one value per particle.
    """
    grid = field.shape[0]
    scale = grid - 1
    ix = np.clip((np.asarray(x) * scale + 0.5).astype(np.int32), 0, grid - 1)
    iy = np.clip((np.asarray(y) * scale + 0.5).astype(np.int32), 0, grid - 1)
    return field[iy, ix]
