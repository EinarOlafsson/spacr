"""Play spaCR's sounds: off by default, rendered off the GUI thread, or silent.

The sounds themselves are synthesized by :mod:`spacr.qt.sound_synth`; this
module decides when one plays and hands it to the operating system through
Qt Multimedia's :class:`~PySide6.QtMultimedia.QSoundEffect`.

Three rules shape everything here, and each is a measurement rather than a
preference:

* **Nothing exists while sound is off.** The master switch defaults to off,
  and until somebody turns it on :func:`apply_sound_preferences` returns
  before constructing anything: no thread, no event filter, no Qt
  Multimedia import. An application-wide event filter costs every event in
  the process (item 380 measured 0.93 us per event per filter), so the
  filter is installed only while a click or hover sound is actually wanted,
  and removed the moment neither is.
* **Qt Multimedia is touched only on the GUI thread.**
  :class:`_SoundPlayer` owns ``QMediaDevices`` and every ``QSoundEffect``
  and lives where the application's event loop does; the dedicated
  ``QThread`` keeps the part that is genuinely slow, which is synthesizing
  a sound set (:class:`_SoundRenderer`, numpy and file writes, nothing
  from Qt Multimedia). Connecting to the sound server is slow -- 354 ms on
  the maintainer's workstation (PipeWire) -- so it is paid ONCE, on an idle
  timer after sound is switched on (:meth:`SoundEngine._warm`), and logged
  rather than felt. Playing an already loaded sound is ``play()``, which
  returns at once: 0.007-0.014 ms on the GUI thread against the
  real stack.

  This is item 444 and it is a rule written in a crash. Building the
  effects on the audio thread instead did move the connection off the GUI
  thread -- and Qt Multimedia's device handling belongs to the thread that
  owns the event loop, so with the FFmpeg backend it enabled socket
  notifiers from the wrong thread. Reopening Preferences with sound on
  then segfaulted three times out of three, and on the maintainer's
  workstation stopped the interface with an empty Preferences frame until
  he force quit.
* **A missing sound stack is silence.** No device, no server, no Qt
  Multimedia, a file that will not load: each ends in a debug log line and
  no sound. Nothing here raises into a caller and nothing opens a dialog.

The hover sound is the risky one, so it is debounced twice: the pointer must
rest on an enabled control for :data:`HOVER_SETTLE_MS` before anything is
considered, and a hover sound never follows another within
:data:`HOVER_COOLDOWN_S` or a click within :data:`HOVER_AFTER_CLICK_S`.
"""
from __future__ import annotations

import atexit
import logging
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import (
    QCoreApplication,
    QEvent,
    QObject,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
    Slot,
)

from .sound_synth import (
    BED,
    DEFAULT_THEME,
    FEEDBACK_EVENTS,
    SOUND_THEMES,
    SoundTheme,
    ensure_rendered,
    sound_names,
)

__all__ = [
    "BED_LEVEL",
    "HOVER_AFTER_CLICK_S",
    "HOVER_COOLDOWN_S",
    "HOVER_SETTLE_MS",
    "InputSoundFilter",
    "SILENT_PRESS_PROPERTY",
    "SoundEngine",
    "SoundSettings",
    "announce_run_end",
    "apply_sound_preferences",
    "perceived_gain",
    "preview_sound",
    "read_sound_settings",
    "shutdown_sound",
    "sound_engine",
    "stop_sound_preview",
    "wav_seconds",
]

LOG = logging.getLogger(__name__)

try:
    from shiboken6 import isValid as _IS_VALID
except Exception:                                            # noqa: BLE001
    _IS_VALID = None

#: How long the pointer must rest on a control before a hover sound is
#: considered. Sweeping across a panel crosses each control in far less.
HOVER_SETTLE_MS = 110

#: The shortest gap between two hover sounds, in seconds.
HOVER_COOLDOWN_S = 0.42

#: A hover never sounds this soon after a click: pressing a button that
#: opens something under the pointer would otherwise answer with two sounds.
HOVER_AFTER_CLICK_S = 0.30

#: A press that reaches the filter twice (a child ignoring it, its parent
#: taking it) is one click, not two.
CLICK_REPEAT_S = 0.035

#: A Qt dynamic property. A control that carries it set to True makes no
#: click sound when it is pressed. Preferences sets it on the Sound tab's
#: Preview buttons: a Preview is the user asking to hear ONE sound, and
#: answering the press with a click as well plays two -- twice the same
#: pluck, for the Click row's own Preview.
SILENT_PRESS_PROPERTY = "spacrSilentPress"

#: The music bed plays at this fraction of the other sounds' level: it is
#: under somebody's work, never over it.
BED_LEVEL = 0.6

#: How long a Preview of the music bed plays before it fades out.
PREVIEW_BED_MS = 9000

#: How long a fade-out takes, and how many steps it is taken in.
FADE_MS = 900
FADE_STEPS = 12

#: How long quitting waits for the audio thread before parking it.
SHUTDOWN_WAIT_MS = 3000

#: How long after sound is switched on the sound server is connected to.
#: Long enough that the Save which switched it on has closed its dialog,
#: and that a launch with sound already on has its window up, so that the
#: one connection -- 354 ms on the maintainer's workstation -- is paid
#: while nobody is waiting on anything. See :meth:`SoundEngine._warm`.
DEVICE_WARM_DELAY_MS = 400

#: ``QSoundEffect.Loop.Infinite``, spelled as the integer it is so that a
#: stand-in effect in a test needs no Qt Multimedia to understand it.
LOOP_FOREVER = -2


@dataclass(frozen=True)
class SoundSettings:
    """What the user has asked to hear, read once and handed around whole.

    :param enabled: the master switch.
    :param volume: master volume as a slider fraction, 0 to 1.
    :param theme: key of the sound set in
        :data:`spacr.qt.sound_synth.SOUND_THEMES`.
    :param click: play a sound when a control is pressed.
    :param hover: play a sound when the pointer rests on a control.
    :param run_finished: play a sound when a run finishes successfully.
    :param run_failed: play a sound when a run stops with an error.
    :param bed: play the looping music bed. Already False here when the
        performance level rests it; see :func:`read_sound_settings`.
    :param music: a WAV of the user's own to play as the bed instead of
        the synthesized one. Empty means spaCR's own. It is also what the
        Resonance backdrop is driven by, because the backdrop follows
        WHATEVER IS PLAYING and there is only ever one thing.
    """

    enabled: bool = False
    volume: float = 0.5
    theme: str = DEFAULT_THEME
    click: bool = True
    hover: bool = False
    run_finished: bool = True
    run_failed: bool = True
    bed: bool = False
    music: str = ""

    def wants(self, event: str) -> bool:
        """Whether ``event`` should make a sound now.

        :param event: ``"click"``, ``"hover"``, ``"run_finished"``,
            ``"run_failed"`` or ``"bed"``.
        :returns: True only when the master switch and the event's own
            switch are both on.
        """
        return bool(self.enabled and getattr(self, event, False))

    @property
    def gain(self) -> float:
        """The linear gain the volume slider stands for."""
        return perceived_gain(self.volume)

    @property
    def watches_input(self) -> bool:
        """Whether the application-wide input filter is needed at all."""
        return bool(self.enabled and (self.click or self.hover))


def perceived_gain(fraction: float) -> float:
    """Map a slider fraction to a linear gain that sounds evenly spaced.

    Loudness is heard roughly logarithmically, so a linear slider would do
    all its audible work in its bottom quarter. The square is the usual
    cheap fit.

    :param fraction: slider position, 0 to 1; clamped.
    :returns: gain, 0 to 1.
    """
    try:
        value = min(1.0, max(0.0, float(fraction)))
    except (TypeError, ValueError):
        return 0.0
    return value * value


def read_sound_settings() -> SoundSettings:
    """Read every sound preference into one :class:`SoundSettings`.

    The music bed is reported off at the performance levels that rest it,
    whatever its own switch says, so nothing downstream has to know those
    levels exist.

    :returns: the settings as stored now.
    """
    from . import preferences as prefs

    return SoundSettings(
        enabled=prefs.get_sound_enabled(),
        volume=prefs.get_sound_volume(),
        theme=prefs.get_sound_theme(),
        click=prefs.get_sound_event_enabled("click"),
        hover=prefs.get_sound_event_enabled("hover"),
        run_finished=prefs.get_sound_event_enabled("run_finished"),
        run_failed=prefs.get_sound_event_enabled("run_failed"),
        bed=(prefs.get_sound_event_enabled("bed")
             and not prefs.sound_bed_rests()),
        music=prefs.get_sound_music_file(),
    )


def _alive(wrapped) -> bool:
    """Whether a PySide wrapper still owns a live C++ object.

    The application-wide filter in :mod:`spacr.qt.widgets.feature_dictionary`
    once segfaulted on the first line of ``eventFilter``, reading the type
    of an event whose C++ half had been freed during a teardown. Asking
    shiboken first turns that case into a no-op.
    """
    if wrapped is None:
        return False
    if _IS_VALID is None:
        return True
    try:
        return bool(_IS_VALID(wrapped))
    except Exception:                                        # noqa: BLE001
        return True


def _delete_now(obj) -> None:
    """Delete a QObject's C++ half immediately, on the calling thread."""
    import shiboken6

    if shiboken6.isValid(obj):
        shiboken6.delete(obj)


def _theme(key: str) -> SoundTheme:
    """The sound set for ``key``, or the default set for an unknown key."""
    return SOUND_THEMES.get(key) or SOUND_THEMES[DEFAULT_THEME]


def wav_seconds(path) -> float:
    """How long a WAV is, from its header alone.

    The Resonance backdrop needs the loop's length to work out where in it
    the playback is, and the header is the cheapest true answer -- no
    samples are read. A file that cannot be opened is reported as zero
    seconds, which reads downstream as "nothing is playing".

    :param path: the file.
    :returns: seconds, or 0.0.
    """
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / float(rate or 1)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not read the length of %s", path, exc_info=True)
        return 0.0


def _forget_bed() -> None:
    """Nothing is playing: the backdrop goes back to idling.

    Cheap enough for any thread -- :func:`spacr.qt.resonance.set_now_playing`
    reads no file for ``None`` -- and deliberately import-guarded, so a
    process that never drew a Resonance backdrop never imports one to be
    told that nothing is playing.
    """
    import sys

    module = sys.modules.get(f"{__package__}.resonance")
    if module is not None:
        module.set_now_playing(None)


class _SoundRenderer(QObject):
    """Synthesizes sound sets into the cache, on the audio thread.

    This is the only part of playing a sound that is slow enough to be
    worth a thread of its own: rendering one set is numpy arithmetic and a
    handful of file writes, and it takes as long as it takes. Nothing here
    touches Qt Multimedia -- see the module docstring's second rule, and
    item 444 for what happened when it did.

    :param cache_root: where rendered files go; ``None`` means
        :func:`spacr.qt.sound_synth.sound_cache_root`.
    """

    #: (theme key, the stems tried, ``{stem: path as a string}``) for one
    #: render. A stem that was tried and is absent from the mapping could
    #: not be rendered, which is how a full disk becomes silence rather
    #: than an exception; the two lists are separate because several
    #: renders can be in flight and each answers only for its own.
    rendered = Signal(str, object, object)

    def __init__(self, cache_root: Optional[Path] = None) -> None:
        """Hold the cache root; render nothing yet."""
        super().__init__()
        self._root = cache_root
        self._stopping = threading.Event()
        self._music = ""
        self._bed_analysis: Optional[Path] = None
        self._bed_seconds = 0.0

    def request_stop(self) -> None:
        """Abandon any render at its next file boundary. Thread-safe."""
        self._stopping.set()

    @Slot(str, object, str)
    def render(self, theme_key: str, names, music: str = "") -> None:
        """Make sure every name in ``names`` exists on disk, then say so.

        One name at a time, so a sound that cannot be written leaves the
        rest of the set playable. The music bed is the one sound that can
        come from outside spaCR: a readable file the user chose is used as
        it is, and one that has gone missing falls through to the
        synthesized bed, because a bed that is silent because somebody
        moved a WAV is worse than spaCR's own music playing instead.

        :param theme_key: the sound set.
        :param names: stems from :func:`spacr.qt.sound_synth.sound_names`.
        :param music: a WAV of the user's own for the bed, or empty.
        """
        self._switch_music(music)
        theme = _theme(theme_key)
        ready: Dict[str, str] = {}
        tried: List[str] = []
        for name in list(names or ()):
            if self._stopping.is_set():
                break
            tried.append(name)
            chosen = self._chosen_music() if name == BED else None
            if chosen is not None:
                ready[name] = str(chosen)
                continue
            try:
                made = ensure_rendered(theme, [name], root=self._root,
                                       should_stop=self._stopping.is_set)
            except Exception:                                # noqa: BLE001
                LOG.debug("could not render the %s sound", name,
                          exc_info=True)
                continue
            ready.update({stem: str(path) for stem, path in made.items()})
        if BED in ready:
            self._measure_bed(Path(ready[BED]))
        self.rendered.emit(theme_key, tried, ready)

    def _switch_music(self, music: str) -> None:
        """Forget the previous bed's measurements when the file changes."""
        music = str(music or "")
        if music == self._music:
            return
        self._music = music
        self._bed_analysis = None
        self._bed_seconds = 0.0

    def _chosen_music(self) -> Optional[Path]:
        """The user's own music file, when they named one that is there."""
        if not self._music:
            return None
        chosen = Path(self._music)
        try:
            return chosen if chosen.is_file() else None
        except OSError:
            LOG.debug("could not read the chosen music file", exc_info=True)
            return None

    def _measure_bed(self, path: Path) -> None:
        """Work out the bed's length and analyse it for the visualiser.

        On this thread and nowhere else:
        :func:`spacr.qt.resonance.ensure_analysis` reads a whole file and
        runs a few thousand FFTs, and says in its own docstring never to
        do that on the GUI thread. Done as the bed is RESOLVED rather than
        as it is started, so it is off the path a user waits on entirely;
        both results are cached against the file, so a second start costs
        nothing.
        """
        if self._bed_analysis is not None and self._bed_seconds > 0.0:
            return
        try:
            from .resonance import ensure_analysis
            from .sound_synth import sound_cache_root

            outside = bool(self._music) and path == Path(self._music)
            folder = (self._root or sound_cache_root()) if outside else None
            self._bed_seconds = wav_seconds(path)
            self._bed_analysis = ensure_analysis(path, loop=True,
                                                 out_dir=folder)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not analyse the music bed", exc_info=True)
            self._bed_analysis = None
            self._bed_seconds = 0.0

    @Slot(float)
    def announce(self, started_at: float) -> None:
        """Tell the Resonance backdrop what is playing and when it began.

        ``started_at`` is taken on the GUI thread at the instant playback
        really started and carried here, so the queue hop does not skew
        it. The read of the analysis file happens on this thread, which is
        what :func:`spacr.qt.resonance.set_now_playing` asks for.

        :param started_at: ``time.monotonic()`` when the bed started.
        """
        try:
            from .resonance import NowPlaying, set_now_playing

            if self._bed_analysis is None or self._bed_seconds <= 0.0:
                set_now_playing(None)
                return
            set_now_playing(NowPlaying(str(self._bed_analysis),
                                       float(started_at),
                                       self._bed_seconds, True))
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not announce the music bed", exc_info=True)

    @Slot()
    def silence(self) -> None:
        """Nothing is playing any more.

        Queued behind :meth:`announce` rather than done on the GUI thread
        alone, so a stop that follows a start always lands after it. The
        player ALSO calls :func:`_forget_bed` directly, because a stop the
        user can see must not wait behind a render.
        """
        _forget_bed()

    @Slot()
    def retire(self) -> None:
        """Come home as the audio thread ends.

        Connected to ``QThread.finished`` directly, so it runs on the audio
        thread in its last moments. Moving to the application's thread
        afterwards means whatever destroys this later -- Python's
        collector, on the GUI thread -- destroys an object of its own
        thread, rather than one belonging to a thread that has gone.
        """
        app = QCoreApplication.instance()
        if app is None:
            return
        try:
            self.moveToThread(app.thread())
        except Exception:                                    # noqa: BLE001
            LOG.debug("the renderer could not come home", exc_info=True)


class _SoundPlayer(QObject):
    """Owns every Qt Multimedia object, on the GUI thread.

    ``QMediaDevices`` and every ``QSoundEffect`` are created and used here,
    and here is wherever the :class:`SoundEngine` lives, which is the
    thread that owns the application's event loop. That is the whole of
    item 444: Qt Multimedia's device handling installs socket notifiers on
    the event loop's thread, and driving them from another one is undefined
    behaviour that segfaulted or wedged the application.

    Nothing here blocks except the first device call, which
    :meth:`warm` pays deliberately at an idle moment; a file that is not
    rendered yet is asked for with :attr:`needs` and played when it
    arrives.

    :param engine: the owning :class:`SoundEngine`; also the Qt parent.
    :param effect_factory: builds one effect given its parent. ``None``
        means ``QSoundEffect``; tests pass a stand-in that records what it
        is asked to do.
    """

    #: (theme key, [stems]) -- files the player wants rendered before it
    #: can play them. The engine forwards these to the audio thread.
    needs = Signal(str, object)

    #: (theme key, whether Qt Multimedia could build an effect at all). A
    #: device that exists but refuses the format still reads True here: the
    #: effect then reports an error status and plays nothing.
    prepared = Signal(str, bool)

    #: A Preview of the music bed has finished fading out. The player knows
    #: only that it faded something; the engine is the one that knows what
    #: the stored settings want playing afterwards.
    bed_faded = Signal()

    #: ``time.monotonic()`` at the instant the music bed really started,
    #: which is when its file finished loading and not when ``play()``
    #: returned. The engine hands it to the renderer, which is the thread
    #: allowed to read the analysis the Resonance backdrop needs.
    bed_started = Signal(float)

    #: The music bed stopped. Queued behind :attr:`bed_started` so the two
    #: can never land out of order.
    bed_stopped = Signal()

    def __init__(self, engine: Optional[QObject] = None,
                 effect_factory: Optional[Callable] = None) -> None:
        """Hold the factory; build nothing, and connect to nothing, yet."""
        super().__init__(engine)
        self._factory = effect_factory
        self._theme_key = ""
        self._music = ""
        self._paths: Dict[str, str] = {}
        self._effects: Dict[str, object] = {}
        self._asked: set = set()
        self._missing: set = set()
        self._wanted: List[str] = []
        self._preparing = False
        self._pending_plays: Dict[str, float] = {}
        self._pending_bed: Optional[float] = None
        self._pending_preview: Optional[tuple] = None
        self._available = True
        self._device_ready = False
        self._bed_playing = False
        self._fade_timer: Optional[QTimer] = None
        self._fade_steps_left = 0
        self._fade_from = 0.0

    @property
    def device_ready(self) -> bool:
        """Whether the sound server has been connected to in this process."""
        return self._device_ready

    @Slot()
    def warm(self) -> float:
        """Connect to the sound server now, and report what it cost.

        Called on an idle timer once sound is switched on, so that the
        connection is paid at a moment nobody is waiting on rather than
        inside the first dialog, button press or run that wants a sound.
        Idempotent, and a failure is recorded once and never retried:
        no device is silence.

        :returns: milliseconds the connection took; 0.0 when there was
            nothing to do.
        """
        if self._device_ready:
            return 0.0
        self._device_ready = True
        if self._factory is not None:
            return 0.0
        start = time.perf_counter()
        try:
            from PySide6.QtMultimedia import QMediaDevices

            QMediaDevices.defaultAudioOutput()
        except Exception:                                    # noqa: BLE001
            LOG.debug("no audio device could be listed; spaCR stays silent",
                      exc_info=True)
            return (time.perf_counter() - start) * 1000.0
        cost = (time.perf_counter() - start) * 1000.0
        LOG.info("The sound server answered in %.0f ms. That connection is "
                 "made once, here, so no sound pays for it again.", cost)
        return cost

    def _make_effect(self):
        """One new effect, parented here so it lives on the GUI thread."""
        if self._factory is not None:
            return self._factory(self)
        from PySide6.QtMultimedia import QSoundEffect

        return QSoundEffect(self)

    def _effect(self, name: str):
        """The loaded effect for ``name``, or ``None`` when there is none.

        ``None`` means either that Qt Multimedia is unusable -- silence
        from here on -- or that the file is not rendered yet, which the
        caller answers by asking for it and playing it when it lands.
        """
        if not self._available:
            return None
        effect = self._effects.get(name)
        if effect is not None:
            return effect
        path = self._paths.get(name)
        if path is None:
            return None
        self.warm()
        try:
            effect = self._make_effect()
            effect.setSource(QUrl.fromLocalFile(path))
        except Exception:                                    # noqa: BLE001
            LOG.debug("no sound output is available; spaCR stays silent",
                      exc_info=True)
            self._available = False
            return None
        self._effects[name] = effect
        return effect

    def _switch_theme(self, key: str) -> None:
        """Forget the previous set's effects and files when the set changes."""
        if key != self._theme_key:
            self._drop_effects()
            self._paths = {}
            self._asked = set()
            self._missing = set()
            self._pending_plays = {}
            self._theme_key = key

    def _switch_music(self, music: str) -> None:
        """Change which file the music bed plays, dropping the old effect.

        The bed is the one sound that can come from outside spaCR, and an
        effect holds its source for its whole life -- so a new file means
        a new effect, not a new URL on the old one. Its path and whatever
        the renderer said about it go too, so the next start asks again.
        """
        music = str(music or "")
        if music == self._music:
            return
        self._music = music
        effect = self._effects.pop(BED, None)
        if effect is not None:
            self._quietly(effect.stop)
            self._dispose(effect)
        self._paths.pop(BED, None)
        self._asked.discard(BED)
        self._missing.discard(BED)
        self._bed_playing = False

    def _ask_for(self, names) -> None:
        """Ask the audio thread to render what is neither here nor pending.

        A stem the renderer has already failed on is never asked for
        again until the set changes or the preferences are applied afresh.
        Without that, an unrenderable sound is an endless round trip:
        ask, fail, notice it is still missing, ask again.
        """
        fresh = [name for name in names
                 if name not in self._paths and name not in self._asked
                 and name not in self._missing]
        if not fresh:
            return
        self._asked.update(fresh)
        self.needs.emit(self._theme_key, fresh)

    def prepare(self, theme_key: str, names) -> None:
        """Load ``names`` so their first play is instant.

        :param theme_key: the sound set.
        :param names: stems from :func:`spacr.qt.sound_synth.sound_names`.
        """
        self._switch_theme(theme_key)
        self._wanted = [name for name in list(names or ()) if name]
        self._missing = set()
        self._preparing = True
        self._build_wanted()

    def _build_wanted(self) -> None:
        """Build every wanted effect whose file is here; ask for the rest.

        Answers with :attr:`prepared` the moment there is nothing left to
        wait for -- including when there is nothing that can be waited
        for, because Qt Multimedia gave up or the files cannot be written.
        A caller watching ``available`` has to be told that too.
        """
        missing: List[str] = []
        for name in self._wanted:
            if not self._available:
                break
            if name in self._missing:
                continue
            if self._effect(name) is None:
                missing.append(name)
        if self._available and missing:
            self._ask_for(missing)
            if self._asked:
                return
        if self._preparing:
            self._preparing = False
            self.prepared.emit(self._theme_key, bool(self._available))

    @Slot(str, object, object)
    def deliver(self, theme_key: str, tried, paths) -> None:
        """Take rendered files from the audio thread and use them.

        Whatever was waiting on a file -- a prepare, a one-shot play, the
        music bed, a Preview -- happens here, on the GUI thread. Only the
        stems this render answers for are resolved: the music bed is
        usually asked for in a second render, and a delivery that closed
        out requests it never carried would report the bed unrenderable
        and leave it silent.

        :param theme_key: the set the files belong to.
        :param tried: the stems this render attempted.
        :param paths: ``{stem: path as a string}`` for those that worked.
        """
        if theme_key != self._theme_key:
            return
        self._paths.update(paths or {})
        done = set(tried or ())
        self._missing.update(name for name in done if name not in self._paths)
        self._asked.difference_update(done)
        plays = {name: self._pending_plays.pop(name)
                 for name in [n for n in self._pending_plays if n in done]}
        preview = None
        bed_gain = None
        if BED in done:
            preview, self._pending_preview = self._pending_preview, None
            bed_gain, self._pending_bed = self._pending_bed, None
        self._build_wanted()
        for name, gain in plays.items():
            self._start(name, gain)
        if preview is not None:
            self.preview_bed(theme_key, preview[0], preview[1], self._music)
        elif bed_gain is not None:
            self.bed(theme_key, True, bed_gain, self._music)

    def _start(self, name: str, gain: float) -> None:
        """Play one loaded sound once; a missing one is simply not played."""
        effect = self._effect(name)
        if effect is None:
            return
        try:
            effect.setVolume(float(gain))
            effect.play()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not play the %s sound", name, exc_info=True)

    def play(self, theme_key: str, name: str, gain: float) -> None:
        """Play one sound once, rendering it first if it is new.

        :param theme_key: the sound set.
        :param name: the sound's stem.
        :param gain: linear gain, 0 to 1.
        """
        self._switch_theme(theme_key)
        if not self._available or name in self._missing:
            return
        if name not in self._paths:
            self._pending_plays[name] = float(gain)
            self._ask_for([name])
            return
        self._start(name, gain)

    def bed(self, theme_key: str, on: bool, gain: float,
            music: str = "") -> None:
        """Start, re-level or stop the looping music bed.

        :param theme_key: the sound set.
        :param on: whether the bed should be playing.
        :param gain: its linear gain.
        :param music: a WAV of the user's own, or empty for spaCR's.
        """
        self._stop_fade()
        if not on:
            self._pending_bed = None
            self._pending_preview = None
            effect = self._effects.get(BED)
            if effect is not None:
                self._quietly(effect.stop)
            if self._bed_playing:
                self._say_stopped()
            self._bed_playing = False
            return
        self._switch_theme(theme_key)
        self._switch_music(music)
        if not self._available or BED in self._missing:
            return
        if BED not in self._paths:
            self._pending_bed = float(gain)
            self._ask_for([BED])
            return
        effect = self._effect(BED)
        if effect is None:
            return
        try:
            effect.setVolume(float(gain))
            if not self._bed_playing:
                effect.setLoopCount(LOOP_FOREVER)
                effect.play()
                self._bed_playing = True
                self._say_started(effect)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not start the music bed", exc_info=True)

    def preview_bed(self, theme_key: str, gain: float,
                    milliseconds: int, music: str = "") -> None:
        """Play the music bed for a few seconds, then fade it out.

        :param theme_key: the sound set.
        :param gain: linear gain.
        :param milliseconds: how long it plays before the fade begins.
        :param music: a WAV of the user's own, or empty for spaCR's.
        """
        self._switch_theme(theme_key)
        self._switch_music(music)
        if not self._available or BED in self._missing:
            return
        if BED not in self._paths:
            self._pending_preview = (float(gain), int(milliseconds))
            self._ask_for([BED])
            return
        effect = self._effect(BED)
        if effect is None:
            return
        try:
            self._stop_fade()
            effect.setVolume(float(gain))
            effect.setLoopCount(LOOP_FOREVER)
            if not self._bed_playing:
                effect.play()
                self._bed_playing = True
                self._say_started(effect)
            self._bed_playing = True
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not preview the music bed", exc_info=True)
            return
        self._fade_from = float(gain)
        self._fade_steps_left = FADE_STEPS
        if self._fade_timer is None:
            self._fade_timer = QTimer(self)
            self._fade_timer.timeout.connect(self._fade_step)
        self._fade_timer.setSingleShot(True)
        self._fade_timer.start(max(0, int(milliseconds)))

    @Slot()
    def _fade_step(self) -> None:
        """Lower the bed one step; stop it after the last.

        The last step announces itself with :attr:`bed_faded`, because a
        Preview that ends must not leave a music bed the settings ask for
        silent until the dialog closes.
        """
        effect = self._effects.get(BED)
        if effect is None:
            return
        self._fade_steps_left -= 1
        if self._fade_steps_left <= 0:
            self._quietly(effect.stop)
            if self._bed_playing:
                self._say_stopped()
            self._bed_playing = False
            self.bed_faded.emit()
            return
        self._quietly(effect.setVolume,
                      self._fade_from * self._fade_steps_left / FADE_STEPS)
        if self._fade_timer is not None:
            self._fade_timer.setSingleShot(True)
            self._fade_timer.start(max(1, FADE_MS // FADE_STEPS))

    def _say_started(self, effect) -> None:
        """Say the bed is playing, now or the moment it really is.

        ``QSoundEffect.play()`` on a source that is still loading QUEUES
        the play until it is ready, so the instant the sound starts is the
        instant the file becomes loaded and not the instant ``play()``
        returned -- and a visualiser driven by the wrong instant is a
        visualiser out of step with the music by however long a twelve
        megabyte WAV takes to decode. An effect that cannot say whether it
        is loaded (the stand-ins the tests pass) is taken at its word.
        """
        loaded = True
        try:
            loaded = bool(effect.isLoaded())
        except Exception:                                    # noqa: BLE001
            loaded = True
        if loaded:
            self.bed_started.emit(time.monotonic())
            return
        try:
            effect.loadedChanged.connect(
                self._bed_loaded, Qt.ConnectionType.UniqueConnection)
        except Exception:                                    # noqa: BLE001
            LOG.debug("this effect cannot say when it has loaded",
                      exc_info=True)
            self.bed_started.emit(time.monotonic())

    @Slot()
    def _bed_loaded(self) -> None:
        """The bed's file finished loading, so this is when it starts."""
        effect = self._effects.get(BED)
        if effect is None or not self._bed_playing:
            return
        try:
            if not effect.isLoaded():
                return
        except Exception:                                    # noqa: BLE001
            return
        self.bed_started.emit(time.monotonic())

    def _say_stopped(self) -> None:
        """The bed stopped: idle the backdrop now, and say so in order.

        Both, deliberately. The direct call is what the user sees, and it
        must not wait behind a render on the audio thread; the signal is
        queued to that thread so that a stop always lands after the start
        it follows.
        """
        _forget_bed()
        self.bed_stopped.emit()

    def _stop_fade(self) -> None:
        """Cancel a pending fade-out."""
        if self._fade_timer is not None:
            self._fade_timer.stop()
        self._fade_steps_left = 0

    @staticmethod
    def _quietly(call, *args) -> None:
        """Call ``call`` and log, rather than raise, whatever it raises."""
        try:
            call(*args)
        except Exception:                                    # noqa: BLE001
            LOG.debug("a sound call failed", exc_info=True)

    def _drop_effects(self) -> None:
        """Stop every effect and delete it now, on the GUI thread.

        Deleted at once rather than with ``deleteLater``. A deferred
        deletion is a promise that the owner outlives the next pass of the
        event loop, and nothing here can keep it: a player collected before
        that pass takes its effects with it while their deletion events are
        still queued. Qt reports "shared QObject was deleted directly" and
        the next ``sendPostedEvents`` segfaults in ``~QObject`` --
        reproduced with the input filter, which went the same way, on
        2026-09-19.
        """
        self._stop_fade()
        effects, self._effects = self._effects, {}
        if self._bed_playing:
            self._say_stopped()
        self._bed_playing = False
        for effect in effects.values():
            self._quietly(effect.stop)
            self._dispose(effect)

    def _dispose(self, effect) -> None:
        """Delete one effect now, on this thread. See :meth:`_drop_effects`
        for why nothing here is ever ``deleteLater``'d by spaCR itself."""
        if isinstance(effect, QObject):
            self._quietly(_delete_now, effect)
        else:
            dispose = getattr(effect, "deleteLater", None)
            if callable(dispose):
                self._quietly(dispose)

    def release(self) -> None:
        """Let go of every effect, and with them the audio device."""
        self._drop_effects()
        self._paths = {}
        self._asked = set()
        self._missing = set()
        self._wanted = []
        self._preparing = False
        self._pending_plays = {}
        self._pending_bed = None
        self._pending_preview = None
        self._theme_key = ""
        self._music = ""
        if self._fade_timer is not None:
            self._quietly(_delete_now, self._fade_timer)
            self._fade_timer = None

class InputSoundFilter(QObject):
    """Application-wide filter that hears presses and hovers on controls.

    Installed only while a click or hover sound is wanted (see
    :meth:`SoundEngine.apply`). The body is two liveness checks and at most
    four comparisons for an event it does not want.

    :param engine: the :class:`SoundEngine` to tell.
    """

    def __init__(self, engine: "SoundEngine") -> None:
        """Remember the engine and the kinds of control that make sounds."""
        super().__init__(engine)
        from PySide6.QtWidgets import (
            QAbstractButton,
            QAbstractSlider,
            QComboBox,
            QMenu,
            QMenuBar,
            QTabBar,
        )

        self._engine = engine
        self._silent = SILENT_PRESS_PROPERTY
        self._pressable = (QAbstractButton, QComboBox, QTabBar,
                           QAbstractSlider, QMenuBar)
        self._hoverable = (QAbstractButton, QComboBox, QTabBar)
        self._menu = QMenu
        self._press = QEvent.Type.MouseButtonPress
        self._release = QEvent.Type.MouseButtonRelease
        self._enter = QEvent.Type.Enter
        self._leave = QEvent.Type.Leave
        self._left_button = Qt.MouseButton.LeftButton

    def eventFilter(self, watched, event) -> bool:    # noqa: N802 - Qt naming
        """Tell the engine about presses and hovers; never consume anything.

        A control carrying :data:`SILENT_PRESS_PROPERTY` is pressed without
        a click sound. The property is read only for a left press on
        something pressable, so it costs nothing on the events the filter
        already ignores.

        :param watched: the object the event is for.
        :param event: the event.
        :returns: always False, so every event continues as it would have.
        """
        if not _alive(event) or not _alive(watched):
            return False
        try:
            kind = event.type()
            if kind == self._press:
                if (isinstance(watched, self._pressable)
                        and event.button() == self._left_button
                        and watched.isEnabled()
                        and not watched.property(self._silent)):
                    self._engine._pressed()
            elif kind == self._release:
                if (isinstance(watched, self._menu)
                        and event.button() == self._left_button):
                    action = watched.activeAction()
                    if (action is not None and action.isEnabled()
                            and not action.isSeparator()
                            and action.menu() is None):
                        self._engine._pressed()
            elif kind == self._enter:
                if isinstance(watched, self._hoverable) and watched.isEnabled():
                    self._engine._entered(watched)
            elif kind == self._leave:
                self._engine._left(watched)
        except (RuntimeError, ReferenceError, AttributeError):
            return False
        return False


class SoundEngine(QObject):
    """Decides when a sound plays, and plays it on the thread it belongs to.

    Two halves, and which thread each lives on is the whole of item 444.
    :class:`_SoundPlayer` holds every Qt Multimedia object and lives here,
    on the GUI thread, because that is the thread that owns the event loop
    Qt Multimedia's device handling attaches to. :class:`_SoundRenderer`
    lives on a dedicated ``QThread`` and synthesizes sound files, which is
    the only part that is slow enough to need one.

    :param parent: owning object, normally the application.
    :param threaded: run the renderer on its own ``QThread``. ``False``
        runs it inline, emitting the same calls in the same order, so a
        test can drive the engine synchronously.
    :param effect_factory: see :class:`_SoundPlayer`.
    :param cache_root: see :class:`_SoundRenderer`.
    """

    _ask_render = Signal(str, object, str)
    _ask_announce = Signal(float)
    _ask_silence = Signal()

    #: (theme key, whether Qt Multimedia could build an effect), from the
    #: player.
    prepared = Signal(str, bool)

    def __init__(self, parent: Optional[QObject] = None, *,
                 threaded: bool = True,
                 effect_factory: Optional[Callable] = None,
                 cache_root: Optional[Path] = None) -> None:
        """Start the audio thread (when threaded) and wire the requests."""
        super().__init__(parent)
        self._settings = SoundSettings()
        self._closed = False
        self._warm_asked = False
        self._music_wanted = ""
        self._player = _SoundPlayer(self, effect_factory)
        self._renderer = _SoundRenderer(cache_root)
        self._thread: Optional[QThread] = None
        if threaded:
            thread = QThread()
            thread.setObjectName("spacr-sound")
            self._renderer.moveToThread(thread)
            thread.finished.connect(self._renderer.retire,
                                    Qt.ConnectionType.DirectConnection)
            self._thread = thread
        self._ask_render.connect(self._renderer.render)
        self._ask_announce.connect(self._renderer.announce)
        self._ask_silence.connect(self._renderer.silence)
        self._player.needs.connect(self._on_needs)
        self._player.bed_started.connect(self._ask_announce)
        self._player.bed_stopped.connect(self._ask_silence)
        self._renderer.rendered.connect(self._player.deliver)
        self._player.prepared.connect(self._on_prepared)
        self._player.bed_faded.connect(self._on_bed_faded)
        if self._thread is not None:
            self._thread.start()

        #: Whether the last preparation could build an effect at all.
        #: ``None`` until the player has answered once.
        self.available: Optional[bool] = None
        #: Built the first time it is needed and kept for the engine's
        #: life; installed and removed, never deleted, for the reason
        #: :meth:`_SoundPlayer._drop_effects` gives.
        self._filter: Optional[InputSoundFilter] = None
        self._filter_on = False
        self._rotation: Dict[str, int] = {}
        self._last_click = float("-inf")
        self._last_hover = float("-inf")
        self._hover_target = None
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(HOVER_SETTLE_MS)
        self._hover_timer.timeout.connect(self._hover_settled)

    @property
    def settings(self) -> SoundSettings:
        """The settings this engine is following."""
        return self._settings

    @property
    def closed(self) -> bool:
        """Whether :meth:`shutdown` has run."""
        return self._closed

    @property
    def filter_installed(self) -> bool:
        """Whether the application-wide input filter is in place."""
        return self._filter_on

    def audio_thread(self) -> Optional[QThread]:
        """The thread sounds are rendered on, or ``None`` when unthreaded."""
        return self._thread

    def player(self) -> "_SoundPlayer":
        """The GUI-thread half: every Qt Multimedia object lives here."""
        return self._player

    @Slot(str, object)
    def _on_needs(self, theme_key: str, names) -> None:
        """Hand the player's request for files to the audio thread."""
        if self._closed:
            return
        self._ask_render.emit(theme_key, list(names or ()),
                              self._music_wanted)

    @Slot(str, bool)
    def _on_prepared(self, theme_key: str, available: bool) -> None:
        """Record whether the device could be used, and pass it on."""
        self.available = bool(available)
        self.prepared.emit(theme_key, bool(available))

    @Slot()
    def _warm(self) -> None:
        """Connect to the sound server, once, at an idle moment.

        The connection costs hundreds of milliseconds and it is a GUI
        thread that has to pay it (item 444). So it is paid HERE, on a
        timer that fires once the application has nothing else queued
        after sound was switched on, rather than inside whatever dialog,
        press or finishing run first wants a sound.
        """
        if self._closed or not self._settings.enabled:
            return
        self._player.warm()

    @Slot()
    def _on_bed_faded(self) -> None:
        """A bed Preview ended: put the stored settings back on the bed."""
        self.stop_preview()

    def apply(self, settings: SoundSettings) -> None:
        """Follow ``settings``: filter, loaded sounds and music bed.

        Switching the master off removes the filter and releases every
        effect, so a disabled engine holds no audio device. The thread
        itself stays, idle, until the application quits.

        Switching it ON also schedules the one device connection, on a
        timer rather than now: see :meth:`_warm`.

        :param settings: what to follow from now on.
        """
        if self._closed:
            return
        self._settings = settings
        self._sync_filter()
        if not settings.enabled:
            self._hover_timer.stop()
            self._hover_target = None
            self._player.release()
            return
        if not self._warm_asked:
            self._warm_asked = True
            QTimer.singleShot(DEVICE_WARM_DELAY_MS, self._warm)
        names: List[str] = []
        for event in FEEDBACK_EVENTS:
            if settings.wants(event):
                names.extend(sound_names(event))
        self._music_wanted = settings.music
        self._player.prepare(settings.theme, names)
        self._player.bed(settings.theme, settings.wants(BED),
                         settings.gain * BED_LEVEL, settings.music)

    def _sync_filter(self) -> None:
        """Install the input filter exactly while it is needed."""
        app = QCoreApplication.instance()
        wanted = self._settings.watches_input and not self._closed
        if wanted and not self._filter_on and app is not None:
            if self._filter is None:
                self._filter = InputSoundFilter(self)
            app.installEventFilter(self._filter)
            self._filter_on = True
        elif not wanted and self._filter_on:
            if app is not None:
                app.removeEventFilter(self._filter)
            self._filter_on = False

    def play(self, event: str) -> bool:
        """Play ``event``'s sound if the settings want it.

        :param event: ``"click"``, ``"hover"``, ``"run_finished"`` or
            ``"run_failed"``.
        :returns: True when a sound was requested.
        """
        if self._closed or not self._settings.wants(event):
            return False
        self._request(self._settings.theme, event, self._settings.gain)
        return True

    def preview(self, event: str, theme_key: str, volume: float,
                music: str = "") -> bool:
        """Play ``event`` once because the user pressed its Preview button.

        Plays whatever the stored switches say: the dialog enables Preview
        only while its own master switch is on, and the press is the ask.

        :param event: any event, the music bed included.
        :param theme_key: the sound set chosen in the dialog.
        :param volume: the dialog's volume slider as a fraction.
        :param music: the music file named on the page, unsaved. A Preview
            has to play what the page says and not what the store says, or
            it is a preview of something else.
        :returns: True when a sound was requested.
        """
        if self._closed:
            return False
        gain = perceived_gain(volume)
        if event == BED:
            self._music_wanted = str(music or "")
            self._player.preview_bed(theme_key, gain * BED_LEVEL,
                                     PREVIEW_BED_MS, str(music or ""))
            return True
        self._request(theme_key, event, gain)
        return True

    def stop_preview(self) -> None:
        """Put the music bed back to what the stored settings say."""
        if self._closed:
            return
        self._music_wanted = self._settings.music
        self._player.bed(self._settings.theme, self._settings.wants(BED),
                         self._settings.gain * BED_LEVEL,
                         self._settings.music)

    def _request(self, theme_key: str, event: str, gain: float) -> None:
        """Play the next variant of ``event``'s sound."""
        names = sound_names(event)
        index = self._rotation.get(event, 0)
        self._rotation[event] = (index + 1) % len(names)
        self._player.play(theme_key, names[index], float(gain))

    def _pressed(self) -> None:
        """A control was pressed: a click, and no hover for a moment."""
        now = time.monotonic()
        self._hover_timer.stop()
        self._hover_target = None
        repeated = now - self._last_click < CLICK_REPEAT_S
        self._last_click = now
        if not repeated:
            self.play("click")

    def _entered(self, widget) -> None:
        """The pointer entered a control: wait to see whether it stays."""
        if not self._settings.wants("hover"):
            return
        self._hover_target = widget
        self._hover_timer.start()

    def _left(self, widget) -> None:
        """The pointer left a widget; forget it if it was the candidate."""
        if widget is self._hover_target:
            self._hover_timer.stop()
            self._hover_target = None

    @Slot()
    def _hover_settled(self) -> None:
        """The pointer rested long enough; sound unless a rule says not."""
        target, self._hover_target = self._hover_target, None
        if target is None or not _alive(target):
            return
        try:
            if not (target.isEnabled() and target.isVisible()):
                return
        except RuntimeError:
            return
        from PySide6.QtWidgets import QApplication
        if QApplication.mouseButtons() != Qt.MouseButton.NoButton:
            return
        now = time.monotonic()
        if now - self._last_hover < HOVER_COOLDOWN_S:
            return
        if now - self._last_click < HOVER_AFTER_CLICK_S:
            return
        self._last_hover = now
        self.play("hover")

    @Slot()
    def shutdown(self, timeout_ms: int = SHUTDOWN_WAIT_MS) -> bool:
        """Stop everything and end the audio thread. Idempotent.

        Connected to ``aboutToQuit``. A render in progress is abandoned at
        its next file boundary; a thread that still will not stop in time is
        parked by :func:`spacr.qt.bridge.drain_thread` rather than
        terminated.

        :param timeout_ms: how long to wait for the thread.
        :returns: True when the thread has stopped.
        """
        if self._closed:
            return True
        self._closed = True
        self._hover_timer.stop()
        self._hover_target = None
        self._settings = SoundSettings()
        self._sync_filter()
        self._player.release()
        self._renderer.request_stop()
        _forget_bed()
        if self._thread is None:
            return True
        from .bridge import drain_thread
        stopped = drain_thread(self._thread, self._renderer, int(timeout_ms))
        _forget_bed()
        return stopped


#: The one engine this process has, once anybody has turned sound on.
_ENGINE: Optional[SoundEngine] = None


def sound_engine() -> Optional[SoundEngine]:
    """The running engine, or ``None`` when sound was never switched on."""
    return _ENGINE


def _stop_at_exit() -> None:
    """Stop the audio thread while the interpreter can still run Python.

    ``aboutToQuit`` covers an application that quits through its event
    loop. Nothing covered a process that ends any other way -- and a
    ``QThread`` whose last reference goes while it is still running aborts
    the process, which is how item 444's run ended: "QThread: Destroyed
    while thread 'spacr-sound' is still running", then a core dump that
    took whatever the run journal had not flushed with it.

    Registered only once sound has actually been switched on, so a user
    who never asked for sound is charged nothing, not even a hook.
    """
    try:
        shutdown_sound()
    except Exception:                                        # noqa: BLE001
        LOG.debug("the sound engine did not stop cleanly at exit",
                  exc_info=True)


#: Whether :func:`_stop_at_exit` is registered. One process, one hook.
_EXIT_HOOK_INSTALLED = False


def _create_engine(app) -> SoundEngine:
    """Build the process's engine and have it stop when the app quits."""
    global _ENGINE, _EXIT_HOOK_INSTALLED
    parent = app if isinstance(app, QObject) else None
    engine = SoundEngine(parent)
    about_to_quit = getattr(app, "aboutToQuit", None)
    if about_to_quit is not None:
        try:
            about_to_quit.connect(engine.shutdown)
        except (RuntimeError, TypeError):
            LOG.debug("could not tie the sound engine to quitting",
                      exc_info=True)
    if not _EXIT_HOOK_INSTALLED:
        _EXIT_HOOK_INSTALLED = True
        atexit.register(_stop_at_exit)
    _ENGINE = engine
    return engine


def apply_sound_preferences(app=None,
                            settings: Optional[SoundSettings] = None
                            ) -> Optional[SoundEngine]:
    """Bring sound in line with the stored preferences.

    Called from :func:`spacr.qt.preferences.apply_preferences_to_app`, at
    launch and after every Save. While sound has never been switched on in
    this process it reads one setting and returns: nothing is imported or
    built for a user who has not asked for sound.

    :param app: the application; the running instance when omitted.
    :param settings: settings to follow instead of the stored ones.
    :returns: the engine, or ``None`` when there is none.
    """
    app = app or QCoreApplication.instance()
    if app is None:
        return None
    engine = _ENGINE
    if settings is None:
        if engine is None or engine.closed:
            from .preferences import get_sound_enabled
            if not get_sound_enabled():
                return None
        settings = read_sound_settings()
    if engine is None or engine.closed:
        if not settings.enabled:
            return None
        engine = _create_engine(app)
    engine.apply(settings)
    return engine


def preview_sound(event: str, theme_key: str, volume: float,
                  app=None, music: str = "") -> bool:
    """Play one sound for a Preview button, starting the engine if needed.

    :param event: any event, the music bed included.
    :param theme_key: the sound set chosen in the dialog.
    :param volume: the dialog's volume slider as a fraction.
    :param app: the application; the running instance when omitted.
    :param music: the music file named on the page, unsaved.
    :returns: True when a sound was requested.
    """
    engine = _ENGINE
    if engine is None or engine.closed:
        app = app or QCoreApplication.instance()
        if app is None:
            return False
        engine = _create_engine(app)
    return engine.preview(event, theme_key, volume, music)


def stop_sound_preview() -> None:
    """End any preview and return to the stored settings."""
    engine = _ENGINE
    if engine is None or engine.closed:
        return
    engine.stop_preview()
    if not engine.settings.enabled:
        engine.apply(engine.settings)


def announce_run_end(status: str) -> bool:
    """Play the run-finished or run-failed sound for a run that ended.

    A run the user stopped is silent: they already know.

    :param status: ``"success"``, ``"failed"`` or ``"cancelled"``.
    :returns: True when a sound was requested.
    """
    engine = _ENGINE
    if engine is None or engine.closed:
        return False
    event = {"success": "run_finished", "failed": "run_failed"}.get(status)
    if event is None:
        return False
    return engine.play(event)


def shutdown_sound(timeout_ms: int = SHUTDOWN_WAIT_MS) -> bool:
    """Stop the process's engine, if there is one, and forget it.

    :param timeout_ms: how long to wait for the audio thread.
    :returns: True when nothing is left running.
    """
    global _ENGINE
    engine, _ENGINE = _ENGINE, None
    if engine is None:
        return True
    return engine.shutdown(timeout_ms)

