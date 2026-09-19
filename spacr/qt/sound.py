"""Play spaCR's sounds: off by default, off the GUI thread, silent when it cannot.

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
* **The audio device is never touched from the GUI thread.** Constructing
  the first ``QSoundEffect`` connects to the sound server, and that took
  354 ms on the maintainer's workstation (PipeWire) and 224 ms on the way
  to failing with no server at all -- a visible freeze either way. So every
  effect lives on a dedicated ``QThread`` (:class:`_AudioWorker`), which is
  also where sounds are synthesized the first time a set is used, and the
  audio devices are listed there before the first effect is built: without
  that, the constructor still stopped the GUI thread from the audio thread
  (see :meth:`_AudioWorker._make_effect`). The GUI thread only emits a
  queued signal naming the sound to play.
* **A missing sound stack is silence.** No device, no server, no Qt
  Multimedia, a file that will not load: each ends in a debug log line and
  no sound. Nothing here raises into a caller and nothing opens a dialog.

The hover sound is the risky one, so it is debounced twice: the pointer must
rest on an enabled control for :data:`HOVER_SETTLE_MS` before anything is
considered, and a hover sound never follows another within
:data:`HOVER_COOLDOWN_S` or a click within :data:`HOVER_AFTER_CLICK_S`.
"""
from __future__ import annotations

import logging
import threading
import time
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
    """

    enabled: bool = False
    volume: float = 0.5
    theme: str = DEFAULT_THEME
    click: bool = True
    hover: bool = False
    run_finished: bool = True
    run_failed: bool = True
    bed: bool = False

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


class _AudioWorker(QObject):
    """Owns every sound effect, on the audio thread.

    Everything that can block -- synthesizing a set, connecting to the
    sound server, loading a file -- happens in these slots, which run on
    the thread this object was moved to. With ``threaded=False`` on the
    engine they run inline instead, which is how tests drive it.

    :param effect_factory: builds one effect given its parent. ``None`` means
        ``QSoundEffect``; tests pass a stand-in that records what it is
        asked to do.
    :param cache_root: where rendered files go; ``None`` means
        :func:`spacr.qt.sound_synth.sound_cache_root`.
    """

    #: (theme key, whether Qt Multimedia could build an effect at all). A
    #: device that exists but refuses the format still reads True here: the
    #: effect then reports an error status and plays nothing.
    prepared = Signal(str, bool)

    #: A Preview of the music bed has finished fading out. The worker knows
    #: only that it faded something; the engine is the one that knows what
    #: the stored settings want playing afterwards.
    bed_faded = Signal()

    def __init__(self, effect_factory: Optional[Callable] = None,
                 cache_root: Optional[Path] = None) -> None:
        """Hold the factory and cache root; build nothing yet."""
        super().__init__()
        self._factory = effect_factory
        self._root = cache_root
        self._theme_key = ""
        self._paths: Dict[str, Path] = {}
        self._effects: Dict[str, object] = {}
        self._available = True
        self._stopping = threading.Event()
        self._bed_playing = False
        self._fade_timer: Optional[QTimer] = None
        self._devices_listed = False
        self._fade_steps_left = 0
        self._fade_from = 0.0

    def request_stop(self) -> None:
        """Abandon any render at its next file boundary. Thread-safe."""
        self._stopping.set()

    def _switch_theme(self, key: str) -> None:
        """Forget the previous set's effects when the set changes."""
        if key != self._theme_key:
            self._drop_effects()
            self._paths = {}
            self._theme_key = key

    def _path(self, name: str) -> Optional[Path]:
        """The file for ``name``, rendering it first if it is missing."""
        if name not in self._paths:
            try:
                ready = ensure_rendered(_theme(self._theme_key), [name],
                                        root=self._root,
                                        should_stop=self._stopping.is_set)
            except Exception:                                # noqa: BLE001
                LOG.debug("could not render the %s sound", name,
                          exc_info=True)
                return None
            self._paths.update(ready)
        return self._paths.get(name)

    def _make_effect(self):
        """One new effect, parented here so it lives on this thread.

        The audio devices are listed once BEFORE the first effect is built.
        Measured 2026-09-19 with the GUI thread ticking every 5 ms: when the
        first Qt Multimedia call on the audio thread was the ``QSoundEffect``
        constructor, the GUI thread stopped for 180-320 ms while the sound
        server was connected; when it was
        ``QMediaDevices.defaultAudioOutput()``, the same connection took
        185-201 ms on the audio thread and the GUI thread's worst gap was
        5.2-8.3 ms, and the constructor that followed took under 1 ms.
        Where the module was imported made no difference either way.
        """
        if self._factory is not None:
            return self._factory(self)
        from PySide6.QtMultimedia import QMediaDevices, QSoundEffect
        if not self._devices_listed:
            QMediaDevices.defaultAudioOutput()
            self._devices_listed = True
        return QSoundEffect(self)

    def _effect(self, name: str):
        """The loaded effect for ``name``, or ``None`` when there is none."""
        if not self._available:
            return None
        effect = self._effects.get(name)
        if effect is not None:
            return effect
        path = self._path(name)
        if path is None:
            return None
        try:
            effect = self._make_effect()
            effect.setSource(QUrl.fromLocalFile(str(path)))
        except Exception:                                    # noqa: BLE001
            LOG.debug("no sound output is available; spaCR stays silent",
                      exc_info=True)
            self._available = False
            return None
        self._effects[name] = effect
        return effect

    @Slot(str, object)
    def prepare(self, theme_key: str, names) -> None:
        """Render and load ``names`` now, so their first play is instant.

        :param theme_key: the sound set.
        :param names: stems from :func:`spacr.qt.sound_synth.sound_names`.
        """
        self._switch_theme(theme_key)
        for name in list(names or ()):
            if self._stopping.is_set():
                break
            self._effect(name)
        self.prepared.emit(theme_key, bool(self._available))

    @Slot(str, str, float)
    def play(self, theme_key: str, name: str, gain: float) -> None:
        """Play one sound once.

        :param theme_key: the sound set.
        :param name: the sound's stem.
        :param gain: linear gain, 0 to 1.
        """
        self._switch_theme(theme_key)
        effect = self._effect(name)
        if effect is None:
            return
        try:
            effect.setVolume(float(gain))
            effect.play()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not play the %s sound", name, exc_info=True)

    @Slot(str, bool, float)
    def bed(self, theme_key: str, on: bool, gain: float) -> None:
        """Start, re-level or stop the looping music bed.

        :param theme_key: the sound set.
        :param on: whether the bed should be playing.
        :param gain: its linear gain.
        """
        self._stop_fade()
        if not on:
            effect = self._effects.get(BED)
            if effect is not None:
                self._quietly(effect.stop)
            self._bed_playing = False
            return
        self._switch_theme(theme_key)
        effect = self._effect(BED)
        if effect is None:
            return
        try:
            effect.setVolume(float(gain))
            if not self._bed_playing:
                effect.setLoopCount(LOOP_FOREVER)
                effect.play()
                self._bed_playing = True
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not start the music bed", exc_info=True)

    @Slot(str, float, int)
    def preview_bed(self, theme_key: str, gain: float, milliseconds: int) -> None:
        """Play the music bed for a few seconds, then fade it out.

        :param theme_key: the sound set.
        :param gain: linear gain.
        :param milliseconds: how long it plays before the fade begins.
        """
        self._switch_theme(theme_key)
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
            self._bed_playing = False
            self.bed_faded.emit()
            return
        self._quietly(effect.setVolume,
                      self._fade_from * self._fade_steps_left / FADE_STEPS)
        if self._fade_timer is not None:
            self._fade_timer.setSingleShot(True)
            self._fade_timer.start(max(1, FADE_MS // FADE_STEPS))

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
        """Stop every effect and delete it now, on this thread.

        Deleted at once rather than with ``deleteLater``. A deferred
        deletion is a promise that the owner outlives the next pass of the
        event loop, and nothing here can keep it: the worker is owned by
        Python, and a worker collected before that pass takes its effects
        with it while their deletion events are still queued. Qt reports
        "shared QObject was deleted directly" and the next
        ``sendPostedEvents`` segfaults in ``~QObject`` -- reproduced with
        the input filter, which went the same way, on 2026-09-19.
        """
        self._stop_fade()
        effects, self._effects = self._effects, {}
        self._bed_playing = False
        for effect in effects.values():
            self._quietly(effect.stop)
            if isinstance(effect, QObject):
                self._quietly(_delete_now, effect)
            else:
                dispose = getattr(effect, "deleteLater", None)
                if callable(dispose):
                    self._quietly(dispose)

    @Slot()
    def release(self) -> None:
        """Let go of every effect, and with them the audio device.

        Called on this worker's thread when sound is switched off.
        """
        self._drop_effects()
        self._paths = {}
        self._theme_key = ""

    @Slot()
    def retire(self) -> None:
        """Release everything as the audio thread ends, then come home.

        Connected to ``QThread.finished`` directly, so it runs on the audio
        thread in its last moments. Moving the worker to the application's
        thread afterwards means whatever destroys it later -- Python's
        collector, on the GUI thread -- destroys an object of its own
        thread, rather than one belonging to a thread that has gone.
        """
        self.release()
        if self._fade_timer is not None:
            self._quietly(_delete_now, self._fade_timer)
            self._fade_timer = None
        app = QCoreApplication.instance()
        if app is not None:
            self._quietly(self.moveToThread, app.thread())


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
    """Decides when a sound plays and asks the audio thread to play it.

    Lives on the GUI thread and does nothing there that can wait: every
    request is a queued signal to :class:`_AudioWorker`.

    :param parent: owning object, normally the application.
    :param threaded: run the worker on its own ``QThread``. ``False`` runs
        it inline, emitting the same calls in the same order, so a test can
        drive the engine synchronously.
    :param effect_factory: see :class:`_AudioWorker`.
    :param cache_root: see :class:`_AudioWorker`.
    """

    _ask_prepare = Signal(str, object)
    _ask_play = Signal(str, str, float)
    _ask_bed = Signal(str, bool, float)
    _ask_preview_bed = Signal(str, float, int)
    _ask_release = Signal()

    #: (theme key, whether Qt Multimedia could build an effect), from the
    #: worker.
    prepared = Signal(str, bool)

    def __init__(self, parent: Optional[QObject] = None, *,
                 threaded: bool = True,
                 effect_factory: Optional[Callable] = None,
                 cache_root: Optional[Path] = None) -> None:
        """Start the audio thread (when threaded) and wire the requests."""
        super().__init__(parent)
        self._settings = SoundSettings()
        self._closed = False
        self._worker = _AudioWorker(effect_factory, cache_root)
        self._thread: Optional[QThread] = None
        if threaded:
            thread = QThread()
            thread.setObjectName("spacr-sound")
            self._worker.moveToThread(thread)
            thread.finished.connect(self._worker.retire,
                                    Qt.ConnectionType.DirectConnection)
            self._thread = thread
        self._ask_prepare.connect(self._worker.prepare)
        self._ask_play.connect(self._worker.play)
        self._ask_bed.connect(self._worker.bed)
        self._ask_preview_bed.connect(self._worker.preview_bed)
        self._ask_release.connect(self._worker.release)
        self._worker.prepared.connect(self._on_prepared)
        self._worker.bed_faded.connect(self._on_bed_faded)
        if self._thread is not None:
            self._thread.start()

        #: Whether the last preparation could build an effect at all.
        #: ``None`` until the worker has answered once.
        self.available: Optional[bool] = None
        #: Built the first time it is needed and kept for the engine's
        #: life; installed and removed, never deleted, for the reason
        #: :meth:`_AudioWorker._drop_effects` gives.
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
        """The thread every effect lives on, or ``None`` when unthreaded."""
        return self._thread

    @Slot(str, bool)
    def _on_prepared(self, theme_key: str, available: bool) -> None:
        """Record whether the device could be used, and pass it on."""
        self.available = bool(available)
        self.prepared.emit(theme_key, bool(available))

    @Slot()
    def _on_bed_faded(self) -> None:
        """A bed Preview ended: put the stored settings back on the bed."""
        self.stop_preview()

    def apply(self, settings: SoundSettings) -> None:
        """Follow ``settings``: filter, loaded sounds and music bed.

        Switching the master off removes the filter and releases every
        effect, so a disabled engine holds no audio device. The thread
        itself stays, idle, until the application quits.

        :param settings: what to follow from now on.
        """
        if self._closed:
            return
        self._settings = settings
        self._sync_filter()
        if not settings.enabled:
            self._hover_timer.stop()
            self._hover_target = None
            self._ask_release.emit()
            return
        names: List[str] = []
        for event in FEEDBACK_EVENTS:
            if settings.wants(event):
                names.extend(sound_names(event))
        self._ask_prepare.emit(settings.theme, names)
        self._ask_bed.emit(settings.theme, settings.wants(BED),
                           settings.gain * BED_LEVEL)

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

    def preview(self, event: str, theme_key: str, volume: float) -> bool:
        """Play ``event`` once because the user pressed its Preview button.

        Plays whatever the stored switches say: the dialog enables Preview
        only while its own master switch is on, and the press is the ask.

        :param event: any event, the music bed included.
        :param theme_key: the sound set chosen in the dialog.
        :param volume: the dialog's volume slider as a fraction.
        :returns: True when a sound was requested.
        """
        if self._closed:
            return False
        gain = perceived_gain(volume)
        if event == BED:
            self._ask_preview_bed.emit(theme_key, gain * BED_LEVEL,
                                       PREVIEW_BED_MS)
            return True
        self._request(theme_key, event, gain)
        return True

    def stop_preview(self) -> None:
        """Put the music bed back to what the stored settings say."""
        if self._closed:
            return
        self._ask_bed.emit(self._settings.theme, self._settings.wants(BED),
                           self._settings.gain * BED_LEVEL)

    def _request(self, theme_key: str, event: str, gain: float) -> None:
        """Ask the worker for the next variant of ``event``'s sound."""
        names = sound_names(event)
        index = self._rotation.get(event, 0)
        self._rotation[event] = (index + 1) % len(names)
        self._ask_play.emit(theme_key, names[index], float(gain))

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
        self._worker.request_stop()
        if self._thread is None:
            self._worker.release()
            return True
        from .bridge import drain_thread
        return drain_thread(self._thread, self._worker, int(timeout_ms))


#: The one engine this process has, once anybody has turned sound on.
_ENGINE: Optional[SoundEngine] = None


def sound_engine() -> Optional[SoundEngine]:
    """The running engine, or ``None`` when sound was never switched on."""
    return _ENGINE


def _create_engine(app) -> SoundEngine:
    """Build the process's engine and have it stop when the app quits."""
    global _ENGINE
    parent = app if isinstance(app, QObject) else None
    engine = SoundEngine(parent)
    about_to_quit = getattr(app, "aboutToQuit", None)
    if about_to_quit is not None:
        try:
            about_to_quit.connect(engine.shutdown)
        except (RuntimeError, TypeError):
            LOG.debug("could not tie the sound engine to quitting",
                      exc_info=True)
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
                  app=None) -> bool:
    """Play one sound for a Preview button, starting the engine if needed.

    :param event: any event, the music bed included.
    :param theme_key: the sound set chosen in the dialog.
    :param volume: the dialog's volume slider as a fraction.
    :param app: the application; the running instance when omitted.
    :returns: True when a sound was requested.
    """
    engine = _ENGINE
    if engine is None or engine.closed:
        app = app or QCoreApplication.instance()
        if app is None:
            return False
        engine = _create_engine(app)
    return engine.preview(event, theme_key, volume)


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

