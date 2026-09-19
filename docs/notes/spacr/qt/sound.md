# Notes from `spacr/qt/sound.py`

The module carries no comments; this is where its reasons live. Item 427,
part A, built 2026-09-19.

## Why the audio device is never touched from the GUI thread

Measured on the maintainer's workstation (PipeWire, PySide6 6.11.2, Qt
Multimedia over FFmpeg 7.1.5), in a probe with `QT_QPA_PLATFORM=offscreen`
and a private `HOME`:

| call, first in the process | time |
|---|---|
| `import PySide6.QtMultimedia` | 7.7 ms |
| `QMediaDevices.audioOutputs()` | 415 ms |
| `QSoundEffect()` (when nothing above ran first) | 354 ms |
| the same with no sound server (`PULSE_SERVER=unix:/nonexistent`) | 224 ms, then `Status.Error` |
| `setSource`, `play` afterwards | 0.1-0.2 ms each |

The first multimedia call is where the sound server is connected to. A
third of a second on the GUI thread is a visible freeze, and it is paid
again on a machine with no server at all, so every effect lives on a
`QThread` named `spacr-sound`.

## Moving it to a thread was not enough on its own

THE FIRST MEASUREMENT OF THIS WAS WRONG, and the way it was wrong is worth
keeping. A probe that built the effect on a worker and ticked a 10 ms timer
on the GUI thread reported "longest gap 10.3 ms" -- because it measured
gaps between ticks AFTER the work started, and the GUI thread had not
ticked at all until the effect was built. 138 ticks in 1.5 s where 150 were
due was the only trace of it. Re-measured with the tick before the work
included:

| first Qt Multimedia call on the audio thread | that call | worst GUI gap (5 ms ticks) |
|---|---|---|
| `QSoundEffect()` (plain Python thread) | 316 ms | 321 ms |
| `QSoundEffect()` (`QThread`) | 180-193 ms | 180-194 ms |
| `QMediaDevices.defaultAudioOutput()`, then `QSoundEffect()` | 185-201 ms, then 0 ms | 5.2-8.3 ms |

Where `PySide6.QtMultimedia` was first imported (GUI thread or audio
thread) made no difference in either row. So `_AudioWorker._make_effect`
lists the devices once before it builds the first effect, and
`test_the_devices_are_listed_before_the_first_effect_is_built` pins the
order. With that in place, measured through the real Preferences dialog
(`QT_QPA_PLATFORM=offscreen`, private `HOME`, volume 0):

| | worst GUI gap | ready after |
|---|---|---|
| first ever: `scipy.signal` imported and the set rendered on the audio thread | 34-52 ms | 1.3-1.8 s |
| first switch-on in a FRESH PROCESS, files already rendered | 30.0, 30.2, 31.6 and 9.5 ms in four runs | 0.22-0.32 s |
| a later switch-on in the SAME process | 5.4 ms, which is the tick itself | 3 ms |

THE MIDDLE ROW WAS WRONG AND IS THE ONE THAT MATTERS, corrected 2026-09-19
after review. It read "every later start: 5.3 ms (the tick itself), 0.26 s",
and 5.3 ms is real -- for a SECOND switch-on inside one process, where the
sound server is already connected. The case a user actually meets, launching
spaCR with sound already on, is the middle row: about 30 ms, once per
process, landing +192 to +282 ms in, which is the moment the audio thread's
first `QMediaDevices.defaultAudioOutput()` returns. Listing the devices
first moved that call OFF the GUI thread and took the gap from 180-320 ms to
about 30; it did not take it to zero, and writing 5.3 ms in this row claimed
it had. Measured four times in fresh interpreters against the same warm
cache, sandboxed `HOME`/`XDG_*`/`SPACR_SOUND_CACHE`, offscreen, volume 0,
5 ms ticks, the tick before the work started included.

What remains of the first-ever gap is synthesis sharing the interpreter
lock with the GUI thread, once per set per machine. Pressing Save with sound
switched on took 406-416 ms against 400-410 ms with it left off: the engine
and its thread add under 10 ms to Save.

MEASURING THIS THROUGH THE PREFERENCES DIALOG DOES NOT WORK and cost an
afternoon. `Save` runs ~390-460 ms of unrelated preference work
synchronously, with no event-loop tick anywhere inside it, so the one gap
in the window is Save's own; the audio thread's connection happens
underneath it and never shows. The rows above therefore drive
`apply_sound_preferences` directly, which is the same call Save makes.

## Why effects are deleted at once and never with `deleteLater`

Found 2026-09-19 while writing the tests, reproduced outside pytest:

    engine.apply(SoundSettings(enabled=True))   # installs the input filter
    engine.shutdown()                          # removes it, deleteLater()
    del engine                                 # Python owns the engine
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)

prints `QObject: shared QObject was deleted directly. The program is
malformed and may crash.` and then segfaults inside `QObject::~QObject`
(native backtrace via gdb). A deferred deletion promises that the owner
outlives the next pass of the event loop; an owner collected by Python
before that pass breaks the promise. With the filter kept as a child for
the engine's whole life and only installed and removed, the same script
survives `keep`, `del` and `gc.collect()` alike. Effects follow the same
rule: stopped and deleted immediately, on the audio thread, by
`_AudioWorker._drop_effects`; the worker is moved back to the application
thread in `retire` as the audio thread ends, so whatever collects it later
collects an object of its own thread.

## Why the input filter exists only while it is wanted

Item 380 measured an application-wide filter that does nothing at 0.93 us
per delivered event, and 135,800 events delivered while one module screen
opens. So `InputSoundFilter` is installed only while the click or the hover
sound is switched on, and removed the moment neither is; with sound off --
every fresh install -- it does not exist. Its path for an event it ignores
is two `shiboken6.isValid` checks and at most four comparisons. Measured
through `QApplication.sendEvent` on the real application, 20,000 events
each way: 5.61 us per event with the filter installed against 4.63 us
without, so the filter costs 0.97 us per delivered event while installed
-- the same order as item 380's do-nothing filter.
`test_an_uninteresting_event_costs_microseconds` holds a direct call under
15 us as a loose ceiling.

## The hover rules, and why each exists

* `HOVER_SETTLE_MS` (110 ms): the pointer must REST on an enabled control.
  Sweeping across a panel crosses each control in well under that, so a
  sweep is silent.
* `HOVER_COOLDOWN_S` (0.42 s): never two hover sounds closer than this,
  whatever the pointer does.
* `HOVER_AFTER_CLICK_S` (0.30 s): a click that opens something under the
  pointer would otherwise be answered twice.
* Only buttons, combo boxes and tab bars hover. A slider does not: dragging
  near one is not a question the user asked.
* A control disabled while the pointer settles is silent; checked again when
  the timer fires, not only when the pointer arrived.

Each rule has a test that fails when the rule is removed (mutation-checked
2026-09-19). The filter's own `isEnabled` check on Enter is an early-out
only: removing it leaves the property intact because `_hover_settled`
checks again.

## The music bed

Plays at `BED_LEVEL` (0.6) of the other sounds' gain, and not at all at the
Laptop and Extra Performance levels (`preferences.SOUND_BED_RESTS_AT`) --
the two levels that switch the animated backdrop off, which is the nearest
thing spaCR has to a reduced-motion preference. A Preview plays it for
`PREVIEW_BED_MS` and fades it out over `FADE_MS` in twelve steps; closing
Preferences ends any preview and, if sound is off, lets go of the device.

THE FADE THAT ENDS A PREVIEW USED TO END THE BED ITSELF, found in review
2026-09-19. For a user who already had the music bed switched on, pressing
Preview stopped their bed for as long as Preferences stayed open: the last
step of the fade calls `stop()`, and nothing afterwards knew the settings
still wanted it playing. The worker cannot know that -- it has the gain it
was handed and no settings -- so the last fade step emits `bed_faded` and
the engine answers it with `stop_preview()`, which is the same "put the
stored settings back on the bed" it already ran when the dialog closed.

## Why a control can ask not to click

`SILENT_PRESS_PROPERTY` (`"spacrSilentPress"`) is a Qt dynamic property the
filter reads on a left press on something pressable, after the `isEnabled`
check and nowhere else, so it costs nothing on the events the filter
already ignores. The Sound tab's Preview buttons carry it: a Preview press
is the user asking to hear ONE sound, and the app-wide filter answering the
same press with a click plays two -- for the Click row's own Preview, the
same pluck twice, 20 ms apart. Found in review 2026-09-19.

`spacr/qt/sound_preferences.py` sets the property by its literal name
rather than importing this module, because building the Sound tab must not
import the sound engine for somebody who has sound switched off. Nothing
pins the two spellings against each other directly; what pins them is
`test_a_preview_press_is_not_also_a_click`, which drives the real dialog
with sound on and hears what comes out.

## A run the user stopped is silent

`announce_run_end("cancelled")` plays nothing: the user who pressed Stop
knows. Success and failure have different figures (see
`spacr/qt/sound_synth.py`) because "a user who hears 'done' and finds a
traceback will not trust the sound again" (427, as filed).

## Who is allowed to import this module

"Nothing exists while sound is off" is a promise about the IMPORT, not only
about the engine, and `spacr/qt/notify.py` was quietly breaking it: its
`announce_pipeline_finished` did `from .sound import announce_run_end`
unconditionally, so the first run that ended in any session imported this
module and `sound_synth` on the GUI thread -- 20.4 ms in a warm app -- for
a user who had never asked for sound. Found in review 2026-09-19.

Both callers outside Preferences now ask `sys.modules` first, the way
`preferences.apply_preferences_to_app` already did: `notify` before the run
sound, and `sound_preferences.SoundPage` before ending a preview. The guard
loses nothing, because an unimported module means no engine, and without an
engine `announce_run_end` and `stop_sound_preview` both return at once --
the engine is only ever built by `apply_sound_preferences`, which runs at
launch and after every Save. `test_a_finishing_run_does_not_import_the_engine_to_stay_silent`
ends two runs in a fresh interpreter and then reads `sys.modules`.
