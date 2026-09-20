# Notes from `spacr/qt/resonance.py`

The module carries no comments; this is where its reasons live. Item 427,
part B, built 2026-09-19.

## What was asked

"The theme should sound like melodic space house" (maintainer, 2026-09-19),
and for the visualiser: "an audio-reactive visualiser (Chladni-like particle
figures) driven by the bed's precomputed envelope/spectrum in sync with
playback position, or a user-chosen local WAV; no system-audio capture. It
idles beautifully in silence, is frame-budgeted inside the existing ambient
producer architecture, and respects the Animation preferences."

The feature file's own paragraph says why Chladni and not something
prettier: "real resonance patterns on a vibrating plate are exactly
particles settling at the nodes, so the physical analogy gives a principled
rule for where particles should go rather than an arbitrary one." That is
the whole design brief and this module is the rule.

## The three decisions that cost the most to get right

### 1. Relaxation from a fixed start, not a simulation carried forward

The obvious way to draw sand on a plate is to integrate it: give every
grain a position and a velocity, push it down the gradient of `|w|`, damp
it, and step it once a frame. That is what a plate does and it is wrong
here, because **every ambient engine promises that a frame is a pure
function of `(seed, clock, size)`** and two tests hold it down:

* `test_the_clock_alone_decides_the_frame` steps one engine twelve times
  and jumps a second straight to the same clock and demands the same
  geometry and the same pixels;
* `test_the_backdrop_survives_a_run` shades the same engine at the same
  clock on two threads and compares the bytes.

A carried-forward simulation satisfies neither, and the second failure is
the dangerous one: the frame the shading thread produces would differ from
the frame the GUI thread would have produced, silently, only under load.

So the figure is recomputed every frame from seeded starting points. It
costs 0.61 ms for 1 300 grains at eight rounds, and it has the better
behaviour anyway: the figure always belongs to the sound that is playing
NOW rather than to the last few seconds of it.

**The lattice is a smaller win than it looks and the reason to keep it is
not speed.** Measured on this machine, eight rounds over 1 300 particles:

| where the field is evaluated | cost per frame |
|---|---|
| a 128x128 lattice (0.154 ms), then integer lookups | 0.61 ms |
| eight trigonometric functions per particle per round | 0.76 ms |

1.26x, not the order of magnitude the first draft of this note claimed
before anybody measured the alternative. What the lattice actually buys is
that the FIELD's share stops depending on the particle count, so the
density setting can be raised without the trigonometry following it: at
density 3.0 the lattice half is still 0.154 ms.

### 2. A Newton step, clamped, with reflection at the rim

Gradient descent on `w²` needs a step size that depends on the mode
numbers, because `|∇w|` scales with `m` and `n`. One Newton step onto the
linearised zero set does not:

    p -= t * w * ∇w / |∇w|²

and it lands on the nodal line in one step wherever the field is locally
straight. `|w|` at the particles goes from 0.203 to 0.013 in eight rounds.

**Two things that went wrong on the way and are now tested against.**

The denominator vanishes at an antinode, where the field is flat and its
gradient has nothing to say about where a nodal line is, so an unclamped
step flings those particles off the plate. Clipping the overshoot parks
them on the rim for good — the gradient at a corner is near zero, so
nothing ever moves them off again — and the first rendering drew four
bright dots in the corners of the plate that no figure explained.
`_MAX_STEP` bounds the step and `_reflect` folds an overshoot back inside
instead of clipping it, which is also what a grain bouncing off the rim
does. `test_nothing_piles_up_on_the_rim_or_in_the_corners_of_the_plate` sweeps
every entry of `MODES` — 1 200 grains per pair, 16 800 in all — because how
bad it gets depends on the mode: the pair the test started with showed five
corner grains where the sweep shows 162. Each guard removed on its own:

| | corners | on the rim |
|---|---|---|
| shipped | 3 | 0.85 % |
| without the step clamp | 17 | 1.05 % |
| clipping instead of reflecting | 7 | 1.97 % |
| neither | 162 | 4.15 % |

### 3. The sign, which is a whole family of figures

The classical square-plate superposition is

    w(x, y) = cos(nπx)cos(mπy) ± cos(mπx)cos(nπy)

and the first version only had the minus. With the minus sign, swapping x
and y exchanges the two terms, so `w(x, x) = 0` identically: **every figure
carries the same leading diagonal**, which is exactly what the first
contact sheet looked like. The plus family has no such line and draws the
crosses, lenses and closed rings. `MODES` alternates them, which doubled
the variety for one multiplication.

## The analysis, and why every band is normalised against itself

`analyse` writes per-frame loudness, four band energies, a spectral
centroid and an onset strength at 60 frames a second. Each row is scaled
against **its own** 98th percentile, not against the loudest thing in the
file. The shaker never rises within forty decibels of the kick, so a band
scaled by the file's peak would be a row of zeros — which is how an audio
visualiser ends up with three bars that move and one that does not.

That normalisation is also why the first version of
`test_each_band_answers_its_own_part_of_the_spectrum` said nothing: it fed
four files of one steady tone each, and a band that is silent throughout
and a band that is loud throughout both come back as a flat row after
normalising. The test now plays the four tones one after another in ONE
file, where the normalisation is what lets each band speak.

The onset gets a 0.28 s release (`ONSET_RELEASE_S`) applied in the
analysis, run twice round for a looping piece so the release that starts on
the last beat is still falling under the first one. Without it a flux spike
one analysis frame wide is a spike the 24 fps painter shows or misses
depending on where its tick falls.

Measured on the reference bed (32 bars, 62.95 s):

| | |
|---|---|
| analysis time | 63 ms |
| file on disk | 89.7 kB, `bed.wav.resonance.npz` |
| frames | 3 777 |
| band 3 (the shaker's) mean, by section | drift 0.09, pulse 0.25, lift 0.32, return 0.17 |

## Sync without a playback position, and without capture

`QSoundEffect` has no position to ask for, and capturing the machine's
audio output is not something a scientific tool should do to somebody's
computer — `test_no_audio_input_is_ever_opened` greps for `QAudioInput`,
`QAudioSource`, `QMediaCaptureSession`, `sounddevice` and `pyaudio` across
the three modules and finds none.

So the audio thread records the instant it started (`NowPlaying.started`,
from `time.monotonic`) and the loop's length, and the position is
arithmetic. Both clocks are real time, so they drift only by the audio
device's own rate error — tens of parts per million, about a millisecond an
hour — and the loop wraps it out.

**The instant recorded is the instant the file becomes LOADED, not the
instant `play()` returned.** `QSoundEffect.play()` on a source that is
still loading queues the play until it is ready, and a twelve-megabyte WAV
takes long enough to decode that a visualiser started on the wrong instant
would be visibly behind. `_AudioWorker._announce_when_loaded` asks
`isLoaded()` and connects `loadedChanged` when the answer is no; a
stand-in effect that cannot say is taken at its word and announced at once.

**Not measured on a real sound server.** Only Linux with a faked sink has
been run here, so the load latency itself is unknown; what is known is that
the code does not depend on it.

## Why the analysis is resolved in `set_now_playing` and not where it is read

`playing_moment` is called once a frame from the backdrop's clock, which
runs on the GUI thread. The first version resolved the analysis there, and
`load_analysis` starts with an `os.stat` — a filesystem call on the GUI
thread, on every frame, which on a network home directory is exactly the
stall `spacr/qt/path_probe.py` exists for. It now resolves once, in
`set_now_playing`, on the audio thread that has just written the file.
`test_reading_the_moment_never_touches_the_filesystem` counts calls to
`os.stat` and `np.load` during one read and requires zero.

That test also records a smaller lesson: the first version made `os.stat`
RAISE, and because `rs.os` is the `os` module the raise landed in the
settings store's teardown rather than in the call under test. Counting is
the right instrument when the thing being patched is global.

## What is not here

* **No streaming service and no bundled recordings.** Decided 2026-09-19.
* **No resampling.** A WAV at a rate other than 48 kHz is analysed at its
  own rate, which is correct; `QSoundEffect` does its own resampling for
  playback.
* **16-bit PCM only.** `read_wav_mono` refuses anything else rather than
  guessing, and `ensure_analysis` turns the refusal into "no analysis",
  which the backdrop reads as silence. A file Qt cannot play is a file this
  cannot analyse, and the two failures agree.

## The sidecar never goes in the user's music folder

`ensure_analysis` writes beside the WAV by default, which is right for
spaCR's own rendered bed: it already lives under `~/.spacr/sounds` and is
swept away with the theme folder when the fingerprint changes.

For a file the user chose it is wrong twice over — a tool asked to READ a
file should not leave a sidecar beside it, and somebody's music folder may
not be writable at all. `spacr.qt.sound` passes `out_dir`, and the name
carries twelve hex digits of the source's absolute path so two files
called `loop.wav` in different folders are two analyses.

Not swept: an analysis of a file the user has stopped using stays in the
cache. They are about ninety kilobytes each and nobody changes their bed
music often enough for it to matter, so there is no reaper. Say so rather
than pretend it is tidy.
