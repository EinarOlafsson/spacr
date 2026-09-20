# Notes from `spacr/qt/night_themes.py`

The module carries no comments; this is where its reasons live. Item 427,
part C, built 2026-09-19.

Asked 2026-09-19 by the maintainer, through the question prompt: "make 10
themes for spacr that should all try and capture a space house theme
vibe. Artists I like include worakls, NTO, rivière monk, BIRRD." And,
separately: "The theme should sound like melodic space house."

## Why there is a module at all, instead of ten more entries in `theme.py`

A night theme is not a palette. It is a palette, an ambient animation with
a colour set, and a sound set, and those three live in three modules that
must not import one another:

* `spacr/qt/theme.py` imports QtGui and QtWidgets.
* `spacr/qt/widgets/ambient.py` imports the whole widget stack.
* `spacr/qt/sound_synth.py` is deliberately Qt-free, and `numpy`-only.
* `spacr/qt/preferences.py` is deliberately QtCore-only, and says so at
  the line where it restates `theme.THEMES` as `PALETTE_THEMES`: "restated
  here so importing this module does not pull in QtGui/QtWidgets".

Something has to hold the fact that Nocturne is indigo AND a starfield AND
108 BPM. Putting it in any of the four drags that module's imports into the
other three. So `night_themes.py` holds the data, imports nothing from
spaCR, and the other four read *from* it. `preferences.py` can then keep its
promise, and `theme.THEMES` and `preferences.PALETTE_THEMES` cannot drift,
because they are now the same tuple plus the same ten keys rather than two
hand-written lists that happen to match.

## Why the palettes are written down rather than generated

They were generated. A hue, a saturation and a shared lightness ramp
produce a whole palette, and that is how these ten were built — the ramp
was lifted off the retired Space palette, which is the night palette the
application already had. But the recipe is not what ships:

* the published colours are then the reviewed colours, which matters for a
  palette, where the answer to "is this right" is "look at it";
* no solver runs at import, which is item 284's two-second start;
* `tests/qt/test_ten_night_themes.py` judges what is in the file rather
  than re-running the generator and agreeing with itself.

Three colours were moved off the recipe by measurement, and only three.

Nocturne's and Aphelion's `accent_lo` failed `bg` versus `accent_lo` at
4.5:1 undressed — a deep blue and a deep violet at the recipe's lightness
of 0.53 are simply too dark to clear AA against a near-black window — so
both were raised, to 0.635 and 0.615, until they passed.

Vesper's `accent_lo` passed undressed and failed DRESSED, at 4.49:1
against 4.5:1, at one hue offset out of sixty — 60°, found by
`test_spaceout_looks_alive.py`. The spaceout dressing re-hues a colour
while holding its luminance, and holding it is an 8-bit rounding away from
exact, so a colour sitting on 4.50 will sometimes land on 4.49. `#dd316a`
became `#dd336c`, one step of lightness, and all sixty offsets pass. A
sweep over ten themes and sixty offsets found one number; that is the
argument for running it rather than reasoning about it.

Every other value in the ten palettes is the recipe's.

## Why `success`, `warning` and `error` are identical in all ten

`#66d68f`, `#eec14f`, `#ff7a70`, in every one of the ten. The four older
themes each have their own, tinted to the palette, and this family
deliberately does not: a red that means *this run failed* must not become a
theme decision. A user who learns one spaCR's error colour has learned all
ten. `fg` is `#ffffff` in all ten for the ordinary reason a night palette
wants it.

`chip_class` is teal in all ten for the same reason it is teal in all four
of the older palettes: it is a role with a meaning, not a decoration.

## How the ten were checked

`theme.contrast_failures` and `theme.page_separation_failures` — the two
published rules the four older themes are judged by — over all ten,
through `theme._PALETTES`, so the rule is judging the shipped colours and
not a copy of them. All ten pass with zero failures, and the sweep is in
the test file rather than only in a probe.

## Why choosing a theme writes three preferences and switches nothing on

`preferences.apply_night_theme` writes the ambient animation, the ambient
palette and the sound set. It is a preset: the values go into the same
three keys the Animation, Animation palette and Sound set controls read
and write, once, at the moment the theme is chosen. Nothing re-imposes
them afterwards, so a user who picks Nocturne and then changes the
animation to Bokeh keeps Bokeh.

It does not touch the sound master. Everything 427 built is off by
default and stays that way; all a theme decides is *which* set would play
if sound were ever switched on.

**And it must not restart a backdrop the user turned off.** This is the
trap in the design and it was found by driving it rather than by reading
it. "No animation" is not a separate flag — it is stored as the animation
NAME, `ambient.NO_ANIMATION`, which is exactly the key a preset would
overwrite. Writing `drift` into it would have handed a moving backdrop to
a user who had turned motion off, through the Animation control or through
Extra Performance, which turns it off the same way.
`preferences.backdrop_is_switched_off` is therefore asked first, and when
it answers yes the two ambient keys are left alone. It reads the STORED
choice and not `get_ambient_enabled()`, which also answers False for
`SPACR_NO_BACKDROP` — a one-process suppression set by
`spacr.qt.crash_recovery` after two failed launches, and a suppression is
not a preference.

What that costs, plainly: a user who later switches the backdrop back on
gets the animation they had before, not the one their theme would have
brought. They can pick it on the control they just used.

## Why the binding is ALSO in the dialog, not only in the setter

`PreferencesDialog._save` writes the Theme control and then writes the
Animation, Animation palette and Sound set controls three lines later. A
preset applied inside `set_theme_choice` alone would have been overwritten
by whatever the untouched combo boxes still held, and the dialog is how
almost every user will meet this. So the Theme combo moves the other three
combos when a night theme is picked, live. The two paths then agree, and —
more to the point — the user SEES what the theme brought and can put any
of it back before pressing Save.

## Why ten themes did not need an eleventh animation

The instruction says "reuse and parameterise the existing producers", and
they were: the seven in `ambient.AMBIENT_THEMES` are untouched and no
eighth was added. Ten themes over seven producers means a producer is used
twice; what is not reused is the PAIR, so no two themes put the same
picture on the screen. Four colour sets were added — `midnight`, `dusk`,
`lowsun`, `deepwater` — and offered per animation by the same rule the
file already applied to `borealis` and `fluor`: a set is offered where the
animation reads as the thing the set is named after.

## Why `theme.py` grew a lazy dressing solve

Ten more themes cost the `spaceout` launcher 445 ms before this was fixed,
and the fix belongs here because this work is what exposed it.

`theme._apply_dressing` solved the page damping and the ink bands for
EVERY theme at the moment `enable_spaceout()` was called, and both solves
walk the sixty hue offsets of `_drift_grid()` per theme — 28 ms and 19 ms
each. Four themes is 190 ms; fourteen is 640 ms. Measured, three runs
each, offscreen:

| | before | after |
|---|---|---|
| `enable_spaceout()`, four themes | 417, 418, 438 ms | — |
| `enable_spaceout()`, fourteen, eager | 864, 876, 866 ms | — |
| `enable_spaceout()`, fourteen, lazy | — | 433 ms |

It is paid before the window appears, by `spacr/qt/spaceout.py`, and
thirteen of the fourteen solves were for a palette that run would never
paint: the process is in ONE theme.

So `theme.DRESSED_EAGERLY` names the four that can be on screen without
anybody choosing, and `theme._dress_theme` solves any other the first time
`palette_for` is asked for it — about 42 ms, once per theme per process.
The guard against recursion is the `_DRESSED` entry, added BEFORE the two
solves and not after: every solver resolves palettes through
`palette_for`, so a solve for a theme re-enters `_dress_theme` for that
theme and finds it already in the set. `palette_for` also only calls it
when `_SOLVE_DRIFT is None` — "no solve is running" — and that second
check is REDUNDANT. Removing it turns no test red, because a solve only
ever asks for the palette of the theme it is solving. It is recorded as
redundant rather than left looking load-bearing, and kept because it costs
one comparison and holds the property if a future solver reaches for a
second theme.

**The change was checked by equality, not by inspection.** Dress lazily,
read all fourteen palettes, then run the eager solve and compare: the
damping tables, the ink bands and all fourteen dressed palettes come out
identical. `test_ten_night_themes.py` holds that down.

## What a night theme does NOT do

* It does not make the backdrop follow the music's tempo. The instruction
  mentions "rings pulsing near 122 BPM"; the ripple producer has no
  tempo input, the Speed preference is the user's, and the one backdrop
  that does answer the music is Resonance, which part B built. Aphelion
  uses Resonance for that reason. The other nine run at whatever speed the
  user set.
* It does not change the light theme's handling anywhere. All ten are
  dark, and `theme.py`'s two `theme == "light"` branches therefore take
  their dark path for them, which is correct. One place is worth knowing
  about: `screens/train_compare.py` resolves `palette_for("light" if theme
  == "light" else "dark")`, so the compare plot is drawn in the DARK
  palette under a night theme rather than in the theme's own. That is
  pre-existing behaviour for every non-light theme, it is not a
  regression, and it is not fixed here.

## Measured

Item 380's paint harness, `tools/perf_paint.py`, offscreen, each theme
with its own animation, 3 s per combination:

| theme | animation | 1080p | 4K |
|---|---|---|---|
| lantern | bokeh | 25.1 fps | 25.1 fps |
| halcyon | blobs | 24.9 | 25.0 |
| solstice | cells | 25.0 | 25.0 |
| undertow | ripple | 24.8 | 24.9 |
| meridian | aurora | 24.9 | 24.8 |
| cirrus | drift | 24.7 | 24.9 |
| nocturne | drift | 24.7 | 25.0 |
| aphelion | resonance | 24.8 | 24.8 |
| pulsar | blobs | 24.7 | 24.8 |
| vesper | aurora | 24.8 | 24.9 |
| dark (control) | blobs | 24.7 | 24.9 |

380's baseline of 2026-09-09 is "24.7 fps at 1080p and 24.7 at 4K, every
theme". All ten sit on it; none of them is cheaper or dearer than the
theme that shipped.

ONE ROW WAS WRONG BEFORE IT WAS RIGHT, and it is the reason the table
above is not the first run. That run put `meridian` + `aurora` at 4K at
17.8 fps while `vesper` + the same animation at the same size read 24.9 —
the same producer twice, one number six frames short. It was the machine:
two other pytest processes were running. Three clean re-runs of meridian,
vesper and dark gave 24.8-25.1 across the board. A single low row in a
sweep taken under load is load until it repeats.
