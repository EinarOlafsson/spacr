# Invariants

Rules that hold in this codebase and are **not visible from the code**.
Every one cost real debugging. Each says how it was found, because the
evidence is what lets you tell a rule that still applies from one that has
quietly stopped.

Where a rule can be machine-checked it is checked by `skill/refresh.py`.
Run that before trusting this file.

---

## 1. A widget QSS rule that is not registered when the stylesheet is built is not in it at all

`spacr/qt/theme.py` composes the application stylesheet by appending every
block registered through `register_widget_qss`. **31 modules register at
import time.** A module not yet imported has not registered, so its rule
is absent, and its widget falls through to the blanket

```
QWidget { background-color: bg }
```

`bg` is the **window colour** — `#000000` on the dark theme. An unstyled
container is not slightly off. It is a solid black rectangle.

**This was the longest-running bug in the project.** `settings_search`
owns `SettingsSearchPane`, the wrapper around the search strip *and* the
settings scroll area — the entire left column of every module screen. Its
rule was correct the whole time and was not in the sheet yet, because the
sheet is applied at launch and the module is imported when the first
module screen is built.

That explained all three reported symptoms, which four earlier fixes did
not: black at launch; cleared by changing the theme or the animation
(both rebuild the sheet); cleared by opening enough screens (something
re-applies it).

Measured: `'SettingsSearchPane' in theme.stylesheet()` was `False` on a
fresh interpreter and `True` after importing the module.

**The rule:** a new module that registers widget QSS **must** be added to
`theme.WIDGET_QSS_MODULES`. `refresh.py` checks this. Do not fix a black
container by styling one more widget — that fixes one screen and leaves
the next.

## 2. `bg` is the window colour, `page` is the surface panels float on

Distinct roles, deliberately. `bg` is `QPalette.Window`, the ink on a
filled accent button (`HighlightedText`), and the blanket rule above.
Thirty-six uses, most of which are not "the page".

`page` was added because `bg` was serving as the page by omission: a
module screen clears its layout containers so the backdrop shows between
the settings cards, and with the ambient animation off there is no
backdrop — so what showed through was `#000000`.

Do not "simplify" these into one role. Giving the page a colour must not
change selected-text rendering or button ink.

## 3. Anonymous `QWidget` is scaffolding and must paint nothing

A plain `QWidget` used as a layout container inherits the blanket rule and
paints the window colour. `spacr/qt/theme.py` has `make_transparent` and
`clear_container_surfaces` for this. Every container between the backdrop
and the eye has to be tagged, or one of them buries the animation.

## 4. Connect **bound methods** to `QThread.finished`, never closures

`spacr/qt/bridge.py :: make_thread` connects `finished → deleteLater`
first. A closure connected afterwards makes the QThread its own receiver,
so the call is discarded during teardown and **the job silently never
retires**. A bound method gives Qt a receiving QObject with GUI-thread
affinity, so the call is queued onto the GUI thread.

Also: never connect `worker.deleteLater` to `worker.finished`. `finished`
is emitted from inside the worker thread, so scheduling the object's C++
deletion from it hands Qt a second owner for an object Python already owns
and the two race — measured at 3 crashes in 8 runs of the stress harness.

## 5. Process-global state leaks between test files

A test that fails in the suite and passes on its own file is an isolation
leak. Three are known:

| State | Leaked by | Status |
|---|---|---|
| The app registry | `register_self_registering_modules()`, called by 10 files, restored by none | fixed — `_restore_app_registry` |
| The font scale | `test_preferences` sets it 10×, restores 0 | fixed — `_font_scale_starts_at_one` + `_restore_font_scale` |
| The app stylesheet | `test_photo_themes` leaves 46 KB of the Cell theme on the session QApplication | **open** — see `instructions/03` |

Do not paper over one with a retry. Find what leaked.

**A blanket stylesheet restore in `tests/qt/conftest.py` does not work** and
was reverted: the session-scoped `qt_theme_applied` applies the sheet once,
and the first test to request it snapshots an empty sheet *before* that
fixture runs, then restores the empty sheet — unthemeing the whole suite.

## 6. Settings: hidden is not the same as absent

`resolve_default_settings(app_key)` produces what a run receives.
`_APP_HIDDEN_KEYS` decides what is *rendered*. They are different
questions:

* A key **absent** from the dict means the pipeline falls back to **its
  own** default, and the two can disagree with nothing saying so.
* A key **hidden** stays in the dict at the value the module needs and is
  simply not shown.

Hiding needs all three of: no widget built (the trailing "Other" section
is built from `self._widgets`), removal from `categories_for_app` (or it
stays findable in the Ctrl+F index and the result scrolls to a control
that is not there), and a forced value in `resolve_default_settings` (or
a settings CSV from an older build re-introduces it).

Removing a key from a layout does **not** hide it — it moves it to
"Additional Settings".

## 7. `QWidget.render()` cannot reproduce paint-ordering bugs

It forces one full synchronous paint. Anything that depends on Qt's
ordinary erase-then-paint cycle — a region erased to the background before
`paintEvent` runs — never happens under it.

**It reported `0.0% black` four times for a screen that was solid black on
the user's display**, and three separate wrong fixes were built on those
clean numbers.

Two further traps when measuring rendering:

* Apply `theme.stylesheet()` to the QApplication first, or everything
  renders in Qt's default palette — a uniform `(239, 239, 239)` — and the
  probe reports a clean page for a screen that is black in the real app.
* Fill the probe image with a colour **nothing else uses** (magenta), not
  with transparent. Filling with 0 and counting "black" pixels scores an
  *unpainted* region as clean, and unpainted is exactly the failure.

Prefer asserting on state — what is in the stylesheet, what the palette
says — over asserting on pixels.

## 8. Never write to the real user preferences

`spacr/qt/preferences.py :: _settings()` resolves to
`~/.config/spacr/qt.conf`. A script run outside pytest bypasses the test
sandbox entirely.

Nine diagnostic scripts called `set_ambient_enabled(False)` while
reproducing a bug. The user's next launch had a flat grey interface, an
opacity setting with nothing to show through, and an animation named in
Preferences that was switched off by a flag they never touched.

Point `_settings` at a temp `QSettings` first, as
`tests/qt/test_preferences_reset.py :: private_store` does.

## 9. The default font scale is 1.5, and the tests pin 1.0

`DEFAULT_FONT_SCALE = 1.5`. The qt suite pins 1.0 for determinism, so
every geometry record describes a configuration no user has. Known and
deliberate; see `instructions/04`. Do not "fix" a geometry test by
changing the product default.

## 10. Decoration must never be load-bearing

A backdrop, a tooltip animation, an icon, a widget QSS block, a
reproducibility manifest: each is wrapped so its failure costs that
feature and nothing else. `open_run` failing prints a warning and the run
proceeds. `_install_ambient` failing leaves the screen exactly as it would
have been. Keep it that way — a decorative fault must not abort a screen
analysis that has been running for hours.

## 11. Cooperative cancellation cannot stop a wedged worker

`RunRegistry.cancel_all` sets a flag, interrupts the thread and waits.
That is the right default — a pipeline killed mid-write leaves a
half-written `.npy`. But a worker wedged in a C extension never checks the
flag, so `closeEvent` refuses to close for as long as it lives.

`spacr/qt/shutdown.py` is the escape: it always asks first, and force is
`os._exit` — **not** `sys.exit`, **not** `QApplication.quit`, because both
of those unwind and can block on the very thread that is already wedged.
A force quit that can hang is not a force quit.

## 12. The desktop installs have no `pip`

The installers build the environment with `uv venv`, which does not seed
pip. `spacr/updater.py :: find_uv()` finds the bootstrapped tool at
`<install root>/bootstrap/uv`. Anything that shells out to
`sys.executable -m pip` in an installed build will fail. See
`instructions/01`.

## 13. Channel order is declared, never inferred from list position

A setting that names channels in a list carries an unstated convention about
which colour each position means, and **the convention will be read
backwards**. It was, for eleven days.

`png_dims=[0,1,2]` meant "0 is blue" for the whole life of the project —
not because anything said so, but because `cv2.imwrite` interprets a
3-channel array as BGR and the writer handed it the array unchanged. Commit
`341f4462` (2026-07-26) read the list the other way, reversed the writer so
`png_dims[0]` landed in red, versioned the format, and migrated existing
folders into the new order. Every crop written or migrated in that window
has its 405/DAPI plane in the red channel; nuclei render red and the 555
plane renders blue.

Nothing was wrong with the code. The list was ambiguous and two readings of
it were each defensible.

**The rule:** where a channel setting decides a *colour*, it is a mapping —
`png_channel_mapping = {'r': 2, 'g': 1, 'b': 0}` — and every part of the
crop path speaks that order. `CropSpec.channels`, `extract_crop`,
`png_view` and `read_crop_png` are all `(red, green, blue)`.
`crops.channels_from_settings` is the single translation from the legacy
list, at the edge.

Where a channel setting decides which channels are *processed* (`channels`
for cellpose and measure) it is a set and stays a list. Forcing r/g/b onto
it would invent a meaning that is not there.

Corollaries worth knowing before touching `crops.py`:

* Formats 1 and 3 hold **identical bytes** for the same mapping. Reversal on
  read is decided by `_FORMAT_IS_DECLARED_ORDER`, not by `fmt != as_format`
  — that comparison reverses between two formats that agree.
* Format 2 is the only one whose pixels are out of step, so it is the only
  input `migrate_crop_folder` has work to do on. Pointing the migrator at a
  legacy folder is correctly a no-op.
* A folder finished with `on_error='skip'` is marked with the **target**
  format and an `unconverted` list. A retry keyed on the *source* format
  silently returns `already` and leaves those files unconverted for ever.
  Measured: the retry converted all three crops instead of the one.
