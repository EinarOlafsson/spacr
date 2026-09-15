# Notes from `spacr/qt/widgets/loading_screen.py`

Prose lifted out of `spacr/qt/widgets/loading_screen.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_role](#_role) (2 entries)
- [SPLASH_BACKGROUND](#splash_background) (1 entry)
- [LoadingScreen.__init__](#loadingscreen__init__) (1 entry)
- [LoadingScreen._load_logo](#loadingscreen_load_logo) (1 entry)
- [LoadingScreen.paintEvent](#loadingscreenpaintevent) (4 entries)

## _role

### lines 61-64

```python
return fallback
```

THE SPLASH MUST NOT FAIL. It is the first thing painted, sometimes before the theme has resolved and always before anything else could report a problem, so a palette lookup that raised would replace it with a traceback.

### line 58 -- item 415, 2026-09-15

```python
value = active_palette().get(name)
```

THE SPLASH WEARS THE THEME THE WINDOW OPENS IN. This used to read `palette_for()`, which takes a theme and defaults to dark, so the startup window was black whatever spaCR was set to. `MainWindow.__init__` also fills its first frame with `splash_bg` (through `splash_role`), under the application's own window text, so a light spaCR -- including the default "Follow system" on a light Windows -- showed #0d0e10 text on #000000 there, 1.09:1. `active_palette()` resolves through `resolve_effective_theme()`, the call `apply_preferences_to_app` made moments earlier in `launch`, so the startup window and the main window read the theme from the same place and cannot disagree. The OS scheme is not read here directly: an explicit theme ignores it, and "Follow system" reads it the one way the main window does, through the application palette. The fallback still holds: `active_palette` falls back to dark, and if that raises too, the `except` below gives the literal back.

Reproduced offscreen before the change for OS light/dark x spaCR light/dark/system; the OS scheme changed nothing on the loading screen, which never read it. `tests/qt/test_the_startup_window_reads_in_every_scheme.py` holds all of it, measured from the pens and from the rendered pixels.

## SPLASH_BACKGROUND

### line 148 -- CI dispatch 35012948690, 2026-09-15

```python
SPLASH_BACKGROUND = _dark_role("splash_bg", "#000000")
```

A MODULE CONSTANT CANNOT FOLLOW THE THEME, so it does not try to. Item 415 (f5d651454) moved `_role` from `palette_for()` to `active_palette()`, and the constant was still defined through `splash_role`. It stopped being a colour and became a snapshot of whichever theme resolved when the process first imported this module: "Follow system" is the default, it resolves through the application palette, and on a Qt worker whose first import ran under Fusion's light palette the constant read `#fafafa` (`test_the_constant_is_renamed_and_the_old_name_still_works`, Qt shard 0). The same snapshot in the app would go stale at the first theme change.

Everything that paints reads `splash_role` at paint time, including `MainWindow`'s first-frame fill, so the constant keeps the meaning it had before 415: the dark palette's `splash_bg`, which is also what the splash falls back to when no theme can be read (`active_palette` falls back to dark). `_dark_role` keeps `_role`'s rule that a lookup never raises. `test_the_constant_does_not_depend_on_the_theme_in_force_at_import` executes the module afresh under a light theme; it fails on the previous definition.

## LoadingScreen.__init__

### lines 174-175

```python
self.setAutoFillBackground(True)
```

Opaque: this covers a partly-built window, and any transparency would show the thing it exists to hide.

## LoadingScreen._load_logo

### lines 227-228

```python
self._logo = None
```

A missing logo must not stop the app from starting. The screen still covers the window and still reports progress.

## LoadingScreen.paintEvent

### lines 256-258

```python
phases = [_translate(p) for p in STRAP_PHASES]
```

Translated at PAINT time, not at import: the loading screen is built before the user's language preference has necessarily been read, and a phase cached in English would stay English.

### lines 269-271

```python
scaled = scaled_for(self._logo, self, side)
```

Scaled inside the paint, so it asks the ratio again on every frame and a splash dragged between screens is right on the next repaint with nothing to subscribe to.

### line 277  _(unsure)_

```python
lit = self.lit_phases()
```

The sentence, one phase at a time.

### lines 293-295

```python
rule_y = baseline + metrics.descent() + scaled_px(12)
```

A hairline under the sentence, filled to the same fraction. Thin on purpose: the sentence is the progress indicator, and a second loud one would compete with it.
