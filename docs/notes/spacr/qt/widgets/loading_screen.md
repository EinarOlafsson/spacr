# Notes from `spacr/qt/widgets/loading_screen.py`

Prose lifted out of `spacr/qt/widgets/loading_screen.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_role](#_role) (1 entry)
- [LoadingScreen.__init__](#loadingscreen__init__) (1 entry)
- [LoadingScreen._load_logo](#loadingscreen_load_logo) (1 entry)
- [LoadingScreen.paintEvent](#loadingscreenpaintevent) (4 entries)

## _role

### lines 61-64

```python
return fallback
```

THE SPLASH MUST NOT FAIL. It is the first thing painted, sometimes before the theme has resolved and always before anything else could report a problem, so a palette lookup that raised would replace it with a traceback.

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
