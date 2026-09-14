# Notes from `spacr/qt/widgets/ai_toggle_label.py`

Prose lifted out of `spacr/qt/widgets/ai_toggle_label.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [AiToggleLabel.__init__](#aitogglelabel__init__) (2 entries)
- [AiToggleLabel.changeEvent](#aitogglelabelchangeevent) (1 entry)
- [AiToggleLabel._apply_elision](#aitogglelabel_apply_elision) (2 entries)
- [AiToggleLabel._refresh_style](#aitogglelabel_refresh_style) (4 entries)

## AiToggleLabel.__init__

### lines 87-88

```python
self.setProperty("_spacr_i18n_text", source_text)
```

Retain canonical English sources so a runtime language switch never translates a translation and never loses the toggle's current state.

### lines 96-98

```python
self._full_text = tr(source_text)
```

The logical text, always. `QLabel.text()` holds whatever fits right now, which may be elided; every caller that asks this widget what it says wants the full thing.

## AiToggleLabel.changeEvent

### lines 116-119

```python
kind = None
```

PySide6 raises when the C++ half of a wrapper is gone, and changeEvent is called during teardown as well as during a Preferences save. An unreadable event means "restyle nothing", not an exception out of a Qt callback.

## AiToggleLabel._apply_elision

### lines 181-188

```python
if not shown.strip("…. \t"):
```

An elision that keeps no character of the label -- "" when the width cannot fit even the ellipsis, "…" when it barely can -- paints a blank toggle. The full text drawn slightly clipped is strictly better: it still says which control this is. Zoom lands here because the enlarged font gets measured against the width the layout granted the smaller one, one relayout behind. The long labels this eliding exists for keep plenty of characters at ELIDE_ABOVE_PX and are untouched by this guard.

### lines 193-195

```python
self._eliding = True
```

`setText` re-enters through the override above; the flag keeps it from mistaking the elided text for a new logical text and truncating the stored copy one character at a time.

## AiToggleLabel._refresh_style

### lines 228-234

```python
"""Re-ink and re-size the label for the current state, theme and zoom.
```

Use the theme-invariant ``button_accent`` for the ON colour so the toggle looks identical in every theme. The OFF colour is the theme's own ``fg``, resolved HERE rather than imported: `active_palette()` reads the preference that is in force right now, so the label inks white on dark and near-black on light. It used to come from `theme.PALETTE`, which is frozen dark — white "AI" on the light theme's #fafafa page.

### lines 254-258

```python
size = font_px("body")
```

Zoom reaches this through `font_px`, not through the application sheet: a per-widget `setStyleSheet` outranks it, so the literal `FONT_SIZE['body']` that used to be here pinned "Live" and "AI" at 13 px whatever the preference said. Padding scales with it or the hit target stops matching the glyphs.

### lines 270-273

```python
if self._restyling or sheet == self.styleSheet():
```

`setStyleSheet` itself posts a StyleChange back to this widget, so `changeEvent` would call straight back in. Both guards matter: the flag stops the immediate recursion, the comparison stops a StyleChange storm when nothing about the answer has changed.

### lines 281-284

```python
self.updateGeometry()
```

The new sheet moves both the font size and the padding, so the `sizeHint` the layout is holding is stale. Without this the widget keeps its old width and the elision below measures the bigger glyphs against it, hiding the text that the zoom just enlarged.
