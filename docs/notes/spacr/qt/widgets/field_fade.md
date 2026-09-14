# Notes from `spacr/qt/widgets/field_fade.py`

Prose lifted out of `spacr/qt/widgets/field_fade.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [field_fade_enabled](#field_fade_enabled) (2 entries)
- [paint_field_fade](#paint_field_fade) (1 entry)
- [_FieldFadeFilter.eventFilter](#_fieldfadefiltereventfilter) (5 entries)
- [install_field_fade](#install_field_fade) (1 entry)
- [Module level](#module-level) (1 entry)

## field_fade_enabled

### lines 69-71  _(unsure)_

```python
def field_fade_enabled() -> bool:
```

The preference, read once and cached

### lines 89-91

```python
_enabled = True
```

An unreadable settings store falls back to the shipped look, not to "off" — the same rule every other preference in this app follows.

## paint_field_fade

### lines 196-198

```python
inset = FIELD_BORDER_PX / 2.0
```

Half-pixel inset so the 1px outline lands ON the widget's edge pixels rather than straddling them, which is what keeps the left end reading as a hard edge rather than a 50 % smear.

## _FieldFadeFilter.eventFilter

### lines 232-233  _(unsure)_

```python
"""Start the fade when the watched field changes.
```

First line of a filter that sees every event in the process:

one enum compare, then out.

### lines 244-247

```python
paint_owner = obj.window()
```

A child wrapper does not keep its top-level Python owner reachable. If cyclic GC collects that owner while a painter is active, Qt deletes the child's native paint device and ``QPainter.end()`` segfaults. Keep the owner alive until the native painter is closed.

### lines 255-257

```python
LOG.exception("Field fade could not paint %s",
```

A cosmetic effect must never be the reason a screen fails to draw. Logged rather than swallowed, so a broken palette is discoverable instead of merely invisible.

### lines 261-263

```python
if painter is not None and painter.isActive():
```

Explicit, not left to refcounting: a painter still active on this widget would break the widget's own paint two lines later, which is a blank field rather than an unstyled one.

### line 267  _(unsure)_

```python
return False
```

False, always: the widget still has to draw its text on top.

## install_field_fade

### lines 283-284  _(unsure)_

```python
app.installEventFilter(_filter)
```

Re-install on the (possibly new) app. Qt ignores a duplicate install of the same filter on the same object.

## Module level

### lines 323-325  _(unsure)_

```python
_FIELD_SELECTORS = (
```

The QSS that gets out of the painter's way
