# Notes from `spacr/qt/button_roles.py`

Prose lifted out of `spacr/qt/button_roles.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_SemanticButtonFilter.classify](#_semanticbuttonfilterclassify) (2 entries)
- [_SemanticButtonFilter._classify_live](#_semanticbuttonfilter_classify_live) (1 entry)
- [_SemanticButtonFilter._settle_after_handler](#_semanticbuttonfilter_settle_after_handler) (2 entries)
- [install_button_roles](#install_button_roles) (1 entry)

## _SemanticButtonFilter.classify

### lines 150-152

```python
"""Give one button its semantic role, if its C++ half is still there.
```

A queued event can outlive the C++ widget while a signal connection still retains its Python wrapper.  Every operation below crosses into Qt, so reject that wrapper at the single entry boundary.

### lines 172-175

```python
if alive(button):
```

Qt 6.6 can delete a dialog button re-entrantly while

``parentWidget()`` delivers another construction event.  Keep real RuntimeErrors visible, but a wrapper that became invalid during that call has no remaining state to classify.

## _SemanticButtonFilter._classify_live

### lines 181-186

```python
if (isinstance(button.parentWidget(), QDialogButtonBox)
```

spaCR's dialog buttons are text, not text-plus-glyph. Qt's platform styles put a standard icon on the standard roles — a cross on Cancel and Close, a downward arrow on Save — which reads as system chrome dropped into the app's own type. Stripped here rather than at each call site, because this filter already sees every button in every dialog, including ones built after startup.

## _SemanticButtonFilter._settle_after_handler

### lines 234-236

```python
running = (
```

Disabled Run/Stop buttons conventionally mean their asynchronous worker is still starting or stopping. Keep the solid operation fill until the owning screen re-enables/clears the button.

### line 244  _(unsure)_

```python
pass
```

Close can delete its dialog before this queued callback runs.

## install_button_roles

### lines 280-282

```python
parent = app if isinstance(app, QObject) else None
```

QApplication is always a QObject in production. Keeping the parent optional also supports lightweight application adapters used by embedding hosts and tests; the attribute below retains the filter.
