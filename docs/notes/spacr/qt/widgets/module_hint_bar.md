# Notes from `spacr/qt/widgets/module_hint_bar.py`

Prose lifted out of `spacr/qt/widgets/module_hint_bar.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ModuleHintBar.__init__](#modulehintbar__init__) (3 entries)
- [ModuleHintBar.event](#modulehintbarevent) (1 entry)
- [ModuleHintBar.show_module](#modulehintbarshow_module) (2 entries)

## ModuleHintBar.__init__

### lines 82-84

```python
self.setOpenExternalLinks(True)
```

The links leave the application, so Qt opens them. `setOpenExternal Links` also makes the label focusable for a keyboard user, which is what makes the words reachable without a pointer at all.

### lines 88-107

```python
line = max(1, self.fontMetrics().lineSpacing())
```

A FIXED HEIGHT, and it is load-bearing rather than tidy.

A strip that takes whatever height its text needs GROWS when a long summary wraps and shrinks when a short one does not -- so moving the pointer between two modules relayouts the page under it. The dock is in that layout. Reported 2026-09-03: "if i hover quickly an element in the dock it blinks blue a bunch of times then stays blue after a while" -- the row moving out from under the pointer and back, delivering an Enter and a Leave each time.

The same lesson twice already: `HintBar` caps itself at three lines ("moving the pointer between two controls whose help differs in length made the whole dialog jump") and the status bar is pinned for it too ("without this the dock flickered on Linux each time one arrived"). This is the third surface and the first to forget.

TWO LINES: the summary and the row of links under it. Measured from the font rather than pinned at a number, so it stays right at any font scale -- a hard number is a promise about text metrics that breaks the moment the scale or the theme's font stack changes.

### line 110

```python
self.setWordWrap(False)
```

And the text is ELIDED into it rather than wrapping past it.

## ModuleHintBar.event

### lines 133-147

```python
if event.type() == QEvent.Type.ToolTip:
```

`QEvent` COMES FROM THE MODULE IMPORT, NOT FROM HERE, and the reason is not tidiness. `event()` runs for EVERY event this widget receives, so a function-local import was a `sys.modules` lookup per event on a hot path -- and, worse, a place the widget could RAISE.

A test that sets `sys.modules["PySide6.QtCore"] = None` to stand in for "no Qt here" made this import throw `ModuleNotFoundError: import of PySide6.QtCore halted` from inside a paint event during pytest-qt's teardown `processEvents()`. That error then repeated for every remaining test in the process: ONE lazy import produced 419 teardown errors in a single sweep chunk.

There was never anything to defer. Line 28 imports QtCore at module scope already -- this class is a QWidget and the module cannot load without it.

## ModuleHintBar.show_module

### lines 172-184

```python
suffix = f" — {mark}" if mark else ""
```

ELIDED TO ONE LINE, because the strip is two lines tall and the second is the links. A module blurb runs to several hundred characters; letting it wrap is what made the strip resize and the page relayout under the pointer. The whole sentence stays in the accessible description below, which is what a screen reader reads.

THE SUMMARY IS ELIDED, NOT THE STAGE. Appending the word and then eliding the pair cut the word off every blurb long enough to need eliding -- which is most of them -- so the one carrier that was supposed to survive for a colour-blind reader was the one thing reliably lost. The stage is short and load-bearing; the sentence is long and already repeated in the accessible description, so the sentence is what gives way.

### lines 191-192

```python
self.setAccessibleDescription(text)
```

The plain sentence stays reachable for a screen reader, which reads the accessible description rather than parsed rich text.
