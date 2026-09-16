# Notes from `spacr/qt/widgets/hint_bar.py`

Prose lifted out of `spacr/qt/widgets/hint_bar.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [HintBar.__init__](#hintbar__init__) (3 entries)
- [HintBar.explain](#hintbarexplain) (1 entry)
- [HintBar.eventFilter](#hintbareventfilter) (1 entry)

## HintBar.__init__

### lines 61-64

```python
self.setAlignment(Qt.AlignJustify | Qt.AlignVCenter)
```

JUSTIFIED, like the tooltips it replaces. Asked for 2026-08-28. Centring is right for one short line and wrong for the three a paragraph takes: a centred block has two ragged edges instead of one, and reads as a caption rather than as prose.

### lines 67-68  _(unsure)_

```python
self.setMinimumHeight(max(28, self.sizeHint().height()))
```

Tall enough for the sentence it will hold, so the window does not resize the moment the pointer touches a control.

### lines 70-73

```python
line = max(1, self.fontMetrics().lineSpacing())
```

AND NO TALLER THAN THREE LINES. The strip took whatever height the longest help needed, so moving the pointer between two controls whose help differs in length made the whole dialog jump. A bounded strip elides instead, and the full text is still in the register.

## HintBar.explain

### lines 94-95  _(unsure)_

```python
if not widget.accessibleDescription():
```

A screen reader reads neither the bar nor a tooltip that is gone, so the sentence is put where assistive technology looks for it.

## HintBar.eventFilter

### lines 142-145

```python
if self._hints.get(obj) and \
```

ONLY IF THIS WIDGET IS THE ONE BEING SHOWN. Two controls side by side send Leave-then-Enter in that order often enough that blanking unconditionally makes the bar flicker to the default between neighbours.
