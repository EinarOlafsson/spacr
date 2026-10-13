# Notes from `spacr/qt/widgets/eliding.py`

Prose lifted out of `spacr/qt/widgets/eliding.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ElidingLabel._refresh](#elidinglabel_refresh) (2 entries)
- [ElidingPushButton.__init__](#elidingpushbutton__init__) (1 entry)
- [ElidingPushButton._refresh](#elidingpushbutton_refresh) (1 entry)

## ElidingLabel._refresh

### lines 99-101

```python
if not self.testAttribute(Qt.WA_Resized):
```

Before the first layout pass the widget still carries Qt's default 100 px size, which would elide almost everything and leave a stale tooltip behind. Wait for a real geometry.

### lines 113-114

```python
self.setToolTip(self._full_text)
```

A user who cannot read the whole name must still be able to discover it — the tooltip is the only place left to put it.

## ElidingPushButton.__init__

### lines 147-149

```python
policy = self.sizePolicy()
```

Horizontally shrinkable: without this the layout treats the size hint as a hard minimum and squeezes the *whole* sidebar instead of shortening one label.

## ElidingPushButton._refresh

### lines 219-220

```python
if not self.testAttribute(Qt.WA_Resized):
```

See ElidingLabel._refresh — don't elide against the default 100 px size a widget carries before its first layout pass.
