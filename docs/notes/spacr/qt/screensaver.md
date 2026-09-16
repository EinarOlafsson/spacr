# Notes from `spacr/qt/screensaver.py`

Prose lifted out of `spacr/qt/screensaver.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Screensaver.__init__](#screensaver__init__) (2 entries)
- [Screensaver.keyPressEvent](#screensaverkeypressevent) (1 entry)
- [show_screensaver](#show_screensaver) (1 entry)

## Screensaver.__init__

### lines 35-38

```python
"""Build the screensaver as its own full-screen window.
```

Qt.Window, and NOT parented into the layout: a child widget cannot go full screen on its own, and a Tool window loses focus to the main window the moment it appears -- which would make "any key" reach the wrong place.

### lines 54-57

```python
self.setCursor(Qt.CursorShape.BlankCursor)
```

THE POINTER IS HIDDEN, which is what makes it read as a screensaver rather than as a window with nothing in it. It comes back with the cursor's own shape on close, because this widget is destroyed rather than restored.

## Screensaver.keyPressEvent

### lines 104-108

```python
def keyPressEvent(self, event) -> None:
```

leaving

ANY key and ANY click, which is what was asked for. Every one of these is a deliberate act by somebody who wants their screen back, so none of them is worth distinguishing.

## show_screensaver

### lines 158-159  _(unsure)_

```python
saver.setFocus(Qt.FocusReason.OtherFocusReason)
```

FOCUS, EXPLICITLY. Without it the key that is meant to close this goes to whatever had focus before, and the screensaver stays up.
