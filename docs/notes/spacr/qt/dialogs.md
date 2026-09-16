# Notes from `spacr/qt/dialogs.py`

Prose lifted out of `spacr/qt/dialogs.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [detach_from_window_manager](#detach_from_window_manager) (3 entries)
- [_field_types](#_field_types) (1 entry)
- [_drag_class._DragTheWindowByTheForm.eventFilter](#_drag_class_dragthewindowbytheformeventfilter) (1 entry)
- [drop_the_explicit_floor](#drop_the_explicit_floor) (1 entry)
- [let_the_content_scroll](#let_the_content_scroll) (3 entries)
- [give_it_a_size_grip](#give_it_a_size_grip) (1 entry)
- [make_the_window_resizable](#make_the_window_resizable) (4 entries)
- [_DetachEveryDialog.eventFilter](#_detacheverydialogeventfilter) (4 entries)
- [detach_all_dialogs._Filter](#detach_all_dialogs_filter) (1 entry)

## detach_from_window_manager

### line 122, trailing  _(unsure)_

```python
except Exception:
```

headless import

### lines 126-128

```python
dialog.setWindowFlags((flags & ~Qt.WindowType.Dialog)
```

`Qt.Dialog` is `Qt.Window | 0x2`, so clearing it clears the Window bit too and it has to be put back. Setting `Qt.Window` alone would leave the dialog bit standing and change nothing.

### lines 132-133

```python
pass
```

Decoration must never be load-bearing (INVARIANTS 10): a dialog that cannot be detached is still a dialog the user can use.

## _field_types

### lines 223-225

```python
nesting = (QAbstractItemView, QAbstractSpinBox, QComboBox, QLineEdit,
```

A spin box OWNS a line edit, an editable combo owns another, and an item view owns a scroll bar and an editor. Counting those would make one field look like three.

## _drag_class._DragTheWindowByTheForm.eventFilter

### lines 398-399  _(unsure)_

```python
"""Move the window when its holder's empty space is dragged."""
```

`getattr`, for `_DragByBackground`'s reason: Qt goes on delivering to a filter whose Python attributes are cleared.

## drop_the_explicit_floor

### lines 494-496

```python
dialog.setMinimumSize(0, 0)
```

ZERO CLEARS IT rather than setting one of its own: Qt tracks whether a minimum was set explicitly, and a zero on either axis takes that mark off, which puts the axis back under the layout's control.

## let_the_content_scroll

### line 520  _(unsure)_

```python
holder.setLayout(layout)
```

Steals the layout from the dialog, and the fields come with it.

### lines 524-526

```python
scroll.setFrameShape(QFrame.Shape.NoFrame)
```

NO FRAME. A sunken border round the whole form is a box drawn inside a card that already has one, and its straight edges are the most visible thing on a translucent window.

### lines 536-540

```python
make_transparent(scroll, holder)
```

THE NEW CONTAINERS PAINT NOTHING. `glass.clear_the_containers` walks the dialog when the card goes in, and it has already run by the time this does -- a scroll area added afterwards is an untagged QWidget, and in a palette whose `bg` is #000000 that is a black rectangle over the card. The viewport is tagged with it; `make_transparent` knows.

## give_it_a_size_grip

### lines 575-577

```python
make_transparent(*grips)
```

The grip is a child of the DIALOG, added after glass tagged the containers, so without this it paints the palette's flat background as a square in the one corner a rounded card is most visible.

## make_the_window_resizable

### lines 596-618

```python
from PySide6.QtWidgets import QWidget
```

AND THE CHILDREN HAVE TO BE STYLED before the layout is asked what it needs, or the floor is measured in the wrong FONT.

This filter runs on the dialog's Polish event, and an event filter is delivered BEFORE the widget's own handler -- so at this moment neither the dialog nor any descendant has had the application stylesheet applied. Measured on Annotate's settings dialog, which has 165 children: sixteen of them change size across that boundary, and they are its eight QComboBoxes, each 29 px in the default "Sans Serif 9" and 30 px in the stylesheet's "Open Sans".

floor read at Polish   480 x 1183   the size it re-opened at floor once on screen   512 x 1191   the size it really needs

Eight rows, eight pixels, and the dialog opened eight pixels short of its own content -- with a scroll bar already showing on a form that fits. `ExecutionProfileDialog` has the same defect with one combo box and one pixel.

`dialog.ensurePolished()` does NOT do this and was measured not to: it sends the very Polish event being filtered, and Qt polishes a parent before its children. The descendants have to be asked themselves.

### lines 623-627

```python
dialog.layout().activate()
```

THE LAYOUT HAS TO HAVE RUN before its floor can be read. Qt sets a window's minimum size from `QLayout.activate`, and on Polish that has not always happened yet: reading it first answered 0x0 for six of these dialogs, and a floor of zero is the same as not remembering one -- they opened at two thirds of the screen.

### lines 635-636

```python
dialog.setProperty(OPENS_AT, floor)
```

The floor it used to open at, kept for the Show that is on its way. See `open_at_its_natural_size`.

### lines 641-644

```python
LOG.debug("could not make a dialog resizable", exc_info=True)
```

INVARIANTS 10: a dialog that cannot be made smaller is still a dialog the user can use. The marker is set above rather than here, so a dialog this failed on half-way is not tried again on its next show -- the half it did is already done.

## _DetachEveryDialog.eventFilter

### lines 715-719

```python
if obj.property(_GLASS_DETACHED):
```

ALREADY DONE BY THE GLASS INSTALLER, which detaches and goes frameless in ONE flags change. Doing it again here recreates the native window a second time, and on some window managers what comes back has square opaque corners behind the rounded card.

### lines 725-730

```python
if (polished or event.type() == QEvent.Type.Show) \
```

AND THE SAME EVENT MAKES IT RESIZABLE. Polish OR Show: the flags above may only be rewritten before the window is mapped, but moving the contents into a scroll area is an ordinary layout change and is safe either way -- which matters for a dialog that builds its form after its first polish, and would otherwise never be reached.

### lines 734-735

```python
if event.type() == QEvent.Type.Show \
```

AND THE SIZE IT OPENS AT IS PUT BACK ON Show, which is the first moment one sticks. See `open_at_its_natural_size`.

### lines 740-742

```python
pass
```

INVARIANTS 10 again: a dialog that cannot be detached is still a dialog. This filter sees every event in the application and must never be the reason one of them is lost.

## detach_all_dialogs._Filter

### lines 779-783

```python
class _Filter(QObject):
```

ONE CLASS, NOT A MIX-IN, and that is a correctness fix rather than a style one. `class F(QObject, _DetachEveryDialog)` puts QObject first in the MRO, so QObject's own `eventFilter` -- which returns False and does nothing -- wins over the one below it. The filter installed, reported success, and silently never fired.
