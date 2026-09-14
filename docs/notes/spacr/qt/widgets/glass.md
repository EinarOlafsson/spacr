# Notes from `spacr/qt/widgets/glass.py`

Prose lifted out of `spacr/qt/widgets/glass.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [clear_the_containers](#clear_the_containers) (1 entry)
- [_ResizeByEdge.eventFilter](#_resizebyedgeeventfilter) (1 entry)
- [_DragByBackground.eventFilter](#_dragbybackgroundeventfilter) (2 entries)
- [make_frameless](#make_frameless) (5 entries)
- [round_the_corners](#round_the_corners) (2 entries)
- [_Backdrop.eventFilter](#_backdropeventfilter) (1 entry)
- [glass](#glass) (7 entries)
- [_say_how_to_close_it](#_say_how_to_close_it) (2 entries)
- [_install_the_backdrop](#_install_the_backdrop) (4 entries)
- [_GlassInstaller.eventFilter](#_glassinstallereventfilter) (1 entry)
- [install_glass_everywhere](#install_glass_everywhere) (1 entry)
- [uninstall_glass_everywhere](#uninstall_glass_everywhere) (1 entry)

## clear_the_containers

### lines 213-214  _(unsure)_

```python
if any(isinstance(parent, OPAQUE)
```

A control's internals are not containers either: a combo's popup view and a spin box's line edit are children of an opaque thing.

## _ResizeByEdge.eventFilter

### lines 307-308

```python
"""Resize the frameless window when an edge is pressed, and shape the cursor.
```

`getattr`, for the reason on `_DragByBackground`: Qt delivers to a filter whose Python attributes have already been cleared.

## _DragByBackground.eventFilter

### lines 391-396

```python
"""Move the frameless dialog when its background is dragged.
```

`getattr`, NOT `self._dialog`. The C++ QObject outlives this

Python object's attributes: during teardown Qt goes on delivering events to a filter whose `__dict__` has already been cleared, and the AttributeError is printed by Qt on every one of them "Error calling Python override of QObject::eventFilter" -- which is noise nobody can act on in a test log.

### lines 410-413

```python
where = event.position().toPoint()
```

ONLY ON THE BACKGROUND. `childAt` answers None when the press is on the dialog itself rather than on something in it, which is exactly the empty space a title bar used to be.

## make_frameless

### lines 474-477

```python
was_showing = not dialog.isHidden()
```

`isVisible()` is not the test: a child of a parent that has never been shown reports False while still being marked visible itself, and hiding one of those would leave it hidden when its parent finally opened.

### lines 479-483

```python
dialog.setAttribute(Qt.WA_TranslucentBackground, True)
```

THE ATTRIBUTE BEFORE THE FLAGS. `setWindowFlags` RECREATES the native window, and a translucency asked for afterwards applies to a window that already exists -- which on X11 means it does not apply at all. Reported 2026-08-22: "there is a box with square edges behind the box with rounded edges."

### lines 485-494

```python
dialog.setWindowFlags((dialog.windowFlags()
```

ONE FLAGS CHANGE, NOT TWO. `spacr.qt.dialogs._DetachEveryDialog` also rewrites the flags on Polish, to turn Qt.Dialog into Qt.Window so a window manager cannot glue the popup to its parent. Each `setWindowFlags` RECREATES the native window, and a window recreated after the translucent one was made is where the square corners came back -- reported as "the rectangular non-rounded black corners are still visible around the preferences and settings windows". So the detach happens here, in the same call, and the marker below tells that filter this dialog is already done.

### lines 500-506

```python
_paint_nothing_behind_the_card(dialog)
```

AND THE STYLESHEET HAS TO AGREE. WA_TranslucentBackground stops Qt filling the window with the palette's base; it does NOT stop the application stylesheet's `QDialog { background: ... }` rule, which paints the square box this is about. Scoped to the dialog itself with `#objectName`-free `QDialog` -- a bare `QDialog` selector in a widget stylesheet applies to that widget and inherits to its QDialog children, of which a settings popup has none.

### lines 509-510

```python
let_the_user_resize(dialog)
```

AND RESIZABLE. Dropping the frame dropped the resize grips with it, so a settings window could be moved and not resized.

## round_the_corners

### lines 544-552

```python
step = 4.0
```

BUILT AT FOUR TIMES THE SIZE AND SCALED BACK. `toFillPolygon` flattens the arcs at a fixed tolerance, and at real size that polygon keeps pixels just outside the curve the card paints so a sliver of whatever drifts behind the card showed along each rounded corner. Flattening a four-times path puts the polygon's error below one pixel once it is scaled down. NOT ERODED. A mask a pixel inside the edge cuts the outermost row all the way round, which takes the rim with it -- the card paints the full rect, so the mask covers the full rect too.

### lines 563-564

```python
LOG.debug("could not round the window corners", exc_info=True)
```

A window that cannot be masked is a window with square corners, which is worse-looking and still perfectly usable.

## _Backdrop.eventFilter

### lines 610-616

```python
"""Refit the backdrop when the dialog is resized or shown.
```

`getattr`, for the reason spelled out on `_DragByBackground`: the C++ QObject outlives this Python object's `__dict__`, so a filter still installed during teardown is asked about events after its attributes are gone. Reading `self._dialog` directly raised AttributeError on every one of them, and Qt printed the whole traceback -- "Error calling Python override of QObject::eventFilter" -- at spaCR startup.

## glass

### lines 645-649

```python
backdrop = _install_the_backdrop(dialog)
```

THE DRIFTING BACKDROP FIRST, so the card has something to be translucent OVER. A translucent panel on an opaque dialog is just a slightly different opaque panel -- what makes the setup screen read as glass is the moving strata showing through it, and that is what "the same transparent background" means here.

### lines 652-654

```python
_paint_nothing_behind_the_card(dialog)
```

THE DIALOG PAINTS NOTHING OF ITS OWN. Without this its square background shows around the card wherever the card does not reach -- the black box behind the periphery.

### lines 658-661

```python
card.lower()
```

BEHIND THE CONTENTS, IN FRONT OF THE BACKDROP, and never in the layout: it is not added to one at all, because a backdrop that took part in a layout would push the dialog's own contents around, and the contents are the point.

### lines 664-667

```python
card.show()
```

SHOWN EXPLICITLY. This runs on the dialog's Show event, so the parent is already visible and a child made now stays hidden until it is told otherwise -- which is a card that is there, sized, and painting nothing.

### lines 669-674

```python
if backdrop is not None:
```

AND THE BACKDROP GOES UNDER THE CARD. `install_ambient` lowers itself to the bottom of the sibling order and so does the card, so whichever is lowered LAST wins the bottom -- and a card under the strata is a card nobody sees, rim and all. One more `lower` on the backdrop puts them in the order the look needs: strata, then the translucent body, then the dialog's own contents.

### lines 681-682

```python
try:
```

AND THE DIALOG'S OWN VERDICT, for the paths that never touch a button -- Escape rejects, and code can accept directly.

### lines 694-695

```python
LOG.debug("could not glass a dialog", exc_info=True)
```

DECORATION IS NEVER LOAD-BEARING. A dialog that cannot be glassed is a dialog that opens looking as it always did.

## _say_how_to_close_it

### lines 739-741

```python
if dialog.findChildren(QDialogButtonBox) or dialog.findChildren(
```

A BUTTON THAT CLOSES IT is any button at all: every dialog button box rejects or accepts, and a bare button in a dialog with no box is what a hand-built OK looks like.

### line 754  _(unsure)_

```python
from ..theme import font_px
```

SMALL, as asked. It is a reminder, not a control.

## _install_the_backdrop

### lines 777-780

```python
if not get_ambient_enabled():
```

THE USER'S OWN ANSWER ABOUT ANIMATED BACKGROUNDS. Somebody who turned the backdrop off on the module screens has not asked for it back in every popup; the card and the rim still apply, so the look is the same one, just still.

### lines 784-787

```python
if theme == "off":
```

AND THEIR ANSWER FOR POPUPS IN PARTICULAR. What belongs behind a screen full of figures is not necessarily what belongs behind a form somebody is reading, so `off` drops the movement and keeps the card and the rim.

### lines 796-799

```python
return install_ambient(dialog, theme=theme, speed=BACKDROP_SPEED,
```

ROUNDED TO THE CARD'S RADIUS, for the reason the setup window needed it: a SQUARE backdrop behind a rounded card is a second surface, and it is exactly the "non rounded edge black box behind the periphery" reported on About spaCR and the live mask settings.

### lines 803-804

```python
LOG.debug("no ambient backdrop for this dialog", exc_info=True)
```

INVARIANTS 10: with no ambient engine the card is still a card, and the dialog still works.

## _GlassInstaller.eventFilter

### lines 826-831

```python
if (event.type() in (QEvent.Type.Polish, QEvent.Type.Show)
```

POLISH FIRST, SHOW AS THE FALLBACK. Polish arrives before a widget is visible, which is when the window flags can be changed without hiding it. Not every dialog is polished before its first show -- one built and exec'd in a single expression may not be -- so Show still catches it, and `make_frameless` puts back what it had to hide.

## install_glass_everywhere

### lines 874-876

```python
if _INSTALLED is not None:
```

A filter belongs to the object it was installed on.  Forgetting only the owner and retaining the filter makes a later application look "already installed" while receiving no events at all.

## uninstall_glass_everywhere

### lines 918-920

```python
application = _INSTALLED_APP or application or QApplication.instance()
```

Remove it from the object that owns it.  The optional argument is retained for callers written before ownership was tracked, and is the fallback for an installation made by such an older module.
