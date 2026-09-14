# Notes from `spacr/qt/widgets/dock.py`

Prose lifted out of `spacr/qt/widgets/dock.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [DockRow.__init__](#dockrow__init__) (9 entries)
- [SectionHeader.__init__](#sectionheader__init__) (1 entry)
- [Dock.__init__](#dock__init__) (8 entries)
- [Dock._light_only](#dock_light_only) (1 entry)
- [Dock.refresh_visibility](#dockrefresh_visibility) (2 entries)
- [Dock.refresh_icons](#dockrefresh_icons) (2 entries)
- [Dock.apply_theme](#dockapply_theme) (2 entries)
- [Dock.fitting_width](#dockfitting_width) (2 entries)

## Module level

### lines 49-56

```python
from PySide6.QtCore import QEvent, Qt, Signal
```

QEvent AT MODULE SCOPE, NOT INSIDE THE CALLBACK. A function-local import in an event handler is not lazy loading: this module is a QWidget module and cannot load without QtCore, so the import bought nothing but a sys.modules lookup on every event -- and it put an EXCEPTION SITE on a path with no way to report one. The same shape in `ModuleHintBar.event` produced 419 errors in one sweep when a test stubbed PySide6.QtCore out of sys.modules and teardown then delivered a paint event.

## DockRow.__init__

### lines 110-114

```python
super().__init__(name.replace("&", "&&"), parent)
```

`&` DOUBLED, OR QT EATS IT. A QPushButton reads a single ampersand as a mnemonic marker, so "Align & Stitch" draws as "Align _Stitch" -- the ampersand gone and the S underlined. The accessible name below keeps the real character, because a screen reader must not say the escape.

### lines 118-120

```python
self.setObjectName("SidebarItem")
```

THE LEGACY OBJECT NAME, deliberately: the theme carries eight `QPushButton#SidebarItem` rules and a rename would silently un-style every row in the dock.

### line 122  _(unsure)_

```python
self.setProperty("moduleNameSource", name)
```

What the bottom strip reads off whatever the pointer is over.

### lines 125-127

```python
self.setProperty("navKey", key)
```

`navKey` is how refresh_icons, the tutorial highlighter and the maturity tests find a row. It is the row's identity to everything outside this module.

### lines 129-132

```python
self.setProperty("moduleAppKey", key)
```

`moduleAppKey` is module_hints.KEY_PROPERTY: the bottom strip's event filter reads it off whatever the pointer is over. Setting it here is what makes the dock explain itself through the SAME mechanism as the menus and the tiles, rather than a second one.

### lines 134-144

```python
self.setProperty("moduleTooltipStyle", "sidebar")
```

AND THE STYLE, without which the two lines below are English for ever. `_refresh_module_help` dispatches on this property: "sidebar" retranslates the accessible name and description on every language change, and anything else falls through to a branch that sets a status tip and leaves both alone. This row was in that fallback, so a screen reader announced every module in English in all nine languages -- the name and summary set below are correct exactly once, at construction, in whatever language the app started in.

It also clears the popup tooltip on each pass, which is what this row already wants: see "NO POPUP TOOLTIP" below.

### lines 146-148

```python
self.setAccessibleName(name)
```

AN ACCESSIBLE NAME EVEN THOUGH THE TEXT IS VISIBLE. The old row painted no text and needed one; a screen reader still needs the full name when a long one has been elided down to fit the column.

### lines 150-154

```python
self.setAccessibleDescription(desc)
```

AND THE DESCRIPTION, which this dock stopped setting when it was rewritten. The summary is drawn nowhere on the row -- it goes to the strip along the bottom -- so for a screen reader the accessible description is the ONLY route to it, and without this a row announced its name and nothing about what the module does.

### lines 159-160

```python
self.setToolTip("")
```

NO POPUP TOOLTIP. The bottom strip is the explanation surface; a popup here would be a second one, in a place the pointer covers.

## SectionHeader.__init__

### lines 210-211

```python
self.setObjectName("SidebarSection")
```

The legacy name, deliberately: the theme styles `SidebarSection` and the maturity test looks headers up by it.

## Dock.__init__

### lines 263-277

```python
from ..theme import make_transparent
```

WITHOUT THIS THE COLUMN PAINTS NOTHING AT ALL. A plain QWidget ignores a stylesheet background unless it is told to draw one, so the ground set in `apply_theme` was being dropped and the translucent panel composited straight onto the window's black base the "black box" behind the dock. THE BLANKET RULE WAS THE BOX. The application sheet carries `QWidget { background-color: bg }`, so any untagged container paints an opaque rectangle -- and a plain QWidget holding a rounded panel is exactly that: a square of `bg` behind rounded corners. Colouring it (black, grey, the page ground) only changes which colour the rectangle is; `make_transparent` stops it painting at all.

`Panel` in `home.py` already does this, and its comment says why in as many words: six untagged wrappers stacked down the aside "read as one large black column behind every panel". The dock is one of them.

### lines 287-290

```python
self._items = self._rows
```

THE OLD PRIVATE NAMES, bound to the same objects. Several suites reach into `_items` and `_section_headers` rather than through `rows()` and `sections()`, and a rename that broke them would be churn with no reader-visible gain.

### lines 294-309

```python
outer = QVBoxLayout(self)
```

A ROUNDED PANEL, NOT A BLACK COLUMN, and it is a CHILD frame rather than the dock's own background for a reason. Instruction 369 took the dock's container off where there is no picture behind it, and tests/qt/test_space_theme.py pins `#Sidebar` transparent on the flat themes to keep it off. Painting the panel here satisfies both: the container stays transparent and the panel is a widget inside it.

The look is HomePanelBox's, deliberately -- asked for on 2026-09-04, "a rectangle with rounded edges like the top box on the Home screen with the spacr logo and text" -- so the two read as the same material rather than as two guesses at one. NO INSET HERE. The dock widget itself is the rounded box now, and a widget's own margins sit inside its background -- an inset here would pad the contents without moving the box off the window edge. The gap around the box is the SLOT's margin; see `MainWindow._dock_slot`.

### lines 326-329

```python
self._scroll = QScrollArea(self)
```

THE ROWS SCROLL AND THE TITLE DOES NOT. Measured at 1440x900 -- the realistic laptop -- a row per module plus a heading per section asks for more height than the window has, and the last few modules were simply unreachable. This is structure, not decoration.

### lines 332-334

```python
make_transparent(self._scroll)
```

A QScrollArea and its VIEWPORT are two widgets and the viewport is the one that paints; `make_transparent` tags both, and forgetting the viewport is the documented way to get this wrong.

### lines 352-354

```python
if section and section != current:
```

A ROW WITH NO SECTION stands above the headings and is never collapsed away. Home is the one that needs it: it is how you get back, so it cannot live inside a category you can shut.

### lines 357-358

```python
header.installEventFilter(self)
```

The heading is a label, so the click comes through the filter rather than a pressed signal.

### lines 363-366

```python
if not self._headers or len(self._headers) == 1:
```

ONLY THE FIRST CATEGORY STARTS OPEN. Every section open at once makes the dock taller than a 900 px laptop screen, which is the failure collapsing was introduced to fix. The first is the pipeline, and it is why the dock is on screen.

## Dock._light_only

### lines 421-422

```python
want = (key is not None and getattr(row, "key", None) == key)
```

Same reason as `refresh_icons`: a row this dock did not build has no `key`, and it is never the one to light.

## Dock.refresh_visibility

### line 536  _(unsure)_

```python
row.setVisible(mature and (not section or section in self._open))
```

A section-less row (Home) has no heading to be shut by.

### lines 542-543  _(unsure)_

```python
header.setProperty("open", section in self._open)
```

The stylesheet and the tests both read `open` off the heading to tell a shut category from one that is merely empty.

## Dock.refresh_icons

### lines 563-565

```python
row.setIconSize(QSize(side, side))
```

ONE SIZE, SET ONCE, FOR EVERY ROW IN EVERY STATE. The old dock grew the icon under the pointer and shrank it again, and that is what relaid the column out and made it blink.

### lines 569-573

```python
key = getattr(row, "key", None)
```

`getattr`, NOT `row.key`. `_rows` is a plain list and callers append to it: three tests put a bare `ElidingPushButton` in to check that a row with no nav key is left alone, and a row this dock did not build has no `key` at all. Asking for one raised `AttributeError` out of a theme refresh.

## Dock.apply_theme

### lines 599-607

```python
"QFrame#DockPanel {"
```

THE BOX IS A FRAME INSIDE A TRANSPARENT CONTAINER, which is `Panel`'s arrangement in `home.py` and the one the request asks for: "cant you just make that same box widget in place of the dock". Same three values as `QFrame#HomePanelBox` `pane_surface('surface_alt')`, `border_soft`, 8 px -- so the dock and the Home boxes stay one material.

The container above it paints nothing. That was the whole bug: a frame cannot round the corners of the widget behind it.

### lines 621-624

```python
f'QPushButton#SidebarItem[hovered="true"] {{ color: {accent}; }}'
```

`[hovered="true"]`, NOT `:hover`. Qt drives `:hover` from

`WA_UnderMouse`, which sticks when a click swaps the screen out from under the pointer -- see `_on_row_hovered`. The dock sets this property itself so at most one row is ever lit.

## Dock.fitting_width

### lines 648-659

```python
widest = 0
```

MEASURED FROM THE FULL NAME, not from `sizeHint()`. These rows ELIDE, so their size hint reports the width of the shortened text ask it how much room it wants and it answers with how much it has already given up. Measured 2026-09-05: "Cellpose Model Comparison Workbench" hinted 263 px, the column sized itself to 275, and the row was still clipped, because 263 was the width of "Cellpose Model Comparis...".

The icon is added explicitly for the same reason. The old dock painted icons and no text, so a width tuned for text alone was right; this one draws both, and the icon's slot is not in a text measurement.

### lines 667-673

```python
room = widest + scaled_px(ICON_PX) + scaled_px(40)
```

THE ALLOWANCE IS MEASURED, not guessed. At 100 % with the longest shipped-length name ("Cellpose Model Comparison Workbench", 243 px of text) the overhead between the column's width and the room the row leaves that text is 58 px: 14 for the panel's border and the scroll area, 44 for the row's own icon slot and padding. 34 + the icon came to 54 and left the name four pixels short -- which is a column that widened for a name and clipped it anyway.
