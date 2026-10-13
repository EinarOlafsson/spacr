# Notes from `spacr/qt/widgets/object_settings_grid.py`

Prose lifted out of `spacr/qt/widgets/object_settings_grid.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_coerce](#_coerce) (2 entries)
- [ObjectSettingsModel.flags](#objectsettingsmodelflags) (1 entry)
- [ObjectSettingsModel.setData](#objectsettingsmodelsetdata) (1 entry)
- [ObjectSettingsModel.headerData](#objectsettingsmodelheaderdata) (1 entry)
- [_GridHeightGrip.__init__](#_gridheightgrip__init__) (1 entry)
- [ObjectSettingsGrid.__init__](#objectsettingsgrid__init__) (10 entries)
- [ObjectSettingsGrid._sync_help_height](#objectsettingsgrid_sync_help_height) (1 entry)
- [ObjectSettingsGrid.eventFilter](#objectsettingsgrideventfilter) (1 entry)
- [ObjectSettingsGrid._offer_tooltip](#objectsettingsgrid_offer_tooltip) (2 entries)
- [ObjectSettingsGrid._apply_help_animation](#objectsettingsgrid_apply_help_animation) (1 entry)
- [ObjectSettingsGrid._write_help](#objectsettingsgrid_write_help) (2 entries)
- [ObjectSettingsGrid._visible_table](#objectsettingsgrid_visible_table) (1 entry)
- [ObjectSettingsGrid.add_organelle](#objectsettingsgridadd_organelle) (3 entries)

## _coerce

### lines 133-135

```python
if raw == "" or raw.lower() in (AUTO_TEXT, OFF_TEXT):
```

BOTH WORDS CLEAR THE CELL. The table draws an unset value as "auto" for most questions and "off" for a channel, and whichever word the user is looking at is the one they will type back.

### lines 147-149

```python
for kind in (int, float):
```

NO TYPE TO COPY. Read a number as a number so a diameter typed into an empty cell is not stored as text, and leave anything else as the string it is -- a model name is a string and always was.

## ObjectSettingsModel.flags

### lines 255-256

```python
return Qt.ItemIsSelectable
```

NOT EDITABLE AND NOT ENABLED: a cell that can be typed into invents a settings key nothing reads.

## ObjectSettingsModel.setData

### lines 312-314

```python
like = next((v for o, v in row.items()
```

THE SAME QUESTION ABOUT ANOTHER OBJECT is the best evidence available about what this one is: a diameter is a diameter whether it is a cell's or a nucleus's.

## ObjectSettingsModel.headerData

### lines 339-343

```python
return None
```

NO TOOLTIP ON THE ROW HEADER. Every cell in the row now carries the full help -- the typed body, the API link and the setting's animation -- and a second, plainer tooltip on the name beside them is the same explanation twice, in the place the pointer crosses on its way to the cell.

## _GridHeightGrip.__init__

### lines 373-374  _(unsure)_

```python
self.setObjectName("ConsoleSectionResizeHandle")
```

The console's handle is styled by this name; the two are the same affordance and should not look like two.

## ObjectSettingsGrid.__init__

### lines 469-483

```python
from .hover_tooltip import (ANIMATION_MARK, API_MARK, PURPLE, TEAL,
```

THE HELP GOES ABOVE THE TABLE, in a band that does not move.

A popup was tried twice and neither placement works over a table. Under the pointer it covers the row being read, which is the one thing the reader is comparing the help against, and it jumps with every cell. Beside the table it is in one place, but "the table" is the whole container -- wider than the columns -- and aiming at the columns' right edge would move the help further right with every organelle added.

A fixed band solves both: one position for the life of the panel, never over the data, and it cannot drift as the table grows. It is reserved at a constant height so a hover does not reflow the form around it -- a help area that resized would push the table under the pointer as the text arrived.

### lines 497-500

```python
self._help.setObjectName("SubtitleSmall")
```

`SubtitleSmall` is the per-setting hint strip's own object name, borrowed rather than invented: this band says the same kind of thing in the same voice, and a new name would be a new themed surface to keep in step with it.

### lines 502-504

```python
self._help.setStyleSheet("background: transparent;")
```

A caption over the backdrop, not a surface. Named widgets keep their fill under the blanket `QWidget { background-color: bg }` rule, which would draw a panel-coloured slab above the table.

### lines 514-519

```python
self._help_links = QWidget(self._help_band)
```

THE SAME TWO WORDS THE POPUP DRAWS, and the same widgets: `API` in teal, `Animation` in purple, borrowed from `hover_tooltip` rather than restyled here. A second implementation of the footer is a second thing to re-theme, and the request was for a link that looks EXACTLY like the one on every other setting -- which is a thing that can only be guaranteed by using it.

### lines 540-541  _(unsure)_

```python
self._help_links.setStyleSheet(
```

The popup styles these two from its own sheet; the band is not a popup, so it carries the same two rules itself.

### lines 551-554

```python
self._help_animation = _AnimationView(self.HELP_ANIMATION_PX,
```

SMALLER THAN THE POPUP'S 220px SQUARE. The band is reserved permanently above the table, so its height is space the settings never get back -- and it may not grow when an animation arrives, or the table moves out from under the pointer that asked for it.

### lines 571-574

```python
install_sorting(self._table)
```

AFTER setModel, as the contract requires: a QTableView is wrapped in a proxy, so the selection model has to be taken afterwards. The stored answers are unaffected -- `table()` reads the model, not the view, so sorting the questions on screen reorders nothing on disk.

### lines 582-587

```python
self._table.setSizePolicy(QSizePolicy.Policy.Expanding,
```

THE TABLE OWNS ITS HEIGHT, and the grip below changes it. Inside a settings panel the table is one row of a scrolling form, so it gets whatever height the form hands it -- which was a QTableView's default and put twenty-odd questions behind an inner scrollbar inside an outer one. It now opens tall enough to show its rows and can be dragged taller or shorter from the grip.

### lines 629-631

```python
self._help_band.installEventFilter(self)
```

ENTERING THE BAND CANCELS THE HIDE, which is what makes the API link reachable at all: the pointer leaves the table to get to it, and leaving the table is what started the countdown.

### lines 636-643

```python
self.installEventFilter(self)
```

THE BAND IS RE-RESERVED THROUGH `eventFilter`, not a `changeEvent` override. The font scale is a preference the user can move while a panel is open, and a band sized once at construction would keep the old height and clip its last lines -- but a new public override is a new symbol in the API surface, which is mirrored symbol-for-symbol into nine locales' catalogs and cannot be translated from here. This class already has an `eventFilter`; filtering itself costs no new surface.

## ObjectSettingsGrid._sync_help_height

### lines 697-699

```python
self._help_band.setFixedHeight(max(lines, self.HELP_ANIMATION_PX))
```

The BAND, not just the prose: the square sits beside the text and is taller than it, so reserving only the text's height would let the band grow the moment an animation arrived.

## ObjectSettingsGrid.eventFilter

### lines 754-755

```python
self._help_hide_timer.start(self.HELP_HIDE_DELAY_MS)
```

LINGER, do not clear. The reader may be on their way to the API link, and the way to it leaves the table.

## ObjectSettingsGrid._offer_tooltip

### lines 774-775

```python
self._help_show_timer.stop()
```

Off a cell but still inside the table: let the last help stand for a moment rather than blanking between rows.

### lines 779-780

```python
self._help_pending = key
```

AFTER A REST, NOT ON ARRIVAL. Dragging across a row of twenty cells rewrote the band twenty times, which reads as flicker.

## ObjectSettingsGrid._apply_help_animation

### lines 827-829

```python
self._help_anim.setVisible(
```

Offered but folded -> the word is the invitation. Showing -> it folds away again. Undecodable -> no word, because a word that visibly does nothing is worse than no word.

## ObjectSettingsGrid._write_help

### lines 862-865

```python
body, url = split_api_link(text)
```

THE ANCHOR BECOMES THE WORD. `format_tooltip` ends the body with "Open spaCR API documentation" as a full anchor; every other surface in spaCR renders that destination as the teal API** word instead, and this one was showing the sentence.

### lines 871-873

```python
animation = None
```

The animation belongs to the SETTING, so it is resolved from the key rather than from the prose, and it is folded away by default exactly as the popup's is -- the word is the invitation.

## ObjectSettingsGrid._visible_table

### lines 1009-1013

```python
for index, role in enumerate(live):
```

AND THE COUNT CAN ASK FOR MORE THAN THE FILE HOLDS. A settings dict carries keys for the slots it has been given, which is usually one; a count of three is then three columns, and two of them have to be made. Seeded from the slot before, so a second organelle starts where the first one is.

## ObjectSettingsGrid.add_organelle

### lines 1112-1122

```python
self._base = from_table(self._model.table(), self._base)
```

THE COUNT IS RAISED, NOT JUST THE TABLE. `number_of_organelles` is what every other reader of these settings goes by -- the flat form, the pipeline, a saved settings file -- so a column added here without it would be a column the rest of the application does not believe in, and would vanish the next time the table was rebuilt from the count. THE EDITS FIRST. What is on screen may differ from `_base` -- every cell the user has typed into lives in the model until something folds it back -- and the rebuild below reads `_base`. Without this line, adding an organelle silently reverts every unsaved edit and seeds the new column from the values on disk.

### lines 1125-1128

```python
table = self._visible_table()
```

REBUILT FROM THE COUNT, not widened from what is on screen. The settings dict already carries this slot's keys -- that is why lowering the count is reversible -- so raising it brings back the answers the slot had rather than a copy of its neighbour's.

### lines 1131-1133

```python
previous = [o for o in self._model.objects()
```

A settings dict that never held this slot at all. Seed it from the organelle before it, so a second mitochondrion starts where the first one is rather than at a default nobody chose.
