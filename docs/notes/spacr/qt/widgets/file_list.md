# Notes from `spacr/qt/widgets/file_list.py`

Prose lifted out of `spacr/qt/widgets/file_list.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_database_tokens](#_database_tokens) (1 entry)
- [suggest_file_pairs](#suggest_file_pairs) (3 entries)
- [_number_unlabelled_plates](#_number_unlabelled_plates) (1 entry)
- [side_for_header](#side_for_header) (1 entry)
- [PairedFileTableWidget.__init__](#pairedfiletablewidget__init__) (3 entries)
- [PairedFileTableWidget](#pairedfiletablewidget) (1 entry)
- [PairedFileTableWidget._fit_columns_to_their_headers](#pairedfiletablewidget_fit_columns_to_their_headers) (1 entry)
- [PairedFileTableWidget._widen_columns_for](#pairedfiletablewidget_widen_columns_for) (3 entries)
- [PairedFileTableWidget.showEvent](#pairedfiletablewidgetshowevent) (1 entry)
- [PairedFileTableWidget.align_download_buttons](#pairedfiletablewidgetalign_download_buttons) (3 entries)
- [PairedFileTableWidget.add_paths_for_side](#pairedfiletablewidgetadd_paths_for_side) (1 entry)
- [PairedFileTableWidget._apply_pinned](#pairedfiletablewidget_apply_pinned) (1 entry)
- [PairedFileTableWidget.attach_database](#pairedfiletablewidgetattach_database) (3 entries)
- [PairedFileTableWidget.missing_databases](#pairedfiletablewidgetmissing_databases) (1 entry)
- [PairedFileTableWidget._database_item](#pairedfiletablewidget_database_item) (1 entry)
- [PairedFileTableWidget.dropEvent](#pairedfiletablewidgetdropevent) (1 entry)
- [PairedFileTableWidget._pick](#pairedfiletablewidget_pick) (1 entry)
- [PairedFileTableWidget._remove](#pairedfiletablewidget_remove) (1 entry)
- [FilePathListWidget.__init__](#filepathlistwidget__init__) (4 entries)
- [FilePathListWidget](#filepathlistwidget) (1 entry)
- [FilePathListWidget.add_paths](#filepathlistwidgetadd_paths) (1 entry)
- [FilePathListWidget._append](#filepathlistwidget_append) (1 entry)
- [FilePathListWidget._follow_path_probes.redraw](#filepathlistwidget_follow_path_probesredraw) (1 entry)
- [FilePathListWidget._start_directory](#filepathlistwidget_start_directory) (1 entry)

## Module level

### lines 69-71

```python
_DATABASE_GENERIC = frozenset({
```

Words every plate's database shares, so they cannot tell two plates apart. 'measurements' is here because spaCR's own layout calls every plate's database <plate>/measurements/measurements.db.

## _database_tokens

### lines 109-114

```python
break
```

THE NEAREST FOLDER THAT SAYS ANYTHING IS ENOUGH. Climbing further collects whatever the tree above happens to be called, and one project folder with a number in its name would invent a plate token for every database underneath it. ``measurements`` and its friends are skipped because they say nothing about WHICH plate -- which is the only reason to look up at all.

## suggest_file_pairs

### lines 156-158

```python
for score in scores:
```

PASS ONE, BY TOKEN, AND IT KEEPS PRIORITY. A user who named their files carefully must not have that overridden by arrival order, so every unambiguous token match is taken before position is consulted at all.

### lines 170-176

```python
leftover = iter(sorted(unused))
```

PASS TWO, BY POSITION. `scores.csv` and `counts.csv` share no token, so the token pass leaves both unmatched -- and the user got TWO half-empty rows with nothing to say the files belonged together. The engine has always accepted row order as the last resort (`load_regression_input_pairs` resolves plate identity "without filename guesses ... own column, partner column, then pair-row order"); it is this proposal step that never got the memo.

### lines 189-196

```python
return _number_unlabelled_plates(_attach_databases(rows, databases))
```

NUMBERED LAST, AFTER THE DATABASES HAVE HAD THEIR SAY. A database folder NAMES a plate -- `_attach_databases` fills an empty plate cell from the folder it matched -- and that is a parsed fact, where a generated number is only a default. Numbering first made every cell truthy and silently took that away: `_attach_databases` fills the cell only `if not row.get("plate")`, so a database could never name a plate again. Caught by `test_a_database_names_the_plate_when_the_csvs_could_not`, which is what that test is for.

## _number_unlabelled_plates

### lines 224-231

```python
if row.get("score") and row.get("count") and not row.get("plate"):
```

A PAIR, OR NOTHING. The request is about two files that do not share a name -- "if they do share a name it can be used if the files do not the name should be generated" -- and "the files" is the pair. A score still waiting for its partner, or a database waiting for its CSVs, is not a plate row yet, and numbering it would assert exactly the fact step 3 of 392 warns about: a label that reads as parsed when nothing parsed it. Those rows keep a blank cell, which is the honest state and the one the user is being asked to resolve.

## side_for_header

### lines 285-290

```python
return "score"
```

csv.Error is 'line contains NUL': a BINARY file was asked this question. It used to escape from inside Qt's drop dispatch, where an exception is a crash rather than an error dialog -- so dropping a measurements database on the input table killed the window. Databases are now routed by extension before they get here (:func:`is_database_path`); this catch is for the next binary.

## PairedFileTableWidget.__init__

### lines 333-337

```python
self._pinned: dict[str, dict] = {}
```

Databases the user placed on a row BY HAND, anchored to that row's identity rather than to its index. Every addition re-proposes the whole table, so without this an explicit attachment would be undone by the next CSV drop -- silently, which is the same class of bug as pairing by list position.

### lines 350-352

```python
self.status = QLabel(self._EMPTY_STATUS, self)
```

What the table just did, in words. A database attached to "the first row without one" that says nothing is a file the user believes is on another plate.

### lines 377-378  _(unsure)_

```python
self.setAcceptDrops(True)
```

The table is the drop target the user aims at, so the widget takes drops and the table does not swallow them first.

## PairedFileTableWidget

### lines 387-389

```python
RULE_COLUMN = 4
```

The read-only column that reports how the plate id was resolved. It moved right when the database column was inserted; it is named here so that the next column to arrive moves one number, not four.

## PairedFileTableWidget._fit_columns_to_their_headers

### lines 431-433

```python
needed = metrics.horizontalAdvance(text) + 18
```

The sort indicator and the section's own padding both sit beside the text; 18 px covers them at every font scale this has been measured at.

## PairedFileTableWidget._widen_columns_for

### lines 475-488

```python
ours = applied.get(column, self._NEVER_SET)
```

NEVER AGAINST THE USER, AND THAT IS WHY THE WIDTH IS

REMEMBERED RATHER THAN A FLAG BEING SET. A once-only guard answers "have we ever done this" when the question is "has the user moved it since": the caption is set in English when the row is built and REPLACED by the language pass afterwards, so a column sized once is sized for the wrong word. "Count" is 100 px and "Contagem" is 151.

THE WIDTH IS RECORDED EVEN WHEN NOTHING IS RESIZED, which is the half that makes the rule work for a column that already fit. Otherwise "we have never touched this one" and "the user has not touched it either" are the same state, and a column dragged narrow before any caption grew would be dragged back the moment one did.

### lines 493-496

```python
applied[column] = self._THE_USERS
```

THE USER HAS MOVED IT, so this column stops being managed -- for good, not until the next caption grows. A table that argued with a drag every time the language changed would be unusable.

### lines 503-508

```python
applied[column] = current
```

A BASELINE FOR A COLUMN THAT ALREADY FITS, which is the half that makes the rule work at all: without it, "we have never touched this one" and "the user has not touched it either" are the same state, and a column dragged narrow before any caption grew would be dragged back the moment one did.

## PairedFileTableWidget.showEvent

### lines 532-533

```python
LOG.debug("could not align the Download row", exc_info=True)
```

A strip that failed to align is a strip in the wrong place; a table that failed to show is no input at all.

## PairedFileTableWidget.align_download_buttons

### lines 553-556

```python
if strip is None:
```

ONE STRIP OR NONE. The four buttons share a parent because

`_install_example_data_button` puts them in one row widget; if that ever stops being true, aligning some of them would leave the rest laid out by a layout that no longer holds them.

### lines 566-569

```python
self._widen_columns_for(columns)
```

BEFORE the strip is handed to the layout: the layout centres a button in whatever the column is, so a column too narrow for its button has to be widened first or the caption is clipped by the alignment that was supposed to make it readable.

### lines 571-574

```python
self._download_columns = list(columns)
```

AND AGAIN WHENEVER A CAPTION CHANGES SIZE. The language pass runs after the row is built, so the first fit is against the English text; `ColumnAlignedRow.invalidate` reports the change and the table -- which owns the header -- acts on it.

## PairedFileTableWidget.add_paths_for_side

### lines 649-650  _(unsure)_

```python
self._repropose()
```

Re-propose so a count dropped after its score lands on the same row: the pairing is by filename token, not by drop order.

## PairedFileTableWidget._apply_pinned

### line 714  _(unsure)_

```python
self._pinned.pop(database, None)
```

The user removed it from the table; the pin goes with it.

## PairedFileTableWidget.attach_database

### lines 787-789

```python
database = os.fspath(path).strip()
```

Stripped, because the table stores text and hands it back stripped: a path that is not equal to its own strip is one this widget could write and then never find again.

### lines 808-810

```python
self.add_paths_for_side([database], "database")
```

After this the path IS somewhere in the table: the side lists are rebuilt from the table itself, so a path that was not already on a row comes back on one of its own.

### lines 833-835

```python
target = self._first_row_without_database()
```

Asked again rather than adjusted by hand: removing a row shifts every index after it, and an off-by-one here attaches a database to the wrong plate, which is the one mistake this column cannot make.

## PairedFileTableWidget.missing_databases

### lines 860-862

```python
if database and not path_probe.exists(database, wait=True):
```

`wait=True`: "is this database missing" is the whole question, and the optimistic default answers "present" for a path never seen -- so a genuinely absent file is never flagged.

## PairedFileTableWidget._database_item

### lines 919-921

```python
item.setForeground(Qt.red)
```

Marked, not discarded: the path may be right and the disk merely not mounted yet, and a silently emptied cell is worse than a red one.

## PairedFileTableWidget.dropEvent

### lines 1016-1020

```python
local = self.table.viewport().mapFrom(self, position)
```

VIEWPORT coordinates, which is what columnAt and rowAt document themselves to take. Mapping into the table itself instead offsets every answer by the header: one whole row down, and one vertical header's width across. Nobody noticed while only the column was read -- columns are wide -- but a row aimed at is a row missed.

## PairedFileTableWidget._pick

### lines 1053-1054

```python
self.add_paths_for_side(paths, side)
```

Through the same seam a drop uses, so the picker cannot pair by the order the file dialog happened to return.

## PairedFileTableWidget._remove

### lines 1129-1130

```python
self._pinned.pop(self._cell(row, self.SIDE_COLUMNS["database"]),
```

A removed row takes its database's pin with it, or the next addition would put a file back that the user just deleted.

## FilePathListWidget.__init__

### lines 1219-1220

```python
self._allow_folders = bool(allow_folders) and not self._single
```

A folder is not a file, and expanding one into "all the CSVs in here" cannot mean anything for a setting that names one of them.

### line 1233

```python
self._list.setAcceptDrops(False)
```

The list itself must not swallow the drop before the widget sees it.

### lines 1265-1266

```python
if self._single:
```

One path has no order, so the two buttons that reorder the list are not built at all rather than built and left doing nothing.

### lines 1273-1275

```python
self._up_button.setMaximumWidth(
```

A CAP THAT CANNOT CUT (193). 30 px keeps the arrows compact, and `sizeHint` is the floor: at a large font or a glyph a theme renders wider, the button grows rather than clipping.

## FilePathListWidget

### lines 1335-1336

```python
_PLACEHOLDERS = {"list of paths", "none", "", "[]"}
```

A settings CSV written before this widget existed can hold the literal placeholder 'list of paths'; it is not a path and must not become one.

## FilePathListWidget.add_paths

### lines 1388-1390

```python
if path_probe.isdir(expanded, wait=True):
```

`wait=True`: the user just dropped or chose this, and is waiting on the result. Without it an unseen path answers "not a directory" and the folder is appended AS a file.

## FilePathListWidget._append

### lines 1428-1430

```python
item.setToolTip(f"{resolved}\n\nThis path does not exist right now.")
```

Marked, not dropped: a settings file may legitimately be edited on one machine and run on another, and silently discarding the path would leave the user staring at an empty list.

## FilePathListWidget._follow_path_probes.redraw

### line 1515  _(unsure)_

```python
pass
```

The widget has gone; the signal outlived it.

## FilePathListWidget._start_directory

### lines 1621-1623

```python
if self._last_directory and path_probe.isdir(self._last_directory,
```

`wait=True`: this answer chooses where a dialog the user is opening RIGHT NOW will land, so an unknown path must not silently mean "not a directory" and drop them at the default location.
