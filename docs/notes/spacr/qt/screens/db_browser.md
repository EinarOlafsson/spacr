# Notes from `spacr/qt/screens/db_browser.py`

Prose lifted out of `spacr/qt/screens/db_browser.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_read_only_uri](#_read_only_uri) (1 entry)
- [quote_ident](#quote_ident) (1 entry)
- [Module level](#module-level) (1 entry)
- [EditRefused](#editrefused) (1 entry)
- [coerce_for_column](#coerce_for_column) (1 entry)
- [ReadOnlyDb.__init__](#readonlydb__init__) (1 entry)
- [ReadOnlyDb.table_info](#readonlydbtable_info) (1 entry)
- [ReadOnlyDb.row_key](#readonlydbrow_key) (2 entries)
- [ReadOnlyDb.select_sql](#readonlydbselect_sql) (1 entry)
- [ReadOnlyDb.chunk](#readonlydbchunk) (1 entry)
- [WritableDb](#writabledb) (1 entry)
- [WritableDb.update_cell](#writabledbupdate_cell) (2 entries)
- [PreviewModel.data](#previewmodeldata) (1 entry)
- [DbBrowserScreen.__init__](#dbbrowserscreen__init__) (10 entries)
- [DbBrowserScreen._build_ui](#dbbrowserscreen_build_ui) (11 entries)
- [DbBrowserScreen.set_database](#dbbrowserscreenset_database) (1 entry)
- [DbBrowserScreen.select_table](#dbbrowserscreenselect_table) (2 entries)
- [DbBrowserScreen.set_column_filter](#dbbrowserscreenset_column_filter) (1 entry)
- [DbBrowserScreen.apply_seed](#dbbrowserscreenapply_seed) (2 entries)
- [DbBrowserScreen.refresh](#dbbrowserscreenrefresh) (1 entry)
- [DbBrowserScreen._fetch_chunk](#dbbrowserscreen_fetch_chunk) (1 entry)
- [DbBrowserScreen._apply_chunk](#dbbrowserscreen_apply_chunk) (2 entries)
- [DbBrowserScreen._apply_count](#dbbrowserscreen_apply_count) (1 entry)
- [DbBrowserScreen._update_sort_state](#dbbrowserscreen_update_sort_state) (1 entry)
- [DbBrowserScreen._on_header_clicked](#dbbrowserscreen_on_header_clicked) (1 entry)
- [DbBrowserScreen._report_table_status](#dbbrowserscreen_report_table_status) (1 entry)
- [DbBrowserScreen._apply_linked_filter](#dbbrowserscreen_apply_linked_filter) (1 entry)
- [DbBrowserScreen._on_view_selection_changed](#dbbrowserscreen_on_view_selection_changed) (1 entry)
- [DbBrowserScreen.on_linked_selection_changed](#dbbrowserscreenon_linked_selection_changed) (1 entry)
- [DbBrowserScreen._collect_filter](#dbbrowserscreen_collect_filter) (1 entry)
- [DbBrowserScreen._confirmation_text](#dbbrowserscreen_confirmation_text) (1 entry)
- [DbBrowserScreen.edit_cell](#dbbrowserscreenedit_cell) (1 entry)
- [DbBrowserScreen._update_edit_ui](#dbbrowserscreen_update_edit_ui) (1 entry)
- [DbBrowserScreen.export_csv](#dbbrowserscreenexport_csv) (1 entry)
- [DbBrowserScreen._start_job](#dbbrowserscreen_start_job) (2 entries)
- [DbBrowserScreen._update_controls](#dbbrowserscreen_update_controls) (1 entry)
- [DbBrowserScreen.closeEvent](#dbbrowserscreencloseevent) (1 entry)
- [_build_lineage](#_build_lineage) (1 entry)

## _read_only_uri

### lines 226-228

```python
return "file:" + _urlquote(str(path).replace("\\", "/"), safe="/:") + "?mode=ro"
```

Percent-escape everything a URI would otherwise treat as syntax ('?', '#', '%') while leaving path separators and the Windows drive colon alone.

## quote_ident

### lines 232-234  _(unsure)_

```python
def quote_ident(name: str) -> str:
```

SQL construction — identifiers validated, values always bound

## Module level

### lines 320-323

```python
_FORBIDDEN_RAW = re.compile(
```

Statements that have no business inside a WHERE clause. The browsing connection is read-only anyway, so this is a second line of defence — it mostly stops a user pasting a whole script into the predicate box and being confused by the error.

## EditRefused

### lines 362-364  _(unsure)_

```python
class EditRefused(Exception):
```

Editing: types, coercion, and the one statement we are willing to run

## coerce_for_column

### lines 449-450

```python
if _INT_RE.match(stripped):
```

NUMERIC, or a column with no declared type: mirror SQLite's own behaviour — store a number when it is one, text otherwise.

## ReadOnlyDb.__init__

### lines 512-513

```python
with self._con() as con:
```

Probe now so "that file isn't a database" surfaces at open time rather than three clicks later.

## ReadOnlyDb.table_info

### line 587  _(unsure)_

```python
return self._execute(
```

above and quoted here.

## ReadOnlyDb.row_key

### lines 649-653

```python
self.check_table(table)
```

A view has no intrinsic row address. Probing ``SELECT _rowid_`` is not portable discovery: newer SQLite releases may accept that expression on a view and return NULL, which made CI arm editing for a result set that no UPDATE can address uniquely. Determine the schema object kind first; only a real table gets the rowid probe.

### lines 665-672

```python
from ...predictions import _rowid_alias
```

SQLite identifiers are case-insensitive, and a table that DECLARES a column named `rowid` makes the bare name resolve to that column rather than to the implicit row id. png_list declares `rowID`, so this probe used to SUCCEED there and hand back 'r1' -- after which editing one cell issued `UPDATE png_list SET c = ? WHERE rowid 'r1'` and rewrote every crop in that plate row, and keyset paging ordered a TEXT column as if it were the row id. Ask for an alias the table does not shadow.

## ReadOnlyDb.select_sql

### lines 705-706  _(unsure)_

```python
sql += f" ORDER BY {quote_ident(key_cols[0])}"
```

key_cols[0], not the literal "rowid" -- png_list declares a rowID column that shadows the bare name.

## ReadOnlyDb.chunk

### lines 806-809

```python
self.check_columns(table, [order_by[0]])
```

Validated against the table's real columns, not merely quoted: this string reaches an ORDER BY clause, and check_columns is the same gate every other column name in this class goes through.

## WritableDb

### lines 931-933  _(unsure)_

```python
class WritableDb:
```

Read-write sqlite access — the opt-in edit path, and nothing else

## WritableDb.update_cell

### lines 1002-1005

```python
with transaction(con):
```

Validation and update share one write transaction. This closes the former check-then-write window where another connection could remove or replace the addressed row between COUNT and UPDATE.

### lines 1020-1021  _(unsure)_

```python
implicit = {"rowid", "oid", "_rowid_"}
```

All three implicit row-id spellings are legal here, not just

"rowid": row_key() returns one the table does not shadow.

## PreviewModel.data

### lines 1347-1349

```python
return repr(value) if isinstance(value, float) else str(value)
```

The editor must start from the *exact* stored value, or a cell the user opens and closes without touching would be written back rounded.

## DbBrowserScreen.__init__

### lines 1467-1472

```python
self.app_key = "db_browser"
```

ITS OWN REGISTRY KEY. `install_folds_on` dispatches on this, so without it the folds declared at the foot of this module could never be handed to the screen that declares them. Passing `app_key` to `ModuleHeader` is not the same thing -- that tells the HEADER which module it titles; this tells the SCREEN what it is.

### lines 1482-1486

```python
self._token: int = 0
```

incremental load state

Every load carries a token. A result whose token is stale (the user switched table or database while it was in flight) is dropped instead of painted — that race is the reason async loading can otherwise feel *worse* than synchronous loading.

### lines 1508-1516

```python
self._jobs: Dict[int, tuple] = {}
```

job id -> (QThread, PipelineWorker) for every job that has been started and whose event loop has not yet exited. This is an ownership table, not a convenience: PySide6 destroys the C QThread as soon as the last Python reference goes, and destroying a *running* QThread aborts the process. A single `self._thread` slot is not enough, because `worker.finished` (which lets the next job start) fires strictly before `thread.finished` (which retires the old one) — so two jobs legitimately overlap for a moment.

### line 1518, trailing

```python
self._thread = None
```

most recent thread, for introspection

### lines 1520-1522

```python
self._pending: Dict[int, tuple] = {}
```

job id -> (result box, completion callback, kind). Keyed by id rather than FIFO because loads legitimately overlap: a chunk the user abandoned can settle *after* the one that replaced it.

### lines 1524-1525  _(unsure)_

```python
self._queue: List[tuple] = []
```

Jobs waiting for the single worker slot: (fn, on_done, kind, token). See _run_job for why only one runs at a time.

### lines 1540-1545

```python
self._syncing_selection: bool = False
```

linked selection state

True while an incoming selection is being written into the view. Echo suppression stops this screen hearing its *own* publications, but not itself re-publishing what it was just told: the round trip would replace the shared selection with the part of it this page happens to have loaded, quietly narrowing a lasso to one chunk.

### lines 1554-1556

```python
from ..dnd import install_dropzone
```

Match the pipeline screens: the database file, its measurements/ folder, and the enclosing run folder can all be dropped anywhere on this screen.

### lines 1564-1565  _(unsure)_

```python
self.link_selection("db_browser")
```

After the UI: both hooks paint into the view, and a filter can already be set by the time this screen opens.

### lines 1567-1569

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## DbBrowserScreen._build_ui

### lines 1582-1587

```python
header = ModuleHeader(
```

A ModuleHeader RATHER THAN A BARE LABEL. It draws the same

`DisplayHeading` this used to build by hand, and it is what every other module page wears -- but the reason for the change is `add_trailing`: the fold strip declared at the foot of this module is hung on a masthead, and a plain QLabel is not one, so Lineage and Tabulate had nowhere to appear.

### line 1607  _(unsure)_

```python
src_row = QHBoxLayout()
```

── Source row ────────────────────────────────────────────────

### line 1643  _(unsure)_

```python
split = QSplitter(Qt.Horizontal, self)
```

── Body splitter: tables | preview ───────────────────────────

### lines 1679-1681

```python
self._view.setObjectName("DbBrowserPreview")
```

Named so the sorting sweep can tell this one view apart: it is the only table in the application that must NOT take Qt's model sort, because it sorts in SQL over rows it has not loaded.

### lines 1684-1685

```python
self._view.setEditTriggers(QAbstractItemView.NoEditTriggers)
```

Read-only in the UI as well as on the connection, until edit mode says otherwise.

### lines 1689-1690  _(unsure)_

```python
self._view.setSortingEnabled(False)
```

Sorting stays off while a table is partially loaded — see _update_sort_state().

### lines 1693-1694

```python
header.setSectionsClickable(True)
```

Our own handler, not Qt's model sort: the sort runs in SQL over the whole table, so it is right however little of the table is loaded.

### lines 1698-1699

```python
header.setStretchLastSection(False)
```

No stretch-last-section: with feature columns the last one would balloon to fill the window while its neighbours stay clipped.

### lines 1702-1706

```python
self._view.selectionModel().selectionChanged.connect(
```

Publish whatever the user picks out, and re-hide the filtered rows whenever the row set moves under the view. `setRowHidden` is positional and Qt clears it on a model reset, so both signals are needed: `modelReset` for a new page, a column search or a sort, and `rowsInserted` for the chunks that arrive as the user scrolls.

### line 1744  _(unsure)_

```python
filt_row = QHBoxLayout()
```

── Filter + export row ───────────────────────────────────────

### lines 1795-1797

```python
self._filter_op.currentTextChanged.connect(
```

Wire the enablement-affecting signals last, once every widget _update_controls touches actually exists. Both signals carry an argument the slot doesn't want, hence the *_ lambdas.

## DbBrowserScreen.set_database

### line 1901  _(unsure)_

```python
self._table_list.setCurrentRow(0)
```

Selecting row 0 fires _on_table_selected, which loads the preview.

## DbBrowserScreen.select_table

### lines 1949-1951

```python
self._clear_sort()
```

A sort column from the previous table would land in this table's ORDER BY, where check_columns rejects it -- so the table would fail to load rather than merely come back unsorted.

### lines 1960-1961  _(unsure)_

```python
for i in range(self._table_list.count()):
```

Keep the current selection in the table list in sync when this was called programmatically.

## DbBrowserScreen.set_column_filter

### line 1992, trailing  _(unsure)_

```python
return
```

textChanged re-enters with the same value

## DbBrowserScreen.apply_seed

### lines 2085-2087

```python
if not self._scroll_column_into_view(column):
```

Scroll the column into view rather than sorting by it: arriving on a re-sorted table would hide which rows were just annotated, which is the thing the user came here to look at.

### lines 2089-2093

```python
if not self.visible_columns():
```

No columns yet means the first chunk is still in flight, which is the normal case for a threaded browser: the seed is handled the moment the table is selected and the rows arrive later. Remember it and scroll when they do, or the scroll silently never happens.

## DbBrowserScreen.refresh

### lines 2155-2156

```python
try:
```

`refresh()` is how a header click re-reads the table, so it must not wipe the indicator that click just set.

## DbBrowserScreen._fetch_chunk

### lines 2190-2191

```python
want_estimate = first and not where
```

max(rowid) is only an estimate of the *table* size; with a filter in play it says nothing, so we don't pretend it does.

## DbBrowserScreen._apply_chunk

### line 2210, trailing  _(unsure)_

```python
return
```

cancelled: a stale load's rows

### lines 2225-2226  _(unsure)_

```python
self._exhausted = True
```

A short chunk is the end of the table — and it makes the count exact for free, no COUNT(*) needed.

## DbBrowserScreen._apply_count

### line 2274, trailing  _(unsure)_

```python
return
```

cancelled

## DbBrowserScreen._update_sort_state

### lines 2316-2317

```python
self._view.setSortingEnabled(False)
```

Never Qt's own: it would reorder the loaded slice underneath the SQL order and the two would disagree.

## DbBrowserScreen._on_header_clicked

### lines 2363-2364

```python
self.refresh()
```

Reload from the top: rows already loaded are the wrong ones now, not merely in the wrong order.

## DbBrowserScreen._report_table_status

### lines 2374-2375

```python
self._set_status(" · ".join(bits) + self._linked_filter_note)
```

A table quietly showing two thirds of its rows is how a count gets reported as the whole population.

## DbBrowserScreen._apply_linked_filter

### lines 2440-2443

```python
for row in range(total):
```

Skipped entirely while nothing is filtered, and this runs once per chunk: a walk over every loaded row on each of the 4 000 chunks of a 400 k-row table is the difference between scrolling and not scrolling.

## DbBrowserScreen._on_view_selection_changed

### lines 2540-2541  _(unsure)_

```python
return
```

No object identity in this table; selecting a row in it is a local act, not something the other views can follow.

## DbBrowserScreen.on_linked_selection_changed

### lines 2556-2558

```python
self._syncing_selection = True
```

Guarded, not merely echo-suppressed: this screen would otherwise re-publish what it was just told, replacing a selection of ninety thousand objects with the hundred of them this page has loaded.

## DbBrowserScreen._collect_filter

### line 2613

```python
return None, (), ""
```

Nothing typed: treat as "no filter" rather than an error.

## DbBrowserScreen._confirmation_text

### lines 2686-2688

```python
key_columns = self._db.row_key(self._table)[1] if self._table else []
```

No table (a database with none) or no key (a view) still gets a statement to look at — the rowid shape, which is what an editable table would use.

## DbBrowserScreen.edit_cell

### line 2808  _(unsure)_

```python
self._set_status(
```

Belt and braces: edit mode is armed for one file only.

## DbBrowserScreen._update_edit_ui

### lines 2876-2878

```python
writable = self._edit_mode and self._table_is_editable()
```

A table with no rowid and no primary key stays read-only even in edit mode: offering a cell editor that always refuses would be a lie told twice.

## DbBrowserScreen.export_csv

### line 2941  _(unsure)_

```python
columns = self._model.visible_columns() or self._all_columns
```

Honour the column search: what you filtered down to is what you get.

## DbBrowserScreen._start_job

### lines 3088-3101

```python
self._next_job_id += 1
```

make_thread deliberately does not connect worker.deleteLater: Python owns this worker, and _retire_job releases its last strong reference on the GUI thread after the event loop exits. Do not try to "defensively" disconnect a slot that is absent — PySide emits a RuntimeWarning for every job, and signal mutation during native teardown is precisely the lifecycle race this ownership scheme avoids. See make_thread's ownership contract. Strong references: PySide6 will not keep the worker alive through the started→run connection alone, and a collected worker means the thread spins forever without ever calling run(). A QThread that loses its last Python reference while running takes the process down with it, so the pair is held until `thread.finished` says the event loop has exited. Same fix as AppScreen._on_run — but held per-job, keyed by job id.

### lines 3110-3120

```python
thread.finished.connect(self._retire_finished_jobs)
```

A BOUND METHOD, not a closure — and the contrast with the line above is the whole point. ``worker`` is moveToThread'd, so a closure on ITS signal runs on the worker thread and re-emitting a Signal is the only safe thing to do from one. ``thread`` is the opposite case: the QThread object is GUI-affine, so PySide6 makes it the receiver for a closure, and ``make_thread`` connects ``thread.finished -> thread.deleteLater`` FIRST. Slots run in connection order, so the DeferredDelete is posted ahead of the closure's metacall and Qt discards queued events for a destroyed receiver: the job was never retired and ``active_jobs()`` never returned to zero.

## DbBrowserScreen._update_controls

### lines 3225-3227

```python
self._table_list.setEnabled(has_db)
```

The table list stays live during a load on purpose: switching table mid-load has to be possible, and the token check makes it safe.

## DbBrowserScreen.closeEvent

### line 3247  _(unsure)_

```python
pass
```

The process-wide link's C++ side is gone (interpreter teardown).

## _build_lineage

### lines 3278-3280

```python
from .map_barcodes import build_registered_screen
```

IMPORTED HERE. This module used `build_registered_screen` without importing it, so every folded module it hosts raised NameError the moment its button was pressed.
