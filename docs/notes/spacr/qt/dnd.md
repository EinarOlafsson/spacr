# Notes from `spacr/qt/dnd.py`

Prose lifted out of `spacr/qt/dnd.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [install_dropzone](#install_dropzone) (2 entries)
- [_DropzoneFilter.__init__](#_dropzonefilter__init__) (2 entries)
- [_DropzoneFilter.eventFilter](#_dropzonefiltereventfilter) (1 entry)
- [_DropzoneFilter._on_drop](#_dropzonefilter_on_drop) (1 entry)
- [_classify_drop](#_classify_drop) (3 entries)
- [_deliver_drop](#_deliver_drop) (1 entry)
- [_drain](#_drain) (1 entry)
- [_queue_drop](#_queue_drop) (3 entries)
- [_answer_drop](#_answer_drop) (2 entries)
- [_route_drop.deliver](#_route_dropdeliver) (1 entry)
- [_route_drop](#_route_drop) (2 entries)
- [_find_console](#_find_console) (1 entry)
- [_report_drop_problem](#_report_drop_problem) (1 entry)
- [_read_settings_csv](#_read_settings_csv) (1 entry)
- [_route_data_csv_to_inputs](#_route_data_csv_to_inputs) (3 entries)
- [_apply_settings_csv](#_apply_settings_csv) (2 entries)
- [has_images_in](#has_images_in) (1 entry)
- [find_image_folders_nearby](#find_image_folders_nearby) (2 entries)

## Module level

### lines 65-66  _(unsure)_

```python
IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".czi",
```

File extensions that count as images for "does this folder have images?" checks. Keep in sync with spacr.io's readers.

## install_dropzone

### lines 129-131

```python
target._dnd_handler = handler
```

Store the handler + owning-screen on the widget itself so the event filter can look them up without capturing them in a closure that would keep the target alive after destruction.

### line 134  _(unsure)_

```python
f = _DropzoneFilter(target)
```

Filter is parented to target — Qt cleans it up when target dies.

## _DropzoneFilter.__init__

### lines 178-180

```python
"""Install on ``target`` and route its drops to that widget's handler."""
```

QObject parenting can synchronously deliver a ChildAdded event to the target.  Set this first so eventFilter is fully initialized even during super().__init__ (standalone tool screens exposed this race).

### line 183, trailing  _(unsure)_

```python
super().__init__(target)
```

parent → auto-cleanup

## _DropzoneFilter.eventFilter

### lines 186-201

```python
"""Accept drags and drops on the target, and decline once it is gone.
```

`getattr`, not `self._target`, and the reason is not defensiveness for its own sake. Qt goes on delivering events to a filter after the target's C++ half is gone, and PySide6 clears the Python wrapper's dict__ when that happens -- so `self._target` raises AttributeError from INSIDE the Qt event loop, which prints

Error calling Python override of QObject::eventFilter()

AttributeError: '_DropzoneFilter' object has no attribute '_target'

once per delivered event, and cannot be caught by any caller because there is no Python caller. A filter whose target is gone has nothing to filter, so declining the event is both correct and quiet.

The same shape as `RunHandle.is_running` swallowing "Internal C object already deleted": the destroyed wrapper IS the answer, not an error condition.

## _DropzoneFilter._on_drop

### lines 266-269

```python
event.acceptProposedAction()
```

Tell the drag source the drop landed as soon as we know we have something to do with it. Doing this only at the very end meant a settings-CSV-only drop (which IS handled below) was reported back to the OS as rejected.

## _classify_drop

### lines 276-278  _(unsure)_

```python
def _classify_drop(paths: Sequence[Path], handler: DropHandler,
```

Drop routing: the scan half runs on a worker, the widget half on the GUI

### lines 319-322

```python
if not multiple and others > 1:
```

Modules that do not handle multi-drop degrade to first-only, and the ones beyond the first are dropped HERE rather than scanned and then discarded -- a second sleeping mount is a second freeze.

### lines 331-334

```python
LOG.debug("drop classification failed for %s", path,
```

A policy that raises used to raise inside Qt's event delivery, where there is no Python caller to catch it. Off the GUI thread it would be swallowed by the runner instead, and the user would be told nothing at all -- so it is carried back as a rejection.

## _deliver_drop

### lines 377-378

```python
for entry in report:
```

CSVs first, as they always were: a settings CSV and a folder in one drop means the folder wins, because it is applied last.

## _drain

### lines 586-594

```python
while queue and queue[0].answered:
```

POP BEFORE DELIVERING, AND RE-READ AFTER. A slot left on the queue while its own delivery is running would be delivered a second time by a scan that lands from inside it, and one that raised would block every later drop on this screen for the life of the window. Taking the whole ready run off in one batch would fix that too, but it would then miss the slots answered DURING the run -- and a delivery that opens a modal dialog is exactly when a slow scan lands. `_run_delivery` swallows what a delivery raises, so one bad drop cannot break the loop.

## _queue_drop

### line 616, trailing  _(unsure)_

```python
except TypeError:
```

not weak-referenceable / not hashable

### lines 618-619

```python
_forget_abandoned(screen, queue)
```

A new drop is the moment to notice that an older one can never be delivered, and to stop holding this one behind it.

### lines 623-625

```python
_drain(queue)
```

Anything that was waiting behind a slot just written off is ready now. The new slot is at the BACK and unanswered, so it holds nothing up and nothing here delivers it.

## _answer_drop

### line 632, trailing  _(unsure)_

```python
if slot is None:
```

untracked screen: nothing to order

### lines 638-642

```python
_run_delivery(slot)
```

This slot lost its place in line: the screen's queue is gone, or a later drop wrote this one off as abandoned (:func:`_forget_abandoned`) and the scan answered after all. Deliver it where it stands -- the ordering guarantee went with its place, and a drop delivered out of order still beats the silent no-op of one never delivered.

## _route_drop.deliver

### lines 689-692

```python
untracked = _PendingDrop(
```

Untracked screen: no queue to order it against, and no

`_answer_drop` to keep a delivery that raises to itself. It is kept here instead -- see :func:`_run_delivery` for why an exception must not leave this callback.

## _route_drop

### lines 710-712

```python
LOG.debug("the drop scanner refused; classifying inline",
```

The scanner refused outright. Answer the slot anyway: an unanswered slot is a screen whose every later drop is silently queued behind it forever.

### lines 718-724

```python
LOG.warning("a dropped path was never classified: %s",
```

``_scan_then`` returns False both for a scan it ran INLINE and for one that RAISED there -- and in the second case it never calls back at all. Left alone, this drop's slot would stay unanswered at the head of the screen's queue and hold every later drop behind it for the life of the window. :func:`_classify_drop` is written never to raise, so this is the guard for the day something beneath it does; the drop is reported rather than silently forgotten.

## _find_console

### lines 756-758

```python
screens = getattr(window, "_screens", {}) or {}
```

Standalone tool screens are hosted alongside AppScreens. Prefer the most recently visited screen so rejected drops never disappear merely because the tool itself has no embedded console.

## _report_drop_problem

### lines 816-817

```python
if not displayed_inline:
```

Standalone tools use a read-only summary/log pane instead of an AppScreen ConsolePanel. Put the same actionable text there as well.

## _read_settings_csv

### lines 924-929

```python
try:
```

spaCR's own save_settings writes Key/Value columns; other tools (and older spaCR CSVs) use setting_key/setting_value. load_settings RAISES on a column mismatch rather than returning something non-dict, so the second form has to be tried in its own except — otherwise the fallback was unreachable and every setting_key/setting_value CSV was reported as a failed import.

## _route_data_csv_to_inputs

### lines 973-979

```python
identifiers = {"name", "gene id", "gene_id", "geneid", "gene", "grna",
```

An ANNOTATION table is neither side of the pairing.

Classifying only count-vs-score meant everything that was not a count became a score, so a gRNA barcode export (name, sequence) landed in the score column of the pairing table. It has no plate, no well and no response; it annotates results after the fit. Recognised by carrying an identifier and no per-well coordinates.

### lines 993-1000

```python
paired = widgets.get("paired_data")
```

THE PAIRED TABLE IS TRIED FIRST, and that ordering is the whole fix.

The regression panel replaced its separate score_data / count_data lists with one paired_data table. This router looked for those two keys as FilePathListWidgets, found neither, and fell through to metadata_files -- the only FilePathListWidget left on the screen. So every CSV dropped on the regression panel went to metadata: score tables, count tables, all of it.

### lines 1007-1008

```python
preferred = ("count_data", "score_data") if is_count else \
```

Most specific first: a count table must not land in the score slot just because that widget happens to come first in the panel.

## _apply_settings_csv

### lines 1039-1041

```python
if not _header_is_settings(header):
```

A dropped file is DATA unless its header says it is settings. Deciding by header rather than by extension is what lets the regression screen accept four score CSVs and four count CSVs by drag and drop.

### lines 1069-1071

```python
failure = str(e)
```

The read succeeded and the screen refused what it read. Same report as a failed read, because to the user it is the same sentence: this CSV did not become settings.

## has_images_in

### lines 1176-1187

```python
def has_images_in(path: Path, min_count: int = 1,
```

Filesystem helpers reused by handlers

WORKER-THREAD ONLY, ALL THREE. They list directories, and the directory is always one the user dropped -- which on the maintainer's machine reaches ``/nas_mnt`` shares behind an ``autofs`` mount that took more than twenty seconds to answer a single stat on 2026-09-04. Called from a handler's ``can_accept`` or ``suggest_alternatives``, they are already on the drop scanner's thread (see :func:`_route_drop`); called from anywhere that draws, they are the freeze. Same contract as the scans in :mod:`spacr.qt.dnd_handlers`: no Qt, no widgets, data out.

## find_image_folders_nearby

### line 1214  _(unsure)_

```python
if path.parent and path.parent.is_dir():
```

One level up: check siblings

### line 1219  _(unsure)_

```python
if path.is_dir():
```

One level down: check immediate children
