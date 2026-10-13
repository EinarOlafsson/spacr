# Notes from `spacr/qt/dnd_handlers.py`

Prose lifted out of `spacr/qt/dnd_handlers.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (14 entries)
- [_add_to_source_set](#_add_to_source_set) (2 entries)
- [_is_alive](#_is_alive) (1 entry)
- [_DropScanner.__init__](#_dropscanner__init__) (2 entries)
- [_DropScanner.eventFilter](#_dropscannereventfilter) (2 entries)
- [_DropScanner.is_busy](#_dropscanneris_busy) (1 entry)
- [_scan_then](#_scan_then) (1 entry)
- [_remember](#_remember) (1 entry)
- [_decide](#_decide) (1 entry)
- [_decide.run](#_deciderun) (2 entries)
- [scan_mask_folder](#scan_mask_folder) (2 entries)
- [scan_mask_drop](#scan_mask_drop) (1 entry)
- [scan_folder_structure](#scan_folder_structure) (2 entries)
- [MaskDropHandler](#maskdrophandler) (1 entry)
- [MaskDropHandler.apply](#maskdrophandlerapply) (1 entry)
- [MaskDropHandler._apply_facts](#maskdrophandler_apply_facts) (2 entries)
- [_report_regex_on_mask](#_report_regex_on_mask) (1 entry)
- [_render_container_report](#_render_container_report) (2 entries)
- [_render_mask_report](#_render_mask_report) (4 entries)
- [_render_folder_structure](#_render_folder_structure) (1 entry)
- [_open_metadata_table](#_open_metadata_table) (4 entries)
- [_open_regex_editor](#_open_regex_editor) (2 entries)
- [MeasureDropHandler](#measuredrophandler) (1 entry)
- [MeasureDropHandler.can_accept](#measuredrophandlercan_accept) (2 entries)
- [MeasureDropHandler.suggest_alternatives](#measuredrophandlersuggest_alternatives) (1 entry)
- [MeasureDropHandler.apply](#measuredrophandlerapply) (1 entry)
- [AnnotateDropHandler](#annotatedrophandler) (1 entry)
- [AnnotateDropHandler.apply](#annotatedrophandlerapply) (1 entry)
- [ClassifyDropHandler](#classifydrophandler) (1 entry)
- [MakeMasksDropHandler](#makemasksdrophandler) (1 entry)
- [MeasurementsDropHandler](#measurementsdrophandler) (1 entry)
- [MeasurementsDropHandler.apply](#measurementsdrophandlerapply) (2 entries)
- [RegressionDropHandler.apply](#regressiondrophandlerapply) (1 entry)
- [PlateQueueDropHandler.apply](#platequeuedrophandlerapply) (2 entries)
- [ModelZooDropHandler.apply](#modelzoodrophandlerapply) (2 entries)
- [_apply_model_zoo_source](#_apply_model_zoo_source) (1 entry)
- [_resolve_for](#_resolve_for) (1 entry)
- [ProjectRootsDropHandler.deliver](#projectrootsdrophandlerdeliver) (1 entry)
- [TableDropHandler.deliver](#tabledrophandlerdeliver) (1 entry)
- [LayerStackDropHandler.deliver](#layerstackdrophandlerdeliver) (1 entry)
- [get_handler](#get_handler) (1 entry)

## Module level

### lines 60-62

```python
from .folder_metadata import IMAGE_EXTS as RASTER_EXTS
```

Two extension sets, deliberately: IMAGE_EXTS (above) is what the filename preview will *sample*, containers included; RASTER_EXTS is what counts as one image on disk.

### lines 167-182

```python
_FOLDER_PROBE = 30
```

Scanning a dropped folder without freezing the window

A drop is delivered inside Qt's event dispatch, so everything a handler does happens on the GUI thread with the event loop stopped. Reading a directory is not "a bit of I/O": a user dropped a 100 000-file plate folder and the window froze for over a second -- three separate recursive walks of the same tree, one to detect a folder layout and two more inside the extraction planner, which called the detector again.

So the walking moves to a worker via :class:`spacr.qt.job_runner.JobRunner`, and only the walking. ``handler.apply`` still runs -- and returns synchronously; what changes is that the answer arrives a moment later, through a completion handler that runs back on the GUI thread and is the only place allowed to touch a widget.

### lines 421-446

```python
DECISION_BUDGET_S = 0.25
```

Deciding whether a drop is acceptable, without waiting for a sleeping mount

``_scan_then`` above is the right shape whenever there is a callback to report into. ``DropHandler.can_accept`` has none: the boolean has to come back before the drop can be routed anywhere, so there is no "later" to report it in.

WHERE THE ACCEPT TESTS ACTUALLY RUN. ``spacr.qt.dnd._route_drop`` hands the whole classification -- ``can_accept``, ``error_message``, ``suggest_alternatives`` -- to a worker through ``_scan_then``, so the ORDINARY path is already off the GUI thread and blocking there is not a freeze, it is what the worker is for. Two paths are not:

``_route_drop``'s own fallback. When the screen cannot hold a scanner it classifies INLINE, on the GUI thread, "better a stall than a drop that reports nothing". ``handler.apply``, which stays on the GUI thread by contract because it touches widgets -- and which asks these same questions again.

So the budget below is real, and it is applied WHERE IT IS NEEDED AND NOWHERE ELSE. Spending it on a worker would be worse than useless: it would answer ``default`` for a share that was going to reply in half a second, turning a correct rejection -- with its message and its "did you mean" list into a silent accept. See :func:`_on_gui_thread`.

### lines 2359-2362  _(unsure)_

```python
_MODEL_SUFFIXES = (
```

Tool and results screens — these are not SettingsWidgets/AppScreens, so each handler calls the screen's small public configuration API directly.

### lines 3787-3790

```python
"classify_merged": ClassifyDropHandler,
```

THE MERGED SCREEN DROPS LIKE THE ONE IT REPLACED. Without a row it falls back to the generic source handler, which replaces `src` with ONE path -- and the merged screen's `src` is a list, so four plates dropped together silently became one.

### lines 3796-3797  _(unsure)_

```python
"regression":      RegressionDropHandler,
```

Not MeasurementsDropHandler: the screen also hosts the sweep card, whose CSV inputs a measurements-only handler turns away. See the class.

### line 3802, trailing  _(unsure)_

```python
"analyze_plaques": MakeMasksDropHandler,
```

plaque images

### line 3803, trailing  _(unsure)_

```python
"train_cellpose":  MakeMasksDropHandler,
```

image + mask pairs

### lines 3805-3809

```python
"cellpose_all":    MakeMasksDropHandler,
```

The "Mask the whole folder" key. It shares the applying half's screen rather than owning one, so nothing asks for its handler today -- but a key that falls through to `SourceDropHandler` answers a dropped folder of images with "here is a source path", which is the wrong reading of the same gesture on the same folder.

### lines 3825-3829

```python
"graph_builder":    TableDropHandler,
```

the layout-aware screens

One table out of a measurement database, or a CSV. All nine expose the same ``load_path(path, table=None)``, which is why one handler covers them: the difference between these screens is what they draw, not what they read.

### line 3842  _(unsure)_

```python
"pipeline_graph":   ProjectFolderDropHandler,
```

A whole project, from anywhere inside it.

### line 3851  _(unsure)_

```python
"profiler":         CoefficientsDropHandler,
```

One artifact out of the layout.

### lines 3861-3863

```python
"parameter_sweep":  SweepInputsDropHandler,
```

Kept although the sweep has no tile any more: the handler is still the one that fills the card, reached through RegressionDropHandler, and get_handler("parameter_sweep") stays a working way to ask for it.

### lines 3919-3921

NOTE: an ``accepts_multiple`` override used to sit here, after the return, indented as if it were a method of this *function*. It was unreachable in both readings, so nothing it claimed was ever true.

## _add_to_source_set

### lines 69-73

```python
def _add_to_source_set(screen, path):
```

Shared setter — every AppScreen exposes the src widget through _settings_model._widgets["src"]; AnnotateScreen / MakeMasksScreen have their own _open_source / _open_folder methods.

### lines 95-98

```python
try:
```

ALREADY IN THE SET IS A SUCCESSFUL DROP. `add_sources` returns how many were NEW, which is a different question: a user who drops plate2 twice has a screen pointing where they pointed it, and reporting failure would put "this module has no source field" in front of them.

## _is_alive

### line 202, trailing

```python
return True
```

a plain Python screen cannot be half-deleted

## _DropScanner.__init__

### lines 235-238

```python
"""Watch ``screen`` for drops, parenting to it when it is a QObject."""
```

Assigned BEFORE super().__init__: parenting can deliver a ChildAdded event synchronously, and this object is an event filter, so it must already be able to answer for itself. (The same race that put the assignment first in ``_DropzoneFilter.__init__``.)

### lines 243-250

```python
self._runner = JobRunner(self, app_key="folder scan",
```

`user_visible=False`: NOTHING THE USER STARTED IN THE HOME SENSE. A drop is a gesture, not a run, and `home._on_runs_changed` filters run banners on exactly this flag -- so without it every drop on every screen flashes a blue "folder scan - running" box across the top of Home, the same mistake the usage poller and the home journal walk each made once. It still turns the activity spinner, because a thread genuinely is running. Nothing else submits to this runner: `_scan_then` is its only caller and it carries drop scans alone.

## _DropScanner.eventFilter

### lines 258-263

```python
"""Shut the scanner down when the screen it serves closes.
```

Every event delivered to the screen comes through here, so the cheap discriminator goes first and the attribute lookup second. ``getattr`` rather than ``self._screen`` for the reason spelled out in ``_DropzoneFilter.eventFilter``: Qt keeps delivering events to a filter after PySide6 has emptied its wrapper's __dict__, and an AttributeError raised there has no Python caller to catch it.

### line 279, trailing

```python
return False
```

never consume the event

## _DropScanner.is_busy

### line 309  _(unsure)_

```python
def is_busy(self) -> bool:
```

state (used by tests and by anything that wants to wait)

## _scan_then

### line 394

```python
LOG.debug("falling back to an inline folder scan", exc_info=True)
```

Qt refused to start a thread. Better a stall than no report.

## _remember

### lines 543-544

```python
excess = len(_decisions) - _DECISION_CAP
```

Still over the cap means every entry is live, which the expiry sweep alone cannot fix. Drop the ones due to expire first.

## _decide

### lines 643-644

```python
LOG.debug("drop decision %r failed", key, exc_info=True)
```

A path that cannot be read is a rejection, not an answer worth keeping: the share may be back in a moment.

## _decide.run

### lines 659-660

```python
_settled_partials[threading.get_ident()] = box
```

Registered BEFORE ``work`` starts, because what it hands back part of the way through is the whole point: see :func:`_settled_so_far`.

### lines 665-667

```python
LOG.debug("drop decision %r failed", key, exc_info=True)
```

An unreadable path is not an acceptable reason to raise out of a Qt event filter, so the optimistic answer stands and the drop is reported on by whatever runs next.

## scan_mask_folder

### line 688

```python
def scan_mask_folder(path, sample: int = 20) -> Dict[str, Any]:
```

the scans themselves. Worker-thread code: no Qt, no widgets, data out.

### lines 708-710

```python
total = sum(1 for p in entries if p.suffix.lower() in RASTER_EXTS)
```

The count deliberately uses the narrower raster set, as it always has: it is quoted as "N of M total sampled" beside a filename-regex preview, and one .nd2 container is not M images yet.

## scan_mask_drop

### lines 779-785

```python
_settled_so_far(dict(out))
```

THE DROP IS DECIDED BY THE LINE ABOVE, and the walk below is no part of deciding it -- it lists the parent, every sibling and every child to fill the "did you mean" list, which is the neighbourhood's cost and not the dropped folder's. Hand the decided half over before paying it, or a caller on the budget gets the optimistic guess and ACCEPTS a folder this function has already read and found empty. See :func:`_settled_so_far`.

## scan_folder_structure

### lines 870-871

```python
template = fm.detect_folder_metadata(path, files=probe)
```

Reached through the module, not a from-import, so that patching ``spacr.qt.folder_metadata.detect_folder_metadata`` still works.

### line 877, trailing

```python
return out
```

the rest of the tree is never walked

## MaskDropHandler

### lines 887-889  _(unsure)_

```python
class MaskDropHandler(DropHandler):
```

Mask — the star handler with regex-preview canvas

## MaskDropHandler.apply

### lines 970-976

```python
_set_src_on(screen, str(path))
```

NOTHING IS KNOWN YET -- the budget in :func:`_decide` ran out and this record is the optimistic guess, not an answer. Fill ``src`` from it, because a person who just let go of a folder is owed the path appearing in the field, and send the question that actually matters -- folder or container? -- to a worker. Deciding it here would read a ``.nd2`` as a folder and print "no images found in the top level of plate.nd2"; see :func:`_mask_drop_unknown`.

## MaskDropHandler._apply_facts

### lines 1005-1013

```python
_scan_then(
```

Read the folder on a worker thread and render the report when it comes back.

What this replaced: ``QTimer.singleShot(50, ...)``, commented "asynchronously so the UI doesn't stall". It is not asynchronous. A single-shot timer defers to the next turn of the event loop and then runs everything ON the GUI thread, with the loop stopped — the freeze just started 50 ms later than the drop, which is why it was never traced back to here.

### lines 1022-1024

```python
_scan_then(
```

A single container file. Describing it is a FILE OPEN, which is the call that froze on the sleeping share, so it goes to a worker too and only the widget half comes back. See :func:`scan_mask_container`.

## _report_regex_on_mask

### line 1064  _(unsure)_

```python
_render_container_report(path, screen, scan_mask_container(path))
```

── Single-file dataset path ─────────────────────

## _render_container_report

### lines 1085-1088

```python
_set_screen_setting(screen, "metadata_type", "auto")
```

Container formats (nd2/czi/lif/multi-page tiff/npz) are expanded to the canonical Yokogawa layout by the pipeline's auto converter. Set metadata_type='auto' so that conversion actually runs, and point src at the containing folder.

### lines 1096-1099

```python
failure = scan.get("error") or ""
```

Preview the planned extraction and let the user edit the plate/well/field/channel assignment before committing. A planner that failed on the worker is reported in the same words it was reported in when the planning happened here.

## _render_mask_report

### line 1133  _(unsure)_

```python
custom = ""
```

Read the user's current custom_regex (may be empty)

### line 1142

```python
if custom:
```

Auto-detect if the user has no custom regex or if it fails

### lines 1168-1169  _(unsure)_

```python
_report_folder_structure(path, screen)
```

Offer folder-structure metadata as an alternative to a filename regex (useful when the plate/well/field/channel live in directory names).

### lines 1179-1188

```python
_log(screen, "→ Confirm the parsed columns above match your naming "
```

Confirm even when nothing looks wrong. A regex that captures every required field can still be capturing the WRONG field -- a well ID read as a field ID validates perfectly and silently mislabels the whole plate. The check that catches that is a person reading the parsed columns, which only happens if they are shown.

Previously this branch pushed the pattern with no prompt, so the editor appeared only when validation failed. That made the common case (a naming dialect that fits) the one case nobody ever verified, and made the prompt read as an error rather than a step.

## _render_folder_structure

### lines 1252-1254

```python
rows = result.get("rows") or []
```

Make the detection actionable: the preview of how each image would be named opens in the editable metadata table so the user can accept or correct it, writing a filename_map.csv the pipeline consumes.

## _open_metadata_table

### lines 1281-1284

```python
dst = Path(dst)
```

``dst`` arrives as the FOLDER the data lives in, but the dialog hands it straight to folder_metadata.save_filename_map(), which treats its argument as the CSV file to open() for writing. Passing a directory made every Apply raise IsADirectoryError and silently write nothing.

### lines 1299-1301

```python
try:
```

Show modeless (never exec()) so the drop handler never blocks — a blocking modal would hang headless/offscreen runs. Keep a reference on the screen so the dialog isn't garbage-collected while open.

### lines 1309-1312

```python
holder = _ORPHAN_DIALOGS
```

The screen refuses new attributes (__slots__, proxy, …). Park the reference module-side: the dialog is parentless here, so without SOME live reference it is collected the moment this function returns and the user never sees it.

### line 1320  _(unsure)_

```python
pass
```

Non-interactive / headless — leave the console report in place.

## _open_regex_editor

### lines 1337-1338

```python
if confirming and fallback:
```

No editor available. A validated pattern is still better than none, so a confirmation that cannot be shown must not lose it.

### lines 1346-1349

```python
if dlg.exec() == QDialog.Accepted and dlg.regex:
```

QDialog.Accepted, not dlg.Accepted: PySide6 exposes the enum on the class, not on instances. This only ever ran when validation failed, so the AttributeError sat here until the editor started opening on every import to confirm a good match.

## MeasureDropHandler

### lines 1381-1384

```python
class MeasureDropHandler(DropHandler):
```

NOTE: ``_count_images`` used to live here, and the report called it right after ``sample_image_names`` -- two listings of the same directory, both on the GUI thread. :func:`scan_mask_folder` produces the sample and the count from one listing, on a worker.

## MeasureDropHandler.can_accept

### line 1406  _(unsure)_

```python
if path.name == "merged" and has_images_in(path, exts=(".tif", ".tiff", ".npy")):
```

Direct: dropped `merged` folder itself

### line 1409  _(unsure)_

```python
merged = path / "merged"
```

Contains: dropped a plate parent that HAS merged/

## MeasureDropHandler.suggest_alternatives

### line 1423  _(unsure)_

```python
if path.is_dir():
```

Look for merged/ under nearby folders

## MeasureDropHandler.apply

### lines 1445-1456

```python
"""
```

The plate folder, not ``merged/`` inside it.

This used to drill *into* ``merged``, and auto-chaining fills the same field with the plate — so dropping a folder and letting the chain fill it produced two different strings for one project. Both run (``spacr.ports.project_root`` hops a trailing ``merged``), which is exactly why the disagreement survived: it only showed up when a settings CSV written by one was compared against the other.

:func:`spacr.chaining.resolve_drop` is the single answer now, and it asks the registry first, so a plate whose merged arrays were written somewhere unusual resolves to where the producer says they are.

## AnnotateDropHandler

### lines 1484-1486  _(unsure)_

```python
class AnnotateDropHandler(DropHandler):
```

Annotate — expects a measurements DB

## AnnotateDropHandler.apply

### lines 1517-1521

```python
"""
```

Drop-db: use its containing plate folder as src. The canonical layout is <plate>/measurements/measurements.db, so climb two levels ONLY when the db really sits in a measurements/ folder — a loose .db (which can_accept also allows) must resolve to its own directory, not that directory's parent.

## ClassifyDropHandler

### lines 1542-1544  _(unsure)_

```python
class ClassifyDropHandler(DropHandler):
```

Classify — same DB requirement as annotate, plus optional model dir

## MakeMasksDropHandler

### lines 1607-1609  _(unsure)_

```python
class MakeMasksDropHandler(DropHandler):
```

Make Masks — image folder, optional companion masks/

## MeasurementsDropHandler

### lines 1748-1750

```python
class MeasurementsDropHandler(DropHandler):
```

Generic "measurements DB" downstream handler — UMAP / ML / regression

## MeasurementsDropHandler.apply

### lines 1816-1817

```python
"""
```

A screen whose inputs are one row per plate wants the database on a PLATE ROW; `src` is not where its measurements live.

### lines 1836-1839

```python
app_key = str(getattr(screen, "app_key", "") or "")
```

Same resolution as auto-chaining, for the same reason as in :meth:`MeasureDropHandler.apply`: the registry knows where the producer actually wrote, and the declared layout answers when no run was ever registered.

## RegressionDropHandler.apply

### lines 2155-2156  _(unsure)_

```python
SweepInputsDropHandler().apply(path, screen)
```

Handed the HOST, not the card: the sweep handler resolves the panel itself, and the console the drop reports to is the host's.

## PlateQueueDropHandler.apply

### lines 2703-2706

```python
try:
```

The two-column spelling is the one spaCR writes; the single-argument call is the documented default and is what a hand-made snapshot is likely to use. Only the SECOND failure means the file is unreadable.

### lines 2728-2731

```python
_log(screen,
```

A partial drop used to report plain success: one unreadable snapshot among several meant that plate quietly never reached the queue, and the user found out when the run they expected was not in the list.

## ModelZooDropHandler.apply

### lines 2866-2870

```python
if ((path.is_file() and path.name.lower().endswith(_MODEL_SUFFIXES))
```

The cheap answers first, inline: a dropped checkpoint, or a checkpoint sitting at the top level of the dropped folder. That is one directory listing which stops at the first hit — and it is what a real model folder looks like, so the common drop stays fully synchronous.

### lines 2875-2880

```python
_scan_then(
```

Nothing up top. Answering "is there one further down?" means walking the entire tree, and it is the NO that costs — ``any()`` short-circuits on a hit but a negative answer visits every file. That is exactly the "dropped a plate folder on the wrong screen" case: 100 000 files, a second of dead window. Off the GUI thread it goes, and the branch it decides goes with it.

## _apply_model_zoo_source

### lines 2895-2898

```python
reason = (getattr(screen, "last_error", "")
```

``apply`` returned long ago, so raising here would surface as an unhandled exception in the Qt event loop instead of being caught by ``dnd._on_drop`` — and the user would be told nothing at all. Report it exactly as that handler would have.

## _resolve_for

### lines 2992-3011

```python
def _resolve_for(handler, app_key: str, path: Path):
```

Layout-aware drops

"It should be possible to drag-n-drop folders and files into every module, and every module should be aware of the spaCR folder structure": drop the project on a screen that reads a database and it finds ``measurements/measurements.db``; drop it on one that reads a table and it offers the tables in that database; drop the database itself and that still works.

None of the layout knowledge lives here. :func:`spacr.chaining.resolve_drop` answers "where is the X in this project?" by asking the artifact registry first -- the same question, through the same call, that auto-chaining asks and falling back to the declared paths in :data:`spacr.ports.PORTS`. Two answers to "where is the database" is how a screen and the run it launches come to disagree, so there is only one.

What a handler adds is the last step: which of this screen's fields the answer goes into.

## ProjectRootsDropHandler.deliver

### lines 3327-3329

```python
"""
```

``add_root`` returns False for a root that is already listed, which is not a failure and must not be reported as one: dropping a folder the browser already watches should be a no-op, not an error dialog.

## TableDropHandler.deliver

### line 3408, trailing  _(unsure)_

```python
if table is False:
```

the chooser was cancelled

## LayerStackDropHandler.deliver

### lines 3601-3602  _(unsure)_

```python
as_labels = (target is not None and target.kind == _kinds.MASKS) or (
```

A file that came out of ``masks/`` is a label array; anything else the user dropped is the image they want to look at.

## get_handler

### lines 3893-3894  _(unsure)_

```python
return cls(app_key)
```

The layout-aware handlers resolve against the module they are installed on, so they need to be told which one that is.
