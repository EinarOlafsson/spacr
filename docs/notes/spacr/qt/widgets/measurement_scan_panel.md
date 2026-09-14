# Notes from `spacr/qt/widgets/measurement_scan_panel.py`

Prose lifted out of `spacr/qt/widgets/measurement_scan_panel.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [joinable_tables](#joinable_tables) (1 entry)
- [merge_across_databases](#merge_across_databases) (9 entries)
- [displayed_plates](#displayed_plates) (1 entry)
- [merge_summary](#merge_summary) (1 entry)
- [step_header](#step_header) (1 entry)
- [WorkflowStep.__init__](#workflowstep__init__) (4 entries)
- [regressable_columns](#regressable_columns) (1 entry)
- [column_run_settings](#column_run_settings) (1 entry)
- [_fit_outcome](#_fit_outcome) (1 entry)
- [DatabaseMergePanel.__init__](#databasemergepanel__init__) (8 entries)
- [DatabaseMergePanel._read_budget](#databasemergepanel_read_budget) (1 entry)
- [DatabaseMergePanel._new_generation](#databasemergepanel_new_generation) (1 entry)
- [DatabaseMergePanel._read_off_thread](#databasemergepanel_read_off_thread) (2 entries)
- [DatabaseMergePanel._run_read](#databasemergepanel_run_read) (1 entry)
- [DatabaseMergePanel._on_read_landed](#databasemergepanel_on_read_landed) (2 entries)
- [DatabaseMergePanel._settle_paths](#databasemergepanel_settle_paths) (1 entry)
- [DatabaseMergePanel._follow_path_probes.corrected](#databasemergepanel_follow_path_probescorrected) (1 entry)
- [DatabaseMergePanel._fill_table](#databasemergepanel_fill_table) (1 entry)
- [DatabaseMergePanel._source_info](#databasemergepanel_source_info) (4 entries)
- [DatabaseMergePanel._offer_tables](#databasemergepanel_offer_tables) (2 entries)
- [DatabaseMergePanel._refresh_steps](#databasemergepanel_refresh_steps) (1 entry)
- [DatabaseMergePanel.describe](#databasemergepaneldescribe) (1 entry)
- [DatabaseMergePanel._plan_lines](#databasemergepanel_plan_lines) (2 entries)
- [DatabaseMergePanel._table_notes](#databasemergepanel_table_notes) (1 entry)
- [DatabaseMergePanel._column_kinds](#databasemergepanel_column_kinds) (1 entry)
- [DatabaseMergePanel.start_merge](#databasemergepanelstart_merge) (1 entry)
- [DatabaseMergePanel.cancel_merge](#databasemergepanelcancel_merge) (1 entry)
- [DatabaseMergePanel._prepare_merge](#databasemergepanel_prepare_merge) (3 entries)
- [DatabaseMergePanel._merge_worker](#databasemergepanel_merge_worker) (1 entry)
- [DatabaseMergePanel._relay_progress](#databasemergepanel_relay_progress) (2 entries)
- [DatabaseMergePanel.show_aggregation_rules](#databasemergepanelshow_aggregation_rules) (3 entries)
- [well_keys](#well_keys) (1 entry)
- [describe_key_overlap._canonical](#describe_key_overlap_canonical) (1 entry)
- [ColumnRegressionPanel.__init__](#columnregressionpanel__init__) (1 entry)
- [ColumnRegressionPanel.start_regressions](#columnregressionpanelstart_regressions) (4 entries)
- [ColumnRegressionPanel._queue_worker](#columnregressionpanel_queue_worker) (1 entry)
- [ColumnRegressionPanel._relay_started](#columnregressionpanel_relay_started) (1 entry)
- [ColumnRegressionPanel._relay_result](#columnregressionpanel_relay_result) (1 entry)
- [ColumnRegressionPanel._on_queue_progress](#columnregressionpanel_on_queue_progress) (1 entry)
- [ColumnRegressionPanel._finish_queue](#columnregressionpanel_finish_queue) (1 entry)
- [MeasurementScanPanel.__init__](#measurementscanpanel__init__) (7 entries)
- [MeasurementScanPanel._add_folding_section](#measurementscanpanel_add_folding_section) (2 entries)
- [MeasurementScanPanel._share_the_height](#measurementscanpanel_share_the_height) (3 entries)
- [MeasurementScanPanel._keep_the_filler_last](#measurementscanpanel_keep_the_filler_last) (1 entry)
- [MeasurementScanPanel._apply_section_layout](#measurementscanpanel_apply_section_layout) (3 entries)
- [MeasurementScanPanel._show_section](#measurementscanpanel_show_section) (1 entry)
- [MeasurementScanPanel.add_section](#measurementscanpaneladd_section) (1 entry)
- [MeasurementScanPanel._on_databases_changed](#measurementscanpanel_on_databases_changed) (1 entry)
- [MeasurementScanPanel.why_nothing_to_scan](#measurementscanpanelwhy_nothing_to_scan) (1 entry)
- [MeasurementScanPanel.scan](#measurementscanpanelscan) (1 entry)
- [MeasurementScanPanel.set_result](#measurementscanpanelset_result) (2 entries)

## Module level

### lines 31-38

```python
from PySide6.QtCore import QEvent, Qt, Signal
```

QEvent AT MODULE SCOPE, NOT INSIDE THE CALLBACK. A function-local import in an event handler is not lazy loading: this module is a QWidget module and cannot load without QtCore, so the import bought nothing but a sys.modules lookup on every event -- and it put an EXCEPTION SITE on a path with no way to report one. The same shape in `ModuleHintBar.event` produced 419 errors in one sweep when a test stubbed PySide6.QtCore out of sys.modules and teardown then delivered a paint event.

### lines 103-105

```python
DEFAULT_ANCHOR = DEFAULT_PRIMARY
```

Instruction 130 B: the databases attached to the input table

### lines 1073-1088

```python
WORKFLOW_STEPS = (
```

STEP 4: PICK A COLUMN AND REGRESS ON IT  (instruction 154 F)

"the point of the measurements tab is to merge measurements so that regression can be run on any column in the databases", as four steps:

1. LOAD the measurement databases 2. MERGE THE TABLES within each database 3. MERGE THE DATABASES into one frame 4. PICK A COLUMN and regress on it

Steps 1-3 were built and step 4 was not, so the tab ended before its own purpose -- which is most of "i dont understand how this is all set up". Everything below is Qt-free on purpose: it is the half worth testing without a widget, and `spacr/umap_search.py` is the house precedent.

## joinable_tables

### lines 320-326

```python
offered = tuple(OBJECT_TABLES) + (PNG_TABLE,)
```

png_list IS OFFERED. `merge_tables.mergeable_tables` has always returned it, and `object_keys` exists specifically to translate its 'o5' spelling of the object key into the integer the object tables use -- so the backend was ready and the panel filtered it back out by intersecting with OBJECT_TABLES, which is the object-ROLE registry and deliberately does not list it. Asked for repeatedly; the answer was always one name missing from a list, not a missing feature.

## merge_across_databases

### lines 488-493

```python
_say("planning the merge")
```

EVERY TABLE IS PLANNED BEFORE ANY IS READ, so the denominator exists before the first row does. `describe_merge` reads sqlite metadata and the distinct plate ids only -- the same call the panel already makes on every click -- so this costs a fraction of a second and buys a progress count that means something. It also moves a missing table's failure to BEFORE the expensive anchor read rather than after it.

### lines 517-520

```python
carried = [name for name in [*keys, SCREEN_COLUMN, SOURCE_COLUMN]
```

WHAT EVERY JOIN IS KEYED ON: the well identity, the screen and the file. `source_database` is in here as well as `screenID` because two databases of one screen are still two files, and a cell in one of them is not the same cell as the identically numbered cell in the other.

### lines 525-528

```python
reserved = set(carried) | {OBJECT_COLUMN}
```

The anchor's own measurements carry its name, exactly as `merge_tables` prefixes its primary -- so `area` from cell and `area` from nucleus can be told apart in the axis picker. A column that ALREADY starts with the table's name is left alone: `cell_area` must not become `cell_cell_area`.

### lines 562-564

```python
skipped[table] = (
```

Measured without a parent mask: the roll-up is not empty, it is UNDEFINED. Named and skipped, as merge_tables does -- one unlinkable table must not cost the user the others.

### lines 572-573

```python
rolled = child.rename(columns={
```

One row per cell already: nothing to aggregate, and putting it through the roll-up rules would answer a question nobody asked.

### lines 585-588

```python
identifiers[table] = tuple(
```

WHAT THE TEXT COLUMNS ACTUALLY GET, recorded rather than inferred. `aggregation_plan` asks the DTYPE first, so a string takes `first` whatever its name -- which is the true answer the plan used to get wrong by matching on names alone.

### lines 601-604

```python
refused[table] = ambiguous
```

REFUSED, NOT PICKED (instruction 79 item 2, and 154 C). The column is left out and named; the other eighty-odd are not lost with it, exactly as an unlinkable table does not cost the user the tables that do link.

### lines 615-617

```python
on = [name for name in carried + [OBJECT_COLUMN]
```

The object key is in both by construction: the anchor was checked for it above, and a child that does not carry it was skipped as unlinkable a few lines up.

### lines 621-623

```python
how = policy.how_for(table)
```

PER TABLE, FROM CARDINALITY -- never one blanket `how`. A cell with no nucleus is not a cell; a cell with no pathogen is an uninfected cell and usually the control population.

## displayed_plates

### lines 660-687

```python
def displayed_plates(plates: Sequence[str]) -> Tuple[str, ...]:
```

Instruction 154 D: a plate called plate1 is shown as plate1

MEASURED, BEFORE DECIDING IT WAS COSMETIC. The panel prints exactly what is stored: a database whose `plateID` column holds `plate1` shows `plate1`, and one that holds `pplate1` shows `pplate1`. Nothing in this file, in `describe_merge` or in `read_merged` adds a prefix to anything.

So the doubling is in the DATA, and that makes it more than cosmetic. Every join INSIDE this merge is safe -- both sides of it read the same stored value out of the same file -- but the merged frame then meets the regression side, where `spacr.utils.correct_metadata` has ALREADY rewritten `pplate1` to `plate1` in `plateID`, `prc` and `prcfo`. Score files stamped `pplate1` meeting count files stamped `plate1` is the recorded failure that produced a zero-row join and died two hundred lines later in a plot; a measurements database stamped `pplate1` meeting a normalised score CSV is the same mismatch from the other direction.

The house rule is to correct the format going forward and migrate the old content rather than preserve the bug, and that is where this ended up: `tabular.read_database` collapses the doubling ON READ, so the plan and the merged frame both name the plate `plate1` and the measurement side meets a score CSV `correct_metadata` has normalised. Naming the stored spelling beside it was what the panel could do while the doubling still reached the frame; it now describes a mismatch that no longer happens, so the panel says nothing and `plate_id_notes` is left as the tripwire for an id that reaches the plan UNREPAIRED.

## merge_summary

### lines 788-790

```python
lines.append(
```

NOT a mean, and never was. A text column takes `first` from its dtype, and saying "the default (mean)" about a file name told the user something about their data that cannot happen.

## step_header

### lines 865-868

```python
label = QLabel(f"{int(number)}. {tr(str(title)).upper()}", parent)
```

THE NUMBER IS NOT PART OF THE TITLE. Upper-casing the composed line asks the catalog for "1. LOAD THE MEASUREMENT DATABASES", which no row can hold, so only the odd word came back translated. Look the title up on its own and number it afterwards.

## WorkflowStep.__init__

### lines 935-936

```python
spoken = f"{self._number}. {tr(self._title)}"
```

SAID ALOUD, AND TRANSLATED. "Toggle" on its own tells a screen-reader user nothing about which of four steps they are on.

### lines 940-942

```python
self._fold.setMinimumSize(scaled_px(22), scaled_px(22))
```

AN ARROW WITH NO TEXT IS 24 px WHATEVER THE FONT, so at a 200 % font scale the one control on the row that has to be hit stays half the size of everything around it.

### lines 947-950

```python
self.label.setCursor(Qt.PointingHandCursor)
```

THE HEADING IS PART OF THE CONTROL. A 22 px arrow beside a heading that ignores clicks is the affordance every other folding heading in the tool does not have -- `Section` and `CollapsibleSection` both put the whole caption on the button.

### lines 957-958

```python
self._body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
```

EXPANDING, so a step that owns the panel's stretch really gets the height rather than sitting at its hint with a gap underneath.

## regressable_columns

### lines 1159-1164

```python
continue
```

A dtype that CLAIMS to be numeric while its values will not compare. No built-in one does -- complex, Int64, bool, sparse, float16 and timedelta were all checked -- but this scans whatever DataFrame the project produced, and a third-party ExtensionArray cannot be ruled out. One such column must not stop the scan finding the others.

## column_run_settings

### lines 1233-1236

```python
pair["score"] = str(score_path)
```

EVERY PLATE'S SCORE IS THE MERGED FRAME. It holds all the plates `source_database` and `plateID` are in it -- so pointing each pair at it and letting the loader align on plate is what keeps the count side paired exactly as the input table pairs it.

## _fit_outcome

### line 1347, trailing  _(unsure)_

```python
except TypeError:
```

a payload with no length

## DatabaseMergePanel.__init__

### lines 1594-1596

```python
self._stop = threading.Event()
```

A plain Event, not a Qt flag: it is read from the worker thread on every stage boundary, and `threading.Event` is the one primitive both sides can touch without a lock.

### lines 1654-1658

```python
self._tables_grip = HeightGrip(self.tables_list, 44, 480, self,
```

THE HANDLE GOES UNDER THE ROW, not inside it: `chooser` is a horizontal row, so a grip added to it would be a nine-pixel column beside the list rather than a border under it. The list is already placed, so this is the one box built with the handle directly instead of through `resizable_box`.

### lines 1707-1708

```python
step = self._add_step(3, layout, stretch=1)
```

STEP 3 IS ITS OWN STEP, so the button that does it is under the heading that names it rather than at the end of step 2's row.

### lines 1717-1719

```python
self.merge_button.clicked.connect(self.start_merge)
```

`start_merge`, NEVER `merge`. `merge` blocks until the whole join is done; on four databases that is minutes with a frozen window, which is the report instruction 154 was filed from.

### lines 1731-1734

```python
self.progress = QLabel("")
```

WHAT STAGE, AND HOW FAR. The plan already prints the row total; this counts against that same number rather than against one invented here, so "120,431 of 226,467" is a claim the user can check against the line above it.

### lines 1746-1751

```python
from .section import Section
```

THE COUNT IS THE SENTENCE; THE LIST IS THE EVIDENCE (154 B). A hundred and seventy column names in a 190-pixel box buried the three lines that matter. `Section` is the house's foldable, collapsed by default, and `add_prose` rather than `add_widget` because this is not a labelled setting row -- see Section.add_prose for what that distinction costs when it is got wrong.

### lines 1760-1763

```python
self.evidence.add_prose(self.details)
```

A GRIP INSIDE THE FOLD, because the fold is the sub-sub-subsection and the box inside it is the thing that was 220 px whatever the font. `add_prose` twice rather than `add_widget`: neither the box nor its handle is a labelled setting row.

### lines 1772-1776

```python
from ..screens.settings_model import retarget_field_tooltips
```

HOVER HELP GOES ON THE SETTING'S NAME, not on the box you type into. A tooltip on an editable field is unreachable the moment the user is editing it -- which is exactly when they wanted it -- and tests/test_tooltips_are_on_the_setting_not_the_field.py is the guard that says so.

## DatabaseMergePanel._read_budget

### lines 1784-1803

```python
@contextmanager
```

reading the databases

WHY THIS SECTION EXISTS. Everything step 1 shows comes out of the attached databases, and every one of those reads -- `mergeable_tables`, `describe_merge`, `column_kinds` -- opens sqlite on a path the user chose. `DatabaseMergePanel.__init__` ends in `refresh()`, so opening the Measurements tab did all of it inline on the GUI thread. On the maintainer's machine one of those paths was an `autofs` mount whose share was asleep and a single stat on it had not returned after twenty seconds; see `READ_BUDGET_S` and `spacr/qt/path_probe.py`.

The shape here is `spacr/qt/chaining.py`'s -- a GUI half that reads widgets and a worker half that touches none -- with one difference, and it is deliberate. `JobRunner` delivers through the event loop, and this panel's answers are read STRAIGHT BACK by its own callers: `refresh()` returns the count the host tab enables the rest of the workflow from, `_prepare_merge` reads `plan_summary()` on the line after it refreshes. So the GUI thread waits here -- but for a fixed fifth of a second and never for a filesystem, which is the difference between a panel that is a moment late and an application that is gone.

## DatabaseMergePanel._new_generation

### lines 1856-1859

```python
wanted = self._rules_wanted
```

Never the one the rules dialog is parked on. That read was asked for by a CLICK, it opens the dialog when it lands, and dropping it here would leave the button saying "reading" with nothing left to answer it.

## DatabaseMergePanel._read_off_thread

### lines 1889-1890

```python
threading.Thread(target=self._run_read,
```

Outside the lock: starting a thread is not something to hold a lock the reader threads need across.

### lines 1901-1902  _(unsure)_

```python
self._painted_pending = True
```

Whatever is drawn now is provisional, so say so: `_on_read_landed` draws it again for real.

## DatabaseMergePanel._run_read

### lines 1949-1950  _(unsure)_

```python
pass
```

The panel's C++ half went with its screen while this read was still parked on a mount that had not woken up.

## DatabaseMergePanel._on_read_landed

### line 1988  _(unsure)_

```python
self._wait_for_rules(None)
```

A click asked for this one and is still waiting for it.

### line 1995  _(unsure)_

```python
pass
```

The panel is gone; the queued signal outlived it.

## DatabaseMergePanel._settle_paths

### lines 2027-2028  _(unsure)_

```python
path_probe.exists(path)
```

Asking is what queues the check; the answer is what is waited for below.

## DatabaseMergePanel._follow_path_probes.corrected

### line 2059  _(unsure)_

```python
pass
```

The panel has gone; the signal outlived it.

## DatabaseMergePanel._fill_table

### lines 2201-2202

```python
item.setFlags(Qt.ItemIsSelectable)
```

Disabled, not removed: the user has to be able to see which plate is missing from this tab and why.

## DatabaseMergePanel._source_info

### lines 2222-2224

```python
offered = set(OBJECT_TABLES) | {PNG_TABLE}
```

png_list too -- see `joinable_tables`. This list is what the row shows as "tables", so leaving it out here made the panel report a database as not having a table it has.

### lines 2237-2240

```python
tables[path] = []
```

The row is still LISTED, with its plate, its file name and its screen: what this panel must never do is leave a plate out. Only the three columns that come from inside the file wait for it.

### lines 2269-2270

```python
detail = out[source.path]
```

Keyed on the path the plan was given, so the row and the summary cannot come apart.

### lines 2277-2279

```python
detail["status"] = (f"holds {', '.join(source.plates)}, not "
```

The row says plate3 and the file holds plate7. Not refused the plate label in the input table is the user's own name for the row -- but never silent either.

## DatabaseMergePanel._offer_tables

### lines 2301-2304

```python
return
```

LEAVE THE CHOOSER ALONE until the databases answer. Clearing it would drop the user's ticks and put them back a moment later, and an empty list is a statement -- "no object table is shared by every database" -- that nothing has been read to support.

### lines 2315-2316  _(unsure)_

```python
item.setCheckState(
```

Everything present is checked: the user asked for the measurements, and a table they did not want is one click.

## DatabaseMergePanel._refresh_steps

### lines 2427-2430

```python
if not self.heading.text():
```

STEP 1's line is `_fill_table`'s, which already counts the attached rows and names the missing ones. `step_states` answers the same question for a headless caller and a test; overwriting the richer sentence with the shorter one would be a step BACKWARDS on screen.

## DatabaseMergePanel.describe

### lines 2479-2481

```python
self._refresh_steps()
```

EVERY CHANGE REPAINTS THE STEPS. `describe` is what a click, a refresh and a new provider all end in, so hooking the state here is what keeps the four headings honest without a second signal path.

## DatabaseMergePanel._plan_lines

### lines 2529-2531

```python
gone = [entry.plate for entry in self._databases
```

BEFORE THE RUN, NOT FOUR MINUTES IN. A row that named a database which is not there is left out of the merge, and left out silently is how a result comes to describe fewer plates than the user thinks.

### lines 2546-2548

```python
return (lines + [f"Reading {anchor} from {len(paths)} "
```

The lines above are already true and already said -- the anchor, and which plates were left out for having no database on disk. Only what has to be read out of the files waits.

## DatabaseMergePanel._table_notes

### lines 2655-2658

```python
lines.append(
```

SAY WHAT A NUMBER CANNOT SAY. A column the database declared no type for cannot be promised either treatment, and an absent answer that reads as a definite one is the false assurance this panel is most careful about.

## DatabaseMergePanel._column_kinds

### lines 2685-2688

```python
continue
```

Left out rather than guessed. A column whose declared type has not been read yet is not a column this panel can promise anything about, which is the same rule the disagreement case below applies -- and the note is redrawn when it lands.

## DatabaseMergePanel.start_merge

### lines 2743-2751

```python
self._merging = False
```

PUT THE PANEL BACK. `_merging` and the running state were set before the submit, because the submit is the part that takes minutes. Left set after a refusal, every later press returns early and the merge can never be started.

The old pragma here claimed JobRunner always returns True. It does not: `submit` returns False whenever it is unthreaded and the work or the completion callback raises, which is exactly how these panels are built in tests.

## DatabaseMergePanel.cancel_merge

### lines 2764-2766

```python
self._jobs.cancel()
```

The worker is asked to stop at its next stage boundary AND its result is dropped on arrival by the runner's generation check, so neither a slow stage nor a fast one can leave a frame behind.

## DatabaseMergePanel._prepare_merge

### line 2782  _(unsure)_

```python
def _prepare_merge(self) -> Optional[Dict[str, Any]]:
```

the three halves of a merge, so both entry points share them

### lines 2800-2801

```python
with self._read_budget(fresh=False):
```

Not a fresh generation: `refresh` on the line above has just read these files, and what it read is what the merge is about to do.

### lines 2810-2811

```python
"destination": self._destination()}
```

READ HERE, ON THE GUI THREAD. The provider is the settings panel, and a worker thread may not touch a widget.

## DatabaseMergePanel._merge_worker

### lines 2836-2841

```python
artefact = ""
```

THE ARTEFACT, WRITTEN ON THIS THREAD (154 F). Two hundred thousand rows of eighty columns is seconds of CSV, and doing it in `_finish_merge` would put those seconds back on the GUI thread which is the exact defect section A was filed about, moved twenty lines later. A merged frame nobody can write is still a merged frame, so a failure here is a NOTE and not a refusal.

## DatabaseMergePanel._relay_progress

### line 2892

```python
def _relay_progress(self, stage: str, done: int, total: int) -> None:
```

progress, across the thread boundary

### line 2904, trailing

```python
except RuntimeError:
```

teardown race

## DatabaseMergePanel.show_aggregation_rules

### lines 3003-3004  _(unsure)_

```python
question = ("rules preview", paths, table, _screen_key(screens))
```

A preview, not the merge: enough rows to know each column's type, which is all the rules need.

### lines 3016-3019

```python
with self._read_budget():
```

The budget is the panel's usual one and NOT a fresh generation: a click is not a reason to re-open every database, and a local disk answers inside it, so the dialog still opens on the click exactly as it always did.

### lines 3025-3028

```python
self._wait_for_rules(None)
```

The read failed rather than being slow. Put the button back BEFORE the message box: a modal opened over a button still saying "reading" leaves it saying that for good.

## well_keys

### lines 3070-3079

```python
def well_keys(frame) -> Tuple[str, Tuple[str, ...]]:
```

Instruction 154 E: a message that asserts a cause it has not checked

"Nothing to scan. Load a run whose wells carry both the gene assignment and the measurements" was shown to the maintainer WITH FOUR MEASUREMENT DATABASES LOADED. It names two things a well must carry, checks neither, and offers no way to give it them. Which half is missing is answerable here the panel holds both halves -- and when both are present and the scan still has nothing, the answer is the KEY, with one example from each side.

## describe_key_overlap._canonical

### lines 3129-3131

```python
def _canonical(keys):
```

NORMALISED, so a `pp` doubling is not reported as a mismatch of wells when it is a mismatch of ONE CHARACTER in the plate id -- which is the failure instruction 154 D is about, seen from here.

## ColumnRegressionPanel.__init__

### lines 3297-3299

```python
self._outcomes_grip.setVisible(False)
```

THE HANDLE FOLLOWS THE BOX IT RESIZES. A grip under a hidden box is a border with nothing above it, and the panel shows the outcomes only once a queue has produced some.

## ColumnRegressionPanel.start_regressions

### lines 3487-3491

```python
self._queue_settings = base
```

SNAPSHOTTED ON THE GUI THREAD. The provider is the live settings panel; reading it from the worker would be touching a widget off the GUI thread, and reading it per fit would let a user editing the panel mid-queue fit twelve different models and compare them as if only the response had changed.

### lines 3494-3498

```python
self._offer_frame(score)
```

THE FRAME IS OFFERED FOR EXACTLY AS LONG AS THE QUEUE THAT READS IT. The merge offers it when it stages it and `_finish_queue` withdraws that offer, so without this a second queue over the same merge would parse the artefact back once per fit -- gigabytes of it -- while the panel above was still holding the very frame it wrote.

### lines 3506-3511

```python
self.progress.setText(f"Queued {len(columns)} fit(s).")
```

THE LABEL ONLY, not `_on_queue_progress`. Calling that here put the first column's `fit_started` out TWICE -- once from here and once from the worker's own progress callback -- which is two rows in the Runs tab for one fit, and the first of them says "running" for ever because the second overwrote its handle. Found by driving the real queue; the tests were green.

### lines 3518-3520

```python
self._running = False
```

Same contract as start_merge above: the running state goes up before the submit, so a refusal has to take it down or the queue can never be started again.

## ColumnRegressionPanel._queue_worker

### line 3552  _(unsure)_

```python
def _queue_worker(self, columns: Sequence[str]) -> Dict[str, Any]:
```

the three halves, so both entry points share them

## ColumnRegressionPanel._relay_started

### line 3573, trailing

```python
except RuntimeError:
```

teardown race

## ColumnRegressionPanel._relay_result

### line 3580, trailing

```python
except RuntimeError:
```

teardown race

## ColumnRegressionPanel._on_queue_progress

### lines 3589-3591

```python
self.fit_started.emit(
```

THE ROW GOES UP BEFORE THE FIT COMES BACK. A twelve-column queue that showed nothing until it ended would be the freeze this whole instruction was filed about, one screen along.

## ColumnRegressionPanel._finish_queue

### lines 3625-3630

```python
from ...frame_handoff import release
```

THE PRODUCER SAYS IT HAS FINISHED. The offer is a weak reference, so this is not the difference between a leak and none -- the merging panel above still owns the frame either way. What it buys is a DETERMINISTIC fallback: after this, anything that reads the merged frame reads the file, rather than getting the object or the file depending on when a garbage collection happened to run.

## MeasurementScanPanel.__init__

### lines 3717-3719

```python
self.databases = DatabaseMergePanel(
```

THE DATABASES COME FIRST, because they are the input to everything below them. Hidden entirely when no plate row has one, so a project that never attached a database sees the tab it has always seen.

### lines 3724-3728

```python
self.databases.step_folds_changed.connect(self.remember_section_layout)
```

A NESTED FOLD OR A DRAGGED BORDER IS PART OF THE SAME ARRANGEMENT as the outer dividers, so it is stored the same way and at the same moment. The panels relay rather than store: the user arranges ONE Measurements tab, and three records that can disagree is the bug that arrangement-per-widget always turns into.

### lines 3730-3745

```python
self._sections = QSplitter(Qt.Vertical, self)
```

EVERY SECTION IS A SPLITTER CHILD, so its borders move and it cannot be squeezed into its neighbour. Reported 2026-08-19: "still cant resize the elements in the measurements tabs. now they overlap in such a way i dont have access to some of them" -- a QVBoxLayout gives the sections whatever height it decides, and adding one more widget to it took the space out of the others.

`setChildrenCollapsible(False)` with a minimum height per section is what makes "not be able to overlap" true rather than merely unlikely: a section can be dragged small, never to nothing.

AND EACH ONE FOLDS. Reported in the same breath: "there are to many elements in the measurements tab". Four panels is too many only when all four are open -- a user fitting a regression does not need the attach-database table on screen. See :class:`~.collapsible_section.CollapsibleSection`.

### lines 3750-3751  _(unsure)_

```python
self._restoring = False
```

Set while the stored layout is being put back, so restoring does not write the half-restored state straight back out again.

### lines 3760-3763

```python
self.regression = ColumnRegressionPanel(
```

STEP 4, WHICH THE TAB USED TO END WITHOUT (154 F). Steps 1-3 merge; merging so that "regression can be run on any column in the databases" is the POINT of the merging, and it was not on this tab at all -- so a user who had merged had no idea what came next.

### lines 3769-3771

```python
self.databases.merged.connect(self._on_merged)
```

A NEW MERGE IS A NEW SET OF COLUMNS. Without this the picker holds the previous merge's columns and every fit reads a file that has been overwritten underneath it.

### lines 3792-3793

```python
self._rank.addItem("effect size", "effect_size")
```

Effect size first, because that is what was asked for and because with enough wells a trivial effect is significant.

## MeasurementScanPanel._add_folding_section

### lines 3842-3843

```python
section.toggled.connect(lambda *_: self._reweigh_and_remember())
```

RE-WEIGH ON EVERY FOLD, because the weights depend on which sections are open -- see `_keep_the_filler_last`.

### line 3847

```python
self._keep_the_filler_last()
```

THE FILLER STAYS LAST, so a section added later still folds upward.

## MeasurementScanPanel._share_the_height

### lines 3875-3878

```python
if not widget.isVisible():
```

A HIDDEN SECTION TAKES NO ROOM. `_show_section` hides the ones with nothing to show, and the splitter forces a hidden child to zero anyway -- so asking for its minimum here only makes the arithmetic disagree with the layout that follows.

### line 3884, trailing  _(unsure)_

```python
sizes.append(0)
```

filled in below

### line 3893  _(unsure)_

```python
filler = self._sections.indexOf(getattr(self, "_filler", None))
```

Nothing open: the whole gap goes under the folded headers.

## MeasurementScanPanel._keep_the_filler_last

### lines 3928-3937

```python
for i in range(self._sections.count()):
```

THE FILLER YIELDS FIRST. Reported 2026-08-20, right after the fold-upward fix landed: "when opened they just open a tiny bit. have them fill the container to the next subsection."

Giving the filler the only stretch made it absorb TOO well -- an opened section took its minimum and the filler kept everything else. An OPEN section stretches, so opening one takes the space back from the gap; a FOLDED section does not, so it still hands its height over. The filler stretches least of the three, which is what makes it the last to get space and the first to give it up.

## MeasurementScanPanel._apply_section_layout

### lines 3983-3986

```python
self.set_section_expanded(title, title not in folded)
```

ONLY the titles that are actually there. A stored layout from a version with a section this one does not have must not be an error, and a NEW section defaults to open rather than to whatever the absent entry would imply.

### lines 3989-3993

```python
if len(sizes) == self._sections.count() - 1:
```

A LAYOUT STORED BEFORE THE FILLER EXISTED is one child short, and dropping it would throw away every arrangement a user already has. The filler takes whatever is left, so it is restored at zero and grows on the first layout pass. (Future-first: what is written from now on carries the filler's own size.)

### lines 3998-4002

```python
steps = layout.get("steps") or {}
```

THE TWO NESTED LEVELS, restored in the same pass as the outer one. Numbered steps and box keys are unique across the tab's panels, so each panel takes the entries it recognises and ignores the rest see `set_step_folds`, which is where "ignores the rest" is spelled out and why it is not an error.

## MeasurementScanPanel._show_section

### lines 4064-4067

```python
self.databases.setVisible(bool(showing))
```

A TITLE THAT NAMES NO SECTION. Every caller passes one from section_titles(), so this is reached only if the two ever disagree -- which is exactly when a header would be left opening onto nothing.

## MeasurementScanPanel.add_section

### lines 4101-4105

```python
from ..screens.settings_model import retarget_field_tooltips
```

HOVER HELP GOES ON THE SETTING'S NAME, not on the box you type into. A tooltip on an editable field is unreachable the moment the user is editing it -- which is exactly when they wanted it -- and tests/test_tooltips_are_on_the_setting_not_the_field.py is the guard that says so.

## MeasurementScanPanel._on_databases_changed

### lines 4132-4134

```python
"""Re-run the scan when the set of databases changes.
```

Shown when there is anything to show -- including rows whose database is missing or absent, because "this plate has none" is exactly what a user opening this tab needs to be told.

## MeasurementScanPanel.why_nothing_to_scan

### line 4181

```python
if attached and merged is not None and len(merged):
```

THE MEASUREMENT HALF, from what this tab is actually holding.

## MeasurementScanPanel.scan

### lines 4238-4245

```python
also = self.what_is_available()
```

A refusal is an ANSWER and it says what to do about it. Shown in full rather than summarised: "the scan failed" would send the user looking for a bug in the software.

AND WHAT ELSE IS HERE. "no 'gene' column" is true and incomplete when four measurement databases are sitting above it whose wells do not meet the loaded run's -- that is a second, checked fact, and the user cannot act on the first without it.

## MeasurementScanPanel.set_result

### lines 4268-4271

```python
table = table.copy()
```

BOTH CORRECTIONS, IN WORDS, ON EVERY ROW. A measurement that passes within its own run and fails across the scan is the single most important thing this feature can tell a user, and it is invisible in two columns of small numbers.

### line 4274, trailing  _(unsure)_

```python
table = table.loc[table.index]
```

keep the frame's own order
