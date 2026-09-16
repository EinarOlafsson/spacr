# Notes from `spacr/qt/widgets/sweep_runs.py`

Prose lifted out of `spacr/qt/widgets/sweep_runs.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [_read_results_table](#_read_results_table) (1 entry)
- [_find_the_run](#_find_the_run) (1 entry)
- [_readable_size](#_readable_size) (1 entry)
- [save_run_states](#save_run_states) (3 entries)
- [SweepRunsPanel.__init__](#sweeprunspanel__init__) (8 entries)
- [SweepRunsPanel.closeEvent](#sweeprunspanelcloseevent) (1 entry)
- [SweepRunsPanel.load](#sweeprunspanelload) (1 entry)
- [SweepRunsPanel._table_arrived](#sweeprunspanel_table_arrived) (2 entries)
- [SweepRunsPanel.set_frame](#sweeprunspanelset_frame) (1 entry)
- [SweepRunsPanel.update_run](#sweeprunspanelupdate_run) (3 entries)
- [SweepRunsPanel.load_run_from_disk](#sweeprunspanelload_run_from_disk) (2 entries)
- [SweepRunsPanel._run_arrived](#sweeprunspanel_run_arrived) (2 entries)
- [SweepRunsPanel.set_loaded_run](#sweeprunspanelset_loaded_run) (2 entries)
- [SweepRunsPanel._announce_the_loaded_run](#sweeprunspanel_announce_the_loaded_run) (3 entries)
- [SweepRunsPanel.the_load_failed](#sweeprunspanelthe_load_failed) (3 entries)
- [SweepRunsPanel._paint_the_loaded_mark](#sweeprunspanel_paint_the_loaded_mark) (1 entry)
- [SweepRunsPanel._settle_the_loaded_run](#sweeprunspanel_settle_the_loaded_run) (1 entry)
- [SweepRunsPanel._rebuild](#sweeprunspanel_rebuild) (5 entries)
- [SweepRunsPanel._describe](#sweeprunspanel_describe) (1 entry)
- [SweepRunsPanel.selected_trial](#sweeprunspanelselected_trial) (1 entry)
- [SweepRunsPanel.selected_runs](#sweeprunspanelselected_runs) (2 entries)
- [SweepRunsPanel.remove_runs](#sweeprunspanelremove_runs) (3 entries)
- [SweepRunsPanel.delete_runs_from_disk](#sweeprunspaneldelete_runs_from_disk) (2 entries)
- [SweepRunsPanel._ask_then_delete](#sweeprunspanel_ask_then_delete) (1 entry)
- [SweepRunsPanel._deletion_finished](#sweeprunspanel_deletion_finished) (1 entry)
- [SweepRunsPanel._on_job_failed](#sweeprunspanel_on_job_failed) (1 entry)
- [SweepRunsPanel._workspace_answer](#sweeprunspanel_workspace_answer) (1 entry)
- [SweepRunsPanel._build_run_menu](#sweeprunspanel_build_run_menu) (5 entries)
- [SweepRunsPanel.load_this_run](#sweeprunspanelload_this_run) (1 entry)
- [SweepRunsPanel._apply_run_menu](#sweeprunspanel_apply_run_menu) (4 entries)
- [SweepRunsPanel._on_states_saved](#sweeprunspanel_on_states_saved) (1 entry)
- [SweepRunsPanel._on_selection](#sweeprunspanel_on_selection) (2 entries)

## Module level

### lines 83-86

```python
"dependent_variable",
```

WHAT WAS FITTED, first. Instruction 154 F queues one run per column of the merged measurements, so a table of those runs differs in the RESPONSE and in nothing else -- and a comparison table whose only varying column is missing is a list of identical-looking rows.

### lines 88-91

```python
"regression_type", "regression_backend", "inference",
```

WHICH ENGINE AND HOW MANY SHUFFLES. Two permutation runs a thousand shuffles apart were one row apart in this table and identical on it, which is the comparison the table exists for. `guide_permutations` is copied only for a run that actually permuted -- see `_run_settings_row`.

### lines 116-119

```python
_DELETION_STARTED = -1
```

What :meth:`SweepRunsPanel.delete_runs_from_disk` answers when the delete has been STARTED rather than finished: the count is not knowable at return time and pretending otherwise would be a lie the caller cannot detect. It is truthy, because "the delete is under way" is what the menu wants to hear.

## _read_results_table

### lines 169-170

```python
return None, str(error) or type(error).__name__
```

NAMED EVEN WHEN THE EXCEPTION IS NOT. An empty message would be read as "no table here yet", which is the one thing it is not.

## _find_the_run

### lines 192-194

```python
try:
```

ITS OWN SETTINGS, so an old run is described by the same columns as a new one. Without them the row is a name and a folder, and two runs cannot be compared on the settings that differ.

## _readable_size

### lines 262-266

```python
raise AssertionError(                                # pragma: no cover
```

NO TRAILING RETURN. "GB" is the last unit and the condition carries `or unit == "GB"`, so the final iteration returns whatever the size is -- the loop cannot fall out of the bottom. Checked against 20,000 values spanning zero to 2**63-1 and negatives; every one returned from inside.

## save_run_states

### lines 307-311

```python
failures.append((path, f"{type(error).__name__}: {error}"))
```

INSIDE THE TRY LIKE EVERYTHING ELSE. This runs on a worker, and a worker that throws never reaches `_on_states_saved` -- so one unaskable path would leave "Saving the state of 3 runs…" on the line for the rest of the session and say nothing about the two that could have been saved.

### lines 315-317

```python
failures.append((path, "the run folder is not on disk any more"))
```

`save_for_run` would CREATE this folder, leaving a directory holding nothing but a workspace file where a deleted run used to be. A run that is gone from disk is not a run to save.

### lines 321-323

```python
written = save_for_run(path, {"save_workspace": "reference"})
```

`reference` rather than `copy`: the run's own files are already on disk beside it, and copying them again to save a state would double a screen's worth of crops.

## SweepRunsPanel.__init__

### lines 505-518

```python
self._jobs = JobRunner(self, threaded=self._threaded,
```

TWO RUNNERS, because one of them is CANCELLED. A tab change can ask for the sweep table again while the last read is still out on a slow mount, and the answer that must win is the newest -- so the table reads have a runner of their own that `load` can cancel without abandoning a delete or a save half-way through.

`user_visible=False` ON BOTH, and the reason is Preferences' rather than the usage poller's: a right-click that saves a bundle or deletes a folder IS something the user started, but it is not a RUN and `home.py` filters that flag to decide which of them gets a blue "<module> — running" banner across the top of Home. None of this work had a banner before, because none of it had a thread; the flag hides nothing the user saw, and without it every right-click on this tab flashes one.

### lines 533-536

```python
self._open = QPushButton("Load run…")
```

A RUN ON DISK IS A FIRST-CLASS RUN (154 G). An earlier session's results folder is opened here, gets a row beside this session's own, and becomes the loaded run -- rather than being something the user can only reach by re-running a fit that already finished.

### lines 552-554

```python
self.table = ResultsTable()
```

The same table widget the results use: it already sorts numerically, filters, and copies as TSV, and a second implementation of those is a second set of bugs.

### lines 556-558

```python
self.table.configure(
```

Its own words. The coefficient table's "type a gene, a guide" and "significant only" belong to a table of findings; over a list of trials the first is wrong and the second cannot do anything.

### lines 563-571

```python
from PySide6.QtWidgets import QAbstractItemView
```

THE TWO GESTURES A USER LOOKS FOR (instruction 146 B): a context menu on the row, and the Delete key on the selection. Multi-select is the QTableWidget default and is kept -- a sweep writes one folder per trial, and clearing up after one by hand twenty times is not a feature. SAID RATHER THAN INHERITED. ExtendedSelection is the QTableWidget default today, and "several rows at once" is a requirement of instruction 146 B rather than a happy accident of a default that could change.

### lines 578-583

```python
self.table.table.doubleClicked.connect(self._on_double_click)
```

DOUBLE-CLICK IS THE GESTURE THAT LOADS (190). It used to be the second way in, beside a selection that also loaded; as of 2026-08-20 it is the only one, because selecting a row to read its name should not cost a multi-second read. It still FORCES the load rather than asking politely, so a run refused for any reason can be insisted on.

### lines 588-593

```python
self._photo = QLabel()
```

THE STILL OF A RUN THAT IS NOT LIVE (instruction 116's last line). A run opened beside the loaded one is photographed when it closes; this is where that photograph is finally SHOWN, beside the row it belongs to. Hidden when there is none, which is most rows -- an empty frame under the table would read as a run that failed to draw rather than as one nobody has opened beside.

### lines 666-671

```python
for runner in (self._jobs, self._load_jobs):
```

THE OTHER END OF EVERY HANDLER BELOW. `JobRunner` calls `on_done` only for a job that SUCCEEDED, so a worker that raises leaves the placeholder on the line and the "Load run…" button disabled by the click that started it -- for the rest of the session. Connected last, after `_open` and the state above exist, because a job can fail the moment it is submitted.

## SweepRunsPanel.closeEvent

### lines 676-680

```python
"""Stop background work and unlink before going away.
```

Qt ABORTS THE PROCESS if a running QThread is destroyed, and a run folder on a sleeping mount is exactly the job that is still going when the user closes the screen it belongs to. Both runners are asked to stop and are waited for a bounded time; a job that outlasts the budget is parked rather than killed mid-delete.

## SweepRunsPanel.load

### lines 718-720

```python
self._load_jobs.cancel()
```

LAST ASK WINS. A tab flipped twice queues two reads of the same folder, and the older one landing second would put a stale table back. Cancelling drops the result rather than joining the thread.

## SweepRunsPanel._table_arrived

### lines 744-748

```python
self._rebuild(f"No results table at {path} yet.")
```

NOT a wipe, and the note goes THROUGH the rebuild. The sweep's table being absent says nothing about the runs this session has made; setting the status here and clearing the table would be how opening the tab before a sweep exists used to empty it.

### lines 754-755

```python
path_probe.isdir(folder)
```

The chooser opens here next time, and asking now means the answer is in the cache by then rather than being stat-ed under the click.

## SweepRunsPanel.set_frame

### lines 765-772

```python
"""Point the panel at a table of runs.
```

Takes THE SWEEP'S HALF of the table and returns whether the tab now shows a row -- which is no longer the same question, because this session's own runs are rows too. Deliberately left as a comment rather than a docstring: a docstring here is public API surface, and `tests/test_api_i18n_extractor.py` holds an exact count of that, so promoting this to a docstring means bumping the count in the same commit. (It used to say the count file "belongs to another session right now" -- true on 2026-08-17, not a constraint today.)

## SweepRunsPanel.update_run

### lines 828-832

```python
folder = str(row.get("folder") or "")
```

A RUN FINISHING IS WHEN A BUNDLE APPEARS, which is the whole reason `_has_workspace` asks the folder rather than the row. A "no workspace" answer cached from a right-click while the run was still going would otherwise grey the restore entry for the rest of the session.

### lines 836-841

```python
self._workspace_pending.pop(folder, None)
```

AND THE PROBE THAT IS STILL OUT IS DROPPED WITH IT. A right-click while the run was going submitted a `has_workspace` on a folder with no bundle in it yet; dropping only the cached answer leaves that probe free to land afterwards and write the same "no" back, which greys the restore entry for the rest of the session -- the exact thing the paragraph above is for.

### lines 843-852

```python
before = self._loaded_key
```

A RUN THAT FINISHES BECOMES THE LOADED RUN, with no step in between (154 G): "i just ran a regression so that should be loaded automatically". The views were being told nothing had been loaded by the run the user had just watched finish.

AND THEY ARE TOLD NOW (157). This line moved the mark and stopped there, so a second run finishing left the first run's coefficients, figures and summary on screen under a mark naming the second. The key the mark was on is carried into the rebuild, which announces the change once everything else has settled.

## SweepRunsPanel.load_run_from_disk

### lines 886-891

```python
start = self._folder if path_probe.isdir(self._folder) else ""
```

NOT `self._folder` DIRECTLY. Qt stats the start directory before it draws the dialog, so handing it a remembered ``/nas_mnt`` path freezes the click that opened the chooser. `path_probe.isdir` answers from its cache and says no to a path it has not seen -- the dialog opens at its default place, and the next click gets the remembered folder back.

### lines 904-909

```python
self._open.setEnabled(False)
```

A SECOND CLICK MUST NOT QUEUE A SECOND WALK. Re-enabled by

`_run_arrived` when the search answers and by `_on_job_failed` when it does not -- and BOTH are needed, because `JobRunner` hands a result to `on_done` only for a job that came back cleanly. A button disabled by a click and re-enabled by nothing is the silent no-op this repository keeps fixing.

## SweepRunsPanel._run_arrived

### lines 929-930

```python
self._rebuild(f"No run in {folder}: none of "
```

NAMED, not a silent no-op. The user picked a folder; being told nothing happened is the failure this repository keeps fixing.

### lines 954-956

```python
self._rebuild(f"Loaded the run in {run_folder}.", since=before)
```

AND THE VIEWS FOLLOW IT -- through the same funnel every other path uses (157). This used to emit the two signals itself, which is how the finishing path came to be the only one that did not.

## SweepRunsPanel.set_loaded_run

### lines 1057-1059

```python
self._source_note = ""
```

The note describes the LAST LOAD, which this supersedes. Left alone it produced "Loaded: ols_1. Loaded the run in .../ols_2." -- one sentence naming two runs.

### lines 1063-1067

```python
self._announce_the_loaded_run(before)
```

AND THE VIEWS FOLLOW THE CHOICE. Moving the mark and leaving the results panel, the summary and the figure grid on the previous run is the failure 154 G is about: the choice has to be visible from the views that depend on it, not only from the tab that sets it.

## SweepRunsPanel._announce_the_loaded_run

### lines 1083-1085

```python
return False
```

The key names a row the composed frame does not hold, which is not a run anybody can be shown. Say nothing rather than hand a view a record it cannot open.

### lines 1088-1098

```python
self._undo_answers = self._loaded_key
```

BOTH NAMES FOR ONE EVENT. `loaded_run_changed` is the question the screen connects ("which run is on screen"); `trial_activated` predates it and is what the sweep's own listeners were written against. They are emitted together and the screen's handler is idempotent on the run already showing, so connecting either -- or both -- costs one load. SET BEFORE THE EMITS. A listener that refuses SYNCHRONOUSLY -- which is how the screen behaved before the read moved to a worker, and how a test drives it -- calls back into `the_load_failed` from inside these two lines, and the token has to already name this announcement or its own refusal is rejected as stale.

### lines 1102-1116

```python
return True
```

THE UNDO IS KEPT, AND IT NAMES WHAT IT ANSWERS.

It used to be spent right here, on the reasoning that the window in which a load can fail IS the emission -- true while the listener read the run synchronously, and false since instruction 159 moved that read onto a worker. The failure now arrives after this method has returned, so an undo consumed here is gone before the only caller that needs it: a run whose folder does not exist kept the mark, which is 157's disagreement pointing the other way.

Kept, but not open-ended. `_undo_answers` records WHICH announcement the undo belongs to, so a listener coming back later -- a click on a trial that failed two choices ago -- cannot drag the mark backwards: `the_load_failed` checks that the mark is still on the run it was told about, and does nothing when it is not.

## SweepRunsPanel.the_load_failed

### lines 1139-1141

```python
answers = getattr(self, "_undo_answers", None)
```

ONLY THE ANNOUNCEMENT THIS ANSWERS. An asynchronous read reports back after the mark may have moved again; rolling back then would undo a choice the user has since made.

### lines 1148-1155

```python
return False
```

NOTHING TO GO BACK TO, so the mark stays on the run that finished. Clearing it here would answer "no run is loaded" immediately after a run the user watched finish -- which is 154 G's report, arrived at from the other direction. The run IS the loaded one; what failed is drawing it, and the status line says so. The rule this rolls back is "the mark must not point at a run OTHER than the one on screen", and with nothing on screen there is no other run to point at.

### line 1158

```python
self._undo_answers = previous
```

Spent: this undo has been used and must not be used twice.

## SweepRunsPanel._paint_the_loaded_mark

### line 1198, trailing  _(unsure)_

```python
index = item.data(0x0100)
```

Qt.UserRole: the frame row

## SweepRunsPanel._settle_the_loaded_run

### lines 1221-1223

```python
self._loaded_key = ""
```

A key naming no row is worse than none: `loaded_run` would answer None while the column showed nothing, and the two disagreeing is how a stale mark survives a reload.

## SweepRunsPanel._rebuild

### lines 1258-1260

```python
trials["run"] = ["trial " + str(value)
```

A trial's name IS its number; saying so in the same column the session's runs use is what lets one glance answer "which run is this row".

### lines 1275-1279

```python
whole = {name for part in frames for name in part.columns
```

WHICH COLUMNS WERE WHOLE NUMBERS BEFORE THE CONCAT. A session run has no `trial_id` and no `n_below_alpha`, so concatenating it in fills those with NaN and pandas promotes the column to float -- and the sweep's trial 1 becomes "1.0", its 12 hits become "12.0". Recorded here, restored below.

### lines 1284-1286

```python
self._settle_the_loaded_run(frame)
```

WHICH ROW IS THE LOADED ONE, decided over the composed frame the session's runs and the sweep's trials are one population, and "there is only one run so it is the loaded one" has to count both.

### lines 1293-1295

```python
record = self.loaded_run()
```

THE LOADED RUN IS THE SELECTED ROW. Refilling the table clears the selection, so without this the highlight jumped off the run being shown every time another one was recorded.

### lines 1304-1309

```python
self._announce_the_loaded_run(before)
```

LAST, AND OUTSIDE `_rebuilding`. A listener re-points the results panel and the figure grid, and one that came back into this method would be refilling a table Qt is still holding items from -- the crash `_paint_the_loaded_mark` exists to avoid. Nothing below this line reads `_loaded_key`, so a listener that hands the mark back (:meth:`the_load_failed`) cannot leave the table half-built.

## SweepRunsPanel._describe

### lines 1368-1370

```python
note += ". No run is loaded — pick one to show it everywhere else"
```

SAID, rather than left as an empty column. Several runs and no choice made is a state the user has to resolve, and a blank column is indistinguishable from a feature that is not working.

## SweepRunsPanel.selected_trial

### line 1388, trailing  _(unsure)_

```python
index = items[0].data(0x0100)
```

Qt.UserRole: the frame row

## SweepRunsPanel.selected_runs

### lines 1393-1401

```python
def selected_runs(self) -> list:
```

delete

Instruction 146, requested 2026-08-18: "the user should be able to delete runs from the figures (currently possible) and from the run tab (not possible)".

TWO DIFFERENT THINGS A USER COULD MEAN, and a single "Delete" that does not distinguish them is how a screen's results are lost. Both are legitimate; the DEFAULT gesture is the safe one.

### line 1414, trailing  _(unsure)_

```python
index = item.data(0x0100)
```

Qt.UserRole: frame row

## SweepRunsPanel.remove_runs

### lines 1490-1496

```python
trials = {str(record.get("trial_id"))
```

BY TRIAL NUMBER AS WELL AS BY KEY, because a sweep trial's

NAME is derived during the rebuild ("trial 2" from `trial_id`) and its row in the raw sweep frame has neither a `run` column nor a `folder` -- so `_row_key` answers "" for every one of them and a match on the key alone removed nothing at all. That is the whole of what "reload brings it back" was promising about rows that had never left.

### lines 1511-1514

```python
self._loaded_key = ""
```

The mark cannot stay on a run that is no longer a row: a key naming nothing makes `loaded_run` answer None while the column shows a tick, and the two disagreeing is how a stale mark survives (see `_settle_the_loaded_run`).

### lines 1521-1522  _(unsure)_

```python
self.runs_removed.emit([dict(record) for record in gone])
```

AFTER the rebuild, so a listener that re-points the results panel is looking at the table as it now is.

## SweepRunsPanel.delete_runs_from_disk

### lines 1548-1549

```python
self._say(self._why_it_cannot_be_deleted(refused))
```

NEVER DELETE WHAT IS RUNNING, and say why rather than ignoring the gesture (instruction 106).

### lines 1563-1568

```python
self._deleted_count = 0
```

NEITHER HALF OF THE QUESTION CAN BE ASKED HERE. Which of these folders is still on disk is a stat each, and what is in one is an `os.walk` of a whole run -- and the answer is needed BEFORE the modal, because the modal is what says what is about to be destroyed. So the description goes to a worker and the confirmation is shown from its callback.

## SweepRunsPanel._ask_then_delete

### lines 1597-1604

```python
self._abandon_waiting()
```

NO IS AN ANSWER, and the line goes back to what it said before the delete was asked for. Saying nothing was already the behaviour -- declining the modal left the status line exactly as it was -- and the placeholder is the one thing that has to be taken back down. NOTHING ELSE WILL: no worker is running any more, so no arrival handler and no `job_failed` is coming, and without this "Working out what these runs hold…" is the last sentence this tab ever shows.

## SweepRunsPanel._deletion_finished

### lines 1632-1633

```python
path_probe.forget(folder)
```

The caches answer "it is there" until told otherwise, and this is the moment they are wrong.

## SweepRunsPanel._on_job_failed

### lines 1737-1742

```python
self._open.setEnabled(True)
```

THE BUTTON FIRST. It is the one thing here that a user cannot work around, and the failure that disabled it is exactly the case `_run_arrived` does not cover. UNCONDITIONALLY, and that is the safe direction: narrowing it to "only the search's own failure" needs an attribution `job_failed` cannot give, and being wrong the other way leaves the button dead for good.

## SweepRunsPanel._workspace_answer

### lines 1779-1780

```python
return self._workspace_answers.get(key, True)
```

Unthreaded the job above has already answered, so this reads the truth rather than the optimism.

## SweepRunsPanel._build_run_menu

### lines 1826-1829

```python
load = None
```

LOAD FIRST, because it is what a user opens this menu for. The menu offered Remove, Open beside and Delete and no way to LOAD -- so the only route to a different run was a single click, and when that was refused there was no second route at all.

### lines 1836-1838

```python
keep = menu.addAction(
```

SAVE, FOR ONE OR FOR SEVERAL. Restore below is single-run because two workspaces cannot both be put on screen; SAVING several is a different thing and is what was asked for.

### lines 1856-1858

```python
restore.setEnabled(False)
```

Instruction 106: OFFERED AND DISABLED, saying why. An entry that appeared only for runs that happen to have a bundle is one nobody learns exists.

### lines 1883-1886

```python
why = self._why_it_cannot_be_deleted(running)
```

GREYED OUT AND SAYING WHY (instruction 106), not silently ignored. Both entries: removing the row of a run this session is still updating would leave `update_run` writing to a handle with nothing to show for it.

### lines 1888-1892

```python
if load is not None:
```

LOAD IS GREYED TOO, and for its own reason rather than the delete reason: a run still going has produced no results table, no figures and no summary, so loading it would put an empty screen under a mark claiming a run. That is worse than a disabled entry, which at least says why.

## SweepRunsPanel.load_this_run

### lines 1941-1942

```python
self.trial_activated.emit(dict(record))
```

Nothing listened to the loaded-run signal; the results panel still has to be told, and `trial_activated` is the other door.

## SweepRunsPanel._apply_run_menu

### lines 1976-1978

```python
self._on_states_saved(([], []))
```

Nothing to hand the writer, and the note still has to be written -- a menu entry that does nothing and says nothing is the failure instruction 106 is about.

### lines 1981-1989

```python
plural = "" if len(folders) == 1 else "s"
```

THE WRITE GOES TO A WORKER. `save_run_states` stats each folder and then writes a bundle into it, both on paths the user chose.

`_start_waiting` RATHER THAN `_say`, because this sentence is a placeholder and has to be registered as one: `_on_job_failed` only takes down a line it can see is outstanding, so a save written with `_say` and then raised on -- `_on_states_saved` rebuilding the table is the throw that reaches it -- would leave "Saving the state of 3 runs…" up for good.

### lines 1999-2001

```python
if self._is_running(records[0]):
```

The menu greys it, and so does this: a menu is one door and

`_apply_run_menu` is the seam tests drive, so a guard on the paint alone would be a guard a test could walk straight past.

### lines 2011-2015

```python
if not self._workspace_answer(str(records[0].get("folder") or "")):
```

THE SAME NON-BLOCKING ANSWER THE MENU WAS DRAWN FROM, rather than a blocking check "to be sure": the menu has already been built and shown from it, so a stat here would only add the freeze back at the click. `AppScreen.restore_run_workspace` is where a bundle that turns out not to be readable is reported.

## SweepRunsPanel._on_states_saved

### line 2039  _(unsure)_

```python
self._workspace_answers.pop(str(folder), None)
```

A save is the moment a cached "no bundle" answer goes stale.

## SweepRunsPanel._on_selection

### lines 2118-2119  _(unsure)_

```python
return
```

The re-select at the end of `_rebuild` is this panel putting the highlight back, not the user choosing a run.

### lines 2124-2138

```python
if not _is_ok(record):
```

PICKING A RUN IS NOT LOADING IT (190). Reported 2026-08-20: "for some reason clicking once on a run shows the results. double click should loade the results".

This used to load on selection, which meant ARROWING DOWN A LIST OF FIVE RUNS LOADED FIVE RUNS -- five multi-second reads nobody asked for, to look at five names. Selection now does what selection does: it shows this run's photograph and its detail, and nothing else. `_load_selected` on double-click is the gesture that costs time, and it was already wired.

THE FAILURE MESSAGE STILL BELONGS TO SELECTION, though. A trial that failed or is still going has no results to show ever, and saying so when it is picked is the difference between a table that ignores clicks and one that explains them -- it costs nothing to say.
