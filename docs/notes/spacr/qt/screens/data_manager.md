# Notes from `spacr/qt/screens/data_manager.py`

Prose lifted out of `spacr/qt/screens/data_manager.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [ConfirmDeleteDialog.__init__](#confirmdeletedialog__init__) (3 entries)
- [ConfirmDeleteDialog._start_the_file_list](#confirmdeletedialog_start_the_file_list) (2 entries)
- [ConfirmDeleteDialog._on_listing_failed](#confirmdeletedialog_on_listing_failed) (1 entry)
- [ConfirmDeleteDialog._stop_the_file_list](#confirmdeletedialog_stop_the_file_list) (1 entry)
- [DataManagerScreen.__init__](#datamanagerscreen__init__) (1 entry)
- [DataManagerScreen._dialog_start](#datamanagerscreen_dialog_start) (1 entry)
- [DataManagerScreen.choose_project](#datamanagerscreenchoose_project) (1 entry)
- [DataManagerScreen.choose_destination](#datamanagerscreenchoose_destination) (1 entry)
- [DataManagerScreen._run](#datamanagerscreen_run) (1 entry)
- [DataManagerScreen._update_controls](#datamanagerscreen_update_controls) (1 entry)
- [DataManagerScreen._follow_path_probes.redraw](#datamanagerscreen_follow_path_probesredraw) (2 entries)
- [DataManagerScreen._follow_path_probes](#datamanagerscreen_follow_path_probes) (1 entry)
- [DataManagerScreen.scan](#datamanagerscreenscan) (1 entry)
- [DataManagerScreen.plan_prune](#datamanagerscreenplan_prune) (1 entry)
- [DataManagerScreen.confirm_and_prune](#datamanagerscreenconfirm_and_prune) (1 entry)
- [DataManagerScreen.closeEvent](#datamanagerscreencloseevent) (2 entries)

## Module level

### lines 113-114

```python
register_widget_qss("DataManager", _data_manager_qss, replace=True)
```

``replace=True`` because this module owns the name: a reimport must re-register the same block rather than raise and leave the screen unstyled.

## ConfirmDeleteDialog.__init__

### lines 208-209

```python
self.listing.setPlainText(
```

The heading only. `describe()` walks the project, and this is the GUI thread.

### line 217  _(unsure)_

```python
self.acknowledged.setEnabled(False)
```

Nobody can have read a list that is not on screen yet.

### lines 232-233

```python
self._start_the_file_list(threaded)
```

Last, because its completion handler touches every widget above and, unthreaded, it runs before this line returns.

## ConfirmDeleteDialog._start_the_file_list

### line 279

```python
def _start_the_file_list(self, threaded: bool) -> None:
```

the file list, off the GUI thread

### lines 298-299

```python
self._show_the_files(((), False))
```

Nothing to walk. `file_list` returns an empty tuple without touching the disk, and a thread costs more than the answer.

## ConfirmDeleteDialog._on_listing_failed

### lines 333-334

```python
return
```

The dialog has already let go of the walk; this is a failure arriving after the shutdown that abandoned it.

## ConfirmDeleteDialog._stop_the_file_list

### lines 371-373

```python
pass
```

The runner's C++ half has gone with the dialog. The threads are still drained by `job_runner.shutdown_all` on the way out of the application.

## DataManagerScreen.__init__

### lines 472-473  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

## DataManagerScreen._dialog_start

### lines 729-730  _(unsure)_

```python
if path_probe.exists(remembered, want_dir=True,
```

`exists(want_dir=True)`, not `isdir()`, only so the default can be chosen per path; the question and the cache key are identical.

## DataManagerScreen.choose_project

### lines 755-756

```python
self._remember_picked(chosen)
```

The dialog has just proved this folder is there, so record it rather than have the cache learn it again.

## DataManagerScreen.choose_destination

### lines 772-774

```python
chosen = QFileDialog.getExistingDirectory(
```

Same reasoning as choose_project, and the same two calls: the last destination is a hint about where to open, never a reason to wait on a filesystem.

## DataManagerScreen._run

### lines 843-845

```python
self._jobs.append((thread, worker))
```

Strong references: PySide6 will not keep the worker alive through the started→run connection alone, and a QThread garbage-collected while still running takes the process down with it.

## DataManagerScreen._update_controls

### lines 923-932

```python
"""Enable the actions for whatever root is set.
```

self._root is whatever folder the user picked or dropped, and this runs on construction, on every checkbox change and on every job settle. A bare os.path.isdir here was therefore a stat on the GUI thread with a user-supplied path: measured 2026-09-04, one on a sleeping /nas_mnt autofs share had not returned after twenty seconds, and a stalled event loop is a freeze with no traceback. Optimistic while the probe is out -- everything these controls start hands the real question to a worker, which reports a bad root through _on_job_error -- and _follow_path_probes greys them once the answer lands.

## DataManagerScreen._follow_path_probes.redraw

### lines 985-990

```python
try:
```

`getattr`, not `self._root`, and for the reason spelled out in `spacr.qt.dnd._DropzoneFilter.eventFilter`: PySide6 CLEARS the Python wrapper's __dict__ when the C++ widget goes, so a plain attribute read on a dead screen raises AttributeError rather than the RuntimeError this guard is named for -- and it raises it inside the Qt event loop, where no caller can catch it.

### lines 996-997  _(unsure)_

```python
pass
```

The screen has gone; the signal outlived it. The enable pass touches widgets, so this is where that lands.

## DataManagerScreen._follow_path_probes

### lines 1000-1001

```python
self._path_probe_redraw = redraw
```

Held on the instance because the connection alone does not keep a plain closure alive.

## DataManagerScreen.scan

### lines 1023-1028

```python
if not root or not path_probe.exists(root, want_dir=True,
```

A guard against no project at all, not an authority on this one: dm.scan_project runs in the worker and is what genuinely fails on a root that is not there. exists(want_dir=True) rather than path_probe.isdir because isdir answers False for a path nobody has probed yet, which would refuse the very first scan of a folder the user just chose.

## DataManagerScreen.plan_prune

### line 1094

```python
if not root or not path_probe.exists(root, want_dir=True,
```

Same guard, same reasoning as scan(): never a stat on this thread.

## DataManagerScreen.confirm_and_prune

### lines 1141-1143

```python
dialog = ConfirmDeleteDialog(plan, self, threaded=self._threaded)
```

The screen's own threading, so a test that drives this screen synchronously gets a dialog whose file list is already filled in rather than one that is still reading the disk.

## DataManagerScreen.closeEvent

### lines 1262-1263

```python
pass
```

Already gone. A screen that refused to close over its own housekeeping would be the worse defect.

### lines 1271-1273

```python
pass
```

The thread's C++ half has already gone. A close handler that let this out would leave the screen half-closed, and the job list below is cleared either way.
