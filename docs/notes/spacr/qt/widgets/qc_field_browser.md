# Notes from `spacr/qt/widgets/qc_field_browser.py`

Prose lifted out of `spacr/qt/widgets/qc_field_browser.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_keys_for_finding](#_keys_for_finding) (1 entry)
- [load_qc_field](#load_qc_field) (2 entries)
- [_mask_colour](#_mask_colour) (1 entry)
- [QCFieldBrowser.__init__](#qcfieldbrowser__init__) (3 entries)
- [QCFieldBrowser._show_target](#qcfieldbrowser_show_target) (1 entry)
- [QCFieldBrowser._recheck_files](#qcfieldbrowser_recheck_files) (1 entry)
- [QCFieldBrowser._on_probe_answered](#qcfieldbrowser_on_probe_answered) (1 entry)
- [QCFieldBrowser._sync_action](#qcfieldbrowser_sync_action) (1 entry)
- [QCFieldBrowser.toggle_quarantine](#qcfieldbrowsertoggle_quarantine) (2 entries)
- [QCFieldBrowser._apply_move](#qcfieldbrowser_apply_move) (3 entries)
- [QCFieldBrowser._on_move_failed](#qcfieldbrowser_on_move_failed) (1 entry)
- [QCFieldBrowser.closeEvent](#qcfieldbrowsercloseevent) (1 entry)

## Module level

### lines 67-70

```python
MAX_DISPLAY_EDGE = 1600
```

A 4096-square camera frame is 32 MB per uint16 plane.  The browser is an overview, not an editor, so prepare a bounded display copy and leave the memory-mapped source immediately.  1600 px still resolves five-pixel QC objects while keeping four image planes plus masks under a modest budget.

## _keys_for_finding

### lines 222-224

```python
if exact or str(getattr(finding, "kind", "")) != "clean":
```

A flag finding carries exact fields.  A positional finding does not: every matching field is part of the pattern and belongs in the browser, including individually-clean fields.

## load_qc_field

### lines 394-396

```python
if stacks and planes > len(stacks):
```

Legacy arrays carry no manifest.  Their contract still appends one mask plane per stack after the intensities, so the discovered stack count is stronger evidence than blindly assuming four channels.

### lines 438-440

```python
del array
```

Drop the memmap before quarantine can be offered.  On Windows an open mapping prevents the atomic rename; the bounded copies above own their bytes independently.

## _mask_colour

### line 645  _(unsure)_

```python
seed = sum((index + 1) * ord(char)
```

Stable and vivid for custom object-role names.

## QCFieldBrowser.__init__

### lines 815-822

```python
self._move_jobs = JobRunner(
```

A THIRD RUNNER, not a share of `_jobs`. The quarantine move used to run inline on the GUI thread; putting it on the loading runner instead would have made `_on_load_failed` report a failed rename as "Could not load this field", and would have let `_show_target`'s `cancel()` orphan a move mid-flight. `user_visible=True` is left at its default deliberately: unlike the two above, this runner carries only work the user asked for by pressing the button, and a move on a slow share is exactly the kind of activity Home should own up to.

### lines 940-945

```python
self.finished.connect(self._on_finished)
```

`closeEvent` is NOT a teardown hook for this dialog: it is

`WA_DeleteOnClose`, and Escape goes through `QDialog.done`, which deletes the widget without ever raising a close event (verified on Qt 6.10). `finished` is emitted on both paths, while the object is still alive, which is the only place a move that committed during the dismissal can still be announced.

### lines 952-956

```python
for widget in self.findChildren(QWidget):
```

Arrow keys belong to field navigation even when a canvas, button or channel picker currently owns focus.  QGraphicsView consumes arrows for scrolling before a dialog-level keyPressEvent can see them, so install one narrow filter across this dialog's children as well as retaining QShortcuts for native shortcut dispatch.

## QCFieldBrowser._show_target

### lines 1008-1010

```python
self._last_active = True
```

A different field knows nothing about the last one's two copies. Back to the module's own defaults until this field's load, a couple of lines below, settles them exactly.

## QCFieldBrowser._recheck_files

### line 1289, trailing  _(unsure)_

```python
self._file_state()
```

queues both probes again; the answers repaint

## QCFieldBrowser._on_probe_answered

### line 1313  _(unsure)_

```python
pass
```

The dialog has gone; the signal outlived it.

## QCFieldBrowser._sync_action

### lines 1340-1344

```python
self._quarantine.setEnabled(False)
```

A move already running owns this field. The button is refused so the same rename cannot be started twice, and the status line is left exactly as it was: the move was instantaneous and silent while it ran on the GUI thread, and it is not this dialog's job to invent a caption for having stopped freezing.

## QCFieldBrowser.toggle_quarantine

### line 1385, trailing  _(unsure)_

```python
if active == quarantined:
```

both present or both absent

### lines 1396-1399

```python
self._sync_action()
```

Still in flight -- repaint the button as refused. When the runner is unthreaded (tests) the handlers above have already run and already repainted, and syncing again here would erase the failure notice one of them just wrote.

## QCFieldBrowser._apply_move

### lines 1411-1415

```python
active_path, quarantined_path = self._paths_for(target)
```

The rename just made the probe cache wrong in both directions, and the truth is already in hand -- prime rather than forget, or the two keys answer with the PRE-MOVE state (that is what `_last_active` carries forward) until a fresh probe lands, and the button offers to quarantine a file that is already in quarantine.

### lines 1421-1424

```python
self._sync_action()
```

A banner link re-pointed the dialog while the move was in flight. The move still happened and the listener above still has to hear about it, but this field's notice belongs to a field that is no longer on screen.

### lines 1432-1433  _(unsure)_

```python
self._show_target(preserve_notice=True)
```

Reload from the new location, both to release stale path text and to prove the reversible move left a readable array.

## QCFieldBrowser._on_move_failed

### lines 1453-1456

```python
for path in self._paths_for(target):
```

Nothing is known about either copy after a half-finished move. Forget rather than prime, and let the background probe settle it; `_last_active` keeps the button showing the pre-move state in the meantime, which is the state a failed move should leave.

## QCFieldBrowser.closeEvent

### lines 1567-1569

```python
pass
```

Escape never reaches this method at all, so the connection is only ever dropped here as a courtesy; Qt has already done it by the time the object is really gone.
