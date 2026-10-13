# Notes from `spacr/qt/startup_benchmark.py`

Prose lifted out of `spacr/qt/startup_benchmark.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [BenchmarkController._ready](#benchmarkcontroller_ready) (1 entry)
- [BenchmarkController._settle_ready](#benchmarkcontroller_settle_ready) (2 entries)
- [BenchmarkController._advance_after_watchdog](#benchmarkcontroller_advance_after_watchdog) (1 entry)
- [BenchmarkController._advance](#benchmarkcontroller_advance) (3 entries)
- [BenchmarkController._timed_out](#benchmarkcontroller_timed_out) (1 entry)
- [BenchmarkController._record_error](#benchmarkcontroller_record_error) (4 entries)

## BenchmarkController._ready

### lines 197-199

```python
QTimer.singleShot(SETTLE_MS, self._settle_ready)
```

Let the 16 ms watchdog report a timer delayed by the click handler before this interval is sealed.  The readiness timestamp remains the first settled paint; only the stall inventory waits two frames.

## BenchmarkController._settle_ready

### lines 240-244

```python
entry["stall_window_started_at"] = start
```

Preserve the exact window used for the derived stall fields.  The readiness timestamp precedes the two-frame settling interval above, so ``started_at``/``at`` alone cannot reproduce this calculation. The parent benchmark driver independently recomputes every value below from the raw watchdog trace and these two boundaries.

### lines 255-258

```python
if self.phase == "module":
```

ONLY A MODULE HAS A DOOR. Home is the screen the window opens on and Preferences is a dialog; stamping either with "sidebar" would be a field that says how it was reached and is wrong about it, which is precisely the failure this field was added to stop.

## BenchmarkController._advance_after_watchdog

### lines 290-291

```python
QTimer.singleShot(SETTLE_MS, _after_beat)
```

Unit environments may not install the production watchdog; ``None`` deliberately falls through after the same two-frame settle.

## BenchmarkController._advance

### lines 311-344

```python
if len(buttons) > 1:
```

NINE KEYS ONCE HAD TWO ROWS, and neither was a mistake: a module folded onto a host's masthead kept its registry row as well, drawn once from the registry and once as an indented child of its host. This driver demanded exactly one and errored on all nine, which is how instruction 284's ratchet stopped measuring nine modules without anyone noticing -- 314's own suspicion, that "the ratchet is not running on the path the user actually takes", made concrete. The fix was to prefer the registry row.

THAT FIX HAS SINCE STOPPED WORKING, AND SO HAS THE ROW IT PREFERRED. Measured 2026-09-11 on a real `MainWindow` with the app drawer open: NO button anywhere in the window carries the `navKey` of `feature_explorer`, `convert`, `external_masks`, `lineage`, `layer_viewer`, `tabulate`, `outliers`, `control_chart`, `train_compare`, `plate_view`, `profiler`, `investigate_hit`, `feature_dict` or `trellis` -- fourteen of forty-five. The `isFoldChild` property this branch discriminated on is READ here and SET NOWHERE in the tree, so the rows are gone and the preference is guarding nothing. The count of unmeasured modules went from nine to fourteen and the artifact said `passed: false` with 463 violations, which reads as a broken application rather than as a driver that cannot find fourteen doors.

A RATCHET THAT ERRORS IS NOT A RATCHET THAT FAILED, and the difference is invisible from the summary line.

SO THE SECOND DOOR IS OPENED HERE, and it is a real one rather than a constructor proxy. Every one of the fourteen is in `TILELESS_APPS`, reached from a host module's button, from Help, or from the command palette -- and the palette reaches ALL of them uniformly: `CommandPalette._nav` is `window._on_nav_selected(key)`, the same slot `Sidebar.nav_selected` fires into. Pressing the button stays the primary path and is unchanged; this is what happens when there is no button to press.

### lines 355-356

```python
self._door = "command palette"
```

WHICH DOOR IS RECORDED, because a number that does not say how it was obtained is the thing this whole comment is about.

### lines 370-373

```python
try:
```

QAbstractButton.click() is the same signal path as a user release: Sidebar.nav_selected -> MainWindow._on_nav_selected.  Calling the screen factory directly is the constructor proxy this benchmark exists to replace.

## BenchmarkController._timed_out

### lines 468-470

```python
QTimer.singleShot(
```

Let an overdue watchdog beat run before sealing the failed interval. Readiness is rejected while this is pending, so the deadline stays decisive even when a paint was queued behind the same long block.

## BenchmarkController._record_error

### lines 482-484

```python
"""Record a failure without losing the timings already taken.
```

``already_stopped`` means the Qt single-shot has fired; its wall timer is independent and must still be cancelled before this method checkpoints or advances.

### lines 531-534

```python
if self.phase == "module":
```

SAME RULE AS THE SUCCESS PATH: only a module was reached through a door, so only a module's record carries one. A refusal for Home or Preferences stamped with the last module's door would be a field that is confidently wrong.

### lines 541-542

```python
self._finish("Home never became interactive")
```

Without a usable Home, no click path exists to benchmark.  Do not disguise that by calling private factories instead.

### lines 551-554

```python
self.current_key = None
```

The screen can finish painting after its overdue timeout has been delivered.  Clear the key before the next settled advance so that late readiness cannot terminate this attempt a second time and skip the following registry row.
