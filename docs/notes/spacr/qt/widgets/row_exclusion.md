# Notes from `spacr/qt/widgets/row_exclusion.py`

Prose lifted out of `spacr/qt/widgets/row_exclusion.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_ExclusionRuleRow.__init__](#_exclusionrulerow__init__) (1 entry)
- [RowExclusionEditor.__init__](#rowexclusioneditor__init__) (2 entries)
- [RowExclusionEditor.set_source](#rowexclusioneditorset_source) (1 entry)
- [RowExclusionEditor._refresh_values](#rowexclusioneditor_refresh_values) (2 entries)
- [RowExclusionEditor._run_pending_loads](#rowexclusioneditor_run_pending_loads) (1 entry)
- [RowExclusionEditor._source_paths](#rowexclusioneditor_source_paths) (1 entry)

## _ExclusionRuleRow.__init__

### line 292

```python
apply_close_mark(remove, tooltip="Remove this exclusion rule")
```

THE APPLICATION'S CLOSE MARK -- see `theme.apply_close_mark`.

## RowExclusionEditor.__init__

### lines 364-367

```python
self._schema_jobs = JobRunner(self, threaded=self._threaded,
```

Two runners, not one. `JobRunner.cancel` abandons *everything* that runner has in flight, and superseding a keystroke's value read must not also abandon the schema read that tells us which databases that column even lives in.

### line 376

```python
self._debounce.timeout.connect(self._run_pending_loads)
```

A bound method of a GUI-thread QObject, per job_runner's rules.

## RowExclusionEditor.set_source

### lines 428-429

```python
self._value_cache.clear()
```

Everything cached describes the *previous* source, and any read still in flight would repopulate it. Drop both.

## RowExclusionEditor._refresh_values

### lines 514-515

```python
row.values.set_options((), selected)
```

Not read yet. Show the selection alone rather than the previous column's values, which are now wrong, and queue the read.

### lines 521-523

```python
self._run_pending_loads()
```

`threaded=False` promises the values are there when the call that asked for them returns, so there is nothing to coalesce and nothing to wait a timer out for.

## RowExclusionEditor._run_pending_loads

### lines 541-543

```python
self._value_jobs.cancel()
```

Supersede: a read started by an earlier keystroke is for a column the user has already moved off. Its thread is asked to stop and its result is dropped on arrival.

## RowExclusionEditor._source_paths

### line 607  _(unsure)_

```python
@staticmethod
```

kept for callers that predate the module-level readers
