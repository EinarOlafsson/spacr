# Notes from `spacr/curation.py`

Prose lifted out of `spacr/curation.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [is_curated](#is_curated) (1 entry)
- [MaskCuration.__init__](#maskcuration__init__) (1 entry)
- [MaskCuration._record](#maskcuration_record) (1 entry)
- [MaskCuration.paint](#maskcurationpaint) (1 entry)
- [MaskCuration.undo](#maskcurationundo) (1 entry)

## is_curated

### lines 166-167  _(unsure)_

```python
return True
```

An unreadable ledger is a reason to be suspicious, not a reason to certify the data as raw.

## MaskCuration.__init__

### lines 512-517

```python
self._listeners: List[Any] = []
```

Views that want to redraw when the LEDGER moves. The layer's own subscribers fire per dab, which is mid-stroke -- a panel listening only to those redraws before end_stroke has recorded anything, and so never shows the entry it is there to show. Bound methods only: a session outlives nothing here, but a lambda would keep a closed panel alive as a receiver.

## MaskCuration._record

### lines 571-572

```python
pass
```

One view's redraw must not take the correction with it the edit has already happened to the data.

## MaskCuration.paint

### lines 663-664  _(unsure)_

```python
self._open = [edit]
```

A bare dab is its own stroke, so it is undoable and recorded like any other.

## MaskCuration.undo

### lines 694-696

```python
for edit in reversed(stroke):
```

Newest dab first: two dabs that overlapped must be reverted in the reverse of the order they were laid down, or the older one's "before" values overwrite the newer one's.
