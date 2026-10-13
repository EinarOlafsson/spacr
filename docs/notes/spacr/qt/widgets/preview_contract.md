# Notes from `spacr/qt/widgets/preview_contract.py`

Prose lifted out of `spacr/qt/widgets/preview_contract.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [preview_cellpose_model](#preview_cellpose_model) (1 entry)
- [LivePreviewContract.preview_running](#livepreviewcontractpreview_running) (1 entry)
- [LivePreviewContract.display_primaries](#livepreviewcontractdisplay_primaries) (1 entry)
- [LivePreviewContract._extra_work_in_flight](#livepreviewcontract_extra_work_in_flight) (1 entry)

## preview_cellpose_model

### line 173, trailing  _(unsure)_

```python
if gpu is not None:
```

an explicit caller still wins

## LivePreviewContract.preview_running

### line 234  _(unsure)_

```python
return False
```

The C++ side is already gone; nothing is in flight.

## LivePreviewContract.display_primaries

### lines 260-261

```python
return "rgb"
```

No QSettings, no Qt: the untransformed image is the honest answer, and never worse than failing to draw one.

## LivePreviewContract._extra_work_in_flight

### line 348  _(unsure)_

```python
def _extra_work_in_flight(self) -> bool:
```

optional: panels whose work is not a QThread
