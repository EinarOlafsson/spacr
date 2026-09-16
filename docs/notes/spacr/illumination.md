# Notes from `spacr/illumination.py`

Prose lifted out of `spacr/illumination.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_read_binned_field](#_read_binned_field) (1 entry)
- [estimate_illumination](#estimate_illumination) (2 entries)
- [IlluminationCorrector._to_input_dtype](#illuminationcorrector_to_input_dtype) (1 entry)
- [SegmentationIlluminationSession.__init__](#segmentationilluminationsession__init__) (1 entry)
- [SegmentationIlluminationSession.mark_completed](#segmentationilluminationsessionmark_completed) (1 entry)
- [SegmentationIlluminationSession._record_stage](#segmentationilluminationsession_record_stage) (1 entry)
- [_env_entries](#_env_entries) (1 entry)
- [enable_illumination_correction](#enable_illumination_correction) (2 entries)
- [_write_qc_figure](#_write_qc_figure) (1 entry)
- [prepare_illumination_model](#prepare_illumination_model) (2 entries)

## _read_binned_field

### lines 663-665

```python
plane = np.median(plane, axis=0)
```

A z-stack is corrected with one 2-D field: the illumination is a property of the optics in x/y. The median over z is what the estimate sees, so an out-of-focus slice does not dominate it.

## estimate_illumination

### lines 760-763

```python
skipped += 1
```

Mixed field sizes in one folder: correcting a 512x512 field with a 1024x1024 gain map is not a thing that can be made to mean anything, so those fields sit the estimate out and the corrector refuses them later by shape.

### line 772, trailing  _(unsure)_

```python
stack = np.stack(stack, axis=0)
```

(K, C, y, x)

## IlluminationCorrector._to_input_dtype

### lines 958-959

```python
lost = int(np.count_nonzero((rounded > info.max) &
```

Only signal this correction pushed out of range counts: a pixel the microscope had already saturated was never recoverable.

## SegmentationIlluminationSession.__init__

### lines 1428-1430

```python
self._write_provenance(self._completed_fields, 'prepared')
```

A fresh preprocessing run invalidates any previous completion claim immediately, but the explicit state says no field has yet been corrected or made durable.

## SegmentationIlluminationSession.mark_completed

### lines 1485-1486

```python
self._write_provenance(completed, 'running')
```

Assign only after os.replace succeeds: the in-memory state must not claim durability that the filesystem refused to record.

## SegmentationIlluminationSession._record_stage

### line 1597

```python
return
```

Provenance must not replace a scientific result or its error.

## _env_entries

### lines 1601-1603  _(unsure)_

```python
def _env_entries(value: str) -> list:
```

Enabling it -- including in worker processes

## enable_illumination_correction

### line 1663, trailing  _(unsure)_

```python
IlluminationModel.load(model_path)
```

fail here, not in a worker

### lines 1680-1684

```python
registered = [entry.name for entry in preprocessing_hooks()]
```

Consulting the registry runs the environment installers, which is how this process ends up with a hook tagged 'env' -- the same tag a worker gets, and the one measure_crop's start-method warning knows not to shout about. If the variable was already read in this process (it is read once) that does nothing, so fall back to installing directly.

## _write_qc_figure

### lines 1977-1985

```python
from .plot import save_figure
```

108 point 6: through the one writer for the resolution rule and the repaint for paper -- but `fmt` STAYS PNG. This path is RETURNED and recorded in the QC metrics under a name ending `.png`, and a format preference that renamed it would rename a value other code reads back.

THE PATTERN, since this is the fourth: routing a save through `save_figure` always gains the DPI and the paper repaint; the FORMAT follows the preference only where nothing depends on the filename. A figure whose name is part of a contract keeps its extension.

## prepare_illumination_model

### lines 2014-2016  _(unsure)_

```python
def prepare_illumination_model(
```

Settings-driven preparation and stage entry points

### lines 2069-2072

```python
model_path = os.path.join(folder, 'illumination_model.npz')
```

Keep Measure's established failure boundary: QC runs against the in-memory estimate, and only a successful QC leaves a reusable model on disk. ``enable_illumination_correction`` used to perform this save after QC; the stage-neutral preparer preserves that ordering.
