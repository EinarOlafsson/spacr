# Notes from `spacr/qt/screens/image_import.py`

Prose lifted out of `spacr/qt/screens/image_import.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ImageImportScreen.__init__](#imageimportscreen__init__) (2 entries)
- [ImageImportScreen._build_ui](#imageimportscreen_build_ui) (7 entries)
- [ImageImportScreen._run_job](#imageimportscreen_run_job) (1 entry)

## ImageImportScreen.__init__

### lines 414-416

```python
from ..dnd import install_dropzone
```

A DROPPED FOLDER IS THE GESTURE THIS MODULE IS FOR. The handler also takes a saved plan, so last week's answers arrive the same way this week's images do.

### lines 425-427

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## ImageImportScreen._build_ui

### lines 434-435

```python
"""Lay out the source row, the scan options, the proposal, the questions and the report."""
```

ITS OWN REGISTRY KEY -- what `install_folds_on` and the drop handlers dispatch on.

### lines 438-440

```python
from .app_screen import ModuleHeader
```

IMPORTED HERE, NOT AT MODULE LEVEL: `app_screen` imports the screen registry, which reaches this module, and a top-level import would close that circle at startup rather than at first build.

### line 468  _(unsure)_

```python
src_row = QHBoxLayout()
```

── Source ────────────────────────────────────────────────────

### line 483  _(unsure)_

```python
opt_row = QHBoxLayout()
```

── Scan options ──────────────────────────────────────────────

### lines 515-516  _(unsure)_

```python
install_sorting(self._table)
```

After setModel: the helper puts a sorting proxy over the model, which replaces the view's model and its selection model.

### line 525  _(unsure)_

```python
self._answers = AnswerModel(self)
```

── The questions ─────────────────────────────────────────────

### line 548  _(unsure)_

```python
dst_row = QHBoxLayout()
```

── Destination + actions ─────────────────────────────────────

## ImageImportScreen._run_job

### lines 1163-1165

```python
self._jobs.append((thread, worker))
```

Strong references: PySide6 will not keep the worker alive through the started→run connection alone, and a QThread garbage-collected while still running takes the process down with it.
