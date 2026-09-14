# Notes from `spacr/qt/screens/agreement.py`

Prose lifted out of `spacr/qt/screens/agreement.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [AgreementScreen.__init__](#agreementscreen__init__) (3 entries)
- [AgreementScreen._build_ui](#agreementscreen_build_ui) (2 entries)
- [AgreementScreen.set_database](#agreementscreenset_database) (2 entries)
- [AgreementScreen._on_pair_row_changed](#agreementscreen_on_pair_row_changed) (1 entry)

## AgreementScreen.__init__

### line 163, trailing  _(unsure)_

```python
self._disagreements = None
```

pd.DataFrame | None

### lines 165-167

```python
self._jobs: List[tuple] = []
```

Ownership list for in-flight (QThread, worker) pairs — a QThread collected while still running takes the process down with it. Same idiom as DbBrowserScreen._jobs.

### lines 183-185

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## AgreementScreen._build_ui

### line 213  _(unsure)_

```python
src_row = QHBoxLayout()
```

── Source row ────────────────────────────────────────────────

### line 233  _(unsure)_

```python
split = QSplitter(Qt.Horizontal, self)
```

── Annotators | results ──────────────────────────────────────

## AgreementScreen.set_database

### lines 445-446

```python
self._set_status(
```

Not a crash and not a dialog: one column is a legitimate state, it just cannot produce an agreement number.

### lines 456-457  _(unsure)_

```python
for i in range(min(2, self._columns_list.count())):
```

Two candidates is the common case — tick them so Compute works on the first click.

## AgreementScreen._on_pair_row_changed

### line 646, trailing  _(unsure)_

```python
self._pair_combo.setCurrentIndex(row)
```

fires _show_confusion
