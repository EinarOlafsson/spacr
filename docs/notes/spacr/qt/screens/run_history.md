# Notes from `spacr/qt/screens/run_history.py`

Prose lifted out of `spacr/qt/screens/run_history.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_bytes](#_bytes) (1 entry)
- [Module level](#module-level) (1 entry)
- [RunHistoryScreen.__init__](#runhistoryscreen__init__) (1 entry)
- [RunHistoryScreen._build_ui](#runhistoryscreen_build_ui) (1 entry)

## _bytes

### lines 97-98

```python
for unit in units[:-1]:
```

The largest unit is left out of the loop and answered below: with it in, the loop always returns and the line after it can never run.

## Module level

### lines 124-125

```python
register_widget_qss(TABS_NAME, _tabs_qss, replace=True)
```

``replace=True``: this module owns the name, so a reimport re-registers rather than raising and leaving the tabs unstyled.

## RunHistoryScreen.__init__

### lines 161-162  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

## RunHistoryScreen._build_ui

### lines 226-234

```python
self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
```

SEVERAL ROWS AT A TIME. Asked for on 2026-08-31: "the ability to select more than one run and right click and delete or open". Deleting runs one at a time is the operation nobody performs what a full disk actually needs is forty of them gone at once.

The detail panes below still describe ONE run (the current row), because "the settings of these six runs" is not a thing a form can show. Extending the selection changes what the ACTIONS operate on, not what is displayed.
