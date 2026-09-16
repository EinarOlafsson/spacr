# Notes from `spacr/qt/settings_diff.py`

Prose lifted out of `spacr/qt/settings_diff.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [DiffRow](#diffrow) (1 entry)
- [_category_map](#_category_map) (1 entry)
- [_normalize](#_normalize) (1 entry)
- [SettingsDiffDialog.__new__](#settingsdiffdialog__new__) (3 entries)

## DiffRow

### line 65, trailing

```python
kind:  str
```

"added" / "removed" / "changed" / "same"

## _category_map

### lines 280-284

```python
mapping.setdefault(str(key), str(name))
```

First bucket wins. `tests/test_settings_categories.py` forbids a key appearing twice, but a plugin merging its own categories in is not covered by that test, and a silent overwrite would move the key under a heading the settings panel does not put it under.

## _normalize

### line 337  _(unsure)_

```python
if s.lower() in ("true", "false"):
```

Bool

## SettingsDiffDialog.__new__

### line 363  _(unsure)_

```python
"""Build and return the settings-diff dialog.
```

Lazy build of the Qt dialog when actually invoked in a GUI.

### lines 408-409  _(unsure)_

```python
colours = {
```

Colour palette per kind — inline styles so it works in both light/dark themes without extra QSS.

### lines 419-421

```python
item = table_item(text)
```

Built and coloured in one pass. Setting the four cells and then reading them back left a `table.item(...) is None` branch that could not happen and was never tested.
