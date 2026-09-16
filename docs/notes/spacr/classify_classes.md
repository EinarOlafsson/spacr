# Notes from `spacr/classify_classes.py`

Prose lifted out of `spacr/classify_classes.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [class_rules](#class_rules) (1 entry)
- [class_names](#class_names) (1 entry)
- [folder_names](#folder_names) (3 entries)
- [_rules_from_annotation](#_rules_from_annotation) (2 entries)
- [_rules_from_metadata](#_rules_from_metadata) (2 entries)
- [normalize_settings](#normalize_settings) (4 entries)
- [assign_classes](#assign_classes) (2 entries)
- [annotation_column_of](#annotation_column_of) (1 entry)

## class_rules

### lines 182-193

```python
names = ", ".join(repr(str(n)) for n in raw)
```

A plain list of names is the OLD shape. It says nothing about which objects belong to which class, so it cannot be turned into rules here -- `normalize_settings` translates it, using the other retired keys, before anything asks.

It does NOT always translate it, and that is why this message says more than "run normalize_settings first". With no basis to derive rules from, normalize_settings deliberately leaves the names alone rather than guessing a column -- so the SHIPPED defaults (classes=['nc','pc'], nothing bound to a column) arrive here having already been normalized, and the old message sent the user to do the one thing they had just done.

## class_names

### lines 238-241

```python
return [str(n) for n in raw]
```

The old shape, either untranslated or left alone because nothing said what its names select. Reading the names off it is always right; `class_rules` refuses it because it cannot make RULES from names, which is a different question.

## folder_names

### lines 268-278

```python
return [str(n) for n in legacy]
```

THE SHAPE OF `classes` SAYS WHICH FILE THIS IS. A list is the pre-split spelling, and in that file `classes` IS the folder list so it decides, including when it is empty, which is a caller saying "this run has no classes" and must keep aborting the run rather than picking up a default.

It has to win over `class_folder_names` rather than lose to it, because the defaults factories inject that key into every settings dict they touch. Losing would mean every settings file ever written silently trained on ['nc','pc'] instead of its own classes, which is the opposite of what the split is for.

### lines 281-287

```python
if isinstance(settings.get(CLASSES), Mapping) and settings.get(CLASSES):
```

Current class definitions take precedence over the recorded output of a previous dataset-generation run.

Only when classes are actually DEFINED. An empty definition means "this settings file says nothing about classes", and must not shadow a recorded folder list -- that would break every settings file written before the Classes editor existed.

### lines 302-303  _(unsure)_

```python
return []
```

A malformed definition is not a reason to refuse to name the folders; whoever needs the rules will raise on its own terms.

## _rules_from_annotation

### lines 337-343

```python
names = folder_names(settings)
```

The names come from `folder_names`, which preserves a pre-split, list-shaped `classes` before consulting `class_folder_names`. Reading `classes` directly stopped working the moment it became the definitions dict: a settings file carrying the retired keys alongside the new default named its derived classes 'negative control' / 'positive control' instead of the names sitting right beside them.

### lines 349-350  _(unsure)_

```python
column = columns[i] if i < len(columns) else columns[0]
```

One column and several values is the common case; several columns pairs them off positionally, which is what the old readers did.

## _rules_from_metadata

### lines 377-382

```python
name = (names[i] if i < len(names)
```

STRIP THE IDENTIFIER SUFFIX BEFORE SHOWING IT. These two settings were renamed to negative_control_id / positive_control_id, and this fallback is a DISPLAY name -- de-underscoring the key verbatim put "negative control id" on a user's class and on the folder written for it. The `_id` says what the setting holds, not what the class is called.

### line 385  _(unsure)_

```python
if isinstance(value, (list, tuple)):
```

A control setting can name several wells.

## normalize_settings

### lines 414-419

```python
if not isinstance(raw, Mapping) or not raw:
```

`not raw` as well as `not isinstance(...)`: the default is now an EMPTY dict meaning "nothing defined yet", and an empty Mapping is still a Mapping -- so testing the type alone would skip the derivation for exactly the settings that need it most, and a plate with `class_metadata` set would train on no classes at all while reporting nothing. Empty means undefined, whichever shape it is empty in.

### lines 427-429

```python
LOG.info("settings name %d class(es) but nothing says which "
```

Names with nothing saying what they select. Left alone rather than invented: a guessed column would train on the wrong labels and report success.

### lines 436-438

```python
names = legacy_names
```

Defaults inject class_folder_names=['nc', 'pc']; it is not evidence that a pre-split file chose those names. Carry the legacy list across the shape migration explicitly, including [] (which means stop).

### lines 447-448  _(unsure)_

```python
out["class_names"] = names
```

`class_names` is retained for older downstream readers, but is always synchronized with the one folder-name contract.

## assign_classes

### lines 480-484

```python
import pandas as pd
```

IMPORTED HERE, NOT AT MODULE SCOPE. Everything else in this file is an annotation, and `from __future__ import annotations` makes those strings -- so a module-level import cost 0.30 s to load pandas for two lines that only run when classes are actually assigned. The Home page reaches this module through the class editor, so every launch paid it.

### lines 498-500

```python
take = hit & ~claimed
```

First rule wins. Two rules matching the same object is a definition the user has to fix, but silently relabelling is worse than keeping the order they wrote.

## annotation_column_of

### lines 522-524  _(unsure)_

```python
def annotation_column_of(settings) -> str:
```

Compatibility values derived from class definitions
