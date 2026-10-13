# Notes from `spacr/qt/settings_pack.py`

Prose lifted out of `spacr/qt/settings_pack.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## settings_from_pack

### lines 202-215

```python
for survivor in _package_renames(key):
```

THE PACKAGE'S OWN RENAME TABLE, consulted after this app's. `PACK_RENAMES` is curated per app and is empty for both of them, which used to mean a pack written before 391 lost every key that instruction renamed: `cell_FT`, `cell_CP_prob` and their nucleus and pathogen siblings were reported as DROPPED while `spacr.settings` knew exactly what each had become. Measured 2026-09-13 on a four-key pack: applied 2, renamed 0, dropped 4, and all four resolvable.

`surviving_setting_name` follows a rename CHAIN, so a key renamed twice still lands. It can return more than one name where a setting was split; a value cannot be sent to two places without inventing a meaning for it, so that case is left to `PACK_RENAMES`, which can say what was intended.

### lines 225-228

```python
settings["src"] = str(src)
```

LAST, and unconditionally. The pack's own `src` is a path on the machine that produced it; applying it would point the run at a folder that is not there, which fails much later and blames the dataset rather than the pack.


---

# Notes from `spacr/qt/settings_pack.py`

Prose lifted out of `spacr/qt/settings_pack.py` by `tools/extract_source_notes.py`.
Ordinary comments move here; tool directives and published attribute documentation stay in the module. The path mirrors the source path, which is how its reasons are found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_defaults_for_pack_shape](#_defaults_for_pack_shape) (4 entries)
- [_classes_from_pack](#_classes_from_pack) (1 entry)
- [settings_from_pack](#settings_from_pack) (1 entry)

## Module level

### lines 68-69

```python
_FORM_RENAMES = {"png_dims": "png_channel_mapping"}
```

Shared with ordinary GUI CSV imports; these form names remain accepted legacy pipeline inputs and therefore are not globally retired settings.

## _defaults_for_pack_shape

### lines 368-369  _(unsure)_

```python
slot_values[target] = raw[target] if target in raw else value
```

An explicit current spelling wins independently of CSV order, including the primary values cloned into newly revealed slots.

### line 379  _(unsure)_

```python
settings[NUMBER_OF_ORGANELLES] = organelle_count(deciding)
```

Infer from the incoming slots, not a default count of zero.

### lines 384-385

```python
count = len(declared_organelle_roles(deciding))
```

Declared slots include saved values above an explicitly lowered count; expanding their schema must not raise that active count again.

### lines 387-388  _(unsure)_

```python
settings.update(slot_values)
```

Newly revealed slots inherit the pack's primary values, just as the pipeline's defaults do. Existing/supplied secondary values still win.

## _classes_from_pack

### lines 401-402  _(unsure)_

```python
context = dict(raw)
```

Match the form's rename-then-compound order without importing a screen. A current spelling wins when an old and current key coexist.

## settings_from_pack

### line 477  _(unsure)_

```python
value = classes
```

An explicitly empty Classes value does not erase the migration.
