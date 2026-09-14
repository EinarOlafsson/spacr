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
