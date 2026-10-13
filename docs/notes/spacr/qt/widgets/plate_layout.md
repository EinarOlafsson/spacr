# Notes from `spacr/qt/widgets/plate_layout.py`

Prose lifted out of `spacr/qt/widgets/plate_layout.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_well_order](#_well_order) (1 entry)
- [to_settings_fragment](#to_settings_fragment) (1 entry)

## _well_order

### lines 209-211

```python
random.Random(int(design.seed)).shuffle(wells)
```

A dedicated Random rather than numpy: this is a shuffle of at most 1536 tuples, and seeding a local instance cannot disturb any other stream in the process.

## to_settings_fragment

### lines 471-478

```python
for role, setting in ((ROLE_POSITIVE, "positive_control_id"),
```

THE ROLE AND THE SETTING ARE TWO DIFFERENT STRINGS NOW, which is what this line was always saying and could not show while they were spelled the same. `ROLE_POSITIVE` is a Qt WIDGET ROLE -- the token `experiment_design`'s stylesheet selects a well by -- and the second element is the SETTING the design writes. Renaming the setting to `positive_control_id` (364) left the role alone; a blanket replace would have taken both, and the wells would have lost their colour with no error anywhere.
