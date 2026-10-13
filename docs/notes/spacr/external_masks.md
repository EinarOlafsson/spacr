# Notes from `spacr/external_masks.py`

Prose lifted out of `spacr/external_masks.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_suggest_object](#_suggest_object) (1 entry)
- [_label_likelihood](#_label_likelihood) (2 entries)
- [default_settings](#default_settings) (1 entry)
- [register_settings](#register_settings) (2 entries)

## _suggest_object

### lines 286-287

```python
stem = cv._split_ext(str(name))[0]
```

Include parent folders: externally generated masks are commonly named ``cell_masks/fov001.tif`` rather than ``fov001_cell_mask.tif``.

## _label_likelihood

### line 307  _(unsure)_

```python
stride = max(int(np.sqrt(array.size / 250_000)), 1)
```

Cap the sample without changing its value distribution materially.

### lines 312-315

```python
compact = len(values) <= max(64, int(sampled.size * 0.002))
```

A normal 8-bit microscopy image often contains all 256 grey values, whereas a label image contains roughly one value per object. Permit large fields with many objects without calling ordinary 8-bit data a mask merely because its value range is bounded.

## default_settings

### lines 509-510  _(unsure)_

```python
"channels": [],
```

Empty means all imported intensity channels. This avoids Measure's four-channel default being invalid for a one- or two-channel import.

## register_settings

### lines 878-885

```python
"z_handling":
```

SHARED WITH `spacr.convert`, WHICH HAS A DIFFERENT DEFAULT. `register_defaults` refuses a second module's differing text for one key, so this string is the only description either consumer gets and has to be true of both. It used to describe External Masks alone: it named 'max' and 'first', called 'max' the default, and never mentioned 'keep' -- while `convert_folder` DEFAULTS to 'keep' and accepts it. A reader of Format Convert was told the wrong default and not told one of the three values existed.

### lines 910-911

```python
tips = {key: value for key, value in tips.items()
```

A future shared importer may establish canonical prose first. Preserve it instead of making module import order decide which wording wins.
