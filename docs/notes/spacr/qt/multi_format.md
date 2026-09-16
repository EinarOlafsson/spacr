# Notes from `spacr/qt/multi_format.py`

Prose lifted out of `spacr/qt/multi_format.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_describe_npz](#_describe_npz) (5 entries)
- [_describe_npy](#_describe_npy) (1 entry)
- [_describe_tif](#_describe_tif) (1 entry)

## _describe_npz

### lines 132-133  _(unsure)_

```python
n_fields = 1
```

Heuristics: (fields, H, W)  or  (fields, H, W, C)  or

(H, W, C)                      or  (H, W)

### line 141  _(unsure)_

```python
a, b, c = shape
```

3D — could be fields OR channels

### line 144  _(unsure)_

```python
n_fields, H, W = a, b, c
```

Fields first, then H, W (channels = 1)

### line 148  _(unsure)_

```python
img_shape = (int(a), int(b))
```

H, W, C

### line 153  _(unsure)_

```python
if len(keys) > 1:
```

Each named key inside the npz is often ONE field

## _describe_npy

### line 179

```python
arr = np.load(p, mmap_mode="r")
```

mmap_mode='r' → don't read the whole array into memory

## _describe_tif

### lines 222-226

```python
series = getattr(tf, "series", None)
```

Axis meaning: tifffile parses ImageJ / OME / plain TIFFs into series, and the series carries the axes string ("ZCYX", "TYX", "QYX" for an unlabelled stack). TiffFile itself has no `axes` / `ImageJ` / `OME` attributes — probing for those always came back empty, so the axis note was never emitted.
