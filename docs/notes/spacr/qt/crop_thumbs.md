# Notes from `spacr/qt/crop_thumbs.py`

Prose lifted out of `spacr/qt/crop_thumbs.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [crop_paths_for_keys.resolve](#crop_paths_for_keysresolve) (1 entry)

## Module level

### lines 61-63

```python
_LIVE_CACHES: "weakref.WeakSet[CropThumbnails]" = weakref.WeakSet()
```

Weak registration makes every live thumbnail cache observable to the process-wide budget without making the cleanup service the reason a closed screen stays alive.

## crop_paths_for_keys.resolve

### lines 329-331

```python
"""Resolve one batch of keys, bisecting when some are missing.
```

Never called with an empty batch: the caller has already refused an empty key list and a bisection of two or more keys cannot produce an empty half.
