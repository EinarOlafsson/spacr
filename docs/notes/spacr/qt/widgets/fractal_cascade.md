# Notes from `spacr/qt/widgets/fractal_cascade.py`

Prose lifted out of `spacr/qt/widgets/fractal_cascade.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_sample](#_sample) (1 entry)
- [render_into](#render_into) (1 entry)

## _sample

### lines 333-335

```python
ring_base = 1.0 / (1.0 + 20.0 * trap_ring)
```

Rational trap profiles: four exponentials per subpixel is the single most expensive thing this kernel could do, and these stay smooth and bounded for a fraction of it.

## render_into

### lines 393-396

```python
toward_x = pointer_x * (pull * 0.30 - push * 0.55)
```

THE POINTER MOVES THE CAMERA, not the field. Folding it into the translation below draws the fold toward the cursor and shoves it away on a click, without a second warp term fighting the one the pattern already has.
