# Notes from `spacr/image_import.py`

Prose lifted out of `spacr/image_import.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_resolve_marker](#_resolve_marker) (1 entry)
- [infer_layout](#infer_layout) (7 entries)
- [read_axes_inside](#read_axes_inside) (3 entries)
- [ImportPlan.files](#importplanfiles) (1 entry)
- [_stitch_fields](#_stitch_fields) (2 entries)
- [apply_import](#apply_import) (3 entries)

## Module level

### lines 51-57

```python
"ImportResult",
```

`apply_import` RETURNS THIS and it was the one sibling type left out. Undocumented, Sphinx could not resolve the bare `ImportResult` in that function's signature against this module and searched every other one instead, finding `spacr.foreign.ImportResult` and `spacr.omero.ImportResult` -- "more than one target found", which `sphinx-build -W` makes fatal. A reader following the return type would have landed on a different importer's result object.

## _resolve_marker

### lines 199-201

```python
partner = {"column": "r", "well": "f", "z": "ch"}.get(first, "")
```

`c` is a column only when an `r` sits beside it, as in Opera's r01c01; otherwise it is a channel. Same shape of question for `w` (a well in CQ1's W1F001, a wavelength in ImageXpress's _w1) and `p`.

## infer_layout

### lines 233-236

```python
tokenised = {}
```

The FOLDER segments are part of the name. A tree that puts the well in a directory encodes exactly as much as one that puts it in the filename, and reading only the basename is why per-well and per-channel trees recover nothing today.

### lines 249-250  _(unsure)_

```python
slots: Dict[int, TokenSlot] = {}
```

Collect what each digit slot takes across the folder, with the alphabetic run in front of it as its marker.

### line 263, trailing  _(unsure)_

```python
continue
```

a constant identifies nothing

### lines 269-279

```python
alpha_values: Dict[int, List[str]] = defaultdict(list)
```

A WELL LETTER IS ONE THAT VARIES. `A01` is a well name; so, by shape alone, are `L01`, `Z01` and `C01` in `plate1_A01_T0001F001L01A01Z01C01` -- and matching on shape marked all four as wells, the last of them overwriting the channel axis. Every plate then had four wells and no channels.

The letter is what separates them: `A` takes A and B across the folder, while `L`, `Z` and `C` are the same letter in every file. A constant letter is part of the convention's punctuation; a varying one is an axis. This is the same variance rule the digit slots use, applied to the half of the name the first cut did not apply it to.

### line 287, trailing  _(unsure)_

```python
continue
```

constant: punctuation, not an axis

### lines 289-291

```python
layout.unplaced[i] = list(dict.fromkeys(values))
```

A varying word rather than a letter -- a dye name in a folder, say. It IS an axis, and one nothing here can name, so it is reported rather than guessed at.

### line 311  _(unsure)_

```python
found["well"] = f"{chr(ord('A') + int(found['row']) - 1)}" \
```

Opera keeps them apart; spaCR's vocabulary is a well name.

## read_axes_inside

### line 370  _(unsure)_

```python
return InsideFile(pages=1 if path.is_file() else 0)
```

Only TIFF carries this. A PNG or JPEG is one plane by construction.

### lines 380-388

```python
return InsideFile(pages=pages, axes=axes, sizes=sizes,
```

DECLARED MEANS A NAMED NON-SPATIAL AXIS WAS FOUND, and nothing weaker. The first version accepted "any axis letter that is not Y, X or S", which let tifffile's own `Q` through -- and `Q` is precisely tifffile's word for "these pages exist and I do not know what they are". An unlabelled three-page stack came back declared, which is the guess this function exists not to make.

is_ome and is_imagej are not sufficient either: a file can carry either container and still not say what its pages mean.

### lines 392-394

```python
return InsideFile(pages=0)
```

Truncated, unreadable, or not really a TIFF. Reported as unknown so the folder-level scan carries on and the caller can list what it could not read.

## ImportPlan.files

### lines 446-448

```python
entry[f"{axis}_count"] = size
```

A COUNT, not a position: the file holds `size` planes of this axis, which is a different fact from "this file is plane 3" and must not be written as one.

## _stitch_fields

### lines 791-794

```python
for _tile, rel in members:
```

NOT WRITTEN AND NOT GUESSED AT. A field whose tiles cannot be read is a field nobody can stitch, and writing one tile of it under the field's name would be a quarter of a field wearing the name of the whole.

### lines 800-804

```python
from .tiff_io import write_tiff
```

`write_tiff`, not `tifffile.imwrite`: a stitched mosaic can have three or four planes in its leading dimension, and tifffile guesses RGB for exactly that shape. The helper declares minisblack/contig so an intensity stack is not written as a colour image.

## apply_import

### lines 880-881  _(unsure)_

```python
tile_fields: Dict[Tuple[object, object, object], int] = {}
```

One field number per (field, tile) pair, assigned in a stable order so two runs of the same import produce the same names.

### lines 892-895

```python
entries = dict(plan.files)
```

THE TILES COME OUT OF THE LOOP FIRST, because a stitched field is one image made of several sources and the loop below writes one image per source. Everything else is untouched: a tree with no tile axis takes exactly the path it took before stitching existed.

### lines 915-918

```python
skipped[rel] = (f"would overwrite {name}, already written from "
```

TWO IMAGES WITH ONE CANONICAL NAME means an axis is missing:

they differ in something the plan did not capture. Overwriting would lose one silently, which is the failure this module exists to prevent, so both are reported instead.
