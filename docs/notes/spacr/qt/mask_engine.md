# Notes from `spacr/qt/mask_engine.py`

Prose lifted out of `spacr/qt/mask_engine.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [canonical_labels](#canonical_labels) (1 entry)
- [paint_disk](#paint_disk) (1 entry)
- [Module level](#module-level) (3 entries)
- [fill_polygon](#fill_polygon) (1 entry)
- [_segment_band](#_segment_band) (1 entry)
- [divide_object](#divide_object) (2 entries)
- [filter_objects](#filter_objects) (2 entries)
- [combine_masks](#combine_masks) (1 entry)
- [magic_wand](#magic_wand) (1 entry)
- [MaskHistory](#maskhistory) (1 entry)

## canonical_labels

### lines 215-217

```python
raise ValueError(
```

Wrapping would fuse object 65536 with object 0 — background — and lose it silently. A mask is uint16 everywhere in spaCR, so this is a mask that cannot be written, not one to truncate.

## paint_disk

### lines 293-295  _(unsure)_

```python
def paint_disk(mask: np.ndarray, cx: int, cy: int, radius: int,
```

Mask edits — brush / erase / object-level ops

## Module level

### lines 333-335  _(unsure)_

```python
DIVIDE_CUT_WIDTH = 1.5
```

Region tools — the free-form outline and the dividing line

### lines 741-743  _(unsure)_

```python
VISIT_BUDGET_FACTOR = 4
```

Magic wand — flood-fill by intensity tolerance (mirrors ModifyMaskApp)

### lines 902-932

```python
RECROP_MIN_SIDE = 32
```

Recrop — cutting one field into the several fields it should have been

Every other tool in this module edits the mask on the field in view. Recrop is the one that changes WHICH field is in view: a staged crop that holds several cells, wells or plaques is not one training example, and curating it as though it were teaches the network that two objects are one picture. So the user boxes each one, every box becomes a field of its own carrying that region of BOTH the image and the draft mask, and the multi-object original is retired rather than curated.

WHAT spaCR CAN AND CANNOT RETIRE. The Make Masks queue is a FOLDER: :func:`list_images` sorts the image files in it and the screen walks that list, with each mask at ``<folder>/masks/<stem>.tif``. spaCR does have a crop DATABASE -- ``png_list`` in ``measurements.db``, which is what the Annotate app and the classifiers read -- but it is keyed on each crop's absolute ``png_path`` and carries no lifecycle column: there is no field in it that can be set to "recropped", and no row that a screen reading a folder has any claim to rewrite. So the original CANNOT be marked retired in spaCR's database the way the standalone marks it in its status CSV.

The nearest thing that is recoverable, and what these functions do, is to move the original out of the enumeration and leave every byte of it on disk: image, mask and curation ledger go into ``<folder>/recropped_originals/`` (the mask keeping its ``masks/`` sub-layout), which :func:`list_images` does not descend into, and :data:`RECROP_MANIFEST` inside that folder records what was moved, which boxes were cut out of it and what the children were called. A recrop drawn wrong is undone by moving two files back; a dataset registered in ``png_list`` can be repointed from the manifest rather than from a guess.

## fill_polygon

### lines 402-404

```python
x, y = pts[:, 0], pts[:, 1]
```

Shoelace area of the closed path. skimage's polygon() hands back the traced pixels themselves for a degenerate outline, which would make a straight drag into a hairline "object".

## _segment_band

### lines 430-431

```python
lo_x = max(0, int(np.floor(min(x0, x1) - half)))
```

Only the segment's bounding box can be within half a width of it, so the distance is computed there instead of over the whole field.

## divide_object

### line 489, trailing  _(unsure)_

```python
continue
```

the line stopped short: not a cut

### line 492, trailing  _(unsure)_

```python
out[body] = 0
```

drop the cut pixels with the rest

## filter_objects

### lines 634-637

```python
labels = canonical_labels(mask)
```

Measured on the canonical labelling, not on the raw array: two separate blobs a brush painted with the same value are one region to regionprops, and their combined area and mean intensity describe neither of them.

### lines 642-644

```python
mean = float(region.intensity_mean
```

scikit-image renamed mean_intensity to intensity_mean and warns on the old spelling; both names are live across the versions spaCR supports, so ask for the new one and fall back.

## combine_masks

### lines 730-732

```python
top = int(out.max()) if out.size else 0
```

Width follows the values, as it does everywhere else a mask is made here: merging 300 detected objects into a uint8 mask and keeping uint8 would wrap object 300 round to 44 and silently fuse it with another.

## magic_wand

### lines 793-809

```python
examined = 0
```

A SECOND BUDGET, ON WORK RATHER THAN ON CHANGES.

`added` counts only pixels that CHANGE state, which is the budget a user thinks in -- "fill at most this many". But a flood that changes nothing never increments it, so `added < max_pixels` stayed true forever and the search walked the entire frame. Erasing where the mask is already empty, or adding over ground the mask already owns, is the most ordinary wrong click there is: measured at 4.8 s on an 800x800 field with max_pixels=100, and roughly half a minute at 2048x2048, with the GUI unresponsive and no way to cancel.

So visits are bounded too. The multiplier is generous on purpose a legitimate fill examines its region AND the out-of-tolerance perimeter around it, and a thin structure can have as much perimeter as area -- so this stops the pathological case without shortening any fill a user would recognise. The floor keeps small budgets workable, since a max_pixels of 10 still needs room to look around.

## MaskHistory

### lines 839-841

```python
class MaskHistory:
```

Undo history — small bounded ring of mask snapshots
