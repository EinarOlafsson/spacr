# Notes from `spacr/qt/mask_engine.py`

## Item 419 point 1 (2026-09-19)

- `canonical_labels`: each id is now labelled inside its own bounding box
  (`scipy.ndimage.find_objects`) rather than across the whole field. The
  pieces are the same and come in the same order, because a crop keeps the
  raster order of the pixels in it and holds every pixel of that id, so the
  kept piece and the minted ids are unchanged; the test
  `test_canonical_labels_is_unchanged_by_the_bounding_box_rewrite` compares
  60 random masks (uint8, uint16, int32; binary, split and touching ids)
  against the whole-field algorithm it replaced. Measured on this machine:
  2048 x 2048 with 400 objects, 3.1 s -> 64 ms; 1024 x 1024 with 200, 382 ms
  -> 15 ms. `filter_objects` and `save_mask` go through it, so both got the
  same speed-up. The int64 copy is only made once a split is found. Masks
  with an id past uint16, and non-integer masks, keep the old whole-field
  path, so their behaviour (including the ValueError) is exactly as before.
- `ObjectLookup.measure`: the mean is `np.mean` of the float32 image over
  the object's pixels taken from its bounding box, which is the arithmetic
  `regionprops(...).intensity_mean` performs for `filter_objects`
  (`image_intensity[image]`, float32, raster order), so the two are equal to
  the last bit rather than approximately.

## Item 419 points 5 and 6 (2026-09-19)

- `dilate_objects`: `skimage.segmentation.expand_labels`, which grows each
  label into background only and gives a contested pixel to the nearer
  label. So no two objects can fuse and the object count cannot change,
  which is what makes it safe on a curated mask: an id that has been got
  right keeps every measurement, track and crop keyed to it.
- `shrink_objects`: each object is eroded against everything that is not
  itself -- background AND its neighbours -- so a pair that was touching
  comes apart. Eroding `mask > 0` as one binary would leave that seam
  untouched, which is the opposite of what Shrink is reached for. It works
  inside each object's own bounding box (`find_objects`), for the same
  reason `canonical_labels` does: cost follows the area of the objects, not
  the area of the field. The distance is Euclidean, matching
  `expand_labels`, so a Shrink undoes a Dilate of the same size on an object
  that had room to grow -- asserted in
  `tests/qt/test_make_masks_shortcuts_otsu_and_object_edits.py`. An object
  thinner than twice the step disappears, which is what erosion means; the
  screen counts what went and says so.
- `_split_touching_objects`: the watershed tail `_classical_region_labels`
  has always ended with, lifted out whole so `_otsu_instances` can reach the
  same recipe. Before this the magnifier's threshold mode split touching
  objects and the Otsu detect button did not, on the same field, and nothing
  said so. `_drop_border_objects` is the new "exclude border" step beside it.
- `_otsu_instances` and `_classical_region_labels` grew keyword-only
  settings (`smoothing`, `fill_holes`, `split_touching`, `exclude_border`)
  and EVERY default reproduces what each did before, so the two functions
  return the same arrays for the calls that already existed: 40 random
  fields x 6 sensitivities x 2 corrections x both sides hash identically
  against `origin/nightly` (`b56c77d46`). The panel, not the engine, is
  where the defaults moved.

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
