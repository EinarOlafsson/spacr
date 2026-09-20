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

## Item 435 (2026-09-20)

- `invert_intensity`: the maintainer asked for "invert so that low intensity
  becomes high intensity and vice versa. 1/intensity i think and then fitted
  to dtype i guess". WHAT IS BUILT IS THE COMPLEMENT, `dtype_max - value`,
  and the reciprocal was considered and NOT built, so that nobody re-derives
  it: `1/value` divides by zero on every background pixel, it squashes the
  bright end so two objects a thousand counts apart come back
  indistinguishable, and it is not reversible, while the complement is what
  every image viewer means by Invert and returns the identical array when it
  is applied twice. He can still have the reciprocal as a SECOND mode if he
  wants a log-like lift of the dim end; it is not this one.
- The range complemented is the dtype's for integers (so `uint16` is
  `65535 - value`, which is "fitted to dtype") and the ARRAY's own for
  floats, which have no dtype maximum worth speaking of. The integer round
  trip is exact; the float one rounds twice and is out by up to one unit in
  the last place of `min + max` -- measured at 0.002 in float32 and 4e-12 in
  float64 over 0..65535, against an interval of one count. Every field this
  editor opens is an integer dtype.
- `_otsu_levels` / `_otsu_values` / `_otsu_histogram`: one reader for the
  numbers the histogram preview marks and the numbers `_otsu_instances`
  cuts at, because a preview that found its own level would be a second
  opinion and could be right while the button was wrong. The dark-side
  two-class level returned is the MIRRORED one, `top - (top - level) *
  correction`, which is the value the detector actually compares against --
  so the marker sits on the cut the user gets rather than on Otsu's own
  number, which is a different place.
- `_otsu_instances` grew `classes`, `foreground_class`, `local` and
  `window`, and the two-class non-local path was rewritten to read its level
  from `_otsu_levels`. That rewrite is byte-identical to what it replaced
  over 3,840 parameter combinations (40 fields x correction x smoothing x
  fill holes x split x exclude border x both sides), which is the check that
  says the shared reader did not move anybody's threshold.
- `_local_otsu_binary` measures its levels on a 256-step rescaling of the
  smoothed field. `skimage.filters.rank.otsu` accepts uint16, but a 16-bit
  rank filter builds a 65,536-bin histogram per pixel, which is not
  pressable on a megapixel field. The cut is made on the same rescaling, so
  nothing is compared across the two.
- `_square_footprint` tries `footprint_rectangle` and falls back to
  `square`: the first arrived in scikit-image 0.25 and the second is
  deprecated there and gone in 0.27, and pinning the package over a helper
  that returns an array of ones would be the wrong trade.
- `invert_mask` keeps its name and its behaviour and gained the docstring
  that says what it is NOT. It is the operation item 435 was filed about --
  flipping the mask on an ordinary field gives one object covering the frame
  -- and it is kept because outlining the space between the cells is a real
  thing to do.

## Item 407, the Overlap rule is counted once over the pixels (2026-09-20)

- `_surviving_region_objects` / `_largest_piece_of_each`: the rule
  `_paste_region_objects` applied per object -- a whole-region comparison for
  every object, and under `clip` a whole connected-component pass for every
  object as well. That is fine for the ten objects in a 128 px box and is not
  fine for the box item 417 allows, which is as wide as the image. Measured on
  this machine, one region, one click, on the GUI thread:

      region   objects   clip        skip        replace
      128 px        10   0.4 ms      0.3 ms      0.3 ms      (was 0.3/0.3)
      256 px        30   1.3 ms      1.1 ms      0.9 ms      (was 7.2/1.2)
      512 px        60   5.2 ms      4.4 ms      3.6 ms      (was 44.8/6.1)
     1024 px       200  29.3 ms     22.2 ms     17.4 ms      (was 595/60.5)
     2048 px       500 117.9 ms     87.4 ms     72.1 ms      (was 6085/821)

  Six seconds of frozen window on a click, at a box size the Size box offers.
  The ids added are identical at every size, which is how the two were
  compared; `test_the_overlap_rule_agrees_with_the_rule_it_replaced` pins it
  against the old rule written out object by object, over random regions,
  for all three rules and three Min areas.
- `_largest_piece_of_each` uses `skimage.measure.label` and NOT
  `scipy.ndimage.label`, and the difference is not a preference. ndimage
  labels connected runs of TRUE, so two different objects that touch become
  one piece and the largest piece of the pair is the largest piece of
  neither. skimage labels connected runs of one VALUE, which is what a piece
  of an object is. The first version used ndimage and the random comparison
  above caught it on two pixels of one seed.
- The rule lives here and not in the screen because the box has to draw what
  a click would add and the click has to add exactly that. Two
  implementations of one rule is a promise the box cannot keep.

## Item 419 points 7, 8 and 9 (2026-09-19)

- `filter_report` / `FilterRemoval` / `filter_objects`: point 7 asks the
  screen to say WHY each object went ("object 22 with area x and intensity y
  was removed by minimum intensity"), and `filter_objects` measured all
  three of those and threw two away. It is now `filter_report` with the
  reasons dropped, so the ids in the mask and the rows the user reads come
  from ONE pass over ONE set of measurements and cannot disagree. Its
  arithmetic is unchanged: the same `regionprops` call, the same bounds, the
  same "0 is off", and `ObjectLookup` still equals it to the last bit, which
  is what makes a row name the numbers the hover readout showed.
- `FilterRemoval.bounds` is a TUPLE and not one name. An object can miss on
  two sides at once -- too small AND too dim -- and saying so is worth a
  word, because an object outside two bounds does not come back by moving
  one of them.
- `split_object_at`: point 8's Ctrl + left click. A watershed on the
  object's own distance to background, inside its bounding box: the recipe
  `_split_touching_objects` already runs on a whole field, on one object.
  THREE DECISIONS, and the first is the one a reader will want:
  - An object with ONE centre is left alone and the screen says so, rather
    than being halved through the click. A single click carries no
    direction; a forced cut would have to invent one. The gesture for a cut
    the user aims is the Divide tool, which already exists.
  - The largest piece keeps the id and the rest are minted above the mask's
    top label. That is `canonical_labels`' own rule and `divide_object`'s,
    so splitting and then saving renumbers nothing.
  - NO PIXEL IS LOST. `_split_touching_objects` drops pieces under
    `min_area`; this does not, and passes `min_area` only as the seed
    spacing. A hand edit moves pixels between ids, and a gesture that
    quietly erased the smaller half would be a delete wearing a split's
    name -- on a field of four hundred objects nobody would notice which.
- `invert_intensity`: point 9 needs an inversion and item 419 left the
  arithmetic "open for the builder to settle". IT IS NOT SETTLED HERE. Item
  435 asked the maintainer the same question on the same day and got an
  answer -- "1/intensity i think and then fitted to dtype i guess" -- and
  landed `invert_intensity` as the dtype complement. Point 9 therefore USES
  THAT FUNCTION AND DEFINES NOTHING. Its first draft reflected about the
  image's own range instead and the two would have collided in this module
  under one name; the branch was rebased onto 435 and its version deleted.
- WHY THE DTYPE COMPLEMENT COSTS POINT 9 NOTHING, since the draft's
  objection -- that a 12-bit field stored in uint16 lands in the top six per
  cent of the range -- is true and sounds like it should matter. It does not
  reach a detector. Both inversions are `a - value` for a constant `a`, and
  BOTH DETECTORS PERCENTILE-NORMALISE before they threshold:
  `_otsu_instances` stretches between its own 1st and 99.8th percentiles
  (and the threshold correction multiplies AFTER that stretch), and Cellpose
  is called with `normalize=True`. A percentile stretch of `a - value` is
  `(p99 - value) / (p99 - p1)`, in which `a` has cancelled. Measured: the
  labels `_otsu_instances` returns are identical arrays under the two
  inversions over 40 random fields x both sides x three corrections. The
  choice is invisible below the display, so the shared name is worth more
  than the wider span.
- The `bounds=` parameter the draft added is gone with it, and so is the
  magnifier's whole-field extremes cache. It existed because a region
  reflected about its OWN extremes is reflected differently wherever the box
  is put, so a crop had to borrow the field's pair to stay a preview of the
  button. The dtype complement is a function of the pixel value alone, so a
  crop inverts identically wherever it is cut and there is nothing to pass.

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
