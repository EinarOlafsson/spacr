# Notes from `spacr/crops.py`

Prose lifted out of `spacr/crops.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (14 entries)
- [_rescale_intensity](#_rescale_intensity) (1 entry)
- [percentile_pair](#percentile_pair) (1 entry)
- [_LabelIndex.__init__](#_labelindex__init__) (1 entry)
- [MergedField.read_window](#mergedfieldread_window) (1 entry)
- [_merged_field_cache_bytes](#_merged_field_cache_bytes) (1 entry)
- [_content_fingerprint](#_content_fingerprint) (2 entries)
- [open_merged_field](#open_merged_field) (1 entry)
- [_region_for](#_region_for) (2 entries)
- [_crop_from_field](#_crop_from_field) (4 entries)
- [png_view](#png_view) (1 entry)
- [narrow_to_uint8](#narrow_to_uint8) (2 entries)
- [read_crop_folder_marker](#read_crop_folder_marker) (1 entry)
- [stamp_crop_folder](#stamp_crop_folder) (2 entries)
- [read_db_crop_format](#read_db_crop_format) (2 entries)
- [stamp_crop_format_in_db](#stamp_crop_format_in_db) (1 entry)
- [read_crop_png](#read_crop_png) (3 entries)
- [_convert_one](#_convert_one) (1 entry)
- [migrate_crop_folder](#migrate_crop_folder) (6 entries)
- [crop_spec_from_settings](#crop_spec_from_settings) (4 entries)
- [path_components](#path_components) (1 entry)
- [reanchor_path](#reanchor_path) (1 entry)
- [ReanchorReport.describe](#reanchorreportdescribe) (1 entry)
- [reanchor_frame](#reanchor_frame) (5 entries)
- [object_label](#object_label) (1 entry)
- [_row_get](#_row_get) (1 entry)
- [PngCropSource.resolve](#pngcropsourceresolve) (1 entry)
- [MergedCropSource.resolve_path](#mergedcropsourceresolve_path) (2 entries)
- [MergedCropSource.spec_for](#mergedcropsourcespec_for) (1 entry)
- [MergedCropSource.get_many](#mergedcropsourceget_many) (1 entry)
- [_has_png_folder](#_has_png_folder) (1 entry)
- [resolve_crop_source](#resolve_crop_source) (7 entries)
- [apply_display_order](#apply_display_order) (1 entry)
- [reconcile_merged_mask_dims](#reconcile_merged_mask_dims) (1 entry)

## Module level

### line 88  _(unsure)_

```python
from .schema import ALL_ROLES, ORGANELLE_ROLES, SEGMENTED_ROLES
```

Normal package import: schema is the dependency-light role registry.

### line 90, trailing  _(unsure)_

```python
except ImportError:
```

exercised by the standalone probe

### lines 91-94

```python
import importlib.util as _importlib_util
```

``tests/test_crops.py`` loads this file directly, without a package, to prove the thumbnail path does not pull in spacr (and therefore torch). Load the same standalone schema source under a private module name; do not duplicate the role vocabulary just to satisfy that import mode.

### lines 886-888

```python
_FIELD_CACHE: "OrderedDict[Tuple[str, int, int], MergedField]" = OrderedDict()
```

A tiny LRU of open fields, so a grid that walks a handful of fields keeps their label indices between calls. Keyed on (path, mtime, size) so a regenerated merged file is never served from a stale entry.

### lines 891-893

```python
_FIELD_CACHE_USED: "Dict[Tuple[str, int, int], float]" = {}
```

Epoch seconds are kept separately so the cache's public values remain real ``MergedField`` objects.  The resource-budget sweep reads this clock and the measured byte count without having to wrap (and thereby change) them.

### lines 1420-1483

```python
CROP_FORMAT_LEGACY_BGR = 1
```

Crop PNG format

A PNG on disk carries no field saying which channel order it was written in, so the format has to be *versioned* or every reader is guessing.

format 1 ("legacy", BGR)

``cv2.imwrite(path, png_channels)``. cv2 reads a 3-channel array as BGR, so ``png_dims[0]`` landed in the file's BLUE slot and ``png_dims[2]`` in its RED one. Every crop written by spaCR before 2026-07-26 is format 1, and every one of them is unmarked.

This was read as a bug and it was not one. Microscope channels come off the scope in wavelength order -- 0 is 405 (blue), 1 is 488 (green), 2 is 555, 3 is 647 -- so a biologist writing ``png_dims=[0,1,2]`` means "405 blue, 488 green, 555 red", which is exactly what format 1 produced. The bytes are right; only the reasoning* for them was accidental.

format 2 ("rgb")

``cv2.imwrite(path, png_channels[..., ::-1])``. Written between 2026-07-26 and 2026-08-06 in the belief that ``png_dims[0]`` ought to be red. It puts the 405/DAPI plane in the red slot, so nuclei come out red and the 555 plane comes out blue. **This is the format that is wrong**, and it is the one read_crop_png reverses.

``migrate_crop_folder`` rewrote format-1 folders into format 2, so a folder that was migrated in that window holds reversed pixels. It is marked, so it is read correctly; but an external image viewer shows it reversed, and re-running the migrator on it now puts it back.

format 3 ("declared_rgb", current)

The user states the mapping outright: ``png_channel_mapping = {'r': 2, 'g': 1, 'b': 0}`` means source channel 2 is red, 1 is green, 0 is blue. The writer assembles the planes in that order and the file's slots hold them. Nothing is inferred from list position, so there is no convention left to get backwards. For the default mapping this is byte-identical to format 1, which is why the two are read the same way.

The marker, in precedence order:

1. the folder sidecar ``.spacr_crop_format.json``, written into the crop folder itself. It is the authority, because it travels with the bytes it describes: copy, move, rsync or zip the folder and the marker goes with it. A database row does not -- crop folders routinely outlive, and get copied away from, the ``measurements.db`` that indexed them. 2. the ``crop_format`` column on ``png_list``, when a database is on hand. Advisory: it makes the format queryable and survives a folder being re-pointed, but it loses to the sidecar when the two disagree, and the disagreement is reported rather than silently resolved. 3. nothing at all -> format 1. Unmarked means legacy, because every crop that exists today is unmarked and every one of them is legacy. This is the only default that cannot corrupt existing data.

16-bit narrowing: crops are ``uint16`` and the files are 16-bit PNGs, but every consumer wants 8-bit RGB. PIL narrows those two different ways -- the high byte for an RGB image, a *clip* at 255 for a single-channel one, which turns any single-channel crop into solid white. spaCR does the narrowing itself now, in exactly one place (:func:`narrow_to_uint8`), with exactly one rule: **take the high byte** (``// 256``). It applies to every channel count and to both formats. The file keeps its full 16 bits; only the view is narrowed, and it is narrowed the same way every time.

### line 1851  _(unsure)_

```python
_FORMAT_CACHE: Dict[str, Tuple[Any, Optional[Dict[str, Any]]]] = {}
```

(folder -> (cache key, marker or None)). Cleared by clear_crop_format_cache.

### lines 1853-1854

```python
_STAMPED_FOLDERS: set = set()
```

Folders this process has already stamped, so the writer pays one stat per folder rather than one per crop.

### lines 1856-1860

```python
_DB_FORMAT_CACHE: Dict[str, Optional[int]] = {}
```

db path -> table-wide crop_format. Held for the life of the process, not keyed on mtime: the annotate GUI writes labels into png_list constantly, and an mtime key would re-run a full-table SELECT DISTINCT for every thumbnail. A dataset's crop format does not change under a running session -- and when spaCR itself changes it, stamp_crop_format_in_db drops the entry.

### lines 2919-2921  _(unsure)_

```python
PATH_ANCHORS: Tuple[str, ...] = ("data", "merged")
```

Re-anchoring a recorded path after the folder has moved

### lines 3688-3700

```python
LOAD_IMAGES = "png"
```

What the two picture sources are CALLED (instruction 171)

One idea had three spellings: 'auto'/'png'/'merged' here, 'pre_generated'/'on_demand'/'generate' in the training settings, and two more proposed for the Cells tab. These are the names a USER sees; the stored values stay 'png' and 'merged', so no settings file already on disk changes meaning.

'auto' is not retired from the code -- it is still the answer to "what is available here" -- it is retired from the panels, where it is not an answer to "which mode do you want".

### lines 3981-3983

```python
"cmy": (np.array([[0.0, 1.0, 1.0],
```

plane 0 -> cyan, 1 -> magenta, 2 -> yellow. Halved because each plane lands in two slots; clipping instead would turn every bright overlap into flat white and hide the colocalisation the figure is about.

### line 3987  _(unsure)_

```python
"deuteranope": (np.array([[1.0, 1.0, 0.0],
```

Red and green collapse: move RED to yellow and leave the other two.

### lines 3994-3995

```python
"tritanope": (np.array([[1.0, 0.0, 0.0],
```

Blue and yellow collapse instead, so blue is the plane that must move to magenta, which keeps it clear of both green and red.

## _rescale_intensity

### lines 300-309

```python
def _rescale_intensity(image, in_range, out_range):
```

Numpy clones of the PNG path's normalisation helpers

These are byte-for-byte reimplementations of

``skimage.exposure.rescale_intensity`` (2-tuple ranges only), ``spacr.utils.normalize_to_dtype`` and ``spacr.utils._get_percentiles``. They exist so this module never has to import skimage or spacr.utils spacr.utils pulls in torch, which must stay off the crop path. ``tests/test_crops.py`` asserts they agree with the originals.

## percentile_pair

### lines 375-377

```python
return (max(0.0, min(100.0, low)), max(0.0, min(100.0, high)))
```

A PERCENTILE OUTSIDE 0-100 IS NOT A PERCENTILE. numpy raises on one, so the alternative to clamping is a montage that dies inside the worker and reports "the montage load failed" without naming the setting.

## _LabelIndex.__init__

### lines 577-579

```python
self._ysum = np.add.reduceat(y.astype(np.float64), starts)
```

float64 accumulation of integer coordinates is exact well past any realistic field size, so this matches ``ys.mean()`` bit for bit which is what ``scipy.ndimage.center_of_mass`` computes.

## MergedField.read_window

### line 831  _(unsure)_

```python
sub = np.asarray(self.array[sy0:sy1, sx0:sx1, int(c)])
```

One plane at a time so the mmap only faults in the window.

## _merged_field_cache_bytes

### lines 915-918

```python
slots = getattr(type(index), "__slots__", None)
```

A label index defines __slots__ and therefore has no __dict__, so vars() raised TypeError here and took the whole memory sweep with it the moment any cached field had been indexed -- i.e. always, after the first crop was cut out of it.

## _content_fingerprint

### lines 1033-1045

```python
if size > window:
```

`> window`, NOT `> window * 2`. The doubled bound left a BLIND BAND: a file between 64 KiB and 128 KiB read its head and never its tail, so everything past byte 65,536 was invisible to the key -- which is the stale-pixels bug this fingerprint exists to stop, surviving in a size band.

Measured: a (4, 100, 100) uint16 field is 80,128 bytes, and changing its LAST pixel left `_cache_key` byte-identical. The first guard written for this could not catch it either: its field is (7, 96, 112) = 150,656 bytes, just above the band.

An overlapping head and tail hashes some bytes twice, which costs nothing and is why the simple bound is the right one.

### lines 1050-1056

```python
return None
```

CANNOT VERIFY IS NOT THE SAME AS CHANGED, and returning a constant here conflated them. "" is not the digest any cached entry carries, so the lookup missed, `MergedField` was rebuilt, and the SAME read error surfaced as `CorruptMergedFile` -- a permission flip on a NAS share, a remount, or transient fd pressure reported as a corrupt file, on a call that used to be a pure cache hit against a mapping that was still perfectly valid.

## open_merged_field

### lines 1096-1100

```python
for other, field in _FIELD_CACHE.items():
```

THE FINGERPRINT COULD NOT BE READ. Fall back to the part of the key that stat still answers: a cached entry for the same path, mtime and size is the best available evidence, and it is exactly the evidence this cache used before the fingerprint existed. Rebuilding instead would raise on the same unreadable file.

## _region_for

### line 1145

```python
ry0 = max(by0 - spec.bbox_buffer, 0)
```

_find_bounding_box: inclusive rectangle, clamped, hard 10 px buffer.

### lines 1172-1176

```python
if px > 0:
```

A radius of 0 means no dilation. scipy reads iterations=0 as "repeat until nothing changes", which grew the region to the whole field and turned the crop into an unmasked window on the middle of the image — for every object under ~25 px at the default ratio. measure.py guards it now, and so does this.

## _crop_from_field

### lines 1210-1213

```python
wy0 = int(centroid[0]) - height // 2
```

_crop_center: a fixed (height, width) window centred on the rounded centroid. The PNG path pads by max(width, height) first, which makes the window guaranteed-complete and zero-filled outside the field -- so in unpadded coordinates it is simply this window, zero-padded at the edges.

### lines 1219-1220

```python
percentile_list = None
```

FOV-wide percentiles have to be measured before the mask is applied, exactly like the PNG path (_get_percentiles on the whole png_channels).

### lines 1228-1230

```python
keep = np.zeros((height, width), dtype=bool)
```

``_region_for`` always returns a region, and the crop window is centred on a point inside its bounds.  The former ``if region is not None:`` and ``if oy1 > oy0 and ox1 > ox0:`` therefore re-checked its contract.

### lines 1245-1246  _(unsure)_

```python
crop = np.dstack((crop, np.zeros_like(crop[:, :, 0])))
```

The PNG path pads a two-channel crop to RGB with a zero third plane before writing, so the file always has three channels.

## png_view

### lines 1368-1369

```python
rgb = np.zeros((eight.shape[0], eight.shape[1], 3), dtype=np.uint8)
```

The PNG path pads a two-channel crop with a zero third plane, so the blue channel is empty -- not the red one, as it was under the bug.

## narrow_to_uint8

### lines 1582-1584  _(unsure)_

```python
def narrow_to_uint8(arr: np.ndarray) -> np.ndarray:
```

Narrowing and the writer's channel order

### lines 1610-1612

```python
return (np.clip(a, 0, 65535) // 256).astype(np.uint8)
```

Anything wider than 8 bit is high-byte narrowed off the 16-bit range: that is what a 16-bit PNG holds, whatever container PIL chose to hand it back in (uint16 for I;16, int32 for I).

## read_crop_folder_marker

### lines 1916-1917

```python
if marker is None or "migration" not in marker:
```

A folder mid-migration changes on every file, so caching it would serve a stale watermark and mis-read the files either side of it.

## stamp_crop_folder

### lines 1999-2002

```python
if existing is None and fmt == CROP_FORMAT_CURRENT:
```

Was `fmt == CROP_FORMAT_RGB`, which silently stopped warning the moment the current format moved to 3. Ask whether this is the format new crops are written in, not which number that happens to be today.

### lines 2005-2008

```python
if stale and read_crop_folder_marker(key, use_cache=False) is None:
```

Re-read the marker before complaining. The writer stamps before its first PNG, so if a sibling measure worker got here first its marker is already on disk and the PNGs we just listed are this run's, not an old dataset's.

## read_db_crop_format

### lines 2036-2038  _(unsure)_

```python
def read_db_crop_format(db_path: str, png_path: Optional[str] = None,
```

Resolving the format of a folder / of one file

### lines 2070-2071  _(unsure)_

```python
rows = conn.execute(
```

The exact path is not in this database (a folder copied somewhere else, say). Fall back to the table-wide answer.

## stamp_crop_format_in_db

### line 2145  _(unsure)_

```python
_DB_FORMAT_CACHE.pop(os.path.abspath(db_path), None)
```

The memoised table-wide answer is now stale.

## read_crop_png

### lines 2281-2284

```python
arr = np.array(img)
```

I;16 (a 16-bit single-channel PNG) must NOT go through convert('L'): PIL clips it at 255 and every crop brighter than that comes back solid white. Take the raw samples and narrow them ourselves.

### lines 2288-2290

```python
arr = narrow_to_uint8(arr)
```

Every branch above yields either a 2-D plane (L / I;16 / F) or three channels (RGB, or anything else converted to it), so there is no other shape to handle here.

### lines 2294-2297

```python
if (_FORMAT_IS_DECLARED_ORDER.get(int(fmt), True)
```

There are still exactly two ORDERINGS, but now three formats, so the reversal is decided by which ordering each format is in -- not by `fmt != as_format`, which would reverse between formats 1 and 3 even though they hold identical bytes.

## _convert_one

### lines 2378-2379  _(unsure)_

```python
return False
```

Single-channel: cv2 did no colour interpretation on the way in, so there is no reversal to undo. Only the marker changes.

## migrate_crop_folder

### lines 2484-2488

```python
raise CropError(
```

Not a corruption -- formats 1 and 3 are both declared order, so nothing would be reversed. It is a LOSS: the marker is the only record that this folder was repaired, and overwriting it with "legacy" makes a repaired folder indistinguishable from one that never needed repairing. Refuse rather than quietly forget.

### lines 2504-2513

```python
retry_only = leftover
```

A previous run finished with on_error='skip'. The folder is repaired apart from these, so retry exactly them: every other crop in here is already back in declared order and reversing it again would undo the repair.

This case has to be tested BEFORE the "nothing to do" check below, because a finished-with-leftovers folder is marked with the TARGET format. Keying the retry on the source format is what made the retry silently return `already` and leave the unconverted files unconverted for ever.

### lines 2516-2519

```python
result.already = True
```

Only format 2 has reversed pixels. Format 1, format 3 and unmarked are all in declared order already, so there is nothing to rewrite -- and rewriting one WOULD reverse a correct folder, which is exactly the damage this function exists to undo.

### line 2582  _(unsure)_

```python
wrote = True
```

A previous run converted it and died before installing it.

### lines 2600-2602

```python
if name in failed_names:
```

Marker first, install second: a crash between them leaves the staging file in place, and "staging file exists" outranks everything else, so the crop is still correctly read as legacy.

### line 2604, trailing  _(unsure)_

```python
failed_names.remove(name)
```

a retry that worked

## crop_spec_from_settings

### lines 2844-2853

```python
if isinstance(size, (int, float)) and not isinstance(size, bool):
```

A BARE NUMBER IS A SQUARE. The annotator's `crop_size` (`img_size` until 2026-09-19) is one integer it is a single spin box -- and mapping it straight onto `png_size` gave this function a scalar, where `size[0]` raises

TypeError: 'int' object is not subscriptable

inside the montage worker, surfacing as "The montage load failed" with no hint that the cause was a settings shape. Normalised here as well as at the mapping, because every caller of this function reaches the same line and a settings CSV can carry the scalar too.

### line 2879  _(unsure)_

```python
dilate = False
```

_measure_crop_core hard-disables dilation for cytoplasm crops.

### lines 2884-2889

```python
low, high = percentile_pair(normalize, (0.0, 100.0))
```

A PAIR WRITTEN AS TEXT IS STILL A PAIR. `_coerce` recovers the spellings `ast.literal_eval` accepts, and leaves the ones it does not -- `[1 99]`, separated by a space rather than a comma -- as a string. Passed through, a non-empty string is TRUTHY but is not a sequence, so the cut fell to the full 0-100 stretch: the user configured a window, the crop ignored it, and nothing said so.

### lines 2899-2903

```python
channels=channels_from_settings(settings),
```

In COLOUR order -- red source, green source, blue source -- which is the order the PNG's slots are in and therefore the order `png_view` and `read_crop_png` both speak. Taking `png_dims` verbatim here is what made the on-demand source and the PNG folder disagree: one was in list order and the other in file order.

## path_components

### line 2991, trailing  _(unsure)_

```python
continue
```

a doubled or trailing separator

## reanchor_path

### lines 3041-3042

```python
for index in range(len(parts) - 2, -1, -1):
```

From the RIGHT, and never the final component: an anchor with nothing after it names a folder, and there would be no file left to re-anchor.

## ReanchorReport.describe

### lines 3080-3088

```python
if self.n_reanchored == 0 and self.n_already == 0:
```

A ROUTE THAT IS NOT ON THIS MACHINE IS NOT N FAILURES.

Measured on the maintainer's screen: all 60,816 `png_path` values re-anchored and all 60,816 `path_name` values could not, because that screen has PNG crops and no `merged/` folder -- and it is completely healthy. Reported together they read as "60,816 of 121,632 could not be re-anchored", which is the false alarm that teaches a reader to ignore the true one. Same distinction `spacr.portable_paths.RerootReport` draws.

## reanchor_frame

### lines 3117-3131

```python
prefixes: List[Tuple[str, str]] = []
```

WHAT ONE FOLDER ANSWERS, THE WHOLE FOLDER ANSWERS. Both of these exist in `spacr.portable_paths.reroot_frame` already, with its own measurement beside them -- 8.2 s over 60,816 rows against 0.6 s once a prefix is known -- and this function, which is the one the cell montage calls, never got them.

WHAT IT COST, measured on the reporter's shape in GitHub issue 116: `_reroot_with_prefix` asks the filesystem about ~22 candidate locations for every path it cannot place, so 16,000 recorded paths produced 360,000 stat calls. His four plates carry roughly a million paths, and he had just RENAMED the databases, so not one of them was already anchored -- about 22 million filesystem probes before the montage selected the few hundred cells it was going to draw. "Show the cells" sat on "reading 4 database(s)" for as long as he left it.

### lines 3167-3175

```python
if outcome != ALREADY_ANCHORED and not os.path.exists(new):
```

THE STRUCTURAL PASS NEVER ASKS THE DISK, and it needs `root` to be the folder that holds `data/`. Measured on the TSG101 screen with 3,000 recorded crops: the plate folder resolves all 3,000, and the SCREEN folder above it, the `measurements/` folder and the database file each resolve NONE -- the rewrite lands somewhere plausible that is not there. A caller holding one of those is not doing anything wrong, so fall back to the resolver that searches the recorded structure under every folder the root could mean and returns only what EXISTS.

### lines 3180-3189

```python
placed = False
```

A PREFIX ALREADY DISCOVERED, FIRST. One string replacement and one stat, against the ~22 stats a fresh search costs.

`placed`, NOT `outcome`, decides whether the search still has to run: the structural pass above returns REANCHORED for a path it rewrote WITHOUT asking the disk, and that path may not exist -- which is the whole reason the search below exists. Reading the search's necessity off `outcome` skipped it exactly when it was needed, and a root one level above the plate stopped resolving.

### lines 3200-3203

```python
if unresolvable.get(folder, 0) < give_up_after:
```

Only the SEARCH is skipped for a folder that has failed its allowance -- the prefix above is still tried for every row, so a folder where one crop is missing and the next is present still places the next one.

### lines 3210-3212

```python
unresolvable.pop(folder, None)
```

It resolves after all: the folder is on this machine and its earlier misses were missing FILES, which is a different fact.

## object_label

### line 3243  _(unsure)_

```python
text = text[1:]
```

`o2`, and the same shape for the other object types.

## _row_get

### line 3269, trailing  _(unsure)_

```python
if name in row:
```

pandas Series

## PngCropSource.resolve

### lines 3418-3426

```python
if path and not os.path.exists(path):
```

A RE-ANCHORED PATH THAT DOES NOT EXIST IS NOT A RESOLUTION. `reanchor_path` rewrites on structure alone and never asks the filesystem, which is right for its callers -- but it needs the root to be the folder holding `data/`, and a caller may hold the screen folder above it, the `measurements/` folder, or the database file. `portable_paths` searches the recorded structure under every folder the root could mean and returns ONLY what exists, so this cannot replace a good path with a worse one. Measured on the TSG101 screen: 0 of 60,816 recorded crops existed, 60,816 of 60,816 after this.

## MergedCropSource.resolve_path

### lines 3505-3510

```python
anchored, outcome = reanchor_path(
```

The anchor first -- it preserves any sub-folder under merged/ -- then the flat basename, which is the older fallback and is kept because a hand-built project may have no `merged` component in the recorded path at all. `basename_any` and not `os.path.basename`: on Linux the latter hands back the whole of C:\lab\exp1\merged\x.npy.

### lines 3537-3540

```python
try:
```

chr(ord('A') + n - 1) walked straight off the end of the alphabet: rowID 'r27' -- an ordinary 1536-plate row -- came back as '[', and the rebuilt path pointed at a file that cannot exist. schema.well_id is bijective base 26, so r27 is 'AA'.

## MergedCropSource.spec_for

### lines 3602-3603  _(unsure)_

```python
b = [_row_get(row, f"bbox-{i}", f"bbox_{i}") for i in range(4)]
```

skimage regionprops stores bbox as (min_row, min_col, max_row, max_col); CropSpec.bbox is (y0, y1, x0, x1).

## MergedCropSource.get_many

### lines 3659-3660

```python
return cast(List[np.ndarray], out)
```

``extract_crops`` supports a separate ``on_error='none'`` API, but this source deliberately uses its default fail-loud contract.

## _has_png_folder

### line 3681  _(unsure)_

```python
if dirpath.count(os.sep) - data.count(os.sep) >= 3:
```

Crop folders sit at <data>/<well>/<class>_png; three levels is plenty.

## resolve_crop_source

### lines 3781-3785

```python
if isinstance(src, (list, tuple)):
```

``src`` is multi-source in several settings panels, and those callers also pass the stored value directly.  Normalise both the mapping form and that bare list/tuple form here; stringifying the latter creates a path containing Python's brackets or parentheses and can never find the experiment it names.

### lines 3804-3816

```python
if choice == "png" and has_png:
```

LOAD IMAGES, AND FALL BACK TO STREAM IMAGES RATHER THAN FAIL LATER.

Instruction 171: "the default should always be loade images which loades from data folder. if that fails it should always try the other."

`choice == "png"` used to return a PngCropSource WITHOUT asking whether `data/` was there, so an explicit request on a screen that has only `merged/` handed back a source that could not read anything and failed later, somewhere with less context.

THE FALLBACK IS RECORDED IN `reason`, which is the condition 171 puts on it: a fallback nobody can see is what makes a user believe they are looking at a crop they are not. Every caller already shows `reason`.

### lines 3832-3841

```python
if ask is not None:
```

THE FALLBACK, AND ONLY AFTER THE USUAL RESOLUTION HAS FAILED.

`ask` is INJECTED rather than imported: this module must not depend on Qt, and a caller with nobody in front of it -- a script, a test, a batch run -- simply passes none and gets the error below, which is what it has always got. That makes "never prompt headless" structural instead of something each call site has to remember.

The program already knows exactly what is missing here, which is why asking is more useful than reporting it.

### lines 3847-3849

```python
return resolve_crop_source(answer, object_type=object_type,
```

Resolved AGAINST THE ANSWER, with no `ask` this time: one question per run, and a wrong answer must not open a second dialog on top of the first.

### line 3863  _(unsure)_

```python
for key in ("png_dims", "png_size", "normalize", "normalize_by", "crop_mode",
```

Anything the caller set explicitly wins over the saved snapshot.

### lines 3875-3876

```python
reason = (f"{LOAD_IMAGES_LABEL} was asked for and there is no "
```

Asked for by name, and `data/` is not there. The other route is, so it draws -- and says that it is not what was asked for.

### Every segmented role's mask plane can be overridden

```python
*(f"{role}_mask_dim" for role in SEGMENTED_ROLES)):
```

2026-09-19. This list, and `spacr.io.CROP_SHAPE_KEYS` that feeds it, named `organelle_mask_dim` and no other slot. A run could not say where Organelle 2's plane was, and an on-demand crop of `organelleb` then depended on the saved snapshot naming it (`KeyError: 'organelleb'` when it did not). Both now name every slot.

## apply_display_order

### lines 3940-3942

```python
return image
```

A greyscale or two-plane crop has no three slots to permute. Left alone rather than refused: the order is a display preference and a single-channel image is not wrong, it simply has nothing to reorder.

## reconcile_merged_mask_dims

### An organelle slot neither side names is left out

```python
if (expected is None and key not in settings
        and role in ORGANELLE_ROLES):
    continue
```

2026-09-19, items 364 and 76. The loop used to set `<role>_mask_dim` for every segmented role, which is 702 organelle slots, `None` for each one the manifest does not record. `get_measure_crop_settings` counts a slot as declared when any of its keys is present, even as `None`, so a two-organelle run reached Measure declaring 702 slots, and the factory filled `<slot>_min_area` and `<slot>_type` for each. Measured with raw TIFFs through Mask and Measure: `measurements.db`'s `settings` table held 2,170 rows, 2,100 of them for the 700 slots the run never had, and `measure_crop_settings.csv` had the same rows. It now holds 70.

The cell, nucleus and pathogen keys are still always set. A manifest without a pathogen has to switch off Measure's default `pathogen_mask_dim` of 6, or plane 6 is measured as pathogens. A slot `settings` carries is still set to the manifest's answer, so an explicit slot the manifest lacks is still refused as a conflict.
