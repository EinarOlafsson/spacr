# Notes from `spacr/spacrops.py`

Prose lifted out of `spacr/spacrops.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_DiskFeatureStore.__init__](#_diskfeaturestore__init__) (1 entry)
- [_DiskFeatureStore.get](#_diskfeaturestoreget) (4 entries)
- [_DiskFeatureStore.put](#_diskfeaturestoreput) (2 entries)
- [spacrStitcher.__init__](#spacrstitcher__init__) (19 entries)
- [spacrStitcher._edge_zncc](#spacrstitcher_edge_zncc) (4 entries)
- [spacrStitcher._get_cellpose_model](#spacrstitcher_get_cellpose_model) (2 entries)
- [spacrStitcher._cellpose_labels](#spacrstitcher_cellpose_labels) (2 entries)
- [spacrStitcher._foreground_mask](#spacrstitcher_foreground_mask) (2 entries)
- [spacrStitcher._outline_mask](#spacrstitcher_outline_mask) (1 entry)
- [spacrStitcher._normalize_to_yx](#spacrstitcher_normalize_to_yx) (8 entries)
- [spacrStitcher._read_plane](#spacrstitcher_read_plane) (3 entries)
- [spacrStitcher._parse_meta](#spacrstitcher_parse_meta) (1 entry)
- [spacrStitcher._detect_and_describe](#spacrstitcher_detect_and_describe) (1 entry)
- [spacrStitcher._compute_features_one](#spacrstitcher_compute_features_one) (3 entries)
- [spacrStitcher.prepare_features](#spacrstitcherprepare_features) (2 entries)
- [spacrStitcher.stitch_pair](#spacrstitcherstitch_pair) (20 entries)
- [spacrStitcher._get_channel_count_tif](#spacrstitcher_get_channel_count_tif) (1 entry)
- [spacrStitcher._read_all_channels_cyx](#spacrstitcher_read_all_channels_cyx) (1 entry)
- [spacrStitcher._plot_sorted_scores](#spacrstitcher_plot_sorted_scores) (2 entries)
- [spacrStitcher._pairs_by_site_window](#spacrstitcher_pairs_by_site_window) (1 entry)
- [spacrStitcher.run_folder](#spacrstitcherrun_folder) (8 entries)
- [spacrStitcher.run_folder._job](#spacrstitcherrun_folder_job) (3 entries)
- [spacrStitcher.build_multichannel_mosaic_from_manifest](#spacrstitcherbuild_multichannel_mosaic_from_manifest) (11 entries)
- [spacrStitcher.build_multichannel_mosaic_from_manifest._get_channel_count_tif_local](#spacrstitcherbuild_multichannel_mosaic_from_manifest_get_channel_count_tif_local) (1 entry)
- [spacrStitcher.build_multichannel_mosaic_from_manifest._read_plane_local](#spacrstitcherbuild_multichannel_mosaic_from_manifest_read_plane_local) (2 entries)
- [spacrStitcher._direction_bin](#spacrstitcher_direction_bin) (1 entry)
- [spacrStitcher._compute_mosaic_transforms](#spacrstitcher_compute_mosaic_transforms) (14 entries)
- [spacrStitcher._compute_mosaic_transforms.edge_ok](#spacrstitcher_compute_mosaic_transformsedge_ok) (4 entries)
- [spacrStitcher.render_mosaic_from_csv._series_dtype](#spacrstitcherrender_mosaic_from_csv_series_dtype) (1 entry)
- [spacrStitcher.render_mosaic_from_csv](#spacrstitcherrender_mosaic_from_csv) (13 entries)
- [spacrStitcher.mosaic_all_channels_from_csv](#spacrstitchermosaic_all_channels_from_csv) (7 entries)
- [StitchedMultiAligner.__init__](#stitchedmultialigner__init__) (2 entries)
- [StitchedMultiAligner._normalize_to_yx](#stitchedmultialigner_normalize_to_yx) (1 entry)
- [StitchedMultiAligner._read_all_channels_cyx](#stitchedmultialigner_read_all_channels_cyx) (1 entry)
- [StitchedMultiAligner.align._series_dtype](#stitchedmultialigneralign_series_dtype) (1 entry)
- [StitchedMultiAligner.align](#stitchedmultialigneralign) (11 entries)
- [stitch_cycle_wells](#stitch_cycle_wells) (27 entries)
- [stitch_cycle_wells._resolve_collision](#stitch_cycle_wells_resolve_collision) (1 entry)
- [get_preprocess_ops_settings](#get_preprocess_ops_settings) (17 entries)
- [FOVAlignAndCropper.__init__](#fovalignandcropper__init__) (1 entry)
- [FOVAlignAndCropper._read_plane](#fovalignandcropper_read_plane) (1 entry)
- [FOVAlignAndCropper.run](#fovalignandcropperrun) (21 entries)
- [align_image_to_stitch](#align_image_to_stitch) (8 entries)
- [align_image_to_stitch._group_by_well._site_key](#align_image_to_stitch_group_by_well_site_key) (1 entry)
- [ops_preprocess](#ops_preprocess) (7 entries)

## Module level

### lines 12-14

```python
from .figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

## _DiskFeatureStore.__init__

### line 34, trailing  _(unsure)_

```python
self._lru_lock = threading.Lock()
```

NEW

## _DiskFeatureStore.get

### line 54  _(unsure)_

```python
with self._lru_lock:
```

LRU hit

### line 60  _(unsure)_

```python
pz = self._npz_path(path)
```

Disk hit (no lock while reading disk)

### lines 75-84

```python
corrupt = isinstance(e, (ValueError, EOFError, zipfile.BadZipFile,
```

A truncated/corrupt NPZ (e.g. a run killed mid-write) must not poison the cache forever: drop it and report a miss so the caller recomputes and rewrites the entry.

But delete ONLY for errors that mean the bytes are bad. A transient OSError, PermissionError or MemoryError on read says nothing about the file's contents, and unlinking there destroys a perfectly good cache entry -- and on a full disk or under memory pressure it would destroy the whole cache, one entry per attempt. Those report a miss and leave the file.

### line 96  _(unsure)_

```python
with self._lru_lock:
```

insert into LRU

## _DiskFeatureStore.put

### line 106  _(unsure)_

```python
np.savez_compressed(self._npz_path(path),
```

Save to disk

### line 115

```python
with self._lru_lock:
```

Insert in RAM LRU

## spacrStitcher.__init__

### lines 221-223

```python
cellpose_model: str = "cpsam",
```

Only read when outline_source='cellpose'. 'cpsam' is the one model Cellpose 4 ships; a path to a .CP_model / .pth checkpoint from Train Cellpose is loaded as given.

### line 231  _(unsure)_

```python
outdir: str = "./sbs_out",
```

IO

### line 235  _(unsure)_

```python
all_scores: bool = False,
```

scoring & control

### line 239  _(unsure)_

```python
feature_cache_mode: str = "disk",          # "ram" | "disk"
```

robustness / scaling controls

### line 240, trailing

```python
feature_cache_mode: str = "disk",
```

"ram" | "disk"

### line 241, trailing  _(unsure)_

```python
feature_cache_dir: Optional[str] = None,
```

where to store DS features if disk

### line 242, trailing  _(unsure)_

```python
max_ram_features: int = 256,
```

LRU size if disk mode

### line 243, trailing

```python
n_workers_features: Optional[int] = None,
```

feature threads

### line 244, trailing  _(unsure)_

```python
pair_batch_size: int = 8000,
```

max pairs processed per batch

### line 245, trailing  _(unsure)_

```python
stream_csv: bool = True,
```

write rows as we go

### line 246, trailing

```python
opencv_threads: int = 1,
```

avoid thread oversubscription

### line 247  _(unsure)_

```python
arr_axes: str = "AUTO",
```

axis/Z/time handling

### line 253  _(unsure)_

```python
ops_gpu: bool = True,
```

hardware

### lines 271-275

```python
self.ops_gpu = bool(ops_gpu)
```

WHETHER THIS RUN MAY TAKE THE CARD. Not "is there one"

`spacr.accelerator` answers that -- but whether this run is allowed to, which is a different question on a machine whose GPU is running somebody else's screen. False keeps every step on the CPU even where a device resolves.

### line 282  _(unsure)_

```python
self._cp_model = None
```

Built lazily by _get_cellpose_model and reused for every tile.

### line 299  _(unsure)_

```python
if self.detector == "ORB":
```

detector init

### line 313

```python
try:
```

OpenCV threads

### line 338  _(unsure)_

```python
self._meta_re = re.compile(
```

default metadata regex (10X_c1_A1_..._Site-5.tif)

### line 344  _(unsure)_

```python
self.arr_axes = str(arr_axes).upper()
```

Axis/Z/time handling

## spacrStitcher._edge_zncc

### line 382  _(unsure)_

```python
a = a.astype(np.float32, copy=False)
```

ensure float32

### line 386  _(unsure)_

```python
ea = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)**2
```

gradient energy images

### line 392  _(unsure)_

```python
if idx.sum() < 25:
```

require some overlap

### line 398  _(unsure)_

```python
ea = ea - ea.mean()
```

ZNCC on gradient energy

## spacrStitcher._get_cellpose_model

### lines 458-460

```python
kwargs = {"gpu": False}
```

ASKED FOR THE CPU, SO TAKE THE CPU. The card may exist and be busy with an AlphaFold screen; "there is a GPU" and "this run may use it" are different questions.

### lines 469-476

```python
self._cp_model = cp_models.CellposeModel(pretrained_model=pretrained, **kwargs)
```

THE DEVICE IS NOT POPPED ANY MORE. `cellpose_kwargs` produces `gpu`, `device` and `use_bfloat16` TOGETHER because they have to agree -- cellpose branches on `gpu` before it looks at `device`, and dropping the resolved device left this call site picking cellpose's default rather than the one the resolver chose. On a machine with two cards that is the wrong card. No model_type= / diam_mean=: Cellpose 4 logs "not used in v4.0.1+" and drops both.

## spacrStitcher._cellpose_labels

### lines 488-492

```python
out = model.eval(x, diameter=self.cellpose_diameter,
```

eval(channels=) went away with Cellpose 4 ("channels deprecated in v4.0.1+"): the network takes up to three channels as given. The old [0, 0] pair reached a parameter Cellpose ignores. diameter, by contrast, is still honoured — the image is rescaled by 30/diameter — so it is exposed as cellpose_diameter.

### lines 495-497

```python
masks = out[0]
```

Cellpose 4 returns (masks, flows, styles); Cellpose 3 returned a fourth `diams`. The old `masks, _, _, _ = ...` unpack would have raised on 4.x even if the Cellpose class had survived.

## spacrStitcher._foreground_mask

### line 509  _(unsure)_

```python
I = img_u8
```

Otsu (default)

### lines 514-515  _(unsure)_

```python
th, _ = cv2.threshold(I, 0, 255, cv2.THRESH_OTSU)
```

cv2.threshold returns (computed_threshold, binarised_image); the first value is the Otsu level we need here.

## spacrStitcher._outline_mask

### line 538  _(unsure)_

```python
if self.dilate_ksize and self.dilate_ksize > 0:
```

NOTE: use dilate_ksize here (bugfix). line_thickness is for edge thickening later.

## spacrStitcher._normalize_to_yx

### line 608  _(unsure)_

```python
"""Reduce any TCZYX array to one 2-D ``(Y, X)`` plane of float32.
```

Choose base axes

### line 623  _(unsure)_

```python
ax = list(axes)
```

Align axes length to array rank

### lines 625-627

```python
while len(ax) > arr.ndim:
```

If too many labels, drop T/C first.  Stop as soon as a pass removes nothing, otherwise an axes string with no T/C to give up (e.g. arr_axes="ZYX" against a 2-D plane) spins forever here.

### line 636  _(unsure)_

```python
while len(ax) > arr.ndim:
```

If still too many, drop from the left (safest for unexpected leading dims)

### line 639  _(unsure)_

```python
while len(ax) < arr.ndim:
```

If too few labels, pad (prefer adding missing T then C at the front)

### line 649  _(unsure)_

```python
slicers = []
```

Build slicers

### line 660  _(unsure)_

```python
if self.mip and sub.ndim == 3:
```

Max-project if Z kept

### line 668, trailing  _(unsure)_

```python
if sub.ndim == 3:
```

defensively drop a small stray axis

## spacrStitcher._read_plane

### line 685  _(unsure)_

```python
axes_hint = "".join(a for a in axes_hint.upper() if a in "TCZYX")
```

Keep only T/C/Z/Y/X symbols

### line 692  _(unsure)_

```python
if axes_hint is None and arr.ndim == 3:
```

If no axes hint and 3-D, pick CYX vs ZYX sensibly

### line 695  _(unsure)_

```python
if re.search(r'(^|[_\-])c\d+([_\-]|$)', fn):
```

If filename contains a channel token like _c1_/_c2_, treat first axis as C

## spacrStitcher._parse_meta

### line 728  _(unsure)_

```python
mw = re.search(r"([A-H]\d{1,2})", fn, re.IGNORECASE)
```

lenient fallbacks

## spacrStitcher._detect_and_describe

### line 750  _(unsure)_

```python
if self.max_keypoints is not None and len(kp) > self.max_keypoints:
```

Top-K by response

## spacrStitcher._compute_features_one

### line 768

```python
Hds = max(1, int(round(H * s)))
```

ensure at least 1 px after DS (robust to extreme s)

### line 774  _(unsure)_

```python
if self.max_keypoints is not None and pts.shape[0] > self.max_keypoints:
```

post-cap (if requested)

### lines 777-779

```python
idx = np.lexsort((-np.arange(pts.shape[0]), -distances))[
```

NumPy's default quicksort does not define which equal-distance point wins. Use the original index as an explicit descending tie-break so minimum and newest NumPy keep the same descriptors.

## spacrStitcher.prepare_features

### line 805  _(unsure)_

```python
todo = []
```

only compute for items missing on disk

### lines 837-839

```python
print(f"[features] WARNING: skipping {os.path.basename(futs[fut])}: {e}",
```

One unreadable tile must not abort a whole plate: report it loudly (never silently) and let the pairs that need it fail individually in run_folder's per-pair handler.

## spacrStitcher.stitch_pair

### line 951  _(unsure)_

```python
force_no_qc: bool = False,
```

NEW: QC gating (safe defaults preserve current behavior)

### line 983  _(unsure)_

```python
fA = self._get_features(pathA, channel_index)
```

DS features (8-bit only for keypoints)

### line 990  _(unsure)_

```python
ptsA, ptsB = self._match(fA, fB)
```

match & model @ DS

### line 1002  _(unsure)_

```python
if inlier_mask is not None and inlier_mask.any():
```

inliers for constrained recompute

### line 1019  _(unsure)_

```python
mA_ds = self._foreground_mask(A_ds8)
```

DS masks & score

### line 1029  _(unsure)_

```python
M_full = M_ds.astype(np.float32).copy()
```

lift DS → full-res

### line 1046  _(unsure)_

```python
pass
```

if threshold is malformed, fall back to current do_qc

### line 1056, trailing  _(unsure)_

```python
colA = np.array([0.0000, 0.4470, 0.6980], np.float32)
```

blue

### line 1057, trailing  _(unsure)_

```python
colB = np.array([0.8350, 0.3650, 0.0000], np.float32)
```

orange

### lines 1063-1066

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 1071-1075

```python
from .plot import save_figure
```

108 point 6: the resolution and the repaint for paper. `fmt` STAYS PNG because the key this is stored under is `qc_outline_png` and the stitch result's schema names it that way -- a preference must not rename a file the rest of the pipeline refers to by extension.

### line 1082  _(unsure)_

```python
if save_stitched is None:
```

decide whether to stitch now

### line 1092  _(unsure)_

```python
A_full = self._read_plane(pathA, ch=channel_index)
```

Read full-res (float32 workspace), but keep input dtypes for final cast

### line 1101  _(unsure)_

```python
corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], dtype=np.float32).reshape(-1, 1, 2)
```

Canvas geometry

### line 1112  _(unsure)_

```python
canvas = np.zeros((Hc, Wc), np.float32)
```

Blend in native intensity space (no normalization)

### line 1115  _(unsure)_

```python
canvas[off_y:off_y + H, off_x:off_x + W] += A_full
```

A contribution

### line 1118  _(unsure)_

```python
M_canvas = (T @ self._affine_to_3x3(M_full))[:2, :]
```

B contribution (warp image + warp 1-mask to get coverage)

### lines 1127-1129

```python
stitched = np.where(wgt > 0, np.divide(canvas, np.maximum(wgt, 1e-6)), 0.0)
```

Pixels no tile covers must stay at background level; dividing a leaked interpolation value by the 1e-6 floor would saturate them to the dtype maximum and draw a white seam.

### line 1144  _(unsure)_

```python
edge_zncc_full = ""
```

optional full-res metrics (unchanged)

### line 1196  _(unsure)_

```python
if (Hc > 0) and (Wc > 0):
```

choose canvas dims for CSV row

## spacrStitcher._get_channel_count_tif

### lines 1233-1243

```python
if axes:
```

A DECLARED AXIS ORDER IS TAKEN AT ITS WORD, and a file that names no channel axis has one channel. That is deliberate, not an oversight: `tifffile` labels a bare 3-D write 'QYX' or 'SYX' "unspecified" -- and a stack of three planes with no metadata could equally be three channels, three z-planes or three timepoints. Guessing turns a z-stack's planes into channels silently, which is worse than declining. Multi-channel tiles must say so: `tifffile.imwrite(path, stack, metadata={"axes": "CYX"})`.

The shape fallback below applies only when the file declares NO axes at all, where a guess is the only thing available.

## spacrStitcher._read_all_channels_cyx

### line 1260, trailing  _(unsure)_

```python
return np.stack(planes, axis=0)
```

(C,H,W)

## spacrStitcher._plot_sorted_scores

### lines 1297-1300

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 1308-1311

```python
from .plot import save_figure
```

108 point 6: the resolution and the repaint for paper. `fmt` STAYS PNG for the same reason as the outline QC above `score_sorted_line.png` is a fixed name the run's own output is checked by, and a preference must not rename it.

## spacrStitcher._pairs_by_site_window

### lines 1362-1373

```python
idx_by_site: Dict[Any, List[int]] = {}
```

EVERY INDEX AT A SITE, not one. A site holds one file PER CHANNEL, and `idx_by_site[s] = i` kept only the last of them -- so every candidate partner was a channel-2 file, two tiles of the same channel were never compared with each other, and the c1/c2 rows in the pairs CSV were the same comparison written twice.

It also ORPHANED a tile. With the files ordered c1_S1, c2_S1, c1_S2, c2_S2, ... the map is {1:1, 2:3, 3:5, 4:7}, so `10X_c1_A1_Site-4.tif` at index 6 had exactly one candidate index 5, which fails `j > i` -- and appeared in no pair at all. No pair means no features, which means no place in the mosaic: a corner arrived with no channel-1 data and no warning.

## spacrStitcher.run_folder

### line 1406  _(unsure)_

```python
mosaic: bool = False,
```

mosaic controls

### line 1411  _(unsure)_

```python
mosaic_all_channels: bool = False,
```

NEW: multi-channel mosaic controls

### line 1415

```python
qc_pairs_threshold: int = 1000,
```

NEW: QC gating controls (do not break existing calls)

### line 1462  _(unsure)_

```python
if not paths:
```

Handle no files

### line 1484  _(unsure)_

```python
self.prepare_features(list(set([p for pair in pairs for p in pair])), channel_index, num_workers=...
```

Precompute DS features (disk-backed)

### line 1579  _(unsure)_

```python
if stitch and (score_threshold is None and self.score_threshold is None):
```

optional second pass (stitch winners and, if many pairs, generate QC only for winners)

### line 1600  _(unsure)_

```python
force_no_qc=False,
```

If too many pairs, now allow QC but only for score≥thr (these are winners anyway)

### lines 1629-1630

```python
mosaic_png = (os.path.splitext(mosaic_out)[0] + ".png") if mosaic_out else None
```

mosaic_out may legitimately be None (manifest-only mode); os.path.splitext(None) would raise.

## spacrStitcher.run_folder._job

### lines 1508-1510

```python
if too_many_pairs:
```

QC gating for the first (threaded) pass

If too many pairs and threshold unknown: suppress QC now If too many pairs and threshold known: only QC when score >= threshold

### line 1527  _(unsure)_

```python
force_no_qc=force_no_qc,
```

NEW controls (default keep behavior)

### lines 1532-1537

```python
print(f"[run_folder] WARNING: pair {os.path.basename(A)} vs "
```

ALWAYS report, never only when verbose. This was gated on self.verbose, so a systematic failure -- every tile unreadable, a bad channel index -- produced a short or empty pairwise CSV and a run that looked like it succeeded. A silently-skipped pair and a pair that genuinely did not overlap were indistinguishable.

## spacrStitcher.build_multichannel_mosaic_from_manifest

### line 1648, trailing

```python
blend: str = "max",
```

"max" or "overwrite"

### line 1687  _(unsure)_

```python
if tmp_dir is None:
```

Default tmp_dir: reuse feature cache root if available

### line 1694  _(unsure)_

```python
rows = []
```

Read manifest rows and basic integrity checks

### lines 1712-1717

```python
raise RuntimeError(
```

Dropping the row silently left the tile out of the stitched image, wrote the file anyway and returned its path as if the mosaic were whole. A hole in a mosaic is data the user never gets back and never gets told about, so this refuses the same way the "no usable rows" check below already does.

### line 1794, trailing  _(unsure)_

```python
off = np.array([[1,0,-x_min],[0,1,-y_min]], dtype=np.float32)
```

canvas origin at (0,0)

### lines 1796-1797  _(unsure)_

```python
mmap_path = os.path.join(tmp_dir, f"_mosaic_{os.path.splitext(os.path.basename(out_tif))[0]}.mmap")
```

Allocate output stack (float32 workspace); pick output dtype later If tmp_dir is set, use an on-disk memmap to reduce RAM.

### line 1807, trailing  _(unsure)_

```python
out_stack[:] = -np.inf
```

sentinel for max blending

### line 1809  _(unsure)_

```python
in_dtypes: List[np.dtype] = []
```

track dtypes across input tiles (to pick safe output dtype)

### line 1835  _(unsure)_

```python
out_dtype = np.result_type(*in_dtypes) if in_dtypes else np.float32
```

Save BigTIFF with axes metadata; choose common dtype over inputs

### line 1843

```python
if out_png:
```

Optional preview (channel 0 min-max normalized, downsampled)

### line 1856  _(unsure)_

```python
if use_memmap and hasattr(out_stack, "flush"):
```

ensure memmap data hits disk

## spacrStitcher.build_multichannel_mosaic_from_manifest._get_channel_count_tif_local

### line 1728  _(unsure)_

```python
def _get_channel_count_tif_local(path: str) -> int:
```

Helpers to read channels from TIFFs (local, minimal axis handling)

## spacrStitcher.build_multichannel_mosaic_from_manifest._read_plane_local

### line 1752, trailing  _(unsure)_

```python
labels = []
```

the hint does not describe this array

### lines 1754-1756

```python
if "C" in labels:
```

Keep the label list in step with the array: slicing out C shifts every axis after it, so the Z index has to be looked up again afterwards or the projection hits the wrong axis.

## spacrStitcher._direction_bin

### line 1903, trailing

```python
ang = np.degrees(np.arctan2(ty, tx))
```

[-180,180]

## spacrStitcher._compute_mosaic_transforms

### line 1960  _(unsure)_

```python
nodes = set()
```

Nodes present

### line 1968  _(unsure)_

```python
step_x, step_y = self._estimate_grid_steps(rows, min_score, angle_tol_deg=max(30.0, angle_tol_deg))
```

Step estimates from high-score pairs

### lines 2018-2020

```python
best_per_node_dir: Dict[Tuple[str,str], Tuple[float, str, str, np.ndarray]] = {}
```

Gather candidate edges. With `cap_one_per_dir` the best per

(tile, direction) wins; without it every edge that passes `edge_ok` survives and the spanning tree chooses among them.

### line 2029  _(unsure)_

```python
for (src, dst, M_src_to_dst) in (
```

B->A from CSV row; also add A->B via inverse

### line 2031, trailing  _(unsure)_

```python
(B, A, self._affine_from_row(r)),
```

B->A

### line 2032, trailing  _(unsure)_

```python
(A, B, None)
```

A->B (inverse later)

### line 2039  _(unsure)_

```python
a,b = float(M_src_to_dst[0,0]), float(M_src_to_dst[0,1])
```

Recover theta, scale of this directed transform (approx)

### lines 2055-2065

```python
all_edges.append((sc, src, dst, M_src_to_dst))
```

`cap_one_per_dir=False` is asked for when the best-scoring edge in a direction is a FALSE match -- a repeated background pattern outscoring the true neighbour. Keeping only the winner then hands the spanning tree the wrong edge and no alternative, which is the whole reason to be able to turn the cap off.

The cap used to run unconditionally: this parameter is declared here and on both public mosaic APIs (align_mosaic_from_csv, ops_align_mosaic), documented on all three, threaded down -- and never read.

### line 2086  _(unsure)_

```python
idx = {p:i for i,p in enumerate(nodes)}
```

Kruskal MST on pruned edges (max spanning)

### line 2112  _(unsure)_

```python
adj: Dict[str, List[Tuple[str, np.ndarray, float]]] = {p:[] for p in nodes}
```

Build adjacency for traversal using the selected MST edges

### lines 2118-2120

```python
adj[src].append((dst, self._invert_affine(M), sc))
```

The BFS below reads adj[u] as (v, M_v_to_u), so each entry must carry the transform *into* the key's frame: from src that is dst->src (the inverse of M), and from dst it is src->dst (M).

### line 2127  _(unsure)_

```python
root = max(nodes, key=lambda p: len(adj[p]))
```

Choose root = node with max degree in MST

### line 2130

```python
T3: Dict[str, np.ndarray] = {}
```

BFS to compute transforms to root (homogeneous 3x3 to avoid shape bugs)

### line 2147  _(unsure)_

```python
T2: Dict[str, np.ndarray] = {k: v[:2,:] for k,v in T3.items()}
```

Convert to 2x3 for rendering

## spacrStitcher._compute_mosaic_transforms.edge_ok

### line 1971  _(unsure)_

```python
def edge_ok(tx, ty, theta, scale, dbin):
```

Helper to check geometry/tolerances for one directed edge (src->dst)

### line 2000  _(unsure)_

```python
if not self.allow_rotation and abs(theta) > float(rot_tol_deg):
```

rotation/scale limits (if disallowed)

### line 2005  _(unsure)_

```python
if dbin in ("R", "L"):
```

step gating

### line 2011, trailing  _(unsure)_

```python
else:
```

U/D

## spacrStitcher.render_mosaic_from_csv._series_dtype

### line 2183  _(unsure)_

```python
def _series_dtype(p: str) -> np.dtype:
```

dtype helpers

## spacrStitcher.render_mosaic_from_csv

### line 2197  _(unsure)_

```python
manifest_only = (out_csv is not None) and (out_tif is None)
```

NEW: manifest-only mode (no mosaic rendering)

### line 2200  _(unsure)_

```python
out_png = None
```

No point accepting a PNG target if we aren't rendering

### line 2203  _(unsure)_

```python
rows: List[Dict] = []
```

Load rows

### line 2221  _(unsure)_

```python
T, used_edges = self._compute_mosaic_transforms(
```

Transforms to a common root using pruned graph

### line 2236  _(unsure)_

```python
all_x, all_y = [], []
```

Determine canvas bounds (+ optionally remember per-node dtype)

### line 2239, trailing  _(unsure)_

```python
node_dtype: Dict[str, np.dtype] = {}
```

only used when writing out_tif

### line 2267  _(unsure)_

```python
if not manifest_only:
```

Only allocate + blend if we are actually rendering

### line 2288  _(unsure)_

```python
warped = cv2.warpAffine(I, M_can, (Wc, Hc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
```

warp image and a 1-mask (coverage)

### line 2296  _(unsure)_

```python
corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], dtype=np.float32).reshape(-1, 1, 2)
```

Top-left of warped bbox (for manifest)

### line 2313  _(unsure)_

```python
if out_csv is not None:
```

Write manifest CSV (works in both modes)

### line 2328  _(unsure)_

```python
if manifest_only:
```

If manifest-only, stop here (no mosaic image)

### line 2330, trailing  _(unsure)_

```python
return out_tif, out_png
```

both None in this mode

### lines 2338-2339

```python
out = np.where(wgt > 0, np.divide(canvas, np.maximum(wgt, 1e-6)), 0.0)
```

Otherwise, build and save mosaic image(s) uncovered canvas stays at background level (see stitch_pair)

## spacrStitcher.mosaic_all_channels_from_csv

### line 2396  _(unsure)_

```python
manifest_only = (out_csv is not None) and (out_tif is None)
```

NEW: manifest-only mode (no mosaic rendering)

### line 2432  _(unsure)_

```python
shapes: Dict[str, Tuple[int, int]] = {}
```

per-node shape (and dtype/channel info only if rendering)

### line 2445  _(unsure)_

```python
if not manifest_only:
```

decide which channels to mosaic (only if rendering)

### line 2458  _(unsure)_

```python
all_x, all_y = [], []
```

determine canvas bounds (from transforms on geometry)

### line 2481  _(unsure)_

```python
if out_csv is not None:
```

optional manifest (works in both modes)

### line 2518, trailing  _(unsure)_

```python
return out_tif
```

None in this mode

### line 2520

```python
if out_tif is None:
```

otherwise, render and save mosaic TIFF

## StitchedMultiAligner.__init__

### line 2611  _(unsure)_

```python
arr_axes: str = "AUTO",
```

axes/time/Z

### line 2645  _(unsure)_

```python
if self.detector == "ORB":
```

detector init

## StitchedMultiAligner._normalize_to_yx

### lines 2727-2729

```python
while len(ax) > arr.ndim:
```

Stop as soon as a pass removes nothing, otherwise an axes string with no T/C to give up (e.g. arr_axes="ZYX" against a 2-D plane) spins forever here.

## StitchedMultiAligner._read_all_channels_cyx

### line 2848, trailing  _(unsure)_

```python
return np.stack(planes, axis=0)
```

(C,H,W)

## StitchedMultiAligner.align._series_dtype

### line 2968  _(unsure)_

```python
def _series_dtype(p: str) -> np.dtype:
```

dtype helpers (local)

## StitchedMultiAligner.align

### line 2990  _(unsure)_

```python
s = float(self.downsample)
```

DS ref for features (8-bit only here)

### line 2998  _(unsure)_

```python
all_arrays: List[np.ndarray] = []
```

Output buffer

### line 3002  _(unsure)_

```python
input_dtypes: List[np.dtype] = [_series_dtype(ref_path)]
```

Keep track of input dtypes to choose a common output dtype

### line 3005  _(unsure)_

```python
A0 = self._read_all_channels_cyx(ref_path)
```

Reference channels (no warp)

### line 3019  _(unsure)_

```python
for k in range(1, len(paths)):
```

Others: estimate M (B -> ref), warp all channels at full-res

### line 3043  _(unsure)_

```python
A_lin = M_ds[:, :2].astype(np.float32)
```

constraints

### line 3056  _(unsure)_

```python
M_full = M_ds.copy()
```

lift to full res

### line 3066  _(unsure)_

```python
B_warp_ds = cv2.warpAffine(Iu8, M_ds, (Wds, Hds), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_C...
```

score on DS (foreground of ref)

### line 3072  _(unsure)_

```python
B_all = self._read_all_channels_cyx(p)  # float32 workspace
```

warp all channels

### line 3073, trailing  _(unsure)_

```python
B_all = self._read_all_channels_cyx(p)
```

float32 workspace

### line 3090  _(unsure)_

```python
out = np.concatenate(all_arrays, axis=0)
```

concatenate (float32 workspace) and SAVE using common input dtype

## stitch_cycle_wells

### line 3155, trailing

```python
collision = settings.get("collision", "rename")
```

{'rename','skip','overwrite'}

### line 3156, trailing

```python
on_missing = settings.get("on_missing", "error")
```

{'error','skip'}

### line 3162  _(unsure)_

```python
plate_id = settings.get("plate") or settings.get("plate_id") or settings.get("experiment") or os....
```

plate id for filenames (plate + well metadata in all CSV names)

### line 3211  _(unsure)_

```python
moved = 0
```

Organize into per-well folders (or create symlinks if not organizing)

### line 3217, trailing  _(unsure)_

```python
link_root = os.path.join(dst_root, "_links")
```

used when do_organize=False

### line 3268  _(unsure)_

```python
dp = os.path.join(link_well_dir, fn)
```

create a symlink into link_well_dir

### line 3311  _(unsure)_

```python
scan_dir = os.path.dirname(well_files[0])
```

Where run_folder will scan (keep your behavior)

### lines 3318-3322

```python
orig_outdir = os.path.join(well_root, well)                  # {src}/{well}/{well}
```

Desired layout images moved to:     {dst_root}/{well}/{well} qc images saved in:  {dst_root}/{well}/qc/pairs stitch outputs in:   {dst_root}/{well}/{well}/stitch csv files in:        {dst_root}/{well}/results

### line 3323, trailing  _(unsure)_

```python
orig_outdir = os.path.join(well_root, well)
```

{src}/{well}/{well}

### line 3324, trailing  _(unsure)_

```python
qc_pairs_dir = os.path.join(well_root, "qc", "pairs")
```

{src}/{well}/qc/pairs

### line 3326, trailing  _(unsure)_

```python
results_outdir = os.path.join(well_root, "results")
```

{src}/{well}/results

### line 3327, trailing  _(unsure)_

```python
feat_cache_dir = os.path.join(well_root, "cache")
```

tidy cache

### line 3341  _(unsure)_

```python
mosaic_tif_sc = os.path.join(stitch_outdir, f"{prefix}_mosaic_full.tif")
```

Single-channel mosaic (if multichannel=False)

### line 3345  _(unsure)_

```python
mosaic_tif_mc = os.path.join(stitch_outdir, f"{prefix}_mosaic_allc.tif")
```

Multi-channel mosaic (if multichannel=True)

### lines 3350-3353

```python
mosaic_out = mosaic_tif_mc if do_mc else mosaic_tif_sc
```

THE PATH IS ALWAYS PREPARED; whether a mosaic is BUILT is decided by `want_mosaic` below. This read the other way round -- asking for a mosaic set the output path to None -- so `write_mosaic=True` was the one value guaranteed to produce nothing.

### lines 3356-3357  _(unsure)_

```python
stitcher = spacrStitcher(
```

Instantiate stitcher with your settings

NOTE: outdir now points at qc/pairs (so qc is not mixed with tiles)

### lines 3367-3369

```python
cellpose_model=settings.get("cellpose_model") or "cpsam",
```

Not str(...): str(None) is the four-character string "None", which _resolve_cellpose_pretrained would report as an unknown model name rather than as "no model named".

### line 3398  _(unsure)_

```python
ch_order = settings.get("channel_indices", None)  # None → infer from tiles
```

Run per-well; enable mosaic here (single- or multi-channel)

### line 3400, trailing  _(unsure)_

```python
ch_order = settings.get("channel_indices", None)
```

None → infer from tiles

### line 3414, trailing  _(unsure)_

```python
meta_regex=meta_re,
```

use compiled regex here

### lines 3415-3419

```python
mosaic=bool(settings.get("write_mosaic", False)
```

`write_mosaic` and `mosaic` are synonyms, and either turns it on. They were separate keys with separate defaults, so the one a reader would reach for -- `write_mosaic` -- was not the one `run_folder` consulted, and this function never built the mosaic its own docstring promises.

### line 3426, trailing  _(unsure)_

```python
mosaic_channel_count=None,
```

infer min across tiles unless order provided

### line 3430  _(unsure)_

```python
moved_tiles: List[str] = []
```

Move (or mirror) tiles into {well_root}/{well}/{...} after stitching

### line 3441  _(unsure)_

```python
if collision == "overwrite" and os.path.exists(dp) and rp == dp:
```

move file into orig_outdir

### line 3449  _(unsure)_

```python
target = os.path.realpath(sp)
```

create a symlink into orig_outdir (keep source untouched)

### line 3460  _(unsure)_

```python
by_well_outpaths[well] = moved_tiles
```

keep return metadata consistent with the new layout

### line 3482  _(unsure)_

```python
organized_summary["by_well"] = by_well_outpaths
```

update organized_summary to reflect the final tile locations

## stitch_cycle_wells._resolve_collision

### line 3229  _(unsure)_

```python
base, ext = os.path.splitext(dst_path)
```

rename: add numeric suffix

## get_preprocess_ops_settings

### line 3494  _(unsure)_

```python
settings.setdefault("phenotype_source", "path")
```

high-level sources

### line 3498  _(unsure)_

```python
settings.setdefault("src", None)
```

IO / basic parsing

### line 3509, trailing

```python
settings.setdefault("collision", "rename")
```

{'rename','skip','overwrite'}

### line 3510, trailing

```python
settings.setdefault("on_missing", "error")
```

{'error','skip'}

### lines 3514-3518

```python
settings.setdefault("ops_gpu", True)
```

HARDWARE. Named `ops_gpu` and not `gpu` on purpose: `gpu` is already declared by Image UMAP, where it means "use the RAPIDS cuML backend", and two modules disagreeing about what a shared key means is the bug `register_defaults` refuses `src` to prevent. Same reasoning, applied before the collision rather than after it.

### lines 3521-3545

```python
settings.setdefault("stitch", False)
```

THE TWO SWITCHES THE PIPELINE'S OWN PURPOSE DEPENDS ON, and they were not here. `stitch_cycle_wells` reads both with a `False` fallback (~:3178 and ~:3181) and this factory never set either, so a settings panel generated from it would not OFFER them -- and a default run therefore wrote no mosaic, after which `align_image_to_stitch` found none and returned `{}` WITH NO ERROR. The pipeline succeeded and produced nothing.

FALSE, WHICH IS WHAT THEY ALREADY WERE. Adding them here changes no run: the fallback at the read sites is `False` and that is what is set. What changes is that a panel generated from this factory now OFFERS them, which was the audit's actual complaint -- the switches existed and were unreachable.

THEY PROBABLY SHOULD DEFAULT TRUE AND THAT IS NOT MINE TO DECIDE. `ops_preprocess`'s docstring is "per-genotype stitching + phenotype alignment", and with `mosaic` off the alignment half has nothing to align to, so the default run succeeds and produces nothing. But flipping it was tried and it turns that silent no-op into a RAISE on input that cannot be mosaicked `test_the_post_stitch_move_never_finds_a_tile_already_at_its_target` goes from passing to `RuntimeError: mosaic_all_channels_from_csv: CSV has no usable rows`. Silence and a crash are both wrong and the choice between them is a product decision about what an OPS run is FOR, made with the maintainer awake. Recorded in instruction 372.

### lines 3549-3551

```python
settings.setdefault("plate", "")
```

Read at ~:2931 as `plate` or `plate_id` or `experiment`, falling back to the destination folder's name. Offered here so a panel can show it; "" is falsy, so leaving it empty keeps the existing fallback exactly.

### line 3554  _(unsure)_

```python
settings.setdefault("do_organize", True)
```

pipeline toggles

### line 3559  _(unsure)_

```python
settings.setdefault("channel_index", 0)
```

alignment / nuclei channel

### lines 3574-3575

```python
settings.setdefault("cellpose_model", "cpsam")
```

Only read when outline_source='cellpose'. 'cpsam' is the stock Cellpose 4 model; a path here loads a checkpoint from Train Cellpose instead.

### lines 3586-3592

```python
settings.setdefault("feature_cache_dir", None)     # per well
```

REMOVED 2026-09-03: `max_qc_plots_total` and `plot_only_above_threshold` were read by nothing. Each appeared exactly once in the package -- on its own `setdefault` here -- so the cap was never applied and the threshold never consulted. Instruction 364's standard is that a setting offered and never acted on is deleted rather than documented, because a tooltip on a dead control teaches the user a lie about what the run will do. Found by 372's audit.

### line 3593, trailing  _(unsure)_

```python
settings.setdefault("feature_cache_dir", None)
```

per well

### lines 3604-3609

```python
settings.setdefault("write_mosaic", False)
```

FALSE, and the docstring of `stitch_cycle_wells` was corrected to match rather than the other way round. Defaulting it True was tried and reverted: `mosaic_all_channels_from_csv` RAISES when the pairs CSV has no usable rows, so a plate with too few overlapping tiles would go from succeeding quietly to failing loudly, for a mosaic nobody asked for. The flag now WORKS when set, which is the fix that was wanted.

### line 3615, trailing  _(unsure)_

```python
settings.setdefault("mosaic_min_score", None)
```

auto elbow

### line 3617  _(unsure)_

```python
settings.setdefault("mosaic_out", None)
```

per-well outputs (filled by caller, if desired)

### line 3622, trailing  _(unsure)_

```python
settings.setdefault("channel_indices", None)
```

infer from first tile

### line 3626  _(unsure)_

```python
settings.setdefault("tmp_dir", None)
```

per-well outputs (filled by caller)

## FOVAlignAndCropper.__init__

### line 3669  _(unsure)_

```python
arr_axes: str = "AUTO",
```

axes/time/Z

## FOVAlignAndCropper._read_plane

### lines 3691-3696

```python
def _read_plane(self, *a, **k):
```

Small proxies for IO helpers.

DELEGATED RATHER THAN INHERITED OR COPIED. The cropper is not a kind of aligner -- it holds one -- so the eight helpers it shares with :class:`StitchedMultiAligner` are forwarded to that instance. One definition, and changing the aligner's reader changes the cropper's too.

## FOVAlignAndCropper.run

### line 3798  _(unsure)_

```python
if csv_path is None:
```

Outputs

### line 3805  _(unsure)_

```python
mosa_all = self._read_all_channels_cyx(stitched_path)   # (C_m, Hm, Wm)
```

Load mosaic (all channels) and nuclei for features

### line 3806, trailing  _(unsure)_

```python
mosa_all = self._read_all_channels_cyx(stitched_path)
```

(C_m, Hm, Wm)

### line 3810  _(unsure)_

```python
s = float(self._aligner.downsample)
```

Feature DS for mosaic

### line 3819  _(unsure)_

```python
s_known = (float(folder_image_scale) if folder_image_scale is not None
```

Known FOV→mosaic scale

### line 3844  _(unsure)_

```python
Wfds = max(1, int(round(Wf * s * s_known)))   # FIXED name
```

DS for FOV features *including known scale*

### line 3845, trailing  _(unsure)_

```python
Wfds = max(1, int(round(Wf * s * s_known)))
```

FIXED name

### line 3846, trailing  _(unsure)_

```python
Hfds = max(1, int(round(Hf * s * s_known)))
```

FIXED name

### line 3847, trailing  _(unsure)_

```python
fov_ds = cv2.resize(fov_nuc, (Wfds, Hfds), interpolation=cv2.INTER_LINEAR)
```

FIXED usage

### line 3853  _(unsure)_

```python
ptsA, ptsB = self._match(Fref, Fb)
```

Match (A=mosaic_ds, B=fov_ds)

### line 3862  _(unsure)_

```python
A_lin = M_ds[:, :2].astype(np.float32)
```

Optional constraints (keep known scale separate from "disallowed scale")

### lines 3876-3881

```python
M_full = M_ds.astype(np.float32).copy()
```

Lift DS → full-res with known scale

DS relation: x_mosa_ds = A_ds * x_fov_ds + t_ds, x_fov_ds  = s*s_known*x_fov_full,  x_mosa_ds = s*x_mosa_full ⇒ x_mosa_full = A_ds*(s_known)*x_fov_full + t_ds/s ⇒ A_full = s_known * A_ds ; t_full = t_ds / s

### line 3887  _(unsure)_

```python
a, b, tx = float(M_full[0, 0]), float(M_full[0, 1]), float(M_full[0, 2])
```

Decompose

### line 3893  _(unsure)_

```python
B_warp_ds = cv2.warpAffine(fov_u8, M_ds, (Wmds, Hmds),
```

Score on DS (foreground of mosaic nuclei)

### line 3902  _(unsure)_

```python
corners = np.array([[0, 0], [Wf, 0], [0, Hf], [Wf, Hf]], dtype=np.float32).reshape(-1, 1, 2)
```

Compute FOV bbox top-left in mosaic coords (using M_full)

### lines 3908-3909  _(unsure)_

```python
fov_all = self._read_all_channels_cyx(p)         # (C_f, Hf, Wf), float32
```

Read full FOV & Mosaic channels and build stacked output:

[FOV channels; mosaic channels warped into FOV frame]

### line 3910, trailing  _(unsure)_

```python
fov_all = self._read_all_channels_cyx(p)
```

(C_f, Hf, Wf), float32

### line 3911, trailing  _(unsure)_

```python
mosa_all_full = mosa_all
```

(C_m, Hm, Wm), already loaded

### line 3916  _(unsure)_

```python
C_f = fov_all.shape[0]
```

Warp mosaic channels into FOV frame

### line 3932  _(unsure)_

```python
w.writerow(dict(
```

Write CSV row

### lines 3943-3945

```python
print(f"[FOVAlignAndCropper.run] Skipping {os.path.basename(p)}: {e}",
```

Skip the FOV, but never silently: an unreadable file or a bad transform would otherwise leave an empty manifest with no explanation at all.

## align_image_to_stitch

### line 3959, trailing

```python
relative_scale: float = 2.0,
```

20× vs 10× → ~2.0; adjust as needed

### lines 4055-4056  _(unsure)_

```python
wells_with_mosaic: Dict[str, str] = {}
```

1) find per-well mosaics built by stitch_cycle_wells

Expected location from your pipeline: <stitch_dst_root>/<WELL>/_stitch/mosaic_allc.tif

### lines 4066-4067

```python
import glob as _glob
```

stitch_cycle_wells writes <well>/stitch/<plate>_<well>_mosaic_allc.tif; without this the two halves of the pipeline never meet.

### line 4074

```python
meta_re = re.compile(meta_regex, re.IGNORECASE)
```

2) group 20× (align) images by well

### line 4079  _(unsure)_

```python
results: Dict[str, Dict[str, str]] = {}
```

3) per-well FOV→mosaic alignment via FOVAlignAndCropper

### line 4086, trailing

```python
continue
```

no 20× images for this well

### line 4088  _(unsure)_

```python
link_well = os.path.join(links_root, well)
```

make a light per-well link folder so paths are clean/reproducible

### lines 4095-4096

```python
aligner = FOVAlignAndCropper(
```

Instantiate the aligner per well so its outdir lands next to that well's mosaic instead of polluting the caller's working directory.

## align_image_to_stitch._group_by_well._site_key

### line 4027  _(unsure)_

```python
def _site_key(p):
```

sort per site if present

## ops_preprocess

### line 4142  _(unsure)_

```python
settings = get_preprocess_ops_settings(settings)
```

Fill in defaults for all stitching / alignment-related keys

### line 4153  _(unsure)_

```python
npy_out_root = os.path.join(phenotype_src, "output")
```

Where to store npy outputs (you can change this if you like)

### line 4157  _(unsure)_

```python
if isinstance(genotype_src, (str, os.PathLike)):
```

Normalize genotype_src into a list of folders

### line 4160  _(unsure)_

```python
subdirs = [
```

List subdirectories; if none, treat the folder itself as one genotype

### line 4182, trailing  _(unsure)_

```python
stitch_settings = dict(settings)
```

shallow copy is fine for simple values

### lines 4193-4194  _(unsure)_

```python
if "align_image_to_stitch" in globals():
```

2) alignment of phenotype images to stitched mosaics

Only run if align_image_to_stitch is available in this module.

### lines 4203-4209

```python
)
```

`qc_outlines` is NOT passed. It was, and that was the whole of its life: the ops defaults set it True, this line read it, align_image_to_stitch accepted it, and nothing switched on the *__qc_outlines.png overlays belong to spacrStitcher and are gated on its own `save_qc`, which defaults False. A key whose only reader hands it to an inert parameter is not a read; use save_qc on the stitching step.
