# Notes from `spacr/object.py`

Prose lifted out of `spacr/object.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [display](#display) (1 entry)
- [merge_split_filter_masks](#merge_split_filter_masks) (2 entries)
- [_run_seg_qc](#_run_seg_qc) (2 entries)
- [_z_stack_plan](#_z_stack_plan) (1 entry)
- [_cellpose_z_segment_fn._segment](#_cellpose_z_segment_fn_segment) (2 entries)
- [_segment_volumes_with_z](#_segment_volumes_with_z) (1 entry)
- [_t_stack_plan](#_t_stack_plan) (1 entry)
- [_reconcile_z_and_t_plans](#_reconcile_z_and_t_plans) (1 entry)
- [_segment_timepoints_with_t](#_segment_timepoints_with_t) (2 entries)
- [generate_cellpose_masks_sam](#generate_cellpose_masks_sam) (21 entries)
- [generate_cellpose_masks](#generate_cellpose_masks) (11 entries)
- [generate_organelle_masks_sam](#generate_organelle_masks_sam) (9 entries)
- [_extract_classical_settings](#_extract_classical_settings) (1 entry)
- [_segment_cellpose](#_segment_cellpose) (2 entries)
- [_segment_unet](#_segment_unet) (1 entry)
- [_segment_spots](#_segment_spots) (1 entry)
- [_network_hysteresis](#_network_hysteresis) (1 entry)
- [_segment_ring](#_segment_ring) (5 entries)
- [generate_cellpose_masks_sam, 2026-09-19](#generate_cellpose_masks_sam-2026-09-19) (1 entry)

## Module level

### line 5

```python
from . import accelerator
```

CUDA, ROCm, Metal or XPU from one resolver -- see instruction 319.

## display

### lines 13-16

```python
def display(*args, **kwargs):
```

IPython may be mid-init (partially imported by another thread) — use a no-op fallback so importing this module never blocks. spaCR only calls display() from notebook contexts anyway; the Qt GUI ignores it.

## merge_split_filter_masks

### lines 125-128

```python
ith = settings.get(f'{object_type}_intensity_threshold', None)
```

ONE ABSOLUTE THRESHOLD IN RAW IMAGE UNITS (391), replacing the method dropdown, the merge percentile and the two filter percentiles. There is no default: the right number depends on the acquisition, and a default would be wrong for every image that is not the one it was chosen on.

### line 216  _(unsure)_

```python
filtered_masks = [
```

Always run serial so progress prints work

## _run_seg_qc

### lines 256-257  _(unsure)_

```python
dst = os.path.dirname(src) or src
```

Same idiom as `count_loc` above: plate-level output lives one level up from the mask source, next to measurements/.

### lines 268-269

```python
print(f"Segmentation QC skipped for {object_type}: {type(exc).__name__}: {exc}")
```

QC is a report, never a gate. A run that has just spent hours segmenting must not lose its masks to a scorecard bug.

## _z_stack_plan

### lines 278-287

```python
def _z_stack_plan(settings):
```

3D (Beta): z-stack plumbing

Everything below is inert unless the `z_stack` setting is on. `_z_stack_plan` returns None in that case and every call site branches on it, so a run that has not opted in executes not one line of z code and produces byte-identical masks to a run from before these settings existed. That property is the acceptance criterion and is asserted in tests/test_zstack.py.

## _cellpose_z_segment_fn._segment

### lines 371-372  _(unsure)_

```python
planes = [array[z] for z in range(array.shape[0])]
```

One 2-D call per plane. Labels come back plane-local; zstack links them.

### line 378  _(unsure)_

```python
output = model.eval(x=[array], **kwargs)
```

Projected 2-D plane: exactly the ordinary single-image call.

## _segment_volumes_with_z

### lines 429-431

```python
intensity = np.stack([
```

The same projection segment_3d just made. The merge/split/filter step scores masks against intensities, so it must see the plane the masks were drawn on, not the volume it came from.

## _t_stack_plan

### lines 440-474

```python
def _t_stack_plan(settings):
```

4D (Beta): the time axis on top of the z axis

The same contract as the 3D block above, one axis further out. Everything here is inert unless the `t_stack` setting is on: `_t_stack_plan` returns None then and every call site branches on it, so a run that has not opted in executes not one line of 4-D code and produces byte-identical masks to a run from before these settings existed. That property is the acceptance criterion and is asserted in tests/test_object_tstack_wiring.py.

What `t_stack` declares, exactly

It reinterprets the **leading axis of the .npz batch as time** rather than as a list of independent fields, and requires a z axis behind it -- a batch is then one `(T, Z, Y, X, C)` acquisition instead of `N` separate `(Z, Y, X, C)` fields. Which of the two leading axes is t and which is z is never guessed; `zstack.plan_4d_from_settings` refuses to build a plan at all until `t_axis_order` (or `t_axis`/`z_axis`) says, because reading one as the other links objects down a z stack and reports them as motion.

How far it gets today, stated plainly

`spacr.io._rename_and_organize_image_files` collapses z into one plane per field while organising the raw files, so an ordinary run's batches are `(N, Y, X, C)` and there is no z axis left by the time segmentation sees them. `_require_t_axis` therefore stops such a run with `TAxisNotPresentError` naming that as the cause, rather than segmenting the projection frame by frame and reporting a 4-D result. Handed a genuine `(T, Z, Y, X, C)` array through the Python API -- write the .npz yourself the path runs end to end into `zstack.segment_4d`.

Linking across t (`zstack.track_4d`) is deliberately *not* wired here: its call site is `spacr.timelapse`, not this module. `t_stack` drives segmentation only, and says so.

## _reconcile_z_and_t_plans

### lines 531-532  _(unsure)_

```python
if timelapse and t_plan.z_axis is not None and t_plan.z_mode != 'project':
```

Only a plan that actually produces (Z, Y, X) volumes is a problem for them; a flat time series, or 'project', leaves the masks 2-D.

## _segment_timepoints_with_t

### lines 629-631

```python
pass
```

A flat time series: there is no z to collapse, so there is no projected copy either and the caller scores against the batch it already has, exactly as the ordinary 2-D path does.

### lines 634-636

```python
intensity = np.stack([
```

The same projection segment_3d just made, one per timepoint. The merge/split/filter step scores masks against intensities, so it must see the plane the masks were drawn on, not the volume it came from.

## generate_cellpose_masks_sam

### lines 716-720

```python
timelapse = settings.get('timelapse', False)
```

`timelapse` is no longer offered by the Mask module's settings panel — it belongs to the Timelapse module (spacr.core.preprocess_generate_masks_timelapse). It is still defaulted by set_default_settings_preprocess_generate_masks and still honoured here, so old settings CSVs and direct API calls keep working; .get() keeps a hand-built dict from raising instead of segmenting.

### lines 737-738  _(unsure)_

```python
z_plan = _z_stack_plan(settings)
```

None unless the user opted into 3D (Beta). Every branch below is guarded on it, so the 2-D path is untouched when it is None.

### lines 741-743

```python
t_plan = _t_stack_plan(settings)
```

None unless the user opted into 4D (Beta). Raises here, before the model is loaded and the first field read, when the axis order is not settled that answer cannot change later in the run.

### lines 747-750

```python
if t_plan is not None:
```

The z mode that actually runs, whichever plan is driving. None means no z code runs at all and the masks are ordinary 2-D ones: either neither plan is set, or the 4-D plan describes a flat (T, Y, X) time series, for which segment_4d makes one plain 2-D call per frame and no z mode enters.

### lines 758-774

```python
from .utils import dense_mask_channel_positions
```

THE cellpose_* KEYS HOLD DENSE STACK POSITIONS, NOT RAW CHANNELS. io.preprocess_img_data writes them as `seen[ch]` -- the position on the merged stack's channel axis, which is built in ROLE order (nucleus, cell, pathogen, organelle), deduplicated.

This fallback used to copy the RAW channel across, which is a different number whenever the roles are not in ascending channel order. It fires more often than it looks: preprocess_img_data returns early once the raw images have been moved into src/orig (so on every re-run), and is skipped entirely when preprocess=False -- both documented workflows. With nucleus_channel=1 and cell_channel=0 the first run records nucleus->0, cell->1; the resumed run wrote nucleus->1, cell->0, and Cellpose segmented nuclei on the cell image and cells on the nucleus image with no warning.

`organelle` is filled in too. It never was, so a resumed run with an organelle had no recorded position at all.

### lines 788-791

```python
settings[f'cellpose_{_role}_channel'] = _dense[_raw]
```

``dense_mask_channel_positions`` walks this same role key before returning, with the same ``int`` coercion.  A numeric raw channel is therefore necessarily present; indexing directly keeps any future drift loud instead of silently leaving the alias unset.

### lines 803-807

```python
model_name = object_settings['model_name']
```

pretrained_model used to be the literal 'cpsam' here, so a checkpoint from spaCR's own Train Cellpose module was discarded and the stock weights ran instead — silently, on the pipeline's DEFAULT path. _resolve_cellpose_pretrained keeps 'cpsam' for the stock case and returns the checkpoint path when the user named one.

### lines 810-815

```python
model_name = settings['pathogen_model']
```

LEGACY ONLY. `pathogen_model` was a second setting naming the same thing as `pathogen_model_name`, and two controls for one value is how a user sets one and wonders why the other wins. It is no longer OFFERED -- see _APP_HIDDEN_KEYS -- and is read here so a settings CSV written before it was retired still segments with the model it names rather than silently falling back to cpsam.

### lines 867-868

```python
_require_t_axis(stack, t_plan, path)
```

Fail before the first timepoint rather than after: whether this array is 4-D cannot change later in the run.

### lines 871-872

```python
_require_z_axis(stack, z_plan, path)
```

Fail before the first field rather than after: whether this array has a z axis cannot change later in the run.

### lines 879-881

```python
batch = stack[i: i+batch_size][..., channels].astype(stack.dtype)
```

(N, Z, Y, X, C) — or (T, Z, Y, X, C) under t_stack, where the leading axis is time: select channels off the trailing axis so the z axis is preserved.

### lines 888-893

```python
batch_filenames = filenames[i: i+batch_size].tolist()
```

In the future drop the npz save file step, just keep it in memory and pass the batch directly to the model. This will save time and disk space. For now, keep it for backwards compatibility and to avoid issues with large batches that might not fit in memory. if stack.shape[3] == 1: batch = stack[i: i+batch_size, :, :, [0]].astype(stack.dtype) else: subset = stack[i: i+batch_size, :, :, channels_to_extract].astype(stack.dtype) batch = subset[:, :, :, channels]

### lines 922-927

```python
diameter=_eval_diameter(
```

Cellpose 4 still honours `diameter` in eval() — it rescales the image by 30/diameter. Only diam_mean at construction is ignored. This was hard-coded to None, so an explicitly-set <obj>_diameter (and anything spacr.diameter proposes) never reached Cellpose. The setting defaults to None, so None here still means "let CPSAM work at native scale".

### lines 938-941

```python
z_eval_kwargs = dict(
```

Same eval kwargs as the 2-D call above, minus the ones zstack sets per mode (x, batch_size, channel_axis, do_3D, anisotropy, z_axis). Shared by the 3-D and 4-D paths so the two cannot drive Cellpose differently.

### lines 956-957  _(unsure)_

```python
masks, t_result, beta_intensity = _segment_timepoints_with_t(
```

The whole batch is one acquisition, not a list of independent fields: its leading axis is time.

### lines 979-982

```python
masks = merge_split_filter_masks(
```

merge/split/filter reason in 2-D: they measure areas in px² and split objects with a 2-D watershed. Handing them a (Z, Y, X) volume would silently apply all of that per plane and tear the 3-D labels apart, so the 3-D modes skip them.

### lines 1032-1034

```python
mask_stack = _trackastra_track_cells(
```

Trackastra takes the raw intensity stack as well as the masks — it uses appearance, not just geometry — so hand it the batch we already loaded rather than masks alone.

### lines 1050-1053

```python
mask_stack = _ultrack_track_cells(
```

Ultrack derives its own candidate objects from a contour map built off these labels, and uses the raw intensities for appearance features while linking, so it gets the same two arrays trackastra does.

### lines 1095-1098

```python
if timelapse and settings.get("motility_analysis", False):
```

Legacy inline hook: the automated motility assay is now the standalone Motility Assay module (app key 'motility'), so the Mask GUI no longer exposes `motility_analysis`. The gate stays for settings CSVs and API callers that still set both flags.

### lines 1114-1119

```python
if not timelapse:
```

Plot and save inside the per-batch loop. Both blocks used to sit one level out, at the .npz level: an .npz holding more batches than `batch_size` therefore ran every batch but only ever wrote the last one's masks to disk (the earlier mask_stacks were rebound and lost), while an empty .npz never entered this loop at all and hit the save block with mask_stack unbound -> NameError.

### lines 1123-1127

```python
reason = (f"z_segmentation_mode='{beta_mode}'"
```

plot_cellpose4_output draws the per-image flow field beside each mask; the z paths call eval once per volume and do not collect one, and in the stitch/volumetric modes the mask is a (Z, Y, X) volume it cannot render beside a 2-D field either.

## generate_cellpose_masks

### lines 1184-1185

```python
_refuse_t_stack(settings, 'object.generate_cellpose_masks')
```

This generator has no 4-D path. Say so rather than returning 2-D masks to a user whose settings said 4-D.

### lines 1194-1198

```python
timelapse = settings.get('timelapse', False)
```

`timelapse` is no longer offered by the Mask module's settings panel — it belongs to the Timelapse module (spacr.core.preprocess_generate_masks_timelapse). It is still defaulted by set_default_settings_preprocess_generate_masks and still honoured here, so old settings CSVs and direct API calls keep working; .get() keeps a hand-built dict from raising instead of segmenting.

### lines 1219-1221

```python
from .utils import dense_mask_channel_positions
```

The same fallback as generate_cellpose_masks_sam, and the same reason it has to go through the ROLE-order positions: these keys hold dense stack positions, not raw channel indices. See the longer note there.

### lines 1235-1236  _(unsure)_

```python
settings[f'cellpose_{_role}_channel'] = _dense[_raw]
```

The map was built from this same numeric role channel immediately above, so absence is impossible unless the two contracts drift.

### lines 1239-1244

```python
channels_to_extract, cellpose_channels = _get_cellpose_channels(settings)
```

_get_cellpose_channels takes the settings dict and returns

(channels_to_extract, cellpose_channels). It used to be called here with four positional arguments (src, nucleus, pathogen, cell) left over from an older signature, which raised TypeError on every single call — this whole generator was unreachable. Same call as generate_cellpose_masks_sam makes, so the two cannot pick different channels for the same settings.

### lines 1339-1348

```python
min_size=object_settings['min_size'],
```

No channels=: Cellpose 4 logs "channels deprecated in v4.0.1+" and never reads it, so the pair configured nothing. The planes are already chosen above by stack[..., channels], which is what the remap was always for. <obj>_min_area is documented as "passed to Cellpose as min_size"; this generator never passed it, so Cellpose used its own default of 15 px and the setting did nothing here. The SAM generator has always passed it.

### lines 1416-1429

```python
flows=[flows],
```

_filter_cp_masks iterates zip(masks, flows[0], batch), i.e. it wants the per-image flow list nested one deep. `flows` here is already that per-image list, so passing it bare made flows[0] the FIRST IMAGE's flow array and the zip ran over its rows: any batch with more fields than the images are tall silently lost the trailing masks, and every plot got a single pixel row where a flow image belonged.

### lines 1443-1444  _(unsure)_

```python
flows=[flows],
```

Nested one deep — see the merge branch above.

### lines 1458-1463

```python
mask_stack = _masks_to_masks_stack(masks)
```

`elif not ...merge`, not a bare `else`: with merge on and filter off the block above has already produced the merged stack, and an unconditional else rebound mask_stack to the raw Cellpose masks right after, throwing the merge away. `merge_pathogens` was therefore a no-op in this generator.

### lines 1466-1469

```python
if timelapse and settings.get("motility_analysis", False):
```

Legacy inline hook: the automated motility assay is now the standalone Motility Assay module (app key 'motility'), so the Mask GUI no longer exposes `motility_analysis`. The gate stays for settings CSVs and API callers that still set both flags.

### lines 1485-1488

```python
if not timelapse:
```

Inside the per-batch loop, for the same reason as in generate_cellpose_masks_sam: at the .npz level only the last batch of a multi-batch file was ever written, and an empty .npz reached the save block with mask_stack unbound.

## generate_organelle_masks_sam

### lines 1544-1545

```python
_refuse_t_stack(settings, 'object.generate_organelle_masks_sam')
```

This generator has no 4-D path. Say so rather than returning 2-D masks to a user whose settings said 4-D.

### lines 1551-1569

```python
from .utils import dense_mask_channel_positions
```

The merged .npz stack only contains the channels that map to ENABLED object types, densely re-indexed. Indexing it with the RAW organelle_channel (e.g. 3) blows up when fewer than 4 objects are active — "index 3 is out of bounds for axis 3 with size N".

THE REMAP HAS TO MATCH HOW THE STACK WAS BUILT, and this used to compute `sorted({nucleus, cell, pathogen, organelle})` instead. The stack is built in ROLE order — io.preprocess_img_data walks nucleus, cell, pathogen, organelle and assigns `seen[ch] = len(mask_channels)` — so the two agree only when the roles happen to be in ascending channel order. With nucleus_channel=2, cell_channel=0, organelle_channel=1 the axis is [2, 0, 1], so raw channel 1 sits at position 2, while the sorted reading said position 1 — the CELL plane. Organelles were segmented on the cell image, on a FIRST run, silently.

`cellpose_organelle_channel` wins when present because io.preprocess_img_data records the dense position it actually used. It is absent on a resumed run, which is why the fallback has to be right rather than merely present.

### lines 1609-1611  _(unsure)_

```python
dl_model = None
```

Load deep-learning model once (if needed)

### lines 1628-1630  _(unsure)_

```python
classical_settings = _extract_classical_settings(settings)
```

Build a serialisable settings subset for worker processes

### lines 1633-1635  _(unsure)_

```python
cell_mask_folder = None
```

Optionally load cell masks for per-cell masking

### lines 1657-1662

```python
fields_skipped = 0
```

A stack every field of which is already on disk is skipped in silence otherwise, and a silent skip is indistinguishable from a crash to whoever is watching the log. Before c305bd1b the whole stack was tested up front and said so; that check was replaced by the per-batch `_check_masks` filter, which validates a truncated `.npy` and is the better test -- but it took the sentence with it.

### lines 1694-1696  _(unsure)_

```python
if cell_mask_folder is not None:
```

Per-cell masking: zero out pixels outside cells

### line 1714  _(unsure)_

```python
masks = _segment_classical_parallel(
```

CPU-bound classical methods — parallelise

### line 1736  _(unsure)_

```python
if not np.any(mask_stack):
```

Stats

## _extract_classical_settings

### line 1845  _(unsure)_

```python
'organelle_adaptive_block_size', 'organelle_adaptive_offset',
```

Irregular

## _segment_cellpose

### lines 1936-1937  _(unsure)_

```python
_extract = sorted({c for c in (settings.get('nucleus_channel'),
```

Remap raw object channels to their dense position in the compacted stack (same reasoning as generate_organelle_masks_sam).

### lines 1978-1979

```python
diameter=settings['organelle_diameter'],
```

No channels=: Cellpose 4 never reads it, so [0, 1] configured nothing. cp_batch already holds the planes this call should see.

## _segment_unet

### lines 2055-2057  _(unsure)_

```python
def _segment_unet(img_batch, model, settings):
```

U-Net semantic segmentation (GPU — not parallelised)

## _segment_spots

### line 2155  _(unsure)_

```python
filtered = white_tophat(img, disk(tophat_radius))
```

Pre-filter: white top-hat enhances bright spots on dark bg

## _network_hysteresis

### line 2344

```python
if low < 1.0:
```

Interpret values <1.0 as percentiles

## _segment_ring

### line 2417  _(unsure)_

```python
img_norm = _normalize_01(img)
```

Step 1: Enhance ring structures using DoG (edge enhancement)

### line 2421  _(unsure)_

```python
if method == 'otsu':
```

Step 2: Threshold the enhanced image

### line 2446  _(unsure)_

```python
binary_edges = closing(binary_edges, disk(1))
```

Cleanup edges

### line 2451  _(unsure)_

```python
if fill_method == 'flood':
```

Step 3: Fill rings to get solid objects

### line 2459  _(unsure)_

```python
labeled = sk_label(filled)
```

Step 4: Remove objects that lack ring morphology

## generate_cellpose_masks_sam, 2026-09-19

The `np.load(path)` that GitHub #121 and #117 report ("This file contains pickled (object) data") was handed a macOS AppleDouble sidecar, `masks/._stack_0_norm.npz`, not an archive spaCR wrote. The three `.npz` listings in this module (`generate_cellpose_masks_sam`, `generate_cellpose_masks`, `generate_organelle_masks_sam`) go through `spacr.io._listdir_visible`; the measurement and the reasons are in `docs/notes/spacr/io.md` under `_listdir_visible`.
