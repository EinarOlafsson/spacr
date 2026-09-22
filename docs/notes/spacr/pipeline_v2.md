# Notes from `spacr/pipeline_v2.py`

Prose lifted out of `spacr/pipeline_v2.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [FilenameMapper.discover](#filenamemapperdiscover) (1 entry)
- [FilenameMapper.save_csv](#filenamemappersave_csv) (1 entry)
- [Module level](#module-level) (1 entry)
- [_resolve_regex](#_resolve_regex) (3 entries)
- [StackFile](#stackfile) (3 entries)
- [stream_originals_to_stack](#stream_originals_to_stack) (6 entries)
- [_record_cellpose_hash](#_record_cellpose_hash) (4 entries)
- [_read_plane](#_read_plane) (2 entries)
- [_as_hwc](#_as_hwc) (1 entry)
- [stream_masks_from_stack](#stream_masks_from_stack) (9 entries)
- [FilenameMapper.discover, 2026-09-19](#filenamemapperdiscover-2026-09-19) (1 entry)

## FilenameMapper.discover

### lines 175-177

```python
keys = {}
```

Assign stable per-field ids so all channels of the same

(plate, well, field, time, z) fall into one stack file. Sort keys: plate → well → field → time → z; then enumerate.

## FilenameMapper.save_csv

### line 205  _(unsure)_

```python
(path.with_suffix(".json")).write_text(json.dumps({
```

Sidecar with the regex used, so `spacr repro` can replay

## Module level

### lines 251-254  _(unsure)_

```python
_CELLVOYAGER = (
```

Regex resolution — copy of spacr.utils._get_regex behaviour, kept local so v2 doesn't import the whole spacr.utils stack at module scope.

## _resolve_regex

### line 282, trailing  _(unsure)_

```python
else:
```

auto

### line 286  _(unsure)_

```python
for pattern, name in candidates:
```

Choose the first regex that matches EVERY file

### line 292  _(unsure)_

```python
best_pattern, best_name, best_hits = candidates[0][0], candidates[0][1], -1
```

Last-ditch: choose the one that matches the MOST files

## StackFile

### lines 304-306  _(unsure)_

```python
@dataclass
```

Pass 1 — stream originals into per-field npy stacks

### line 324, trailing  _(unsure)_

```python
shape:     Tuple[int, int, int]
```

(H, W, C) at write time

### line 325, trailing  _(unsure)_

```python
channels:  List[str]
```

human names, in the same order

## stream_originals_to_stack

### line 366  _(unsure)_

```python
by_ch = {r.channel: r for r in recs}
```

Group by channel number for this field

### lines 369-373

```python
ref_shape = None
```

Determine the field's true plane shape/dtype from ANY present channel first, so a zero plane synthesised for a missing channel matches — even when the missing channel is the FIRST requested one (otherwise np.stack raises "all input arrays must have the same shape"). Cache the read so present planes aren't read twice.

### lines 392-393

```python
LOG.warning("field %s missing channel %d — inserting zeros",
```

Missing channel — synthesise a zero plane matching the field's real shape so downstream tools don't crash.

### line 411  _(unsure)_

```python
(dst / "channel_order.json").write_text(json.dumps({
```

Global sidecar describing the C axis

### line 414, trailing  _(unsure)_

```python
"mask_channels":  [],
```

filled in by stream_masks_from_stack

### line 418  _(unsure)_

```python
mapper.save_csv(src / "filename_map.csv")
```

Save filename map at the plate root

## _record_cellpose_hash

### lines 429-430  _(unsure)_

```python
ckpt_paths = []
```

Cellpose's model object usually exposes `pretrained_model`

(list of paths) or `.cp.pretrained_model`.

### line 445  _(unsure)_

```python
ckpt_paths = [Path(p) for p in ckpt_paths
```

Filter to real existing files

### lines 450-452

```python
try:
```

Push to the OPEN run journal, if any. We do this via a thread-local convenience — see spacr.run_journal for the active-run registry.

### lines 461-464

```python
LOG.warning("model %r was not recorded in the run journal (%s); "
```

Provenance, not results — a journal that will not take the record must not stop the segmentation. But an unrecorded model is a run whose manifest cannot say which weights produced the masks, and that is exactly the question asked six months later.

## _read_plane

### line 484  _(unsure)_

```python
if arr.ndim == 3:
```

Reduce to 2-D (grayscale)

### lines 486-487  _(unsure)_

```python
arr = arr[..., 0]
```

H, W, C → take the first channel (spacr's convention for single-channel writes)

## _as_hwc

### lines 492-494  _(unsure)_

```python
def _as_hwc(arr: np.ndarray) -> np.ndarray:
```

Pass 2 — stream Cellpose masks back into the same stacks

## stream_masks_from_stack

### lines 590-592

```python
import torch
```

Resolve legacy names and fine-tuned checkpoint paths through the same adapter as V1. The old V2 branch silently loaded stock cpsam for every model_name, so a V1 run using a trained checkpoint could never match.

### lines 608-610

```python
_record_cellpose_hash(model, model_name)
```

Record the exact model checkpoint hash into the active run journal, if one is open. Downstream reviewers can then trace any mask back to the specific weights that produced it.

### lines 621-622  _(unsure)_

```python
npz_path = scratch / f"batch_{batch_start:04d}.npz"
```

Optionally persist the batch as NPZ for debugging. Deleted after run unless keep_npz=True.

### lines 629-631

```python
selected_images: List[np.ndarray] = []
```

Prepare the same list-of-images batch V1 hands to Cellpose. Besides being faster, keeping the call boundary identical matters for exact V1/V2 reproducibility on CPSAM.

### lines 648-653

```python
if postprocess_settings is not None:
```

V1 segments the percentile-normalised float batch under masks/*.npz, not the raw uint16 planes later retained in merged/.  V2 deliberately retains those raw planes, but must still present the same pixels to Cellpose or small synthetic fields produce materially different masks.  Reuse V1's normaliser with channel roles remapped onto this compact Cellpose input (object first, optional nucleus second).

### lines 739-740  _(unsure)_

```python
for sf, arr, mask in zip(batch, loaded, masks_per_field):
```

Append the mask channel to each stack file and update the StackFile bookkeeping.

### line 760  _(unsure)_

```python
try:
```

Best-effort scratch cleanup

### lines 766-767  _(unsure)_

```python
sidecar = stacks[0].path.parent / "channel_order.json"
```

The empty case returned before Cellpose was loaded, so stacks[0] is available here without a second, unreachable emptiness check.

### lines 774-777

```python
LOG.warning("channel_order.json at %s was not updated with "
```

The masks are written either way, so this does not fail the stage — but channel_order.json is what every later reader uses to know which plane is a mask, and a sidecar that silently did not get the entry makes the stack self-describing and wrong.

## FilenameMapper.discover, 2026-09-19

```python
and not p.name.startswith(".")
```

The v2 Mask pipeline lists the raw folder itself. On a macOS external volume (GitHub #121 and #117) every raw tiff has an AppleDouble sidecar, `._<name>.tif`. The CellVoyager pattern starts `(?P<plateID>.*)_`, so the sidecar matched as plate `._plate1`, got a field of its own, and `stream_originals_to_stack` stopped on it: `TiffFileError: not a TIFF file: header=b'\x00\x05\x16\x07'`. Measured on the code before this line. Dot-files are left out inline rather than through `spacr.io._listdir_visible` because discovery runs before anything here needs torch, and `spacr.io` imports it at module level. Reasons for the dot-file rule are in `docs/notes/spacr/io.md` under `_listdir_visible`.
