# Notes from `spacr/spacr_cellpose.py`

Prose lifted out of `spacr/spacr_cellpose.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [display](#display) (1 entry)
- [parse_cellpose4_output](#parse_cellpose4_output) (5 entries)
- [identify_masks_finetune](#identify_masks_finetune) (5 entries)
- [generate_masks_from_imgs](#generate_masks_from_imgs) (2 entries)
- [check_cellpose_models](#check_cellpose_models) (1 entry)

## display

### lines 10-13

```python
def display(*args, **kwargs):
```

IPython may be mid-init (partially imported by another thread) — use a no-op fallback so importing this module never blocks. spaCR only calls display() from notebook contexts anyway; the Qt GUI ignores it.

## parse_cellpose4_output

### lines 89-101

```python
if isinstance(masks, np.ndarray) and masks.ndim == 2:
```

A BARE 2-D EVAL RETURNS ONE IMAGE'S FLOWS, FLAT.

Handed a single (H, W) array, ``CellposeModel.eval`` returns a 2-D ``masks`` and a ``flows`` list holding the three arrays for that ONE image -- an RGB rendering, the (2, H, W) vectors and the cell-probability map -- rather than a list with one entry per image. ``len(masks)`` is then the image HEIGHT, so both branches below go looking for H entries in a list of three and a field that segmented perfectly raises.

The check goes FIRST because a 2-D mask is one image whatever the flows look like. Nothing else reaches it: a list of 2-D arrays fails the isinstance, and a batched (N, H, W) stack has ndim 3.

### line 108  _(unsure)_

```python
try:
```

Determine number of images

### line 114  _(unsure)_

```python
if len(flows) == 4 and all(isinstance(f, np.ndarray) for f in flows):
```

Case A: batched format (4 arrays stacked over batch)

### line 125  _(unsure)_

```python
elif len(flows) == num_images:
```

Case B: per-image format

### line 148  _(unsure)_

```python
raise ValueError(f"Unrecognized Cellpose flows format: type={type(flows)}, len={len(flows) if has...
```

Unrecognized structure

## identify_masks_finetune

### lines 181-183

```python
if not cellpose_gpu():
```

ONE RESOLVER, NOT A CUDA TEST. `torch.cuda.is_available()` answers "is there CUDA", and this line meant "is there a GPU" -- which on a Mac or a ROCm box is a different answer. See instruction 319.

### lines 189-191

```python
if settings['custom_model'] is None:
```

'cpsam' unless the user pointed at a checkpoint. custom_model wins when set (its existence was checked above); otherwise model_name is resolved, which maps a pre-SAM name forward and reports it once.

### lines 197-202

```python
model = cp_models.CellposeModel(pretrained_model=pretrained,
```

No model_type= / diam_mean= : Cellpose 4 logs "not used in v4.0.1+" and drops both. diameter is NOT dropped — it is passed to eval() below, where the image is rescaled by 30/diameter, and that still works. gpu= AND device= TOGETHER. Cellpose branches on `gpu` before it looks at `device`, so passing a device without the flag still takes the CPU path -- which is exactly what pinned every Mac to the CPU.

### lines 208-211

```python
print("grayscale=True has no effect under Cellpose 4: the channel "
```

The [cytoplasm, nucleus] channel pair went away with Cellpose 4: eval(channels=...) logs "channels deprecated in v4.0.1+" and uses the first three channels regardless. Say so rather than printing a channel pair the network never sees.

### lines 226-227  _(unsure)_

```python
print(f"Either no images were found in {settings['src']} or all images have masks in {dst}")
```

NB: use the local ``dst`` — there is no settings['dst'] key, so the old settings['dst'] raised KeyError on this no-images path.

## generate_masks_from_imgs

### lines 342-343  _(unsure)_

```python
print("grayscale=True has no effect under Cellpose 4: the channel "
```

See identify_masks_finetune: eval(channels=) is deprecated and ignored by Cellpose 4, so there is no channel pair left to force.

### lines 360-366

```python
else:
```

orig_dims is deliberately NOT recomputed from `images` here. The loader was handed target_height/target_width, so it has already resized them; measuring them now records the TARGET size as the original, which makes the `resize back to orig_dims` below a no-op and writes every mask at target resolution instead of the source's. identify_masks_finetune keeps the loader's dims for exactly this reason.

## check_cellpose_models

### lines 429-431

```python
cellpose_models = ['cpsam']
```

Cellpose 4 ships one stock model, so "check the models" is a list of one. It is left as a list rather than collapsed so a future release that ships more than one needs no other change here.
