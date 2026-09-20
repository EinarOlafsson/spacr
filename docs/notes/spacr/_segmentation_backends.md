# Notes for `spacr/_segmentation_backends.py`

The module carries no comments; the reasons behind it live here and in its
docstrings. Item 423 (features/future/423_third_party_segmenters_in_the_model_zoo.txt)
is where the history is.

## Why every optional backend has an environment of its own

Maintainer's decision, 2026-09-19: "Isolated env per backend! But with the
addition of adding cellpose 3 and its cyto, nucleus, and cyto2 and cyto3
models. The cellpose 3 backend will allow more models from the biomass.io to
be added to the model zoo." ("biomass.io" is bioimage.io.)

Two facts made the old install -- `pip install "spacr[<extra>]"` into the
environment spaCR runs in -- unsafe, both read on 2026-09-19:

* DINOCell 0.74 pins exact versions of about ninety packages on PyPI,
  `torch==2.10.0`, `numpy==2.4.3` and `cellpose==4.0.9` among them. spaCR's
  own environment had torch 2.13.0, numpy 2.5.2 and cellpose 4.2.1.1, so pip
  would have replaced all three underneath a running spaCR.
* Cellpose 3 and Cellpose 4 are both the import name `cellpose`. One process
  cannot hold both, so Cellpose 3 can only ever run out of process.

## Why venv and not conda

`venv` is in the standard library on Linux, macOS and Windows, needs no conda
on the machine, and built an environment in 1.4 s here. pip then fills it
from PyPI, where all three projects publish. Its one limit: the environment's
Python is a Python already on the computer -- spaCR's own when it is inside
the range the backend's pins install on, else `python3.X` on PATH, else the
`py -3.X` launcher on Windows. When there is none, the row says "not
installable here" and names the range; Debian and Ubuntu need the
`python3.X-venv` package for `ensurepip`, and the preflight says so.

The ranges: Cellpose 3.1.1.3 pins `numpy<2.1`, which has no wheels past
Python 3.12, so 3.9 to 3.12; DINOCell's `numpy==2.4.3` needs 3.11 or later;
SAMCell asks for 3.8 or later and is held to spaCR's own 3.9 floor.

## PyTorch

The backend gets the same kind of torch spaCR has, read from the local
version tag of spaCR's own torch: `+cu124` means the `cu124` wheel index,
`+cpu` the `cpu` one, and a plain version (PyPI's own build, CUDA on Linux)
means PyPI. `SPACR_BACKEND_TORCH_INDEX` overrides it, and `pypi` there means
PyPI. The real install below used the CPU index to keep the download small.

## The protocol

One JSON object per line each way over the worker's stdin and stdout,
version 1. The worker duplicates its stdout for replies and points file
descriptor 1 at stderr before importing anything, so a library that prints
cannot corrupt a reply. Images and masks travel as `.npy` files in a
temporary folder. A failure is a reply carrying the exception's type,
message and traceback; spaCR raises `_BackendError` with the message
verbatim and keeps the rest on the exception.

`-I` matters: the worker is this file, run from spaCR's package folder, whose
`io.py` and other modules would shadow the standard library if the script's
folder were first on `sys.path`. Isolated mode leaves it off, and ignores
`PYTHONPATH` and the user's site-packages too.

## Where a backend's weights go, and why it is not the person's cache

Every backend here downloads weights the first time it segments, and each
package has its own idea of where. Left alone, Cellpose 3 writes to
`~/.cellpose`, DINOCell's `hf_hub_download` writes to `~/.cache/huggingface`
and SAMCell's `torch.hub` writes to `~/.cache/torch`. All three are OUTSIDE
the folder Uninstall removes, and outside the folder the preflight measures
free space in.

So `_worker_env` points each one inside the environment:
`CELLPOSE_LOCAL_MODELS_PATH` for Cellpose 3 and `HF_HOME` for DINOCell. Two
things follow, and both are the reason:

* Uninstall gives the disk back. 423's promise is "Uninstall deletes the
  environment and everything downloaded into it", and DINOCell's checkpoint
  is 383 MB of that.
* `_preflight` refuses an install when `shutil.disk_usage(root).free` is
  below `spec.size_gb`. If the weights land on another filesystem, that check
  is about the wrong disk.

Measured on 2026-09-19, before and after, in a sandboxed HOME with the user
cache emptied first: a DINOCell run put 402,337,098 bytes under
`$XDG_CACHE_HOME/huggingface` and none inside the environment; with `HF_HOME`
set it put the same 402,337,098 bytes inside the environment and left the
user cache at 0 bytes, and Uninstall then took both away.

### `HF_HOME` is necessary and not sufficient

Setting `HF_HOME` was the first fix and it was half of one. `_clean_env`
forwards the whole inherited environment minus `_STRIPPED_VARIABLES`, and
`huggingface_hub` reads `HF_HUB_CACHE` from the environment BEFORE it derives
anything from `HF_HOME`: `HF_HUB_CACHE` falls back to the legacy
`HUGGINGFACE_HUB_CACHE`, which falls back to `$HF_HOME/hub`. Measured against
huggingface_hub 0.36.2 on 2026-09-19, with `HF_HOME` inside the environment
and `HF_HUB_CACHE=/mnt/elsewhere/hf`, `constants.HF_HUB_CACHE` is
`/mnt/elsewhere/hf`; the legacy name alone does the same.

Whoever exports those variables is exactly the person this redirect is for --
someone whose Hugging Face cache outgrew their home disk and who moved it --
so the fix would have missed its own case. `_worker_env` now drops
`_HF_CACHE_VARIABLES` (both hub names, both assets names, and the xet cache)
alongside setting `HF_HOME`, and
`test_a_relocated_hugging_face_cache_does_not_win_over_the_environment`
exports all five and fails if any survives.

### SAMCell is not done this way yet, and it has TWO caches, not one

It belongs to item 405, deliberately left there rather than changed untested
from 404's lane. 405 should know that `TORCH_HOME` alone will not finish it:

* `_samcell_weights_path` fetches `samcell-generalist.pt` through
  `torch.hub.get_dir()`, which `TORCH_HOME` moves. 375,043,010 bytes on this
  machine, in `~/.cache/torch/hub/checkpoints`.
* `_SamCellBackend` then builds `FinetunedSAM(_SAMCELL_BASE_MODEL)`, and
  samcell's `FinetunedSAM.__init__` is `SamModel.from_pretrained(sam_model)`
  with `_SAMCELL_BASE_MODEL = "facebook/sam-vit-base"`. That is transformers
  pulling a second model into the SAME Hugging Face cache DINOCell's
  checkpoint was just moved out of. 374,986,732 bytes on this machine, in
  `~/.cache/huggingface/hub/models--facebook--sam-vit-base`.

So SAMCell leaves about 750 MB in the person's home across two caches, and
needs `TORCH_HOME`, `HF_HOME` and the `_HF_CACHE_VARIABLES` drop together.
Setting only `TORCH_HOME` would move half of it and read as finished. Both
numbers above are `du -sb` on this machine's existing caches, not estimates,
but neither has been measured through an isolated environment -- that is 405's
to do, and its `size_gb` should be checked against the pair.

## Measured on 2026-09-19

In a sandboxed HOME, on the CPU, with the real packages from PyPI:

| what | result |
|---|---|
| cold install of Cellpose 3 | 258.7 s, 1.6 GB: cellpose 3.1.1.3, torch 2.14.0+cpu, numpy 2.0.2 |
| uninstall | 0.7 s |
| reinstall from pip's cache | 29.2 s |
| reinstall pressed in the dialog | 55.4 s; the GUI timer fired 55 times while pip ran |
| cyto3 on a 128 x 128 field of five blobs | 5 objects, 15.4 s with the weight download and worker start |
| nuclei on the same field | 5 objects, 21.2 s, likewise |
| `generate_cellpose_masks_sam`, cellpose3, 256 x 256, nine discs | 9 objects, 7.3 s |
| bioimage.io "CellPose(cyto3)" through the zoo | 26,566,255 bytes, checksum-checked, 7 of the 9 discs |

Two things the real run found that no stand-in would have:

* **`packaging` is installed with Cellpose 3.** fastremap 1.20, which
  Cellpose 3 needs, imports `packaging` without declaring it. The first real
  install passed its self-test (`import cellpose`) and the first segment then
  failed with `ModuleNotFoundError: No module named 'packaging'`. The
  self-test now imports what the adapter imports (`cellpose.models`).
* **A missing Cellpose 3 model is refused.** Handed a path that does not
  exist, `CellposeModel` logs "pretrained_model path does not exist, using
  default model" and loads cyto3. A typo in a model path would have
  segmented with cyto3 without a word, so both the client and the worker
  refuse it.

### DINOCell, the same day, the same way

Item 404's half of the seam, run for real for the first time. Same sandboxed
HOME, same CPU wheel index:

| what | result |
|---|---|
| cold install of DINOCell | 541.2 s, 1,797,510,073 bytes: dinocell 0.74, torch 2.10.0+cpu, torchvision 0.25.0, numpy 2.4.3, cellpose 4.0.9, on spaCR's own Python 3.12.13 |
| its checkpoint | `KadenStillwagon/DINOCell`, `DINOCell_demo_model.pt`, 401,672,491 bytes, MIT on the model card |
| `generate_cellpose_masks_sam`, dinocell, 256 x 256, nine discs | 9 of 9 objects, 180.7 s with the checkpoint download, the worker start and the model load |
| the same field again, same worker | 9 of 9 objects, 15.7 s |
| the environment afterwards | 2,199,847,171 bytes, the extra 402,337,098 being the checkpoint inside it |
| uninstall | environment gone, checkpoint gone, row back to installable, user cache untouched |

The mask came back `uint16` at the field's own 256 x 256 — DINOCell's flows
are predicted at its 512 crop, so a field narrower than one tile is upscaled
(aspect ratio kept, where DINOCell's own `_resize` would square it) and the
labels are resampled back. `generate_cellpose_masks_sam` wrote
`cell_mask_stack/plate1_A01_1.npy` and the `object_counts` row
`("plate1_A01_1.npy", "cell_before_filtration", 9)`, and spaCR's Cellpose 4
was never constructed: the run held a `CellposeModel` that raises if anything
builds it.

**15.7 s of that 180.7 s is the segmentation.** The rest is the 383 MB
download, the interpreter start and the ViT load, all of which happen once per
worker. That is the number to carry into any estimate, and it is why the
worker is kept alive between fields.

### The self-test now imports what the adapter imports, for every backend

Cellpose 3's `packaging` (below) was the first instance. DINOCell had a second
one waiting: its adapter calls `cellpose.plot.dx_to_circ` to build the flow
image `parse_cellpose4_output` hands on, and `cellpose.plot` was not in its
probe. `tests/test_backends_live_in_their_own_environment.py` now parses each
adapter's own source and fails when a module it imports is missing from its
spec's `probe` — `torch` and `numpy` excepted, since every backend environment
installs both and `_worker_device` imports torch before any probe runs. Run
against the probe as it was, that test names `cellpose.plot` exactly.

bioimage.io's "CellPose(cyto3)" upload (famous-fish) is byte-identical to
Cellpose's own cyto3 (SHA-256 `2dc3087a...`). It found 7 of the nine discs
where the named model found 9 because a named model estimates the diameter
with Cellpose's size model, and a checkpoint runs at the diameter it was
trained at unless the object's diameter is set.

## Worker lifetime

One worker per backend, shared by everything in the process that segments
with it; the magnifier asks on every mouse move, so the model loads once.
A worker idle for ten minutes is shut down, giving back the memory the model
holds, and the next request starts it again; so does a worker that died.

HANDING A WORKER OUT COUNTS AS USING IT. `last_used` used to move only when
a reply arrived, and `busy` is only true while a request is in flight, so
between `_worker_for` releasing its lock and the caller's first write there
was an instant in which a worker idle past the threshold was neither. The
reaper runs on its own thread and could close it exactly there; the write
then raised and the user was told "the backend stopped (exit code 0)". A
mask run with more than ten minutes between batches is the run that reaches
it. `_worker_for` stamps `last_used` under the lock instead.

A caller that CACHES the model, as Make Masks' Mode box does, has the other
half of this problem: the environment can be uninstalled from the Model Zoo
while the model object is still held. `make_masks._backend_model` remembers
the folder each model was built from and drops the model when that folder
is gone, so the next request says the backend is missing rather than that a
file is.

## On a CUDA PyTorch

Measured 2026-09-19 in a sandboxed HOME on this machine's GPU
(gpu_queue/423-isolated-backends-1.md). The environment built with
`cellpose==3.1.1.3`, `torch` 2.14.0 and numpy 2.0.2 and reported device
`cuda`; the install took 2079 s, which is the CUDA wheel download and not
the GPU. `generate_cellpose_masks_sam` with
`segmentation_backend='cellpose3'` and `cell_model_name='cyto3'` found 64
of 64 discs in a 1024 x 1024 field in 21.4 s, the worker reporting device
`cuda`. Uninstalling afterwards removed the folder. Nothing was trained.

## SAMCell's checkpoints

`generalist` was trained on LIVECell and the Cellpose cytoplasm set; `cyto`
on the Cellpose cytoplasm set only (the SAMCell 1.2.0 README).
