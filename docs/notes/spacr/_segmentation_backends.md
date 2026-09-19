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
