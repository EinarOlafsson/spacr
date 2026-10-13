# Notes from `spacr/accelerator.py`

Prose lifted out of `spacr/accelerator.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_cuda_or_rocm](#_cuda_or_rocm) (2 entries)
- [_mps](#_mps) (2 entries)
- [_why_metal_is_unavailable](#_why_metal_is_unavailable) (1 entry)
- [_xpu](#_xpu) (1 entry)
- [resolve](#resolve) (2 entries)
- [torch_device](#torch_device) (1 entry)
- [empty_cache](#empty_cache) (1 entry)
- [capabilities](#capabilities) (1 entry)

## _cuda_or_rocm

### lines 172-176

```python
version_module = getattr(torch, "version", None)
```

ROCm builds set torch.version.hip and leave torch.version.cuda None. `torch.version` is fetched defensively rather than dotted into: a partially-built torch, and any stand-in that implements only the `cuda` namespace, has no `version` at all -- and an AttributeError here would demote a working CUDA card to the CPU.

### lines 187-189

```python
float64=True, autocast=True)
```

ROCm is a full CUDA-shaped backend: double precision and autocast both work, which is why it needs no capability carve-outs the way Metal does.

## _mps

### lines 203-209

```python
if backend.is_built():
```

Built but unavailable is a real and confusing state, and it has SEVERAL causes that a user can act on differently. This used to answer all of them with "this system does not offer a Metal device", which on an Intel Mac with Intel graphics is simply false -- the machine has a Metal device, drives its display with it, and torch still cannot use it. Reported by the maintainer on a 2020 Intel Mac.

### line 222

```python
float64=False, autocast=False, fallback=True)
```

MEASURED, not assumed -- see this module's docstring.

## _why_metal_is_unavailable

### lines 268-269

```python
if release:
```

ASKED FIRST, because on macOS 12.2 an Apple Silicon Mac reaches here too and "upgrade macOS" is the true answer for it as well.

## _xpu

### lines 342-344

```python
return Accelerator(kind="xpu", device="xpu", label=f"{name} (Intel XPU)",
```

Intel's XPU backend has no float64 on most consumer parts and its autocast support depends on the torch build, so both are claimed conservatively: a wrong "yes" here is a crash in a training run.

## resolve

### lines 432-433  _(unsure)_

```python
LOG.debug("accelerator probe failed", exc_info=True)
```

A backend that throws while being ASKED whether it exists is exactly the half-installed case this must survive.

### lines 446-451

```python
found = replace(found, fallback=bool(
```

THE FLAG ITSELF IS SET IN `spacr/__init__.py`, NOT HERE. torch reads it when the MPS backend registers, which is at `import torch` -- long before this resolver runs. Setting it now would look like it worked and change nothing; measured. What is recorded here is only whether the fallback is in force, so a caller can report it.

## torch_device

### lines 515-517  _(unsure)_

```python
def torch_device():
```

The shorthands call sites actually want

## empty_cache

### lines 711-713

```python
return made
```

THE CALL IS RETURNED, NOT A GENERIC PHRASE. Preferences shows this verbatim, and "torch.cuda.empty_cache()" is the line a user can look up; "device cache released" is a euphemism they cannot check.

## capabilities

### lines 744-747

```python
("UMAP / t-SNE / clustering", found.is_cuda,
```

cuML IS NOT PORTABLE, and saying so here is the point. RAPIDS ships for CUDA only -- there is no AMD, Intel or macOS build so this row is red on every machine in this list except NVIDIA, and a user on Metal should not wait for it to get faster.
