# Notes from `spacr/doctor.py`

Prose lifted out of `spacr/doctor.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_declared_requirement](#_declared_requirement) (1 entry)
- [Module level](#module-level) (2 entries)
- [_import_spacr](#_import_spacr) (1 entry)
- [check_gpu](#check_gpu) (4 entries)

## _declared_requirement

### lines 198-199  _(unsure)_

```python
if "extra" in marker:
```

Requirements guarded by `extra == "..."` belong to an extra, not to the core dependency set the check is asking about.

## Module level

### lines 209-221

```python
_OPERATORS = ("===", "==", "!=", "<=", ">=", "~=", "<", ">")
```

Version comparison, PEP 440 subset, stdlib only.

``packaging`` would be one import away and is installed in practically every environment — which is exactly the reasoning that put five undeclared module-scope imports into this project (requests via huggingface-hub, joblib via scikit-learn, ...), each one upstream decision away from an ImportError. A tool whose job is to explain broken environments cannot itself depend on an environment being unbroken, so the comparison is implemented here.

The subset covers what spaCR's own metadata uses: `>= <= == != < > ~= ===`, comma-joined clauses, `.*` prefix matching, local segments (`2.9.1+cu128`) and pre/post/dev suffixes. Anything outside it returns ``None`` — "cannot tell" — rather than a guess.

### lines 1071-1074

```python
("vispy", "vispy"),
```

VisPy backs the installed application's default Mandelbrot renderer. It remains available through the historical ``fractal`` extra spelling, but it is a core dependency now; a missing install is therefore a broken environment, not an optional feature the doctor may report as absent.

## _import_spacr

### lines 593-595  _(unsure)_

```python
def _import_spacr() -> Any:
```

2-5. which spacr am I actually running, and where did it come from

## check_gpu

### lines 1273-1285

```python
try:
```

ANOTHER VENDOR'S ACCELERATOR IS NOT A CUDA FAILURE, and this is the one place where confusing the two is user-facing. Everything below diagnoses CUDA and ends in `nvidia-smi`, and the very first branch "a CPU-only torch: segmentation and training will be very slow" -- is the exact verdict a Mac gets. On the machine this was written on that sentence was wrong by two orders of magnitude: the AMD card it does not mention segments 139x faster than the CPU it is warning about.

ASKED BEFORE `built`, because a stock macOS torch has no CUDA version at all and would never reach a check placed lower. NOT taken when the accelerator IS CUDA -- that path must keep every diagnostic below, including the allocation probe that catches a driver mismatch `torch.cuda.is_available()` reports as fine. Instruction 319.

### lines 1289-1292

```python
found = inspect_torch(torch)
```

ASKED ABOUT THIS torch, not the cached answer for this machine: the torch above comes through `_import_torch` precisely so the diagnosis can be exercised against a stand-in, and a cached global would report the developer's own card instead.

### lines 1307-1308

```python
return Result("gpu", WARN,
```

FOUND AND UNUSABLE is its own verdict rather than "no GPU":

the fix differs and the reader can act on it.

### lines 1314-1316

```python
pass
```

Silent on purpose: this module has no logger, and a doctor has to keep reporting on a machine where something is broken. Falling through to the CUDA diagnosis is the right behaviour anyway.
