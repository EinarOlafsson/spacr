# Notes from `spacr/openmp_guard.py`

Prose lifted out of `spacr/openmp_guard.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [single_threaded_openmp.__enter__](#single_threaded_openmp__enter__) (1 entry)

## Module level

### lines 35-40

```python
_OPENMP_MARKERS = (
```

Substrings that identify an OpenMP runtime image. `libgomp` is GCC's and `libiomp5` is Intel's; mixing any two of the three is the same hazard. Wheels may content-hash a bundled runtime before its extension (PyTorch 2.1 ships ``libgomp-a34b3233.so.1``). Match that spelling as well as the ordinary ``libgomp.so.1`` form, or the supported dependency floor looks as though no runtime is resident even while OpenMP is active.

## single_threaded_openmp.__enter__

### lines 319-320  _(unsure)_

```python
continue
```

A runtime without the symbols, or one that will not dlopen. Leave it alone; the others still help.
