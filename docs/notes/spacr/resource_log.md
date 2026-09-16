# Notes from `spacr/resource_log.py`

Prose lifted out of `spacr/resource_log.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_memory](#_memory) (1 entry)
- [tree_sample](#tree_sample) (1 entry)
- [summarise](#summarise) (1 entry)
- [ResourceSampler.sample_once](#resourcesamplersample_once) (1 entry)
- [ResourceSampler._loop](#resourcesampler_loop) (1 entry)

## _memory

### line 272  _(unsure)_

```python
LOG.debug("no private memory figures for pid %s",
```

The private figures need permissions the resident one does not.

## tree_sample

### lines 443-444  _(unsure)_

```python
missed += 1
```

A child exiting between enumeration and reading is what a short-lived worker DOES. Skip it, keep the rest, and say so.

## summarise

### lines 504-506

```python
burned = []
```

Cumulative counters, so the largest sample is the run's total rather than a sum over samples, which would count the same seconds again once a second.

## ResourceSampler.sample_once

### lines 721-722  _(unsure)_

```python
self._ensure_log()
```

The header is written before the first reading is taken, so a file always names its measure before it carries a figure.

## ResourceSampler._loop

### lines 759-760

```python
LOG.debug("a reading failed; sampling continues",
```

A sampler that dies on a bad reading stops recording the run at exactly the point the run started going wrong.
