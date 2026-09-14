# Notes from `spacr/benchmark.py`

Prose lifted out of `spacr/benchmark.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_rss_bytes](#_rss_bytes) (2 entries)
- [recommend_workers](#recommend_workers) (1 entry)

## _rss_bytes

### line 107, trailing  _(unsure)_

```python
except ImportError:
```

Windows

### lines 110-112

```python
import sys
```

ru_maxrss is KILOBYTES on Linux and BYTES on macOS. Getting this backwards is a factor of 1024 in a memory budget, which would either recommend one worker on a large machine or forty on a small one.

## recommend_workers

### lines 237-239

```python
workers = max(1, min(cores, maximum))
```

No measurement, or the work was too small to register above the interpreter. Fall back to the core count rather than inventing a memory bound from a number that is not there.
