# Notes from `spacr/read_background.py`

Prose lifted out of `spacr/read_background.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [resolve_exclusions](#resolve_exclusions) (1 entry)
- [suggest_threshold](#suggest_threshold) (3 entries)

## resolve_exclusions

### lines 99-101

```python
return {str(e) for e in wanted}
```

A resolver that cannot run must not silently exclude NOTHING that would leave a known contaminant in the denominator. Fall back to the exact names, which is the subset everybody agrees on.

## suggest_threshold

### lines 209-218

```python
keep = values[values < max(middle, 1e-12) * float(outlier_factor)]
```

THE OUTLIERS ARE REMOVED BEFORE THE QUANTILE IS TAKEN, and leaving them in was a real fault rather than a rounding one: the quantile is contaminated by exactly the guides it is supposed to exclude. Caught on a 42-guide fixture where one outlier at 9% dragged the 99th percentile to 6.6% -- a threshold that would delete most of a real library. On a 1,325-guide screen the same outlier hid inside the quantile instead, which is worse, because nothing looked wrong.

`factor` matches `suspicious`, so the guides excluded here are the guides that function reports. One rule, applied twice.

### line 220, trailing  _(unsure)_

```python
if keep.size == 0:
```

every guide is an outlier

### lines 231-232

```python
"guides_needing_their_own": float(values.size - keep.size),
```

The ones a single threshold cannot serve, whatever it is set to: they were left out of the estimate and clear it anyway.
