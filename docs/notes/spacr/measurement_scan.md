# Notes from `spacr/measurement_scan.py`

Prose lifted out of `spacr/measurement_scan.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [simes_p_value](#simes_p_value) (2 entries)
- [effective_number_of_tests](#effective_number_of_tests) (2 entries)
- [_build_design](#_build_design) (1 entry)
- [scan_measurements](#scan_measurements) (5 entries)

## simes_p_value

### lines 238-240  _(unsure)_

```python
def simes_p_value(p_values) -> float:
```

The two statistics that are not in multiple_testing

### lines 263-264

```python
return float(np.min(ordered * ordered.size / ranks))
```

No cap is needed: the last term is m * p_(m) / m == p_(m) <= 1, so the minimum is never above 1 and a clip here would be code no input reaches.

## effective_number_of_tests

### lines 293-296

```python
keep = np.isfinite(np.diag(correlation))
```

Drop a column on its OWN diagonal, not on whether its row is clean. A column with no variance has an undefined correlation with everything, and testing the row would throw away every column it touches -- which is every column -- for one constant neighbour.

### lines 303-306

```python
correlation = np.nan_to_num(correlation, nan=0.0)
```

Two columns can still have no wells in common (pandas correlates pairwise-complete), leaving a NaN between two perfectly good columns. Read that as uncorrelated: it counts them as separate tests, which is the conservative direction for a correction.

## _build_design

### lines 399-401

```python
terms = levels[1:]
```

No controls named, or none of the named ones are in this frame. The first level becomes the baseline -- an arbitrary choice, which is exactly why naming the controls matters.

## scan_measurements

### lines 499-503

```python
genes_dropped: Dict[str, int] = {}
```

DROP THE GENES NOTHING CORROBORATES, AND SAY WHICH.

Named rather than quietly filtered: a gene missing from the result with no explanation reads as a gene with no effect, which is the opposite of what happened to it.

### lines 509-512

```python
thin = thin[[gene for gene in thin.index if gene not in controls]]
```

A control is the BASELINE, not a candidate. Dropping it for being thin would silently move the baseline to whichever gene sorted first, and every effect in the table would then be measured from somewhere the caller did not choose.

### lines 549-552

```python
if present.sum() < 3:
```

Only the "there is nothing here at all" case is decided here. Whether the wells that ARE present can carry the design is _fit_columns' question, because it is the one that knows the rank of the sub-design after the empty gene levels have dropped out.

### lines 567-568

```python
groups: Dict[bytes, list] = {}
```

Group by which wells are present, so the common case -- every measurement complete -- costs one matrix factorisation, not hundreds.

### lines 589-595

```python
scale = float(np.nanstd(column[present]))
```

A residual of essentially nothing is not a huge effect, it is a column the design already IS -- a per-well aggregate that turns out to be the guide assignment, say. Dividing by it gives an effect size of 1e8 that sits at the top of the table for ever and is not a measurement of anything. Judged RELATIVE to the response's own spread, because "small" has no absolute meaning for a measurement whose units the scan does not know.
