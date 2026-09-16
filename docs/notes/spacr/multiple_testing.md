# Notes from `spacr/multiple_testing.py`

Prose lifted out of `spacr/multiple_testing.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [canonical_method](#canonical_method) (1 entry)
- [estimate_pi0](#estimate_pi0) (2 entries)
- [storey_qvalue](#storey_qvalue) (1 entry)
- [adjust_p_values](#adjust_p_values) (1 entry)
- [_beta_uniform_fit](#_beta_uniform_fit) (1 entry)
- [local_fdr](#local_fdr) (1 entry)

## canonical_method

### line 205  _(unsure)_

```python
for spec in METHODS.values():
```

statsmodels spellings that are not already canonical keys.

## estimate_pi0

### line 248  _(unsure)_

```python
return 1.0
```

Too few tests to read a null plateau off the histogram.

### line 252

```python
tail = pi0_grid[grid >= float(np.median(grid))]
```

Take the minimum of the tail estimates: stable, and never above 1.

## storey_qvalue

### line 282  _(unsure)_

```python
q_sorted = np.minimum.accumulate(raw[::-1])[::-1]
```

Enforce monotonicity from the largest P value downwards.

## adjust_p_values

### lines 335-336  _(unsure)_

```python
adjusted[finite] = corrected
```

fdr_tsbh returns adjusted values already scaled by the estimated null count; statsmodels' own rejection call is authoritative for every method.

## _beta_uniform_fit

### lines 415-417

```python
values = np.clip(values, np.finfo(float).tiny, 1.0)
```

A P VALUE OF EXACTLY ZERO IS A REAL RESULT UNDERFLOWING, not a mistake, and log(0) would take the whole fit with it. Clamped to the smallest positive double, which is below any P value a screen can produce.

## local_fdr

### lines 476-481

```python
out[finite] = 1.0
```

TOO FEW TESTS TO READ A DENSITY OFF. Returning 1 everywhere says "nothing here is distinguishable from null", which is the conservative truth; returning a fitted curve would be showing the user a shape estimated from a dozen numbers as if it were the screen's. Callers gate the option on the same count -- see :data:`LOCAL_FDR_MIN_TESTS`.
