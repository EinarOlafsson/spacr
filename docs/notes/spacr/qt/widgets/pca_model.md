# Notes from `spacr/qt/widgets/pca_model.py`

Prose lifted out of `spacr/qt/widgets/pca_model.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_apply_nan_policy](#_apply_nan_policy) (3 entries)
- [pca](#pca) (4 entries)

## _apply_nan_policy

### line 811, trailing  _(unsure)_

```python
reason = None
```

per-feature, below

### line 813  _(unsure)_

```python
keep_feature = fraction < 1.0
```

A feature nobody has a value for has no mean to impute with.

### line 816, trailing  _(unsure)_

```python
else:
```

NAN_COMPLETE keeps every feature and pays for it in rows.

## pca

### line 992  _(unsure)_

```python
tolerance = largest * max(n, p) * float(np.finfo(float).eps)
```

NO `largest <= 0` OR `rank < 1` GUARD. Both were marked

### lines 994-1004

```python
tolerance = largest * max(n, p) * float(np.finfo(float).eps)
```

`_drop_constant` above has already refused a matrix with no variance in it, with a message that names the offending features "every selected feature is constant over the analysed objects", or "only 1 feature varies... PCA needs two". A matrix that reaches here therefore has at least two varying features, so its largest singular value is positive and its rank is at least one.

Checked rather than assumed: identical columns, one constant column, and denormal values all raise from `_drop_constant`; perfectly collinear columns get through and decompose, which is correct -- collinearity reduces the rank to 1, not to 0.

### lines 1012-1015

```python
for i in range(k):
```

SVD signs are arbitrary; pin them so the same data draws the same picture every time and a figure in a report matches the screen it came from. Convention: the largest-magnitude loading of each component is positive.

### lines 1026-1027

```python
column_sd = standard.std(axis=0, ddof=1)
```

Feature-component correlations, computed rather than derived, so the arrows in the biplot cannot drift out of step with the scores.
