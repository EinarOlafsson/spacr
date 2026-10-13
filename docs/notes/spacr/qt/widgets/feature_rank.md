# Notes from `spacr/qt/widgets/feature_rank.py`

Prose lifted out of `spacr/qt/widgets/feature_rank.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_ranks](#_ranks) (2 entries)
- [mutual_info_of](#mutual_info_of) (1 entry)
- [Module level](#module-level) (1 entry)
- [_class_levels](#_class_levels) (1 entry)
- [_score_one](#_score_one) (1 entry)
- [_null_threshold](#_null_threshold) (3 entries)

## _ranks

### lines 117-119  _(unsure)_

```python
def _ranks(values: np.ndarray) -> np.ndarray:
```

The statistics, each over two 1-D arrays of finite values

### lines 131-133

```python
start = 0
```

Average within each run of equal values. A tied pair contributes exactly half to the U statistic, which is what makes AUC 0.5 for two identical distributions rather than something that depends on the sort order.

## mutual_info_of

### lines 211-212  _(unsure)_

```python
return 0.0
```

Fewer than two distinct bins: the feature is (almost) constant, and a constant explains nothing.

## Module level

### lines 648-649

```python
_UNLABELLED = object()
```

Object identity, rather than a user-representable string, distinguishes a missing class label from a real class whose name happens to be empty.

## _class_levels

### lines 679-681

```python
text = np.asarray([
```

Convert rows and level names through exactly the same path. Pandas' vectorised datetime astype omits midnight while str(Timestamp) includes it, causing a date column not to match its own advertised levels.

## _score_one

### lines 736-737  _(unsure)_

```python
for level in (levels if len(levels) > 2 else levels[1:]):
```

Two classes need one comparison, not two: "a against b" and "b against a" are the same separation with the direction flipped.

## _null_threshold

### lines 773-774

```python
labelled = _labelled(keys)
```

Rows without a class enter no real score, so they cannot enter the null experiment that calibrates that score either.

### lines 787-795

```python
unmeasured = 0
```

A SHUFFLE THAT MEASURED NOTHING IS NOT A SHUFFLE THAT MEASURED ZERO. `top` used to start at 0.0 and only ever be raised by a finite score, so a permutation in which every _separation came back NaN -- each candidate class empty once the finite mask was applied -- contributed a literal "chance reached zero" to the null distribution. On a sparsely measured table a sizeable fraction of the null could be those spurious zeros, which deflates the 95th percentile and makes above_null() list features that never beat chance. Starting at None and appending only a real measurement drops such a shuffle instead of scoring it.

### line 814

```python
notices.append(
```

Say what the number cannot say: the null is thinner than asked for.
