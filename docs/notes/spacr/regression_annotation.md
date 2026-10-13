# Notes from `spacr/regression_annotation.py`

Prose lifted out of `spacr/regression_annotation.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [missing_requirement](#missing_requirement) (1 entry)
- [AnnotationRequest](#annotationrequest) (1 entry)
- [Module level](#module-level) (1 entry)
- [wells_selected](#wells_selected) (1 entry)
- [prepare](#prepare) (3 entries)
- [xgboost_available](#xgboost_available) (1 entry)
- [_score_holdout](#_score_holdout) (1 entry)
- [_run_pu_learning](#_run_pu_learning) (1 entry)
- [_run_neighbour_propagation](#_run_neighbour_propagation) (3 entries)

## missing_requirement

### lines 361-364

```python
annotated = usable_annotations(frame, label_column)
```

THE REFERENCE LABEL, WHICH EVERY STRATEGY NEEDS. It is what the hold-out is scored against, so a strategy that fits nothing still cannot run without one -- `prepare` refuses the same table for the same reason.

## AnnotationRequest

### lines 409-411  _(unsure)_

```python
@dataclass
```

What a run is asked for

## Module level

### lines 726-728  _(unsure)_

```python
GROUP_SEPARATOR = "\x1f"
```

Wells, labels, and the hold-out no strategy may choose from

## wells_selected

### lines 774-775

```python
as_text = values.astype(str)
```

ONE ANSWER PER DISTINCT GROUP, not per cell: a screen has half a million objects and a few hundred wells.

## prepare

### lines 989-1004

```python
labelled = np.flatnonzero(np.asarray(known, dtype=bool))
```

THE SPLIT RUNS OVER THE LABELLED ROWS ONLY. Stratifying over rows that carry no annotation would balance the hold-out on a label nobody wrote, and the wells it drew would be drawn for the wrong reason. NO `labelled.size < 4` GUARD. Both routes into `known` already guarantee at least four:

the SCORE route refuses earlier, at `pool.size < 4` -- "Only N scored cell(s) are in the chosen wells" -- and `known` is drawn from that pool; the ANNOTATION route only supplies labels when `known.sum() >= 4`, and otherwise falls back to the score route above.

So this raise could not fire. Argued, then searched: 45 combinations of well count, cell count and n_positive all hit an earlier guard, none reached this one.

### lines 1006-1008

```python
try:
```

THE SPLITTER REFUSES A DESIGN IT CANNOT MAKE HONEST, and its refusal is the message a user needs; it is re-raised in this module's own type so a caller has one exception to catch.

### lines 1020-1022

```python
outside = ~np.isin(groups.astype(str), list(holdout_groups))
```

WHOLE WELLS, not whole labels. A strategy that could pick an unannotated cell out of a hold-out well would be choosing inside the group its own score is measured on.

## xgboost_available

### lines 1059-1061  _(unsure)_

```python
def xgboost_available() -> bool:
```

The model, and what a fit is allowed to claim

## _score_holdout

### lines 1272-1281

```python
auc = None
```

NOT ONLY AN EXCEPTION. A hold-out that came out all-positive or all-negative -- routine on a small screen -- made scikit-learn raise ValueError up to 1.6 and makes it return NaN with an UndefinedMetricWarning from 1.7. Both mean "undefined", and only the first was being turned into None.

NaN here is worse than it looks. `lift_over_chance` falls back to balanced accuracy on None and cannot on NaN, so it returns NaN and that lift is the number the named-method leak check compares. `summary` prints "ROC AUC nan" where it means "n/a".

## _run_pu_learning

### lines 1921-1924

```python
try:
```

THE LABELLING RATE IS ESTIMATED WHERE THE MODEL DID NOT FIT. Estimating it on the rows the model was fitted on returns the model's own confidence rather than the rate, and the rescaling it produces is therefore always about 1.

## _run_neighbour_propagation

### lines 2189-2190

```python
pool = np.union1d(pool, np.intersect1d(np.asarray(train, dtype=int),
```

THE SEEDS STAY IN THE POOL whatever the sample took: a seed that fell out of it could label nothing, which would look like a tight cut.

### lines 2206-2207

```python
observed = distances[:, 1:].reshape(-1)
```

THE CUT IS A QUANTILE OF THE DISTANCES ACTUALLY OBSERVED, so it is a number this screen produced rather than one carried in from another.

### lines 2254-2256

```python
share = crossed / float(max(1, reached))
```

A LABEL THAT CROSSED A WELL BOUNDARY is the one to watch: it says the nearest cell in feature space was in another well, which is either the phenotype repeating or the plate showing through.
