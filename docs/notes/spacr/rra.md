# Notes from `spacr/rra.py`

Prose lifted out of `spacr/rra.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_rho](#_rho) (1 entry)
- [rank_aggregate](#rank_aggregate) (2 entries)

## _rho

### lines 64-68

```python
scores = np.where(sorted_ranks <= alpha, scores, 1.0)
```

OUTSIDE THE TOP alpha IS NOT CONSIDERED, and 1.0 is how a minimum ignores it: a probability can never exceed 1, so a masked position can never be the minimum unless every position is masked -- which happens only when a gene has no guide in the top alpha at all, and rho = 1 is then the right answer rather than a missing value.

## rank_aggregate

### lines 154-156

```python
order = np.argsort(score if tail == "neg" else -score, kind="stable")
```

A HIGH SCORE IS THE WORST RANK FOR DEPLETION and the best for enrichment; `argsort` of the negated score is the whole difference between the two tails.

### lines 171-173

```python
p[index] = (1.0 + np.sum(null <= rhos[index])) / (null.size + 1.0)
```

+1 IN BOTH PLACES. A permutation P value of exactly zero claims a precision the permutation count does not have, and it is the value that survives an FDR correction to become a "finding".
