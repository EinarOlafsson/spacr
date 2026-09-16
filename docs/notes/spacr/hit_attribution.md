# Notes from `spacr/hit_attribution.py`

Prose lifted out of `spacr/hit_attribution.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [build_hit_cell_frame](#build_hit_cell_frame) (1 entry)
- [crossfit_candidate_probabilities](#crossfit_candidate_probabilities) (1 entry)
- [_refitted_permutation_p_values](#_refitted_permutation_p_values) (1 entry)

## build_hit_cell_frame

### lines 242-243  _(unsure)_

```python
frame["candidate_for_review"] = frame["target_guide_fraction"] > 0
```

High is always more hit-like. For a negative hit the descending=False rank above gives the lowest raw score the highest percentile.

## crossfit_candidate_probabilities

### lines 619-620  _(unsure)_

```python
sample_weight[labels[train]] = 0.5 + np.sqrt(
```

Fractions modulate evidence among positive bags but are not treated as known cell-label proportions.

## _refitted_permutation_p_values

### lines 796-798

```python
continue
```

A sparse permutation can leave one training fold with only one bag class. It is an unidentified null draw, not evidence; omit it and report the completed count explicitly.
