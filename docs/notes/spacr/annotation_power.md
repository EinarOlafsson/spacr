# Notes from `spacr/annotation_power.py`

Prose lifted out of `spacr/annotation_power.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [annotatable](#annotatable) (1 entry)
- [screen_size_for](#screen_size_for) (2 entries)
- [_specificity_for](#_specificity_for) (1 entry)

## annotatable

### lines 123-125

```python
cells_reachable += int(round(size * best))
```

An upper bound on the cells this well can yield: the largest clearing guide's share of it. Generous on purpose -- it is a CEILING, and a ceiling that flattered would be worthless.

## screen_size_for

### lines 192-193

```python
needed_per_well = float(1.0 / floor) if floor > 0 else float("inf")
```

For a typical guide to clear the floor its share must be at least `floor`, and in a well of `k` guides the typical share is about 1/k.

### line 195  _(unsure)_

```python
needed_wells = (library * coverage / needed_per_well
```

Same library, same wells-per-guide, fewer guides in each well.

## _specificity_for

### line 228  _(unsure)_

```python
false_positive = pi * se * (1.0 - t) / (t * (1.0 - pi))
```

Solve P(g|+) = t for (1 - sp).
