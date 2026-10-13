# Notes from `spacr/sudoku.py`

Prose lifted out of `spacr/sudoku.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [similarity_graph](#similarity_graph) (2 entries)
- [constrain_to_fractions](#constrain_to_fractions) (2 entries)
- [sudoku](#sudoku) (4 entries)
- [sudoku_all](#sudoku_all) (3 entries)

## similarity_graph

### line 184  _(unsure)_

```python
distances, indices = distances[:, 1:], indices[:, 1:]
```

Column 0 is the cell itself.

### lines 197-198  _(unsure)_

```python
graph = graph.minimum(graph.T)
```

Both directions present: the elementwise minimum of W and W.T is zero wherever one direction is missing.

## constrain_to_fractions

### lines 265-267  _(unsure)_

```python
def constrain_to_fractions(mass: np.ndarray,
```

4. the well constraint -- the sudoku step

### lines 314-315

```python
empty = block.sum(axis=1) <= 0
```

A cell no anchor reached gets the well's prior rather than zero: it is a cell, it carries something, and "no idea" is the answer.

## sudoku

### line 411

```python
reach = mass.sum(axis=1)
```

THE TWO SCORES, BEFORE ANY NORMALISATION.

### lines 417-419

```python
typical = float(np.median(reach[reach > 0])) if np.any(reach > 0) else 0.0
```

Reach relative to the typical cell: an absolute cut-off would depend on the graph's size and its edge weights, which are not the user's to reason about.

### line 428, trailing  _(unsure)_

```python
called.append(ABSTAIN)
```

unlike every anchor

### line 432, trailing  _(unsure)_

```python
called.append(ABSTAIN)
```

a coin flip

## sudoku_all

### lines 508-517

```python
here = sudoku(values[live], np.asarray(scores)[live],
```

EVERY GUIDE IN THE RUN, ONE GUIDE COMMITTED. Running this with `[guide]` alone is degenerate and was, briefly, the bug the benchmark caught: with ONE column, `constrain_to_fractions` normalises each row over a single guide, so every posterior is exactly 1.0, every cell clears the decision bar, and the first guide claims the entire screen. It scored at the null.

A posterior is a COMPARISON. Comparing a guide against nothing returns the prior, which is the same lesson `attribute_well` records for its own per-well call.

### lines 528-530

```python
if here.guides[position] == guide:
```

Only THIS guide's cells are claimed this round. The others were computed to make the comparison honest and are left for their own round, when the pool they compete over is smaller.

### lines 538-540

```python
break
```

THE STOPPING RULE, REACHED NOT COUNTED. Nothing this round cleared the decision bar, so nothing later will either -- the pool only shrinks and the guides only get less confident.
