# Notes from `spacr/group_lasso.py`

Prose lifted out of `spacr/group_lasso.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [fit](#fit) (2 entries)
- [choose_lambda](#choose_lambda) (2 entries)

## fit

### lines 139-141

```python
steps = []
```

THE STEP SIZE PER BLOCK is 1 / L where L is the block's largest squared singular value -- the exact Lipschitz constant of that block's gradient, so the proximal step is a descent step without any line search.

### lines 158-159

```python
partial = residual + sub @ current
```

Add this block back before its own gradient step, which is what makes this coordinate descent rather than one global step.

## choose_lambda

### lines 297-298

```python
floor *= 0.01
```

NOTHING ON THE PATH SAID ANYTHING. Reach a further decade down rather than handing back a penalty already known to be empty.

### lines 300-303

```python
return smallest
```

Even four decades below the ceiling selects nothing. The smallest penalty tried is the one with any chance, and the caller's own guard will refuse the fit if it still says nothing -- which is the honest outcome for a design with no group signal in it at all.
