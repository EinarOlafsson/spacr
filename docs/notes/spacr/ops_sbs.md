# Notes from `spacr/ops_sbs.py`

Prose lifted out of `spacr/ops_sbs.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [compensate_crosstalk](#compensate_crosstalk) (3 entries)
- [assign_reads_to_cells](#assign_reads_to_cells) (1 entry)

## compensate_crosstalk

### lines 191-194

```python
axes[channel, channel] = 1.0
```

No spot claims this channel. Leave its axis as the identity rather than inventing one: a base absent from this field is a fact about the field, and a fabricated axis would rotate every other read to accommodate it.

### lines 205-208

```python
return data
```

Two dyes indistinguishable in this field. Correcting with a pseudo-inverse would quietly produce confident nonsense, so the uncorrected values are returned and the caller's quality scores will show what happened.

### lines 210-212

```python
from .ops_accel import matmul
```

THE BIG MULTIPLY, on whatever hardware there is. `correction` is 4 x 4 and `flat` is every spot of every cycle, so this is the one step in the chain whose cost scales with the plate.

## assign_reads_to_cells

### lines 356-359

```python
if sum(1 for value in counts.values() if value == agreeing) > 1:
```

A TIE IS NOT A WINNER. `max` picks one arbitrarily, so a cell whose reads split evenly between two barcodes would be assigned on dictionary order -- which is exactly the silent misassignment the fraction test exists to prevent.
