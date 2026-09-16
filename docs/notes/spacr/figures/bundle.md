# Notes from `spacr/figures/bundle.py`

Prose lifted out of `spacr/figures/bundle.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [statistics_rows](#statistics_rows) (2 entries)
- [save](#save) (1 entry)

## statistics_rows

### lines 71-76

```python
state = "holds" if assumption.passed else "does not hold"
```

READ THE CHECK'S OWN VERDICT (`passed`), never re-derive it from the p-value. The normality check compares the worst of k groups against a BONFERRONI threshold, and a caller re-deriving `p_value >= 0.05` silently discards the correction -- which in this codebase sent 18% of four-group comparisons on normal data to a rank test instead of 5%.

### lines 79-81

```python
state = "could not tell"
```

A CHECK THAT COULD NOT SEE IS NOT A CHECK THAT PASSED, and a file recording it as "holds" would be the more misleading of the two.

## save

### lines 187-190

```python
from ..tabular import write_table
```

`write_table`, not `to_csv`: the point of the bundle is that the numbers behind a figure can be read back and recomputed, and a well column spelled three ways across the codebase is the first thing that stops. Canonical on the way out is what makes the pair re-readable.
