# Notes from `spacr/permutation_qc.py`

Prose lifted out of `spacr/permutation_qc.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## position_effect

### lines 91-100

```python
within = total - between
```

ETA-SQUARED ALONE CANNOT BE COMPARED TO A FIXED THRESHOLD, and doing so was this module's first bug: under the null it has an expected value of about (k-1)/(n-1), so with twelve levels pure noise scores 0.046 and any tolerance near 0.05 flags it. Caught immediately, on the control case that was supposed to pass.

THE F TEST IS THE COMPARISON THAT KNOWS THIS. It divides the between -level variance by the within-level variance with the right degrees of freedom, so "how many levels" is already accounted for and the answer is a p-value that means the same thing at any k.

### lines 111-113

```python
"omega_squared": float(max(
```

UNBIASED, so it can be reported beside the p-value without contradicting it: omega-squared subtracts the variance the null would have produced anyway.
