# Notes from `spacr/guide_concordance.py`

Prose lifted out of `spacr/guide_concordance.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## guide_support

### lines 114-116

```python
direction = np.sign(np.mean(finite)) or 1.0
```

Sign of the MEAN, not of the strongest guide: asking whether the guides agree with each other is the question, and letting the largest one define "correct" would make disagreement invisible.

### lines 125-127

```python
"n_guides_significant": int(np.nansum(p_values <= alpha)),
```

`nansum` over a comparison is safe -- NaN <= alpha is False so this counts zero significant guides, which is correct when there are no p values to be significant.

### lines 135-143

```python
"best_guide_p": _best_p(p_values),
```

`nanmin` OF AN ALL-NaN SLICE WARNS AND RETURNS NaN, and every gene warns separately -- three lines of RuntimeWarning per run on a screen where the answer is simply "there are no guide p values". THE ABSENCE IS EXPECTED, NOT EXCEPTIONAL: a mixed model makes the guide a RANDOM effect, so each guide gets a shrunken BLUP and no p value at all, which the run already says in words. Asking the question and reporting NaN quietly is the honest response; warning about it is noise that hides real warnings.
