# Notes from `spacr/sp_stats.py`

Prose lifted out of `spacr/sp_stats.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [choose_p_adjust_method](#choose_p_adjust_method) (6 entries)
- [perform_normality_tests](#perform_normality_tests) (3 entries)
- [perform_statistical_tests](#perform_statistical_tests) (2 entries)
- [perform_posthoc_tests](#perform_posthoc_tests) (4 entries)
- [chi_pairwise](#chi_pairwise) (6 entries)

## Module level

### lines 25-32

```python
_ENGINE_TEST_NAMES = {
```

The engine names the tests for a figure legend; this module's callers write them into screen CSVs and have done since before the engine existed. Mapped rather than renamed, so every existing reader of ``Test Name`` keeps working and mapped onto the spelling ``spacrGraph`` already prints, so the two report vocabularies in this package agree. Pinned by tests/test_one_engine_decides_which_test_applies.py, which asserts that every test the engine can run is named here: a test the engine learns must not arrive in a CSV under a name nobody chose.

## choose_p_adjust_method

### line 69, trailing  _(unsure)_

```python
num_comparisons = (num_groups * (num_groups - 1)) // 2
```

Number of pairwise comparisons

### line 71  _(unsure)_

```python
if num_comparisons <= 10 and num_data_points > 5:
```

Decision logic for choosing the adjustment method

### line 73, trailing  _(unsure)_

```python
return 'holm'
```

Balanced between power and Type I error control

### line 75, trailing  _(unsure)_

```python
return 'fdr_bh'
```

FDR control for large number of comparisons and small sample size

### line 77, trailing  _(unsure)_

```python
return 'sidak'
```

Less conservative than Bonferroni, good for independent comparisons

### line 79, trailing  _(unsure)_

```python
return 'bonferroni'
```

Very conservative, use for strict control of Type I errors

## perform_normality_tests

### line 121  _(unsure)_

```python
print(f"Skipping normality test for group '{group}' on column '{column}' - Not enough data.")
```

Shapiro-Wilk needs three points to have a statistic at all.

### lines 148-150

```python
column_verdicts.append(
```

The verdict is the engine's own, taken across the groups together never re-derived from the per-group p-values above, because that would throw away the Bonferroni correction the check applies.

### lines 154-155

```python
is_normal = bool(column_verdicts) and all(column_verdicts)
```

No column examined is not evidence of normality. `all([])` is True, and returning True there would license a parametric test off an empty call.

## perform_statistical_tests

### line 224, trailing  _(unsure)_

```python
continue
```

Extend as needed

### lines 231-234

```python
test_results.append({
```

Fewer than two groups, or a group too small to test. Refusing is the engine's design: a comparison that could not be made is not a comparison with an unknown answer. Reported as a row rather than raised, because the caller is usually writing a CSV per column.

## perform_posthoc_tests

### line 286, trailing  _(unsure)_

```python
num_data_points = len(df[data_column].dropna()) // num_groups
```

Assuming roughly equal data points per group

### line 290  _(unsure)_

```python
tukey_result = pairwise_tukeyhsd(df[data_column], df[grouping_column], alpha=0.05)
```

Tukey's HSD automatically adjusts p-values

### line 295, trailing  _(unsure)_

```python
'Original p-value': None,
```

Tukey HSD does not provide raw p-values

### line 306, trailing  _(unsure)_

```python
if i < j:
```

Only consider unique pairs

## chi_pairwise

### line 349, trailing  _(unsure)_

```python
raw_p_values = []
```

Store raw p-values for correction later

### line 353, trailing  _(unsure)_

```python
num_data_points = raw_counts.sum(axis=1).mean()
```

Average total data points per group

### lines 365-366  _(unsure)_

```python
kept = pair.loc[:, (pair != 0).any(axis=0)]
```

A category neither group observed contributes nothing to this pair and is exactly what makes the expected frequency zero.

### line 389, trailing  _(unsure)_

```python
if contingency_table.shape[1] == 2:
```

Fisher's Exact Test for 2x2 tables

### line 392, trailing  _(unsure)_

```python
else:
```

Chi-Square Test for larger tables

### lines 405-407

```python
raw = np.asarray(raw_p_values, dtype=float)
```

Apply p-value correction over the pairs that were actually testable. Correcting across untestable pairs would inflate the family size and penalise the real comparisons for tests that never ran.
