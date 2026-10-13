# Notes from `spacr/figures/stats.py`

Prose lifted out of `spacr/figures/stats.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [check_normality](#check_normality) (3 entries)
- [check_equal_variance](#check_equal_variance) (2 entries)
- [_hedges_g](#_hedges_g) (1 entry)
- [compare](#compare) (1 entry)
- [_run](#_run) (1 entry)
- [_difference_ci](#_difference_ci) (1 entry)
- [table](#table) (1 entry)

## check_normality

### lines 218-223

```python
flat = [group for group in groups if float(np.ptp(group)) == 0.0]
```

A GROUP WITH NO SPREAD IS NOT A NORMAL GROUP. scipy hands back p = 1.0 and a NaN statistic for constant input rather than raising, and 1.0 read as a p-value says "as consistent with normal as data gets" -- so a column where every object in an arm was called class 0 used to PASS the normality check and license a parametric test. Real input in this package, not a contrived one.

### lines 232-234

```python
worst_p, worst_stat, tested = float("inf"), float("nan"), 0
```

Start above 1.0 so the first group always records its statistic. Starting AT 1.0 meant a group whose p came back exactly 1.0 never updated `worst_stat`, and the result reported a NaN statistic beside a real p.

### lines 250-261

```python
threshold = 0.05 / max(tested, 1)
```

THE MINIMUM OF k TESTS IS NOT A p-VALUE.

Taking the worst group and comparing it to 0.05 tests normality k times and reports the most extreme, which is a multiple-comparison problem in the assumption check itself: with four normal groups of 40 there is a ~19% chance the worst one falls below 0.05 by luck, and the whole comparison then flips to a rank test on data that was fine.

Bonferroni across the groups. Conservative in the direction that matters -- it makes "not normal" harder to claim, and the cost of wrongly claiming it is only a little power, while the cost of the opposite is a parametric test on data that does not support one.

## check_equal_variance

### lines 292-294

```python
with np.errstate(invalid="ignore", divide="ignore"):
```

Levene's denominator is zero when every group is constant. The NaN it produces is handled below, so the numpy warning on the way there is noise in a caller's console, not information.

### lines 301-306

```python
if not np.isfinite(p):
```

NaN IS NOT A SMALL p. Levene returns NaN rather than raising when every group is constant (its denominator is zero), and `nan >= 0.05` is False, so the check used to write "variances differ (p < 0.05)" into a results table on the strength of a number that does not exist. The branch it picks is the safe one either way; the sentence a reviewer reads was a false statement.

## _hedges_g

### lines 335-336  _(unsure)_

```python
return d * (1 - 3 / (4 * total - 9)), "Hedges' g"
```

Hedges' correction. On the replicate counts this field actually uses, Cohen's d is biased upward by several percent.

## compare

### lines 391-394

```python
normal = normality.passed
```

READ THE CHECK'S OWN VERDICT. Re-deriving it here from `p_value >= 0.05` is what discarded the Bonferroni correction the normality check applies across groups, and sent 18% of four-group comparisons on perfectly normal data to a rank test instead of 5%.

## _run

### lines 457-460

```python
pooled = np.concatenate((arrays[0], arrays[1]))
```

SciPy 1.18 returns NaN from the asymptotic tie correction when the pooled sample is entirely constant.  The two empirical distributions are identical in that case: every pair is a tie, U is half of n1*n2, and the two-sided p-value is exactly 1.

## _difference_ci

### lines 512-515

```python
if not np.isfinite(se) or se == 0:
```

Bail on a degenerate spread BEFORE the degrees of freedom are computed. With zero variance the Welch df is 0/0, which prints a RuntimeWarning on the way to a number this function is about to discard -- and two constant arms is a real case, not a contrived one.

## table

### lines 571-572

```python
key = "".join(ch for ch in assumption.name.split()[0].lower()
```

A column name that goes into a CSV a reviewer will open:

"shapiro", not "Shapiro-Wilk" or "shapiro-wilk".
